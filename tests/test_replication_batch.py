from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from theseus.replication_batch import (
    DEFAULT_MAX_BATCH_BYTES,
    DEFAULT_MAX_BATCH_EVENTS,
    NO_REASON_GIVEN,
    BatchRejected,
    parse_batch,
)
from theseus.replication_events import MAX_REASON_CHARS
from theseus.stimulus_log import StimulusEvent


def line(n: int, origin: str = "kitchen-surrogate", **overrides) -> str:
    fields = {
        "id": f"01PRODUCERID{n:014d}",
        "ts": "2026-09-04T16:00:00+00:00",
        "actor": "sensor",
        "type": "observation",
        "content": {"n": n},
        "origin": origin,
        "seq": n,
    }
    fields.update(overrides)

    return json.dumps(fields)


def body(*lines: str) -> str:
    return "\n".join(lines) + "\n"


HOST = "local"


def parse(text, **kwargs):
    """`parse_batch` on behalf of a host whose own origin is `local`.

    `host_origin` is required by the parser and is the same value in almost every test, so it
    is defaulted here rather than repeated thirty times. A test that cares — the one about a
    batch claiming the host's own name — passes it explicitly, and it still wins.
    """
    kwargs.setdefault("host_origin", HOST)
    return parse_batch(text, **kwargs)


def test_a_normal_batch_parses_into_events():
    events = parse(body(line(1), line(2), line(3)))

    assert [e.seq for e in events] == [1, 2, 3]
    assert {e.origin for e in events} == {"kitchen-surrogate"}
    assert events[0].ts == datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc)


def test_a_trailing_newline_is_optional():
    assert len(parse(line(1) + "\n" + line(2))) == 2


def test_blank_lines_are_ignored():
    """A JSONL producer that ends with a blank line has not malformed anything."""
    assert len(parse(line(1) + "\n\n" + line(2) + "\n\n")) == 2


def test_an_empty_body_is_rejected():
    """Nothing to commit is not the same as a batch, and answering 2xx would advance the
    surrogate's cursor past events it never sent."""
    with pytest.raises(BatchRejected) as caught:
        parse("   \n\n")

    assert caught.value.status == 400


def test_a_line_that_is_not_json_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse(body(line(1), "{not json", line(3)))

    assert caught.value.status == 400
    assert "line 2" in caught.value.reason


def test_a_line_missing_a_required_field_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse(body(line(1).replace('"actor": "sensor", ', "")))

    assert caught.value.status == 400


def test_a_line_without_a_seq_is_rejected():
    """Read-side validation: `replication_events` validates only what this node builds, so
    the ingress re-applies the rules to anything off the wire."""
    with pytest.raises(BatchRejected) as caught:
        parse(body(line(1, seq=None)))

    assert caught.value.status == 400
    assert "seq" in caught.value.reason


@pytest.mark.parametrize("seq", ["7", 0, -3, True, 1.5])
def test_a_seq_that_is_not_a_positive_integer_is_rejected(seq):
    with pytest.raises(BatchRejected) as caught:
        parse(body(line(1, seq=seq)))

    assert caught.value.status == 400


@pytest.mark.parametrize("origin", [None, "", "   ", 42])
def test_a_line_without_a_real_origin_is_rejected(origin):
    with pytest.raises(BatchRejected) as caught:
        parse(body(line(1, origin=origin)))

    assert caught.value.status == 400


def test_a_batch_spanning_two_origins_is_rejected():
    """The brief's batch comes from exactly one producer. Two would make one
    all-or-nothing write span two dedupe streams."""
    with pytest.raises(BatchRejected) as caught:
        parse(body(line(1), line(2, origin="android-01")))

    assert caught.value.status == 400
    assert "one origin" in caught.value.reason


def test_seqs_must_ascend():
    with pytest.raises(BatchRejected) as caught:
        parse(body(line(2), line(1)))

    assert caught.value.status == 400
    assert "ascending" in caught.value.reason


def test_a_repeated_seq_in_one_batch_is_rejected():
    """Two events claiming one seq makes the high-water mark ambiguous about which was
    committed."""
    with pytest.raises(BatchRejected) as caught:
        parse(body(line(1), line(1)))

    assert caught.value.status == 400


def test_a_gap_inside_a_batch_is_accepted():
    """Deliberately *not* contiguous. A surrogate that evicted seqs under storage pressure
    holds a buffer with real holes; rejecting it would tell the surrogate to abandon data it
    still has, which is the opposite of what the abandon rule is for."""
    events = parse(body(line(1), line(2), line(90)))

    assert [e.seq for e in events] == [1, 2, 90]


def test_too_many_events_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse(body(*[line(n) for n in range(1, 5)]), max_events=3)

    assert caught.value.status == 413
    assert "3" in caught.value.reason


def test_too_many_bytes_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse(body(line(1), line(2)), max_bytes=50)

    assert caught.value.status == 413


def test_the_byte_limit_is_measured_on_the_encoded_body():
    """A limit that counted characters would let a multi-byte payload through at several
    times the size the host meant to accept."""
    fat = parse  # alias for readability
    with pytest.raises(BatchRejected):
        fat(body(line(1, content={"m": "é" * 200})), max_bytes=300)


def test_limits_default_to_the_module_constants():
    assert DEFAULT_MAX_BATCH_EVENTS > 0
    assert DEFAULT_MAX_BATCH_BYTES > 0
    assert len(parse(body(line(1)))) == 1


def test_a_rejection_carries_a_reason_short_enough_to_record():
    """The surrogate copies this into a `replication.batch_rejected` event, which is bounded
    at `MAX_REASON_CHARS` and lands on a permanent tape."""

    with pytest.raises(BatchRejected) as caught:
        parse(body(line(1), "{not json"))

    assert 0 < len(caught.value.reason) <= MAX_REASON_CHARS


def test_bytes_and_str_bodies_behave_the_same():
    text = body(line(1), line(2))

    assert [e.seq for e in parse(text)] == [
        e.seq for e in parse(text.encode("utf-8"))
    ]


def test_a_line_deep_enough_to_exhaust_the_stack_is_a_4xx_not_a_crash():
    """20,000 levels is 40 KB — one percent of the byte limit — and `json.loads` recurses
    once per level. `RecursionError` is not a `ValueError`, so it is not a `JSONDecodeError`
    and the obvious handler misses it. Escaping here means the endpoint answers 500, the
    surrogate reads that as transient, and it resends this batch forever."""
    depth = 20_000
    body = line(1).replace('"content": {"n": 1}', '"content": ' + "[" * depth + "]" * depth)
    assert len(body) < DEFAULT_MAX_BATCH_BYTES

    with pytest.raises(BatchRejected) as excinfo:
        parse(body)

    assert excinfo.value.status == 400
    assert "nests too deeply" in excinfo.value.reason


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_a_line_separator_inside_a_value_does_not_split_the_line(separator):
    """Not hypothetical: the line is built by the same `to_json` a Theseus surrogate runs.
    It serialises with `ensure_ascii=False`, so these three reach the wire raw, and
    `str.splitlines()` breaks on all three where `split("\\n")` does not. Tearing one line
    into two invalid halves answers 400 — do not retry — so a transcript containing a line
    separator would be discarded permanently, silently, and only for that content.

    The body must come from `to_json`, **not** from this module's `line()` helper:
    `json.dumps` defaults to `ensure_ascii=True` and would escape the separator, leaving
    nothing for `splitlines` to split — and the test would then pass against its own
    reverted fix, which is worse than not having it."""
    message = f"before{separator}after"
    event = StimulusEvent(
        id="01PRODUCERID00000000000001",
        ts=datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc),
        actor="george",
        type="exchange",
        content={"message": message},
        origin="kitchen-surrogate",
        seq=1,
    )
    wire = event.to_json()
    assert separator in wire  # the precondition: raw on the wire, not escaped
    assert len(wire.splitlines()) == 2  # and `splitlines` would indeed tear it

    events = parse(wire + "\n")

    assert len(events) == 1
    assert events[0].content == {"message": message}


def test_a_body_that_is_not_utf8_is_rejected_rather_than_repaired():
    """`errors="replace"` would write the replacement character onto an append-only tape,
    where nothing downstream can tell it from content the producer meant. Bytes that are not
    UTF-8 will not become UTF-8 by being resent."""
    body = line(1).encode("utf-8").replace(b"sensor", b"sen\xffor")

    with pytest.raises(BatchRejected) as excinfo:
        parse(body)

    assert excinfo.value.status == 400
    assert "UTF-8" in excinfo.value.reason


def test_a_seq_above_the_ceiling_is_rejected():
    """A mark of 10**100 is not a counter; accepting it discards that origin's entire
    future, because nothing it ever sends again clears the high-water mark."""
    with pytest.raises(BatchRejected, match="ceiling"):
        parse(line(1, seq=10**100))


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"content": None}, "non-object content"),
        ({"content": "just a string"}, "non-object content"),
        ({"type": None}, "no usable type"),
        ({"type": ""}, "no usable type"),
        ({"id": {}}, "no usable id"),
        ({"actor": 7}, "no usable actor"),
        ({"ts": "not a timestamp"}, "unparseable ts"),
        ({"ts": None}, "no usable ts"),
    ],
)
def test_the_rest_of_the_envelope_is_checked_too(overrides, expected):
    """`from_json` indexes these straight out of the parsed dict, so without a check they
    reach the tape as a `None` type or a dict id."""
    with pytest.raises(BatchRejected, match=expected):
        parse(line(1, **overrides))


def test_a_naive_ts_is_rejected_rather_than_read_in_the_hosts_zone():
    """`_aware` attaches the *host's* zone to a naive timestamp — right for an old local
    line, wrong for a surrogate elsewhere, and it moves the event hours from where it
    belongs in the Assembler's sort."""
    with pytest.raises(BatchRejected, match="no UTC offset"):
        parse(line(1, ts="2026-09-04T16:00:00"))


def test_a_batch_claiming_the_hosts_own_origin_is_a_4xx():
    """Without this it surfaces as a ValueError out of `append_many`, which the endpoint
    answers as a 5xx and the surrogate retries forever — a misconfiguration no retry fixes."""
    with pytest.raises(BatchRejected) as excinfo:
        parse(line(1, origin="local"), host_origin="local")

    assert excinfo.value.status == 400
    assert "own origin" in excinfo.value.reason


def test_batch_rejected_marks_a_truncated_reason():
    """The reason is copied verbatim onto the surrogate's tape. A reader that cannot tell
    "the host said exactly this" from "the host said this and more" is being misled."""
    rejected = BatchRejected(400, "x" * 5000)

    assert len(rejected.reason) == MAX_REASON_CHARS
    assert rejected.reason.endswith("…")


MAX_CONTENT_DEPTH_BOUNDARY = 99  # the deepest `nested()` the parser must still accept


def nested(depth: int) -> str:
    """One line whose `content` holds a list nested `depth` deep.

    `content` itself is level 1, so the innermost value sits at level `depth + 1` — which is
    why the accepted/rejected boundary below lands on 99/100 rather than 100/101.
    """
    return line(1).replace(
        '"content": {"n": 1}', '"content": {"m": ' + "[" * depth + "]" * depth + "}"
    )


def commit(body, tmp_path):
    """Parse a body and append it, the way the ingress will.

    Returns the rejection status, or None if the batch committed. Anything that is neither —
    an exception escaping either call — fails the test, because that is the `5xx` the whole
    module exists to prevent.
    """
    from theseus.stimulus_log import StimulusLog

    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    try:
        events = parse(body, host_origin=log.origin)
    except BatchRejected as rejected:
        return rejected.status
    log.append_many(events)
    return None


# --- The contract: anything the parser accepts, the log can commit ---------------
def test_a_lone_surrogate_in_content_is_a_4xx_not_a_crash(tmp_path):
    """`\\ud800` is six ASCII characters on the wire, legal JSON grammar, and valid UTF-8 —
    so it passes every field check. It has no UTF-8 encoding, so the log's own `to_json`
    raises `UnicodeEncodeError` at write time. Escaping means the endpoint answers 500, the
    surrogate reads that as transient, and it resends a 145-byte body forever."""
    body = line(1).replace('"content": {"n": 1}', '"content": {"m": "\\ud800"}')

    assert commit(body, tmp_path) == 400


@pytest.mark.parametrize("depth", [9993, 9994, 9995])
def test_the_band_where_dumps_gives_out_before_loads_is_a_4xx(tmp_path, depth):
    """`json.loads` and `json.dumps` share one C-stack budget and do not spend it
    identically, so there is a band — measured at 9993-9995, a 20 KB body under half a
    percent of the byte limit — where the batch parses and the write raises. The older
    deep-nesting test uses depth 20,000, past the band, which is how this survived."""
    assert commit(nested(depth), tmp_path) == 400


def test_content_at_the_depth_limit_is_accepted(tmp_path):
    """The bound has to be a bound, not a wall a legitimate payload hits."""
    assert commit(nested(MAX_CONTENT_DEPTH_BOUNDARY), tmp_path) is None


def test_content_past_the_depth_limit_is_rejected_at_a_fixed_bound():
    """Refused at a fixed depth rather than wherever CPython's stack happens to give out.
    The stack limit moves with how much stack is left when the call happens, and a bound
    that moves is not a bound a surrogate can be held to."""
    with pytest.raises(BatchRejected, match="deeper than"):
        parse(nested(MAX_CONTENT_DEPTH_BOUNDARY + 1))


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_float_in_content_is_rejected(token, tmp_path):
    """`json.loads` accepts these bare tokens and `json.dumps` re-emits them, so the line
    lands on the tape as something no JSON reader outside Python will parse — forever, on an
    append-only file. Python's own `read_all` round-trips it, which is what makes it
    dangerous: nothing here would notice."""
    body = line(1).replace('"content": {"n": 1}', '"content": {"m": ' + token + "}")

    assert commit(body, tmp_path) == 400


def test_everything_the_parser_accepts_can_actually_be_committed(tmp_path):
    """The module's contract in one test. Each body below satisfies every field rule; the
    ones that are unwritable must be rejected, and the ones that are writable must commit —
    but nothing may escape either call."""
    bodies = [
        line(1),
        nested(5),
        nested(MAX_CONTENT_DEPTH_BOUNDARY),
        nested(MAX_CONTENT_DEPTH_BOUNDARY + 1),
        nested(9994),
        nested(20_000),
        line(1).replace('"content": {"n": 1}', '"content": {"m": "\\ud800"}'),
        line(1).replace('"content": {"n": 1}', '"content": {"m": NaN}'),
    ]

    outcomes = [commit(body, tmp_path / str(n)) for n, body in enumerate(bodies)]

    assert outcomes == [None, None, None, 400, 400, 400, 400, 400]


# --- host_origin is not optional ------------------------------------------------
def test_host_origin_is_required():
    """There is no caller for whom "do not check" is the right answer. Optional, it was one
    forgotten keyword argument away from the 5xx-retry-forever it exists to prevent."""
    with pytest.raises(TypeError):
        parse_batch(body(line(1)))


# --- The origin has to be the name it looks like --------------------------------
@pytest.mark.parametrize("origin", [" local ", "local ", " local", "\tlocal"])
def test_an_origin_padded_with_whitespace_is_rejected(origin):
    """Rejected rather than trimmed: trimming silently rewrites the name a producer chose,
    and accepting it as-is files those events under a second origin that reads identically
    to the first everywhere a human looks — on a permanent tape. It also walks straight past
    the host-origin guard, since `" local "` is not `"local"`."""
    with pytest.raises(BatchRejected, match="whitespace"):
        parse(line(1, origin=origin), host_origin="local")


# --- The body itself ------------------------------------------------------------
def test_a_body_that_is_neither_text_nor_bytes_is_a_4xx():
    """`bytearray` has no `.encode`. A caller-shape mistake, but one that would otherwise
    escape as an `AttributeError` before any try block."""
    with pytest.raises(BatchRejected, match="not usable text or bytes"):
        parse(bytearray(line(1), "utf-8"))


def test_a_str_body_carrying_a_lone_surrogate_is_a_4xx():
    with pytest.raises(BatchRejected, match="not usable text or bytes"):
        parse(line(1).replace("sensor", "sen\ud800or"))


# --- Constructing the do-not-retry signal must not fail -------------------------
def test_batch_rejected_replaces_an_unusable_reason_rather_than_raising():
    """A reason comes from data, and this exception is the signal that stops a poison batch.
    Raising while constructing it would escape as the 500 that creates one. Losing a little
    information in a case that should not arise beats losing the channel in one that might."""
    assert BatchRejected(400, "   ").reason == NO_REASON_GIVEN
    assert BatchRejected(400, None).reason == NO_REASON_GIVEN
    assert BatchRejected(400, 7).reason == NO_REASON_GIVEN


def test_batch_rejected_still_refuses_a_status_outside_the_4xx_class():
    """The status is the opposite case: it is a literal at every call site, so a wrong one
    is a programmer error that must be loud rather than data that must be tolerated."""
    for status in (200, 500, 503):
        with pytest.raises(ValueError, match="4xx"):
            BatchRejected(status, "whatever")


# --- The writability contract, continued ----------------------------------------
@pytest.mark.parametrize(
    "ts",
    [
        "0001-01-01T00:00:00+01:00",
        "0001-01-01T00:00:00+00:01",
        "0001-01-01T00:00:00+05:30",
        "9999-12-31T23:59:59-05:00",
    ],
)
def test_a_timestamp_that_cannot_be_converted_to_utc_is_a_4xx(ts, tmp_path):
    """`to_json` raises `OverflowError` for any `ts` whose UTC conversion leaves the
    `datetime` range. `OverflowError` is an `ArithmeticError`, not a `ValueError`, so it slips
    past the obvious handler and escapes as a 500 — the surrogate reads transient and retries
    a 162-byte body forever.

    Not only adversarial. .NET's `DateTime.MinValue` is `0001-01-01T00:00:00`, so an
    uninitialised timestamp from a Windows surrogate anywhere east of Greenwich is exactly
    this, and the Phase-1 target surrogate is a Windows desktop."""
    assert commit(line(1, ts=ts), tmp_path) == 400


def test_a_negative_offset_at_year_one_still_works(tmp_path):
    """The bound is real, not a blanket ban on old timestamps: it is the UTC conversion that
    overflows, so the other direction is fine."""
    assert commit(line(1, ts="0001-01-01T00:00:00-08:00"), tmp_path) is None


# --- Markers off the wire -------------------------------------------------------
def gap_line(n=1, **content_overrides):
    content = {
        "origin": "kitchen-surrogate",
        "from_seq": 5,
        "to_seq": 9,
        "reason": "storage_pressure",
        "span_start": "2026-09-04T16:00:00+00:00",
        "span_end": "2026-09-04T16:05:00+00:00",
        "declared": True,
    }
    content.update(content_overrides)
    return line(n, type="stimulus.gap", content=content)


def rejection_line(n=1, **content_overrides):
    content = {
        "origin": "kitchen-surrogate",
        "from_seq": 5,
        "to_seq": 9,
        "status": 400,
        "reason": "malformed",
    }
    content.update(content_overrides)
    return line(n, type="replication.batch_rejected", content=content)


def test_a_well_formed_declared_gap_is_accepted():
    assert len(parse(gap_line())) == 1


def test_a_well_formed_rejection_marker_is_accepted():
    assert len(parse(rejection_line())) == 1


def test_a_surrogate_may_not_claim_the_hosts_word_for_an_undiagnosed_hole():
    """`inferred` is host-minted precisely so a hole the host *diagnosed* can be told apart
    from one a surrogate *reported*. A surrogate able to claim it erases the only distinction
    the gap vocabulary carries."""
    with pytest.raises(BatchRejected, match="only the host mints"):
        parse(gap_line(reason="inferred"))


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"reason": "because"}, "unknown gap reason"),
        ({"reason": None}, "unknown gap reason"),
        ({"declared": False}, "declared="),
        ({"declared": None}, "declared="),
        ({"origin": ""}, "no usable origin"),
        ({"origin": 7}, "no usable origin"),
        ({"from_seq": 9, "to_seq": 5}, "inverted range"),
        ({"from_seq": 0}, "seqs start at 1"),
        ({"to_seq": True}, "non-integer to_seq"),
        ({"span_start": "2026-09-04T17:00:00+00:00"}, "inverted span"),
        ({"span_start": "2026-09-04T16:00:00"}, "no UTC offset"),
        ({"span_end": "not a time"}, "unparseable span_end"),
        ({"span_end": 7}, "non-string span_end"),
    ],
)
def test_a_malformed_declared_gap_is_rejected(overrides, expected):
    """`replication_events` validates only markers *this* node builds; both modules'
    docstrings say the ingress must re-apply those rules to anything off the wire. Until it
    did, every one of these landed on a permanent tape."""
    with pytest.raises(BatchRejected, match=expected):
        parse(gap_line(**overrides))


@pytest.mark.parametrize("missing", ["origin", "from_seq", "to_seq", "reason", "span_start"])
def test_a_declared_gap_missing_a_field_is_rejected(missing):
    content = {k: v for k, v in {
        "origin": "kitchen-surrogate", "from_seq": 5, "to_seq": 9,
        "reason": "storage_pressure",
        "span_start": "2026-09-04T16:00:00+00:00",
        "span_end": "2026-09-04T16:05:00+00:00", "declared": True,
    }.items() if k != missing}

    with pytest.raises(BatchRejected, match=f"no {missing}"):
        parse(line(1, type="stimulus.gap", content=content))


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"status": 500}, "a 5xx was never a rejection"),
        ({"status": 200}, "a 5xx was never a rejection"),
        ({"status": "400"}, "non-integer status"),
        ({"reason": ""}, "no reason"),
        ({"reason": None}, "no reason"),
        ({"reason": "x" * 5000}, "over the 500 bound"),
        ({"from_seq": 9, "to_seq": 5}, "inverted range"),
    ],
)
def test_a_malformed_rejection_marker_is_rejected(overrides, expected):
    """A `5xx` is retried, not rejected — one recorded here would claim a batch was abandoned
    while the surrogate is still trying to send it. And the reason is a remote host's words
    being copied onto an append-only tape by way of the surrogate, so the same bound applies
    on the way in as on the way out."""
    with pytest.raises(BatchRejected, match=expected):
        parse(rejection_line(**overrides))


def test_an_ordinary_event_is_not_held_to_the_marker_schema():
    """Only the two marker types are checked. An observation whose content happens to carry a
    `status` or a `reason` is just an observation."""
    assert len(parse(line(1, content={"status": 500, "reason": "", "from_seq": 9}))) == 1
