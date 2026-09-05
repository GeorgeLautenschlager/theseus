from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from theseus.replication_batch import (
    DEFAULT_MAX_BATCH_BYTES,
    DEFAULT_MAX_BATCH_EVENTS,
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


def test_a_normal_batch_parses_into_events():
    events = parse_batch(body(line(1), line(2), line(3)))

    assert [e.seq for e in events] == [1, 2, 3]
    assert {e.origin for e in events} == {"kitchen-surrogate"}
    assert events[0].ts == datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc)


def test_a_trailing_newline_is_optional():
    assert len(parse_batch(line(1) + "\n" + line(2))) == 2


def test_blank_lines_are_ignored():
    """A JSONL producer that ends with a blank line has not malformed anything."""
    assert len(parse_batch(line(1) + "\n\n" + line(2) + "\n\n")) == 2


def test_an_empty_body_is_rejected():
    """Nothing to commit is not the same as a batch, and answering 2xx would advance the
    surrogate's cursor past events it never sent."""
    with pytest.raises(BatchRejected) as caught:
        parse_batch("   \n\n")

    assert caught.value.status == 400


def test_a_line_that_is_not_json_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1), "{not json", line(3)))

    assert caught.value.status == 400
    assert "line 2" in caught.value.reason


def test_a_line_missing_a_required_field_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1).replace('"actor": "sensor", ', "")))

    assert caught.value.status == 400


def test_a_line_without_a_seq_is_rejected():
    """Read-side validation: `replication_events` validates only what this node builds, so
    the ingress re-applies the rules to anything off the wire."""
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1, seq=None)))

    assert caught.value.status == 400
    assert "seq" in caught.value.reason


@pytest.mark.parametrize("seq", ["7", 0, -3, True, 1.5])
def test_a_seq_that_is_not_a_positive_integer_is_rejected(seq):
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1, seq=seq)))

    assert caught.value.status == 400


@pytest.mark.parametrize("origin", [None, "", "   ", 42])
def test_a_line_without_a_real_origin_is_rejected(origin):
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1, origin=origin)))

    assert caught.value.status == 400


def test_a_batch_spanning_two_origins_is_rejected():
    """The brief's batch comes from exactly one producer. Two would make one
    all-or-nothing write span two dedupe streams."""
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1), line(2, origin="android-01")))

    assert caught.value.status == 400
    assert "one origin" in caught.value.reason


def test_seqs_must_ascend():
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(2), line(1)))

    assert caught.value.status == 400
    assert "ascending" in caught.value.reason


def test_a_repeated_seq_in_one_batch_is_rejected():
    """Two events claiming one seq makes the high-water mark ambiguous about which was
    committed."""
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1), line(1)))

    assert caught.value.status == 400


def test_a_gap_inside_a_batch_is_accepted():
    """Deliberately *not* contiguous. A surrogate that evicted seqs under storage pressure
    holds a buffer with real holes; rejecting it would tell the surrogate to abandon data it
    still has, which is the opposite of what the abandon rule is for."""
    events = parse_batch(body(line(1), line(2), line(90)))

    assert [e.seq for e in events] == [1, 2, 90]


def test_too_many_events_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(*[line(n) for n in range(1, 5)]), max_events=3)

    assert caught.value.status == 413
    assert "3" in caught.value.reason


def test_too_many_bytes_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1), line(2)), max_bytes=50)

    assert caught.value.status == 413


def test_the_byte_limit_is_measured_on_the_encoded_body():
    """A limit that counted characters would let a multi-byte payload through at several
    times the size the host meant to accept."""
    fat = parse_batch  # alias for readability
    with pytest.raises(BatchRejected):
        fat(body(line(1, content={"m": "é" * 200})), max_bytes=300)


def test_limits_default_to_the_module_constants():
    assert DEFAULT_MAX_BATCH_EVENTS > 0
    assert DEFAULT_MAX_BATCH_BYTES > 0
    assert len(parse_batch(body(line(1)))) == 1


def test_a_rejection_carries_a_reason_short_enough_to_record():
    """The surrogate copies this into a `replication.batch_rejected` event, which is bounded
    at `MAX_REASON_CHARS` and lands on a permanent tape."""

    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1), "{not json"))

    assert 0 < len(caught.value.reason) <= MAX_REASON_CHARS


def test_bytes_and_str_bodies_behave_the_same():
    text = body(line(1), line(2))

    assert [e.seq for e in parse_batch(text)] == [
        e.seq for e in parse_batch(text.encode("utf-8"))
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
        parse_batch(body)

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

    events = parse_batch(wire + "\n")

    assert len(events) == 1
    assert events[0].content == {"message": message}


def test_a_body_that_is_not_utf8_is_rejected_rather_than_repaired():
    """`errors="replace"` would write the replacement character onto an append-only tape,
    where nothing downstream can tell it from content the producer meant. Bytes that are not
    UTF-8 will not become UTF-8 by being resent."""
    body = line(1).encode("utf-8").replace(b"sensor", b"sen\xffor")

    with pytest.raises(BatchRejected) as excinfo:
        parse_batch(body)

    assert excinfo.value.status == 400
    assert "UTF-8" in excinfo.value.reason


def test_a_seq_above_the_ceiling_is_rejected():
    """A mark of 10**100 is not a counter; accepting it discards that origin's entire
    future, because nothing it ever sends again clears the high-water mark."""
    with pytest.raises(BatchRejected, match="ceiling"):
        parse_batch(line(1, seq=10**100))


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
        parse_batch(line(1, **overrides))


def test_a_naive_ts_is_rejected_rather_than_read_in_the_hosts_zone():
    """`_aware` attaches the *host's* zone to a naive timestamp — right for an old local
    line, wrong for a surrogate elsewhere, and it moves the event hours from where it
    belongs in the Assembler's sort."""
    with pytest.raises(BatchRejected, match="no UTC offset"):
        parse_batch(line(1, ts="2026-09-04T16:00:00"))


def test_a_batch_claiming_the_hosts_own_origin_is_a_4xx():
    """Without this it surfaces as a ValueError out of `append_many`, which the endpoint
    answers as a 5xx and the surrogate retries forever — a misconfiguration no retry fixes."""
    with pytest.raises(BatchRejected) as excinfo:
        parse_batch(line(1, origin="local"), host_origin="local")

    assert excinfo.value.status == 400
    assert "own origin" in excinfo.value.reason


def test_host_origin_is_not_checked_unless_it_is_given():
    assert len(parse_batch(line(1, origin="local"))) == 1


def test_batch_rejected_refuses_a_status_outside_the_4xx_class():
    """This exception *is* the do-not-retry signal. Raising it with a 5xx would tell the
    surrogate to abandon a batch it should have retried, and nothing downstream could
    catch it."""
    for status in (200, 500, 503):
        with pytest.raises(ValueError, match="4xx"):
            BatchRejected(status, "whatever")


def test_batch_rejected_marks_a_truncated_reason():
    """The reason is copied verbatim onto the surrogate's tape. A reader that cannot tell
    "the host said exactly this" from "the host said this and more" is being misled."""
    rejected = BatchRejected(400, "x" * 5000)

    assert len(rejected.reason) == MAX_REASON_CHARS
    assert rejected.reason.endswith("…")


def test_batch_rejected_refuses_an_empty_reason():
    with pytest.raises(ValueError):
        BatchRejected(400, "   ")
