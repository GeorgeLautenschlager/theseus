"""Turning a replication request body into events, or refusing it.

This module owns the protocol's `4xx` class, and that class has a specific meaning: a `4xx`
tells the surrogate *do not retry*. It advances its cursor past the batch, emits a
`replication.batch_rejected` marker so the hole is visible on its tape, and moves on. So
everything answered here must be genuinely unfixable by sending it again — malformed,
oversized, or the wrong shape — and anything merely transient must never reach this module.

The class exists so one poison batch cannot wedge the channel forever. That is the whole of
its job: without it, a surrogate retries an unacceptable batch until its budget runs out
while everything behind it waits.

Validation here is the read-side counterpart to `replication_events`, whose constructors
validate only what *this* node builds. A remote surrogate goes through none of those, so the
rules are re-applied to anything arriving over the wire rather than a second definition of
well-formed being invented at the endpoint.

Pure: no I/O, no log, no marks, no FastAPI. The endpoint is the only thing that touches those.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from theseus.replication_events import (
    BATCH_REJECTED,
    DECLARED_REASONS,
    GAP,
    INFERRED_REASON,
    MAX_REASON_CHARS,
    clean_reason,
)
from theseus.stimulus_log import StimulusEvent

# A batch is bounded twice, because the two limits fail differently: a count keeps one
# commit from stalling the cognitive loop, and a byte size keeps a single event with a
# hundred-megabyte payload from doing the same with one line. Both are configurable; these
# are the defaults a LAN surrogate can rely on.
DEFAULT_MAX_BATCH_EVENTS = 500
DEFAULT_MAX_BATCH_BYTES = 4 * 1024 * 1024

# A seq is a counter, and a counter that arrives as 10**100 is not a counter — it is a mark
# that no real event can ever exceed, so accepting it discards that origin's whole future.
# int64 is the ceiling every store, wire format and database this could pass through shares,
# and it is beyond any producer that increments once per event.
MAX_SEQ = 2**63 - 1

# How deep a `content` payload may nest. This is a protocol limit deliberately far below the
# depth anything here actually breaks at, because the depth it breaks at is not a contract:
# `json`'s own limit is a C-stack guard, so it moves with how much stack is left when the
# call happens, and a bound that moves is a bound a surrogate cannot be held to. A chat
# message, a tool result and a sensor capture are all a handful of levels deep; 100 is
# generous for every one of them and small enough to be checked without recursing.
MAX_CONTENT_DEPTH = 100

# The envelope fields that must be present, non-empty strings on the wire. `origin` and
# `seq` are checked separately, with reasons of their own.
_REQUIRED_STRINGS = ("id", "actor", "type")

# What a rejection says when its reason is unusable. See `_safe_reason`.
NO_REASON_GIVEN = "rejected (no reason given)"


class BatchRejected(Exception):
    """A batch that will never be acceptable, however many times it is sent.

    Carries the status the endpoint should answer with and the reason the surrogate will
    copy onto its own tape — so the reason is written for that reader, and bounded by the
    same rule `replication_events` bounds a declared reason with, marked where it was cut.

    Two rules pull in opposite directions here and are resolved deliberately. The status is
    *checked*, and constructing this with a `5xx` raises: that is a programmer error at a
    call site, every call site passes a literal, and a `5xx` smuggled into the do-not-retry
    signal would tell a surrogate to abandon a batch it should have retried. The reason is
    *not* checked, and an unusable one is replaced rather than rejected: a reason comes from
    data, and an exception raised while building the signal that prevents a poison batch
    would itself be the 500 that makes one.
    """

    def __init__(self, status: int, reason: str) -> None:
        if (
            not isinstance(status, int)
            or isinstance(status, bool)
            or not 400 <= status < 500
        ):
            raise ValueError(
                f"BatchRejected is the 4xx class; {status!r} is not a 4xx status"
            )
        self.status = status
        self.reason = _safe_reason(reason)
        super().__init__(f"{status}: {self.reason}")


def parse_batch(
    body: str | bytes,
    *,
    host_origin: str,
    max_events: int = DEFAULT_MAX_BATCH_EVENTS,
    max_bytes: int = DEFAULT_MAX_BATCH_BYTES,
) -> list[StimulusEvent]:
    """Parse a JSONL replication body into events, or raise `BatchRejected`.

    Accepts one event per line, ascending by `seq`, all from one origin. Blank lines are
    ignored — a producer that ends with a newline has malformed nothing.

    Seqs must **ascend**, but need not be contiguous. The brief describes a batch as a
    contiguous range, and a well-behaved surrogate sends one; but a surrogate that evicted
    events under storage pressure holds a buffer with real holes in it, and answering that
    with a `4xx` would tell it to abandon data it still has — the opposite of what the
    abandon rule is for. Ascending is what ordering and the dedupe straddle actually need.

    `host_origin` is this log's own origin name, and a batch claiming it is rejected here. It
    is required rather than optional because there is no caller for whom "do not check" is
    the right answer: a surrogate misconfigured with the host's name puts two numbering
    authorities on one seq stream, and without this check it surfaces further down as an
    unhandled `ValueError` out of `append_many`, which the endpoint answers as a `5xx` and
    the surrogate retries forever. An optional guard against that would be one forgotten
    keyword argument away from the failure it prevents.

    **The contract this module owes the ingress: anything it accepts, the log can commit.**
    Validating the envelope is not enough on its own — a body can satisfy every field rule
    and still be unwritable, and the failure then lands past this module as a `5xx`. So the
    last thing each line goes through is the serialisation the log will actually perform.
    """
    try:
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
    except (AttributeError, UnicodeEncodeError) as exc:
        raise BatchRejected(400, f"batch body is not usable text or bytes: {exc}") from exc

    if len(raw) > max_bytes:
        raise BatchRejected(
            413, f"batch is {len(raw)} bytes, over the {max_bytes} byte limit"
        )

    # Strict, not `errors="replace"`. A body that is not UTF-8 will not become UTF-8 by
    # being sent again, and substituting the replacement character writes corruption onto an
    # append-only tape that nothing downstream can tell apart from content the producer
    # meant.
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BatchRejected(
            400, f"batch is not valid UTF-8 (byte {exc.start}: {exc.reason})"
        ) from exc

    # `split("\n")`, never `splitlines()`. JSONL defines exactly one separator, and
    # `splitlines` invents six more: `StimulusEvent.to_json` serialises with
    # `ensure_ascii=False`, so a Theseus surrogate puts U+2028, U+2029 and U+0085 on the
    # wire raw inside string values, and splitting on those tears a well-formed line into
    # two invalid halves. The answer would be a `400` — do not retry — so any transcript
    # containing a line separator would be discarded permanently, silently, and only for
    # certain content.
    lines = [
        (number, stripped)
        for number, line in enumerate(text.split("\n"), 1)
        if (stripped := line.strip())
    ]
    if not lines:
        raise BatchRejected(400, "batch is empty")
    if len(lines) > max_events:
        raise BatchRejected(
            413,
            f"batch carries {len(lines)} events, over the {max_events} event limit",
        )

    events = [_parse_line(number, text) for number, text in lines]
    _check_one_origin(events, host_origin)
    _check_ascending(events)
    return events


def _parse_line(number: int, text: str) -> StimulusEvent:
    """Validate the wire values *before* building an event from them, and the event before
    returning it.

    The first ordering is the point of the module, not an accident. `StimulusEvent.from_json`
    is generous by design — it coerces an absent or empty `origin` to the reading log's own
    origin, because that is the right reading for a line this node wrote before the envelope
    existed. Applied to an untrusted body it would turn a surrogate's missing origin into
    the *host's* own name, which is precisely the collision that puts two numbering
    authorities on one origin. So the raw values are checked first, and only then parsed.
    """
    raw = _loads(number, text)
    if not isinstance(raw, dict):
        raise BatchRejected(400, f"line {number} is not a JSON object")

    _check_origin(number, raw.get("origin"))
    _check_seq(number, raw.get("seq"))
    _check_envelope(number, raw)
    _check_marker(number, raw)

    try:
        event = StimulusEvent.from_json(text)
        # The writability round-trip, and the reason this function's contract is "the log can
        # commit this" rather than "the fields look right". Two things reach here having
        # satisfied every field rule and still cannot be written:
        #
        #   - a lone surrogate escape (`"\ud800"`), which is six ASCII bytes on the wire and
        #     legal JSON grammar, but has no UTF-8 encoding — `UnicodeEncodeError`;
        #   - a `content` deep enough that `json.dumps` gives out where `json.loads` did not,
        #     a band that exists because the two share one C-stack budget and do not spend it
        #     identically — `RecursionError`.
        #
        # Both were reproduced escaping this module as unhandled exceptions, at 145 bytes and
        # 20 KB respectively. Unhandled means the endpoint answers `5xx`, which a surrogate
        # reads as transient, so it resends the one batch that can never succeed while
        # everything behind it waits. Doing the log's own serialisation here converts both
        # into the `4xx` they always were. It costs one extra serialisation per line, which
        # is the cheapest possible price for the module's central promise.
        event.to_json().encode("utf-8")
    except (KeyError, ValueError, TypeError, RecursionError, OverflowError) as exc:
        # `OverflowError` is an `ArithmeticError`, not a `ValueError`, so it needs naming
        # separately: `to_json` raises it for any `ts` whose UTC conversion leaves the
        # `datetime` range — a positive offset at year 1, a negative one at year 9999.
        # That is not only an adversarial input. .NET's `DateTime.MinValue` is
        # `0001-01-01T00:00:00`, so an uninitialised timestamp from a Windows surrogate
        # anywhere east of Greenwich emits exactly it, in 162 bytes.
        raise BatchRejected(400, f"line {number} is not a usable event: {exc}") from exc
    return event


def _loads(number: int, text: str) -> Any:
    """`json.loads` with its recursion made part of the `4xx` class.

    `json.loads` recurses once per level of nesting, so a deeply nested body raises
    `RecursionError` — which is not a `ValueError` and so is not a `JSONDecodeError`.
    Measured against the unguarded parser: 20,000 levels is 40 KB, one percent of the byte
    limit, and escaped as an unhandled exception. The endpoint would answer `500`, the
    surrogate would read that as transient, and it would resend that batch forever while
    every event behind it waited. A batch too deep to parse is as permanently unacceptable
    as one that is not JSON at all, and belongs in the same class.

    The stack has already unwound to this frame by the time the handler runs, so building
    the rejection here is safe.
    """
    try:
        return json.loads(text)
    except RecursionError as exc:
        raise BatchRejected(400, f"line {number} nests too deeply to parse") from exc
    except ValueError as exc:  # JSONDecodeError is a ValueError
        raise BatchRejected(400, f"line {number} is not JSON: {exc}") from exc


def _check_origin(number: int, origin: Any) -> None:
    if not isinstance(origin, str) or not origin.strip():
        raise BatchRejected(
            400, f"line {number} has no usable origin (got {origin!r})"
        )
    if origin != origin.strip():
        # Rejected rather than trimmed. Trimming would silently rewrite the name a producer
        # chose; accepting it as-is files those events under a second origin that reads
        # identically to the first everywhere a human looks, on a permanent tape — and it
        # walks straight past the host-origin guard below, since `" local "` is not `"local"`.
        raise BatchRejected(
            400,
            f"line {number} has an origin padded with whitespace ({origin!r}); it would "
            f"stand beside its own trimmed name as a second, indistinguishable stream",
        )


def _check_seq(number: int, seq: Any) -> None:
    """`bool` is an `int` in Python, and `true` on the wire would sail past a `< 1` guard
    and then compare as 1 against a high-water mark."""
    if seq is None:
        raise BatchRejected(400, f"line {number} carries no seq")
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise BatchRejected(400, f"line {number} has a non-integer seq ({seq!r})")
    if seq < 1:
        raise BatchRejected(400, f"line {number} has seq {seq}; seqs start at 1")
    if seq > MAX_SEQ:
        raise BatchRejected(
            400,
            f"line {number} has seq {seq}, above the {MAX_SEQ} ceiling; a mark that high "
            f"would discard everything that origin ever sends afterwards",
        )


def _check_envelope(number: int, raw: dict[str, Any]) -> None:
    """The rest of the envelope, which `from_json` would take on trust.

    `from_json` indexes these straight out of the parsed dict, so `"type": null` or
    `"id": {}` becomes an event with a `None` type or a dict id, appended to the tape and
    read back by everything downstream. None of that is fixable by resending, so it is a
    `4xx` and not something to discover later in the Assembler.

    `id` is validated and then discarded — `append_many` re-mints it, because identity across
    nodes is `(origin, seq)` and never `id`. That is deliberate: this module states the wire
    contract a non-Python surrogate must satisfy, and a field the host happens not to keep is
    still a field a producer must send correctly if the two are to agree on the format.
    """
    for field in _REQUIRED_STRINGS:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise BatchRejected(
                400, f"line {number} has no usable {field} (got {value!r})"
            )
    content = raw.get("content")
    if not isinstance(content, dict):
        raise BatchRejected(
            400, f"line {number} has a non-object content (got {content!r})"
        )
    _check_content(number, content)
    _check_ts(number, raw.get("ts"))


def _check_content(number: int, content: dict[str, Any]) -> None:
    """One walk over `content`, for two things `json.loads` accepts and the tape should not
    keep.

    **Depth.** Nesting past `MAX_CONTENT_DEPTH` is refused here, at a fixed bound, rather
    than being left to whichever of `json.loads` and `json.dumps` gives out first. Those two
    share one C-stack budget and do not spend it identically, so there is a band — measured
    at depths 9993 to 9995, a 20 KB body — where the load succeeds and the write raises. The
    writability round-trip in `_parse_line` catches that band, but only by accident of stack
    depth, and an accident is not something a surrogate can be held to. This bound is.

    **Non-finite floats.** `json.loads` accepts the bare `NaN` and `Infinity` tokens, and
    `json.dumps` re-emits them, so the line lands on the tape as something no JSON reader
    outside Python will parse — forever, on an append-only file. Python's own `read_all`
    round-trips it, which is what makes it dangerous: nothing here would notice, and the
    export tool or the non-Python surrogate that eventually chokes has no way back.

    Iterative on purpose: a recursive check on deeply nested input would be one more thing
    that raises `RecursionError` in the middle of deciding whether something is too deep.
    """
    stack: list[tuple[Any, int]] = [(content, 1)]
    while stack:
        value, depth = stack.pop()
        if depth > MAX_CONTENT_DEPTH:
            raise BatchRejected(
                400,
                f"line {number} nests content deeper than {MAX_CONTENT_DEPTH} levels",
            )
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise BatchRejected(
                400,
                f"line {number} carries {value} in content; it is not JSON any reader "
                f"outside Python can parse, and the tape is append-only",
            )


def _check_marker(number: int, raw: dict[str, Any]) -> None:
    """Re-apply `replication_events`' rules to a marker that arrived over the wire.

    Those constructors validate only what *this* node builds; a remote surrogate goes
    through none of them. Both modules' docstrings say the ingress must re-apply them at the
    door rather than inventing a second definition of a well-formed marker — and until this
    check existed only the *envelope* was re-applied, so a surrogate could put an inverted
    range, an inverted span, a `5xx` in a field documented as `4xx`-only, a 5000-character
    reason where the bound is 500, or `reason: "inferred"` onto a permanent tape.

    That last one is the reason this is not merely tidiness. `INFERRED_REASON` is host-minted
    precisely so a hole the host *diagnosed* can be told apart from one a surrogate
    *reported*; a surrogate able to claim it erases the only distinction the gap vocabulary
    carries, and an agent reading its own tape could no longer tell "my sensor told me it
    dropped these" from "something over there may be dead".

    A malformed marker is a `4xx` like any other malformed line: no amount of resending
    makes it well-formed, and the alternative is writing a lie that cannot be taken back.
    """
    if raw["type"] == GAP:
        _check_marker_origin(number, raw)
        _check_marker_range(number, raw)
        _check_declared_reason(number, raw)
        _check_marker_span(number, raw)
    elif raw["type"] == BATCH_REJECTED:
        _check_marker_origin(number, raw)
        _check_marker_range(number, raw)
        _check_rejection_status(number, raw)
        _check_rejection_reason(number, raw)


def _marker_field(number: int, raw: dict[str, Any], field: str) -> Any:
    content = raw["content"]
    if field not in content:
        raise BatchRejected(
            400, f"line {number} is a {raw['type']} with no {field}"
        )
    return content[field]


def _check_marker_origin(number: int, raw: dict[str, Any]) -> None:
    origin = _marker_field(number, raw, "origin")
    if not isinstance(origin, str) or not origin.strip():
        raise BatchRejected(
            400,
            f"line {number} is a {raw['type']} about no usable origin (got {origin!r})",
        )


def _check_marker_range(number: int, raw: dict[str, Any]) -> None:
    """Inclusive at both ends, so a one-event hole has them equal. An inverted or
    below-1 range describes no span of events that could exist."""
    bounds = []
    for field in ("from_seq", "to_seq"):
        value = _marker_field(number, raw, field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise BatchRejected(
                400, f"line {number} has a non-integer {field} ({value!r})"
            )
        if value < 1:
            raise BatchRejected(
                400, f"line {number} has {field} {value}; seqs start at 1"
            )
        bounds.append(value)
    if bounds[1] < bounds[0]:
        raise BatchRejected(
            400,
            f"line {number} has an inverted range: from_seq {bounds[0]} is above "
            f"to_seq {bounds[1]}",
        )


def _check_declared_reason(number: int, raw: dict[str, Any]) -> None:
    """A gap off the wire is a gap somebody *declared*, so it must say which kind, and it
    must not claim the host's word for a hole nobody declared."""
    reason = _marker_field(number, raw, "reason")
    if reason == INFERRED_REASON:
        raise BatchRejected(
            400,
            f"line {number} claims reason {INFERRED_REASON!r}, which only the host mints; "
            f"a declared gap must say which of {DECLARED_REASONS} it was",
        )
    if reason not in DECLARED_REASONS:
        raise BatchRejected(
            400,
            f"line {number} has an unknown gap reason ({reason!r}); expected one of "
            f"{DECLARED_REASONS}",
        )
    declared = raw["content"].get("declared")
    if declared is not True:
        raise BatchRejected(
            400,
            f"line {number} declares reason {reason!r} but carries declared={declared!r}; "
            f"a reader that finds those disagreeing cannot tell what it is looking at",
        )


def _check_marker_span(number: int, raw: dict[str, Any]) -> None:
    span = []
    for field in ("span_start", "span_end"):
        value = _marker_field(number, raw, field)
        if not isinstance(value, str):
            raise BatchRejected(
                400, f"line {number} has a non-string {field} ({value!r})"
            )
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise BatchRejected(
                400, f"line {number} has an unparseable {field} ({value!r})"
            ) from exc
        if parsed.tzinfo is None:
            raise BatchRejected(
                400,
                f"line {number} has a {field} with no UTC offset ({value!r}); it would be "
                f"read in the host's zone, not the producer's",
            )
        span.append(parsed)
    if span[1] < span[0]:
        raise BatchRejected(
            400,
            f"line {number} has an inverted span: span_start {span[0]} is after "
            f"span_end {span[1]}",
        )


def _check_rejection_status(number: int, raw: dict[str, Any]) -> None:
    """`5xx` is retried, not rejected. One recorded here would claim a batch was abandoned
    while the surrogate is in fact still trying to send it."""
    status = _marker_field(number, raw, "status")
    if isinstance(status, bool) or not isinstance(status, int):
        raise BatchRejected(
            400, f"line {number} has a non-integer status ({status!r})"
        )
    if not 400 <= status < 500:
        raise BatchRejected(
            400,
            f"line {number} records status {status}; a rejection is 4xx — permanently "
            f"unacceptable — and a 5xx was never a rejection at all",
        )


def _check_rejection_reason(number: int, raw: dict[str, Any]) -> None:
    """Bounded here as well as on the write side: this is a remote host's words being
    copied onto a permanent, append-only tape by way of the surrogate."""
    reason = _marker_field(number, raw, "reason")
    if not isinstance(reason, str) or not reason.strip():
        raise BatchRejected(
            400, f"line {number} records a rejection with no reason ({reason!r})"
        )
    if len(reason) > MAX_REASON_CHARS:
        raise BatchRejected(
            400,
            f"line {number} records a {len(reason)}-character reason, over the "
            f"{MAX_REASON_CHARS} bound",
        )


def _check_ts(number: int, ts: Any) -> None:
    """A producer timestamp, with the offset the wire format requires.

    A naive `ts` is not merely imprecise here. `stimulus_log._aware` attaches the *host's*
    zone to it, which is the right reading for an old local line and the wrong one for a
    surrogate in another zone — the event lands hours from where it belongs in the
    Assembler's chronological sort, silently. The wire format is fully specified, so a `ts`
    without an offset is a producer bug and no amount of resending fixes it.
    """
    if not isinstance(ts, str):
        raise BatchRejected(400, f"line {number} has no usable ts (got {ts!r})")
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError as exc:
        raise BatchRejected(400, f"line {number} has an unparseable ts ({ts!r})") from exc
    if parsed.tzinfo is None:
        raise BatchRejected(
            400,
            f"line {number} has a ts with no UTC offset ({ts!r}); a naive timestamp would "
            f"be read in the host's zone, not the producer's",
        )


def _check_one_origin(events: list[StimulusEvent], host_origin: str) -> None:
    origins = sorted({event.origin for event in events})
    if len(origins) != 1:
        raise BatchRejected(
            400, f"a batch must come from one origin, got {origins}"
        )
    if origins[0] == host_origin:
        raise BatchRejected(
            400,
            f"batch claims this host's own origin ({host_origin!r}); a surrogate must send "
            f"under its own name, or two producers number one seq stream",
        )


def _check_ascending(events: list[StimulusEvent]) -> None:
    for earlier, later in zip(events, events[1:]):
        if later.seq <= earlier.seq:
            raise BatchRejected(
                400,
                f"seqs must be ascending; {later.seq} follows {earlier.seq}",
            )


def _safe_reason(reason: Any) -> str:
    """`clean_reason`, made incapable of failing.

    `clean_reason` raises on an empty or non-string reason, which is right where it is used —
    a marker this node builds with no reason in it is a bug worth stopping for. It is wrong
    here. A reason reaching `BatchRejected` came from data, and this exception is the signal
    that stops a poison batch: an exception raised while constructing it would escape as the
    `500` that creates one. Substituting a placeholder loses a little information in a case
    that should not arise; raising loses the channel in a case that might.
    """
    try:
        return clean_reason(reason)
    except (ValueError, TypeError):
        return NO_REASON_GIVEN
