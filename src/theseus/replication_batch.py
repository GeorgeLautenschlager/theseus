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
from datetime import datetime
from typing import Any

from theseus.replication_events import clean_reason
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

# The envelope fields that must be present, non-empty strings on the wire. `origin` and
# `seq` are checked separately, with reasons of their own.
_REQUIRED_STRINGS = ("id", "actor", "type")


class BatchRejected(Exception):
    """A batch that will never be acceptable, however many times it is sent.

    Carries the status the endpoint should answer with and the reason the surrogate will
    copy onto its own tape — so the reason is written for that reader, and bounded by the
    same rule `replication_events` bounds a declared reason with, marked where it was cut.

    The status is checked because this exception *is* the protocol's "do not retry" signal:
    raising it with a `5xx` would tell the surrogate to abandon a batch it should have
    retried, and there is no later layer that could catch the mistake.
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
        self.reason = clean_reason(reason)
        super().__init__(f"{status}: {self.reason}")


def parse_batch(
    body: str | bytes,
    *,
    max_events: int = DEFAULT_MAX_BATCH_EVENTS,
    max_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    host_origin: str | None = None,
) -> list[StimulusEvent]:
    """Parse a JSONL replication body into events, or raise `BatchRejected`.

    Accepts one event per line, ascending by `seq`, all from one origin. Blank lines are
    ignored — a producer that ends with a newline has malformed nothing.

    Seqs must **ascend**, but need not be contiguous. The brief describes a batch as a
    contiguous range, and a well-behaved surrogate sends one; but a surrogate that evicted
    events under storage pressure holds a buffer with real holes in it, and answering that
    with a `4xx` would tell it to abandon data it still has — the opposite of what the
    abandon rule is for. Ascending is what ordering and the dedupe straddle actually need.

    `host_origin`, when given, is this log's own origin name, and a batch claiming it is
    rejected here. A surrogate misconfigured with the host's name puts two numbering
    authorities on one seq stream, which is unfixable by retrying and so belongs in the
    `4xx` class — without this it would surface further down as an unhandled `ValueError`
    out of `append_many`, which the endpoint would answer as a `5xx` and the surrogate would
    retry forever.
    """
    raw = body if isinstance(body, bytes) else body.encode("utf-8")
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
    """Validate the wire values *before* building an event from them.

    This ordering is the point of the module, not an accident. `StimulusEvent.from_json`
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

    try:
        return StimulusEvent.from_json(text)
    except (KeyError, ValueError, TypeError, RecursionError) as exc:
        raise BatchRejected(400, f"line {number} is not a usable event: {exc}") from exc


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
    _check_ts(number, raw.get("ts"))


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


def _check_one_origin(events: list[StimulusEvent], host_origin: str | None) -> None:
    origins = sorted({event.origin for event in events})
    if len(origins) != 1:
        raise BatchRejected(
            400, f"a batch must come from one origin, got {origins}"
        )
    if host_origin is not None and origins[0] == host_origin:
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
