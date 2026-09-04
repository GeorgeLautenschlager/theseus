"""Vocabulary for the holes in a replicated stream.

Gaps are normal operation on this protocol, not errors — a surrogate may be an edge device
kilometres from the nearest tower, and the host is enriched by as much as it manages to
deliver and unbothered by the rest. But the protocol keeps a distinction worth preserving:

- A **declared** gap carries a marker the surrogate emitted with the abandoned range in
  hand — "you weren't hearing from me between 16:02 and 16:40, the link was down." The
  agent can reason about it.
- An **inferred** gap is a seq jump the host noticed with no marker to explain it: the
  surrogate died mid-buffer, or something is genuinely broken.

Same hole, different diagnosis. Neither is rejected; both go on the tape as ordinary
events, because a hole the agent can read about is worth more than a silence it cannot
account for.

This module is the shared vocabulary only — constructors and validation. Nothing here
emits, appends or transports anything; the ingress and the replicator own that. What it
does own is refusing to build a malformed marker: an invalid reason, an inverted range or
a value that would not survive the wire raises here, at the mistake, rather than
serialising into the log where it is permanent. Every rejection is a `ValueError`,
including the ones that are arguably type errors: an ingress catching malformed input at
one boundary should not need two except clauses to do it.

**Validation here is write-side only.** These constructors are the schema for events this
node *builds*. An event arriving over the wire has been through none of them — a remote
surrogate POSTs JSONL, and nothing in this module stops a buggy or hostile one claiming
`reason: "inferred"` on a marker of its own. Read-side validation belongs to the ingress
(#30), which is where an untrusted body actually arrives; it must apply these same rules
rather than inventing a second definition of a well-formed marker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

# Event types. These strings are wire protocol — a surrogate and a host built from
# different releases still have to agree on them, so they are pinned by a test.
GAP = "stimulus.gap"
BATCH_REJECTED = "replication.batch_rejected"

# Why a surrogate abandoned a range, in its own words.
DECLARED_REASONS = ("link_down", "retry_exhausted", "storage_pressure")
DeclaredReason = Literal["link_down", "retry_exhausted", "storage_pressure"]

# What the host writes when it sees a jump nobody declared. Not declarable by a surrogate:
# a surrogate claiming it would erase the distinction these events exist to carry.
#
# `reason` is authoritative and `declared` is a convenience derived from it — they can
# never disagree on anything this module builds. A reader that finds them disagreeing is
# looking at an event some other implementation wrote, and should believe `reason`.
INFERRED_REASON = "inferred"

# A rejection reason is a remote host's own words copied onto a permanent, append-only
# tape. Bounded so a verbose — or hostile — responder cannot write an unbounded string
# into the log. Over-long reasons are truncated rather than rejected: a rejection that
# cannot be recorded is a silence, which is the failure this event type exists to prevent.
MAX_REASON_CHARS = 500


def declared_gap(
    *,
    origin: str,
    from_seq: int,
    to_seq: int,
    reason: DeclaredReason,
    span_start: datetime,
    span_end: datetime,
) -> dict[str, Any]:
    """Content for a `stimulus.gap` the surrogate declared about its own stream.

    `from_seq` and `to_seq` are **inclusive** — a one-event hole has them equal.

    The surrogate always knows what it dropped: `seq` is assigned at local write time, so
    eviction or abandonment happens with the range in hand. That holds for `link_down`
    too. The link being down never stops a surrogate observing or allocating seqs, so a
    `link_down` gap is a range that was buffered and then given up on, not an interval in
    which nothing was seen.
    """
    if reason not in DECLARED_REASONS:
        raise ValueError(
            f"unknown declared reason {reason!r}; expected one of {DECLARED_REASONS}. "
            f"{INFERRED_REASON!r} is host-minted — use inferred_gap()."
        )
    _check_origin(origin)
    _check_range(from_seq, to_seq)
    start, end = _utc_span(span_start, span_end)
    return {
        "origin": origin,
        "from_seq": from_seq,
        "to_seq": to_seq,
        "reason": reason,
        "span_start": start,
        "span_end": end,
        "declared": True,
    }


def inferred_gap(
    *,
    origin: str,
    from_seq: int,
    to_seq: int,
    span_start: datetime,
    span_end: datetime,
) -> dict[str, Any]:
    """Content for a `stimulus.gap` the host minted because nobody declared one.

    `from_seq` and `to_seq` are **inclusive**.

    `declared: False` is the whole payload of this event: it says the hole was diagnosed,
    not reported, and that whatever is on the other end may be in trouble.
    """
    _check_origin(origin)
    _check_range(from_seq, to_seq)
    start, end = _utc_span(span_start, span_end)
    return {
        "origin": origin,
        "from_seq": from_seq,
        "to_seq": to_seq,
        "reason": INFERRED_REASON,
        "span_start": start,
        "span_end": end,
        "declared": False,
    }


def batch_rejected(
    *,
    origin: str,
    from_seq: int,
    to_seq: int,
    status: int,
    reason: str,
) -> dict[str, Any]:
    """Content for a `replication.batch_rejected`, emitted locally by the surrogate.

    `from_seq` and `to_seq` are **inclusive**.

    The `4xx` class exists so one malformed or oversized batch cannot wedge the channel
    forever. Recording the rejection makes the resulting hole visible on the tape instead
    of a silence. A `5xx` is retried rather than rejected, so it is not a valid status
    here — writing one would claim a batch was abandoned while the surrogate is still
    trying to send it.
    """
    _check_origin(origin)
    _check_range(from_seq, to_seq)
    _check_int("status", status)
    if not 400 <= status < 500:
        raise ValueError(
            f"status must be 4xx — permanently unacceptable — but got {status!r}"
        )
    return {
        "origin": origin,
        "from_seq": from_seq,
        "to_seq": to_seq,
        "status": status,
        "reason": _clean_reason(reason),
    }


def _check_origin(origin: str) -> None:
    """An origin is the key everything downstream dedupes and routes on, so a blank or
    non-string one is a misconfiguration rather than a value to record."""
    if not isinstance(origin, str) or not origin.strip():
        raise ValueError(f"origin must be a non-empty name (got {origin!r})")


def _check_int(name: str, value: int) -> None:
    """`bool` is an `int` in Python, and `True` would serialise as `true` onto a wire field
    a non-Python surrogate parses as a number — so it is excluded explicitly."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer (got {value!r})")


def _check_datetime(name: str, value: datetime) -> None:
    """A span arrives from a JSON body as an ISO *string*, and handing that straight in is
    the obvious mistake. Caught here so it names the field, rather than surfacing as an
    `AttributeError` about `astimezone` from somewhere in the middle of the call."""
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime (got {value!r})")


def _check_range(from_seq: int, to_seq: int) -> None:
    """The abandoned range, inclusive. A single-event hole is `from_seq == to_seq`."""
    _check_int("from_seq", from_seq)
    _check_int("to_seq", to_seq)
    if from_seq < 1:
        raise ValueError(f"from_seq must be 1 or greater (got {from_seq!r})")
    if to_seq < 1:
        raise ValueError(f"to_seq must be 1 or greater (got {to_seq!r})")
    if to_seq < from_seq:
        raise ValueError(
            f"inverted seq range: from_seq {from_seq} is above to_seq {to_seq}"
        )


def _utc_span(span_start: datetime, span_end: datetime) -> tuple[str, str]:
    """The wall-clock span the hole covers, as UTC ISO strings.

    Normalisation happens *before* the ordering check, not after: a naive datetime is
    host-local by the same rule `StimulusEvent.to_json` uses, and comparing one against an
    aware datetime raises an opaque `TypeError` naming neither field. An ingress holding
    `datetime.now(timezone.utc)` beside a naive timestamp parsed from a surrogate's payload
    is exactly that case.

    Converting here also means `content` is JSON-native: `json.dumps` inside
    `StimulusEvent.to_json` cannot encode a datetime, and failing there would be a long way
    from the mistake.
    """
    _check_datetime("span_start", span_start)
    _check_datetime("span_end", span_end)
    start = span_start.astimezone(timezone.utc)
    end = span_end.astimezone(timezone.utc)
    if end < start:
        raise ValueError(f"inverted span: span_start {start} is after span_end {end}")
    return start.isoformat(), end.isoformat()


def _clean_reason(reason: str) -> str:
    """The host's stated reason: stripped, and bounded per `MAX_REASON_CHARS`.

    An over-long reason is marked where it was cut. This is an evidentiary event — a
    reader that cannot tell "the host said exactly this" from "the host said this and
    more" is being quietly misled about what it has.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"reason must be a non-empty string (got {reason!r})")
    reason = reason.strip()
    if len(reason) <= MAX_REASON_CHARS:
        return reason
    return reason[: MAX_REASON_CHARS - 1] + "…"
