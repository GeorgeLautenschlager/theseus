"""Vocabulary for the holes in a replicated stream.

Gaps are normal operation on this protocol, not errors — a surrogate may be an edge device
kilometres from the nearest tower, and the host is enriched by as much as it manages to
deliver and unbothered by the rest. But the protocol keeps a distinction worth preserving:

- A **declared** gap carries a marker the surrogate emitted with the abandoned range in
  hand — "I wasn't observing from 16:02 to 16:40, the link was down." The agent can reason
  about it.
- An **inferred** gap is a seq jump the host noticed with no marker to explain it: the
  surrogate died mid-buffer, or something is genuinely broken.

Same hole, different diagnosis. Neither is rejected; both go on the tape as ordinary
events, because a hole the agent can read about is worth more than a silence it cannot
account for.

This module is the shared vocabulary only — constructors and validation. Nothing here
emits, appends or transports anything; the ingress and the replicator own that. What it
does own is refusing to build a malformed marker: an invalid reason or an inverted range
raises here, at the mistake, rather than serialising into the log where it is permanent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Event types. These strings are wire protocol — a surrogate and a host built from
# different releases still have to agree on them.
GAP = "stimulus.gap"
BATCH_REJECTED = "replication.batch_rejected"

# Why a surrogate abandoned a range, in its own words.
DECLARED_REASONS = ("link_down", "retry_exhausted", "storage_pressure")

# What the host writes when it sees a jump nobody declared. Not declarable by a surrogate:
# a surrogate claiming it would erase the distinction these events exist to carry.
INFERRED_REASON = "inferred"


def declared_gap(
    *,
    origin: str,
    from_seq: int,
    to_seq: int,
    reason: str,
    span_start: datetime,
    span_end: datetime,
) -> dict[str, Any]:
    """Content for a `stimulus.gap` the surrogate declared about its own stream.

    The surrogate always knows what it dropped — `seq` is assigned at local write time, so
    eviction or abandonment happens with the range in hand.
    """
    if reason not in DECLARED_REASONS:
        raise ValueError(
            f"unknown declared reason {reason!r}; expected one of {DECLARED_REASONS}. "
            f"{INFERRED_REASON!r} is host-minted — use inferred_gap()."
        )
    return {
        "origin": _origin(origin),
        **_seq_range(from_seq, to_seq),
        "reason": reason,
        **_span(span_start, span_end),
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

    `declared: False` is the whole payload of this event: it says the hole was diagnosed,
    not reported, and that whatever is on the other end may be in trouble.
    """
    return {
        "origin": _origin(origin),
        **_seq_range(from_seq, to_seq),
        "reason": INFERRED_REASON,
        **_span(span_start, span_end),
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

    The `4xx` class exists so one malformed or oversized batch cannot wedge the channel
    forever. Recording the rejection makes the resulting hole visible on the tape instead
    of a silence. A `5xx` is retried rather than rejected, so it is not a valid status
    here — writing one would claim a batch was abandoned while the surrogate is still
    trying to send it.
    """
    if not 400 <= status < 500:
        raise ValueError(
            f"status must be 4xx — permanently unacceptable — but got {status!r}"
        )
    return {
        "origin": _origin(origin),
        **_seq_range(from_seq, to_seq),
        "status": status,
        "reason": reason,
    }


def _origin(origin: str) -> str:
    if not origin:
        raise ValueError("origin must be a non-empty name")
    return origin


def _seq_range(from_seq: int, to_seq: int) -> dict[str, int]:
    """The abandoned range, inclusive. A single-event hole is `from_seq == to_seq`."""
    if from_seq < 1:
        raise ValueError(f"from_seq must be 1 or greater (got {from_seq!r})")
    if to_seq < from_seq:
        raise ValueError(
            f"inverted seq range: from_seq {from_seq} is above to_seq {to_seq}"
        )
    return {"from_seq": from_seq, "to_seq": to_seq}


def _span(span_start: datetime, span_end: datetime) -> dict[str, str]:
    """The wall-clock span the hole covers, normalised to UTC.

    `content` is serialised by `json.dumps` inside `StimulusEvent.to_json`, which cannot
    encode a datetime — so these become ISO strings here rather than failing at write
    time, a long way from the mistake.
    """
    if span_end < span_start:
        raise ValueError(
            f"inverted span: span_start {span_start} is after span_end {span_end}"
        )
    return {
        "span_start": span_start.astimezone(timezone.utc).isoformat(),
        "span_end": span_end.astimezone(timezone.utc).isoformat(),
    }
