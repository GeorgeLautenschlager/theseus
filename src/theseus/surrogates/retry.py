"""The retry budget, as arithmetic rather than as a loop. Values decided in #39.

Pure arithmetic with no I/O and no clock: the drain loop (Task 3) consumes
`backoff_delay` and `is_too_old` and owns all the transport and bookkeeping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable


@dataclass(frozen=True, slots=True)
class RetryBudget:
    """How hard to try, and how stale is too stale. Values decided in #39."""

    max_attempts: int = 5
    base_seconds: float = 2.0
    multiplier: float = 3.0
    jitter: float = 0.25
    ceiling_seconds: float = 120.0
    max_age: timedelta = timedelta(hours=6)

    def __post_init__(self) -> None:
        """Reject a budget that would silently do nothing.

        `max_attempts=0` is the dangerous one: the drain's retry loop is a `range` over it,
        so a zero body never executes — no send, no abandon, no cursor advance, no
        `stopped_on`. The drain reports a clean pass having shipped nothing, forever, which
        is the head-of-line stall arriving through configuration rather than through a
        host. The rest are rejected because a negative delay or a jitter above 1 is a
        misconfiguration, and clamping one silently to zero hides it.
        """
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts must be 1 or greater (got {self.max_attempts!r}); a budget "
                f"of zero attempts never sends, never abandons and never advances"
            )
        if self.base_seconds < 0:
            raise ValueError(f"base_seconds must not be negative (got {self.base_seconds!r})")
        if self.multiplier < 1:
            raise ValueError(
                f"multiplier must be 1 or greater (got {self.multiplier!r}); below 1 the "
                f"backoff shrinks toward zero and stops being a backoff"
            )
        if not 0 <= self.jitter <= 1:
            raise ValueError(f"jitter must be between 0 and 1 (got {self.jitter!r})")
        if self.ceiling_seconds < 0:
            raise ValueError(
                f"ceiling_seconds must not be negative (got {self.ceiling_seconds!r})"
            )
        if self.max_age <= timedelta(0):
            raise ValueError(
                f"max_age must be positive (got {self.max_age!r}); a zero or negative "
                f"budget abandons every batch before it is sent"
            )


def backoff_delay(
    attempt: int,
    budget: RetryBudget,
    *,
    random_fn: Callable[[], float] = random.random,
) -> float:
    """Seconds to wait before `attempt`, jittered, never above the ceiling."""
    raw = budget.base_seconds * budget.multiplier ** (attempt - 1)
    # Clamp after jittering, not before: a delay already at the ceiling must not come
    # out 25% above it.
    jittered = raw * (1 + budget.jitter * (2 * random_fn() - 1))
    return max(0.0, min(budget.ceiling_seconds, jittered))


def is_too_old(oldest_event_ts: datetime, now: datetime, budget: RetryBudget) -> bool:
    """Whether a batch's oldest event has aged past the budget.

    Exactly at the boundary is not too old, matching the other limits in this
    protocol.

    A naive `oldest_event_ts` is read as **host-local**, via `astimezone()` — the same
    reading `StimulusLog._aware` and `replication_events._utc_span` already give one.
    Stamping UTC on it instead would be a different claim, and a wrong one: west of
    Greenwich it makes an event look hours *older* than it is, which against a six-hour
    budget abandons batches that had hours of life left. Comparing naive against aware
    raises a `TypeError` naming neither value, and a surrogate must not die of that
    either.
    """
    if oldest_event_ts.tzinfo is None:
        oldest_event_ts = oldest_event_ts.astimezone()
    return (now - oldest_event_ts) > budget.max_age
