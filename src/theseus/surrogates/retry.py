"""The retry budget, as arithmetic rather than as a loop. Values decided in #39.

Pure arithmetic with no I/O and no clock: the drain loop (Task 3) consumes
`backoff_delay` and `is_too_old` and owns all the transport and bookkeeping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


def backoff_delay(
    attempt: int,
    budget: RetryBudget,
    *,
    random_fn: Callable[[], float] = random.random,
) -> float:
    """Seconds to wait before `attempt`, jittered, never above the ceiling."""
    raw = budget.base_seconds * budget.multiplier ** (attempt - 1)
    # ponytail: clamp after jittering, not before — a delay already at the ceiling
    # must not come out 25% above it.
    jittered = raw * (1 + budget.jitter * (2 * random_fn() - 1))
    return max(0.0, min(budget.ceiling_seconds, jittered))


def is_too_old(oldest_event_ts: datetime, now: datetime, budget: RetryBudget) -> bool:
    """Whether a batch's oldest event has aged past the budget.

    Exactly at the boundary is not too old, matching the other limits in this
    protocol. A naive `oldest_event_ts` is treated as host-local UTC, the way
    `StimulusEvent.to_json` and `replication_events._utc_span` already treat
    naive timestamps — comparing naive against aware raises a `TypeError`
    naming neither value, and a surrogate must not die of that.
    """
    if oldest_event_ts.tzinfo is None:
        oldest_event_ts = oldest_event_ts.replace(tzinfo=timezone.utc)
    return (now - oldest_event_ts) > budget.max_age
