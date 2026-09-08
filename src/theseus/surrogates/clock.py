"""Time, injected — so backoff is testable without waiting for it.

The retry budget sleeps between attempts; a clock that cannot be faked turns every
exhaustion test into a wall-clock prayer. Tests inject a fake that records sleeps and
advances `now`; production gets the real one.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """The real one. `now()` is timezone-aware UTC, because every ts in this system is —
    a naive datetime would compare wrongly against event timestamps, which are all aware."""

    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
