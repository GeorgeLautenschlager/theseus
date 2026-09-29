from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from theseus.stimulus_log import StimulusLog
from theseus.wake_policy import DESKTOP_OBSERVATION


@dataclass(frozen=True)
class DesktopState:
    # None denotes unavailable, not an empty title or an unlocked session.
    window_id: int | None = None
    pid: int | None = None
    process_name: str | None = None
    title: str | None = None
    activity: str = "unknown"
    locked: bool | None = None


class DesktopSource(Protocol):
    def sample(self) -> DesktopState: ...


class DesktopObserver:
    """Sample cheaply; append full snapshots only on transition or heartbeat."""

    def __init__(
        self, log: StimulusLog, source: DesktopSource, *,
        poll_seconds: float = 1.0, heartbeat_seconds: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in (poll_seconds, heartbeat_seconds)):
            raise ValueError("observation intervals must be finite and positive")
        self._log, self._source = log, source
        self._poll_seconds, self._heartbeat_seconds = poll_seconds, heartbeat_seconds
        self._monotonic, self._wall_clock = monotonic, wall_clock
        self._previous: DesktopState | None = None
        self._published_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle = threading.Lock()

    def sample_once(self) -> None:
        try:
            state = self._source.sample()
        except Exception:
            logging.getLogger(__name__).exception("Desktop sampling failed")
            state = DesktopState()
        now = self._monotonic()
        reason = (
            "initial" if self._previous is None else
            "changed" if state != self._previous else
            "heartbeat" if now - self._published_at >= self._heartbeat_seconds else None
        )
        if reason is None:
            return
        self._log.append(actor="desktop_observer", type=DESKTOP_OBSERVATION, content={
            "v": 1, "source": "windows.desktop", "scope": "current_session",
            "captured_at": self._wall_clock().isoformat(), "reason": reason,
            "state": asdict(state),
        })
        self._previous, self._published_at = state, now

    def start(self) -> None:
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                return
            self._previous = None
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="desktop-observer", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    raise RuntimeError("Desktop observer did not stop")
                self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception:
                logging.getLogger(__name__).exception("Desktop observation could not be published")
            if self._stop.wait(self._poll_seconds):
                break
