"""TimeObserver — wakes on an interval and nudges the Core to orient on new stimuli.

Unlike TerminalChatObserver and WebChatUIObserver, which react to an external event
(a keystroke, an HTTP request), TimeObserver reacts to the passage of time: it wakes
every `interval_seconds`, checks whether anything new has landed in the StimulusLog
since it last looked, and — if so — asks the Core to run a cognitive cycle.

Concurrency policy: skip-on-contention. It calls the Core's non-blocking `try_orient`;
if a cycle is already in flight the attempt is a silent no-op until the next wake. Safe
because the StimulusLog is durable — whichever cycle runs next assembles its context
from the whole log and sees everything that accumulated.

Checkpoint: in memory only, initialised to the log's tip at `start()`, so a fresh run
reacts to what arrives *during* the run, not to prior-run backlog. No cross-restart
persistence.

Self-authored events are ignored: the Core writes its own `decision` and `tool_result`
events into the same log, so counting those as "new" would make every cycle's own
output re-trigger the next wake forever. Only entries from an actor other than the Core
count as a stimulus worth waking for.
"""

from __future__ import annotations

import threading
from typing import Callable

from theseus.stimulus_log import StimulusLog


class TimeObserver:
    def __init__(
        self,
        stimulus_log: StimulusLog,
        try_orient: Callable[[], bool],
        self_actor: str,
        interval_seconds: float = 60.0,
    ):
        """
        Args:
            stimulus_log: the log to watch for new entries.
            try_orient: the Core's non-blocking entry point (OODACore.try_orient).
            self_actor: the Core's own actor name (OODACore.name). Entries this actor
                writes are ignored when deciding whether to wake the Core.
            interval_seconds: seconds between wakes. Defaults to 60.
        """
        self.stimulus_log = stimulus_log
        self.try_orient = try_orient
        self.self_actor = self_actor
        self.interval_seconds = interval_seconds
        self._checkpoint: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Initialise the checkpoint to the current tip and spawn the wake loop on a
        daemon thread. One TimeObserver runs one thread — don't call start() twice."""
        events = self.stimulus_log.read_all()
        self._checkpoint = events[-1].id if events else None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Signal the wake loop to exit and best-effort join it. Interrupts the interval
        immediately via the Event, so shutdown doesn't wait out a full interval."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        # Event().wait(interval) is both the sleep and the interruptible shutdown
        # signal: returns True the instant stop() fires, False on timeout (a wake).
        while not self._stop.wait(self.interval_seconds):
            self._tick()

    def _tick(self) -> None:
        """One wake: fire the Core iff a new, externally-authored entry has appeared
        since the last checkpoint, then advance the checkpoint to the current tip so the
        same entries never count twice. The Core's own output from the cycle this may
        trigger lands beyond the checkpoint but is excluded by the self_actor filter."""
        events = self.stimulus_log.read_all()
        if not events:
            return
        checkpoint = self._checkpoint
        has_new_external = any(
            (checkpoint is None or event.id > checkpoint)
            and event.actor != self.self_actor
            for event in events
        )
        self._checkpoint = events[-1].id
        if has_new_external:
            self.try_orient()
