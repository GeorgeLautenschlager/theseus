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

Scheduled tasks: on each tick, before checking the log, the observer also checks
`schedule_path` (a SCHEDULE.md-style file) for due tasks. The file format, due-ness
rules, and firing persistence live in `theseus.schedule.Schedule` — see that module's
docstring for the grammar. A due task becomes a StimulusLog entry (actor "schedule"),
which then flows through the normal new-external-event check below and wakes the Core
like any other stimulus.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from theseus.schedule import Schedule
from theseus.stimulus_log import StimulusLog


class TimeObserver:
    def __init__(
        self,
        stimulus_log: StimulusLog,
        try_orient: Callable[[], bool],
        self_actor: str,
        interval_seconds: float = 60.0,
        schedule_path: str | Path = "SCHEDULE.md",
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        """
        Args:
            stimulus_log: the log to watch for new entries.
            try_orient: the Core's non-blocking entry point (OODACore.try_orient).
            self_actor: the Core's own actor name (OODACore.name). Entries this actor
                writes are ignored when deciding whether to wake the Core.
            interval_seconds: seconds between wakes. Defaults to 60.
            schedule_path: path to the SCHEDULE.md-style file checked each tick for
                due tasks. Defaults to "SCHEDULE.md" in the working directory. A
                missing file is treated as "no scheduled tasks" — not an error.
            now: clock used to evaluate schedule due-ness; must return a
                timezone-aware UTC datetime. Defaults to real UTC time; injectable
                for tests.
        """
        self.stimulus_log = stimulus_log
        self.try_orient = try_orient
        self.self_actor = self_actor
        self.interval_seconds = interval_seconds
        self.schedule_path = Path(schedule_path)
        self.now = now
        self._schedule = Schedule(self.schedule_path, now=now)
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
        since the last checkpoint. The checkpoint only advances past entries the observer
        is done with — everything read when a cycle actually ran, or the whole log when
        there was nothing external to act on. On contention (try_orient returns False) it
        is left untouched, so the skipped entries are retried on the next wake rather than
        silently dropped. The Core's own output is excluded by the self_actor filter, so a
        triggered cycle never re-triggers the next wake."""
        self._schedule.fire_due(self.stimulus_log)
        events = self.stimulus_log.read_all()
        if not events:
            return
        checkpoint = self._checkpoint
        new_external = any(
            (checkpoint is None or event.id > checkpoint)
            and event.actor != self.self_actor
            for event in events
        )
        tip = max(event.id for event in events)
        if not new_external:
            self._checkpoint = tip
            return
        if self.try_orient():
            self._checkpoint = tip
