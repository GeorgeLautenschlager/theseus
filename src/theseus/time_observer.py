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
`schedule_path` (a SCHEDULE.md-style file) for due tasks. Each line is a checkbox with
a schedule spec and a task description, e.g.:

    - [ ] once @ 2026-08-05 14:00: Water the plants
    - [ ] daily @ 09:00: Check email
    - [ ] weekly @ Monday 09:00: Submit timesheet

All times are UTC. A due task is turned into a StimulusLog entry — actor "schedule",
content `{"message": "Time to {task}"}` — which then flows through the normal
new-external-event check below and wakes the Core like any other stimulus. Firing is
persisted back into the file so a restart doesn't replay it: `once` lines get checked
off (`[x]`); `daily`/`weekly` lines get a trailing `<!-- last-fired: ... -->` marker
so their next occurrence still fires. `[x]` lines and unparseable lines are ignored.
"""

from __future__ import annotations

import calendar
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from theseus.stimulus_log import StimulusLog

_SCHEDULE_LINE_RE = re.compile(
    r"^-\s\[( |x)\]\s(once|daily|weekly)\s@\s(.+?):\s(.+?)"
    r"(?:\s<!--\slast-fired:\s(\S+)\s-->)?$"
)
_WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}


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
        self._check_schedule()
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

    def _check_schedule(self) -> None:
        """Fire a StimulusLog entry for each due line in schedule_path, then persist
        that firing back into the file (checkbox flip for "once", last-fired marker
        for "daily"/"weekly"). A missing file or a malformed line is not an error —
        scheduling is optional, and one bad line must not block the others or crash
        the wake loop."""
        if not self.schedule_path.exists():
            return
        now = self.now()
        lines = self.schedule_path.read_text(encoding="utf-8").splitlines()
        changed = False
        for i, line in enumerate(lines):
            match = _SCHEDULE_LINE_RE.match(line)
            if not match or match.group(1) == "x":
                continue
            freq, spec, task, last_fired_raw = match.group(2, 3, 4, 5)
            occurrence = self._occurrence(freq, spec, now)
            if occurrence is None or now < occurrence:
                continue
            last_fired = datetime.fromisoformat(last_fired_raw) if last_fired_raw else None
            if last_fired is not None and last_fired >= occurrence:
                continue

            self.stimulus_log.append(
                actor="schedule",
                type="scheduled_task",
                content={"message": f"Time to {task}"},
            )
            if freq == "once":
                lines[i] = f"- [x] {freq} @ {spec}: {task}"
            else:
                lines[i] = f"- [ ] {freq} @ {spec}: {task} <!-- last-fired: {now.isoformat()} -->"
            changed = True
        if changed:
            self.schedule_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _occurrence(freq: str, spec: str, now: datetime) -> datetime | None:
        """The scheduled datetime `now` must reach for this line to be due: the exact
        target for "once", or the current period's occurrence for "daily"/"weekly".
        Returns None if `spec` is malformed."""
        try:
            if freq == "once":
                return datetime.strptime(spec, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            if freq == "daily":
                hour, minute = _parse_hhmm(spec)
                return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if freq == "weekly":
                weekday_name, hhmm = spec.split(" ", 1)
                weekday = _WEEKDAYS[weekday_name.strip().lower()]
                hour, minute = _parse_hhmm(hhmm)
                days_since = (now.weekday() - weekday) % 7
                occurrence_date = (now - timedelta(days=days_since)).date()
                return datetime(
                    occurrence_date.year,
                    occurrence_date.month,
                    occurrence_date.day,
                    hour,
                    minute,
                    tzinfo=timezone.utc,
                )
        except (ValueError, KeyError):
            return None
        return None


def _parse_hhmm(spec: str) -> tuple[int, int]:
    hour_str, minute_str = spec.strip().split(":")
    hour, minute = int(hour_str), int(minute_str)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid time of day: {spec!r}")
    return hour, minute
