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
    - [ ] monthly @ 1 09:00: Pay rent
    - [ ] quarterly @ 1 09:00: File quarterly report
    - [ ] annually @ 12-25 09:00: Send holiday cards
    - [ ] every 30 minutes: Check queue depth

`once` fires at its exact timestamp. `daily`/`weekly`/`monthly`/`quarterly`/`annually`
each fire once per period (day/week/month/quarter/year) at the given time, looking back
if needed to the most recent occurrence at or before "now" — so a task remains due for
the rest of its period once its time has passed, until fired. `monthly` takes a
day-of-month (1-31); `quarterly` takes a day-of-month within the first month of each
fixed quarter (quarter-start months are January/April/July/October); `annually` takes an
`MM-DD` date. All three clamp an out-of-range day to the last valid day of the target
month (e.g. day 31 in February becomes February 28, or 29 in a leap year). `every <N>
seconds|minutes|hours` fires on a rolling interval rather than a calendar period: a
never-fired `every` task is due the first time the observer sees it, and then again
every `N` <unit> after that.

All times are UTC. A due task is turned into a StimulusLog entry — actor "schedule",
content `{"message": "Time to {task}"}` — which then flows through the normal
new-external-event check below and wakes the Core like any other stimulus. Firing is
persisted back into the file so a restart doesn't replay it: `once` lines get checked
off (`[x]`); every other frequency gets a trailing `<!-- last-fired: ... -->` marker so
its next occurrence still fires. `[x]` lines and unparseable lines are ignored.
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
    r"^-\s\[( |x)\]\s(once|daily|weekly|monthly|quarterly|annually)\s@\s(.+?):\s(.+?)"
    r"(?:\s<!--\slast-fired:\s(\S+)\s-->)?$"
)
_SCHEDULE_EVERY_RE = re.compile(
    r"^-\s\[( |x)\]\severy\s(\d+)\s(seconds?|minutes?|hours?):\s(.+?)"
    r"(?:\s<!--\slast-fired:\s(\S+)\s-->)?$"
)
_WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}
_INTERVAL_UNITS = {
    "second": timedelta(seconds=1),
    "minute": timedelta(minutes=1),
    "hour": timedelta(hours=1),
}


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
        for everything else). A missing file or a malformed line is not an error —
        scheduling is optional, and one bad line must not block the others or crash
        the wake loop."""
        if not self.schedule_path.exists():
            return
        now = self.now()
        lines = self.schedule_path.read_text(encoding="utf-8").splitlines()
        changed = False
        for i, line in enumerate(lines):
            parsed = self._parse_line(line)
            if parsed is None:
                continue
            freq, spec, task, last_fired_raw = parsed
            last_fired = datetime.fromisoformat(last_fired_raw) if last_fired_raw else None
            occurrence = self._occurrence(freq, spec, now, last_fired)
            if occurrence is None or now < occurrence:
                continue
            if last_fired is not None and last_fired >= occurrence:
                continue

            self.stimulus_log.append(
                actor="schedule",
                type="scheduled_task",
                content={"message": f"Time to {task}"},
            )
            lines[i] = self._rewrite_line(freq, spec, task, now)
            changed = True
        if changed:
            self.schedule_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _parse_line(line: str) -> tuple[str, str, str, str | None] | None:
        """Parse one line into (freq, spec, task, last_fired_raw). Returns None for a
        checked-off ([x]) or unparseable line — either is silently skipped, not an
        error."""
        match = _SCHEDULE_LINE_RE.match(line)
        if match:
            checkbox, freq, spec, task, last_fired_raw = match.groups()
            return None if checkbox == "x" else (freq, spec, task, last_fired_raw)
        match = _SCHEDULE_EVERY_RE.match(line)
        if match:
            checkbox, n, unit, task, last_fired_raw = match.groups()
            return None if checkbox == "x" else ("every", f"{n} {unit}", task, last_fired_raw)
        return None

    @staticmethod
    def _rewrite_line(freq: str, spec: str, task: str, now: datetime) -> str:
        if freq == "once":
            return f"- [x] once @ {spec}: {task}"
        marker = f" <!-- last-fired: {now.isoformat()} -->"
        if freq == "every":
            return f"- [ ] every {spec}: {task}{marker}"
        return f"- [ ] {freq} @ {spec}: {task}{marker}"

    @staticmethod
    def _occurrence(
        freq: str, spec: str, now: datetime, last_fired: datetime | None
    ) -> datetime | None:
        """The scheduled datetime `now` must reach for this line to be due: the exact
        target for "once", the current period's occurrence for the calendar
        frequencies, or `last_fired + interval` (or "always due" if never fired) for
        "every". Returns None if `spec` is malformed."""
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
            if freq == "monthly":
                day_str, hhmm = spec.split(" ", 1)
                day = _parse_day_of_month(day_str)
                hour, minute = _parse_hhmm(hhmm)
                return _day_of_month_occurrence(now.year, now.month, day, hour, minute)
            if freq == "quarterly":
                day_str, hhmm = spec.split(" ", 1)
                day = _parse_day_of_month(day_str)
                hour, minute = _parse_hhmm(hhmm)
                quarter_start_month = ((now.month - 1) // 3) * 3 + 1
                return _day_of_month_occurrence(now.year, quarter_start_month, day, hour, minute)
            if freq == "annually":
                date_str, hhmm = spec.split(" ", 1)
                month, day = _parse_month_day(date_str)
                hour, minute = _parse_hhmm(hhmm)
                candidate = _day_of_month_occurrence(now.year, month, day, hour, minute)
                if candidate > now:
                    candidate = _day_of_month_occurrence(now.year - 1, month, day, hour, minute)
                return candidate
            if freq == "every":
                n_str, unit_word = spec.split(" ", 1)
                interval = _parse_interval(int(n_str), unit_word)
                if last_fired is None:
                    return datetime.min.replace(tzinfo=timezone.utc)
                return last_fired + interval
        except (ValueError, KeyError):
            return None
        return None


def _parse_hhmm(spec: str) -> tuple[int, int]:
    hour_str, minute_str = spec.strip().split(":")
    hour, minute = int(hour_str), int(minute_str)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid time of day: {spec!r}")
    return hour, minute


def _parse_day_of_month(day_str: str) -> int:
    day = int(day_str)
    if not (1 <= day <= 31):
        raise ValueError(f"invalid day of month: {day_str!r}")
    return day


def _parse_month_day(spec: str) -> tuple[int, int]:
    month_str, day_str = spec.split("-")
    month = int(month_str)
    if not (1 <= month <= 12):
        raise ValueError(f"invalid month: {month_str!r}")
    return month, _parse_day_of_month(day_str)


def _day_of_month_occurrence(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    """`day`/`hour`:`minute` in `year`-`month`, UTC, clamping `day` to the last valid day
    of that month if it overflows (e.g. day 31 in February -> February 28)."""
    last_day = calendar.monthrange(year, month)[1]
    return datetime(year, month, min(day, last_day), hour, minute, tzinfo=timezone.utc)


def _parse_interval(n: int, unit_word: str) -> timedelta:
    if n <= 0:
        raise ValueError(f"invalid interval count: {n}")
    unit = unit_word.rstrip("s")
    if unit not in _INTERVAL_UNITS:
        raise ValueError(f"invalid interval unit: {unit_word!r}")
    return n * _INTERVAL_UNITS[unit]
