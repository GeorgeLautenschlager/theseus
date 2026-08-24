"""Schedule — the SCHEDULE.md grammar: parsing, due-ness, firing, persistence.

Each line is a checkbox with a schedule spec and a task description, e.g.:

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
never-fired `every` task is due the first time it is checked, and then again every `N`
<unit> after that.

All times are UTC. A due task is turned into a StimulusLog entry — actor "schedule",
content `{"message": "Time to {task}"}`. Firing is persisted back into the file so a
restart doesn't replay it: `once` lines get checked off (`[x]`); every other frequency
gets a trailing `<!-- last-fired: ... -->` marker so its next occurrence still fires.
`[x]` lines and unparseable lines are ignored.

`TimeObserver` polls a Schedule on a timer; `Autocore` consults one inside its own loop.
Both share this class so there is exactly one grammar and one parser.
"""

from __future__ import annotations

import calendar
import re
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


class Schedule:
    def __init__(
        self,
        path: str | Path,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        """
        Args:
            path: the SCHEDULE.md-style file. A missing file is treated as
                "no scheduled tasks" — not an error.
            now: clock used to evaluate due-ness; must return a timezone-aware UTC
                datetime. Defaults to real UTC time; injectable for tests.
        """
        self.path = Path(path)
        self.now = now

    def fire_due(self, stimulus_log: StimulusLog) -> None:
        """Fire a StimulusLog entry for each due line, then persist that firing back
        into the file (checkbox flip for "once", last-fired marker for everything
        else). A missing file or a malformed line is not an error — scheduling is
        optional, and one bad line must not block the others or crash the caller."""
        if not self.path.exists():
            return
        now = self.now()
        lines = self.path.read_text(encoding="utf-8").splitlines()
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

            stimulus_log.append(
                actor="schedule",
                type="scheduled_task",
                content={"message": f"Time to {task}"},
            )
            lines[i] = self._rewrite_line(freq, spec, task, now)
            changed = True
        if changed:
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def next_occurrence(self, now: datetime | None = None) -> datetime | None:
        """Earliest future due time across all unchecked, parseable lines — strictly
        after `now`, except a rolling `every` task whose interval has already elapsed
        (or that never fired), which is due immediately and yields `now` itself.
        Returns None when no line yields a candidate (missing file included). Lets a
        caller sleep exactly until something is scheduled to happen."""
        if now is None:
            now = self.now()
        if not self.path.exists():
            return None
        candidates: list[datetime] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            parsed = self._parse_line(line)
            if parsed is None:
                continue
            freq, spec, task, last_fired_raw = parsed
            last_fired = datetime.fromisoformat(last_fired_raw) if last_fired_raw else None
            candidate = self._next_occurrence_of(freq, spec, now, last_fired)
            if candidate is not None:
                candidates.append(candidate)
        return min(candidates, default=None)

    def lint_lines(self) -> list[str]:
        """Lines someone meant as schedule entries but got the grammar wrong: they
        start with `- [`, are not checked off, and fail to parse. Ordinary prose is
        left alone. Surfacing these lets the agent rewrite them itself."""
        if not self.path.exists():
            return []
        bad: list[str] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("- ["):
                continue
            if stripped.startswith("- [x]"):
                continue
            if self._parse_line(line) is None:
                bad.append(line)
        return bad

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

    @staticmethod
    def _next_occurrence_of(
        freq: str, spec: str, now: datetime, last_fired: datetime | None
    ) -> datetime | None:
        """The next time this line comes due strictly after `now` — the forward-looking
        counterpart of `_occurrence` (which looks back to decide current due-ness).
        An `every` task whose interval has already elapsed is due immediately: `now`.
        Returns None if `spec` is malformed or the line can never fire again (a `once`
        whose moment has passed)."""
        try:
            if freq == "once":
                if last_fired is not None:
                    return None
                target = datetime.strptime(spec, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                return target if target > now else None
            if freq == "daily":
                hour, minute = _parse_hhmm(spec)
                candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return candidate if candidate > now else candidate + timedelta(days=1)
            if freq == "weekly":
                weekday_name, hhmm = spec.split(" ", 1)
                weekday = _WEEKDAYS[weekday_name.strip().lower()]
                hour, minute = _parse_hhmm(hhmm)
                days_until = (weekday - now.weekday()) % 7
                date = (now + timedelta(days=days_until)).date()
                candidate = datetime(date.year, date.month, date.day, hour, minute, tzinfo=timezone.utc)
                return candidate if candidate > now else candidate + timedelta(days=7)
            if freq == "monthly":
                day_str, hhmm = spec.split(" ", 1)
                day = _parse_day_of_month(day_str)
                hour, minute = _parse_hhmm(hhmm)
                candidate = _day_of_month_occurrence(now.year, now.month, day, hour, minute)
                if candidate > now:
                    return candidate
                year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
                return _day_of_month_occurrence(year, month, day, hour, minute)
            if freq == "quarterly":
                day_str, hhmm = spec.split(" ", 1)
                day = _parse_day_of_month(day_str)
                hour, minute = _parse_hhmm(hhmm)
                quarter_start = ((now.month - 1) // 3) * 3 + 1
                candidate = _day_of_month_occurrence(now.year, quarter_start, day, hour, minute)
                if candidate > now:
                    return candidate
                year, month = (now.year + 1, 1) if quarter_start == 10 else (now.year, quarter_start + 3)
                return _day_of_month_occurrence(year, month, day, hour, minute)
            if freq == "annually":
                date_str, hhmm = spec.split(" ", 1)
                month, day = _parse_month_day(date_str)
                hour, minute = _parse_hhmm(hhmm)
                candidate = _day_of_month_occurrence(now.year, month, day, hour, minute)
                if candidate > now:
                    return candidate
                return _day_of_month_occurrence(now.year + 1, month, day, hour, minute)
            if freq == "every":
                n_str, unit_word = spec.split(" ", 1)
                interval = _parse_interval(int(n_str), unit_word)
                if last_fired is None:
                    return now
                candidate = last_fired + interval
                return candidate if candidate > now else now
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
