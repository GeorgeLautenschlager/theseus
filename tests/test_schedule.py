from __future__ import annotations

from datetime import datetime, timezone

from theseus.schedule import Schedule
from theseus.stimulus_log import StimulusLog


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def make(tmp_path, schedule_text=None, now=None):
    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    path = tmp_path / "SCHEDULE.md"
    if schedule_text is not None:
        path.write_text(schedule_text, encoding="utf-8")
    schedule = Schedule(path, now=(now if now is not None else (lambda: datetime.now(timezone.utc))))
    return schedule, log, path


# --- fire_due ---


def test_due_daily_task_fires_once_and_persists_marker(tmp_path):
    schedule, log, path = make(
        tmp_path,
        "- [ ] daily @ 09:00: Check email\n",
        now=lambda: utc(2026, 8, 3, 9, 5),
    )
    schedule.fire_due(log)
    events = log.read_all()
    assert len(events) == 1
    assert events[0].actor == "schedule"
    assert events[0].type == "scheduled_task"
    assert events[0].content == {"message": "Time to Check email"}
    assert "last-fired: 2026-08-03T09:05:00+00:00" in path.read_text()

    schedule.fire_due(log)  # same day, already fired -> no refire
    assert len(log.read_all()) == 1


def test_due_once_task_fires_and_checks_off(tmp_path):
    schedule, log, path = make(
        tmp_path,
        "- [ ] once @ 2026-08-03 09:00: Water the plants\n",
        now=lambda: utc(2026, 8, 3, 9, 5),
    )
    schedule.fire_due(log)
    assert len(log.read_all()) == 1
    assert "- [x] once @ 2026-08-03 09:00: Water the plants" in path.read_text()

    schedule.fire_due(log)
    assert len(log.read_all()) == 1


def test_malformed_and_checked_lines_are_skipped_without_error(tmp_path):
    schedule, log, path = make(
        tmp_path,
        "- [ ] every other tuesday: broken\n"
        "- [x] daily @ 09:00: already done\n"
        "just some prose\n",
        now=lambda: utc(2026, 8, 3, 9, 5),
    )
    schedule.fire_due(log)
    assert log.read_all() == []


def test_missing_file_is_a_noop(tmp_path):
    schedule, log, _ = make(tmp_path, schedule_text=None)
    schedule.fire_due(log)
    assert log.read_all() == []


# --- next_occurrence ---


def test_next_occurrence_daily_before_and_after_time(tmp_path):
    schedule, _, _ = make(tmp_path, "- [ ] daily @ 09:00: Check email\n")
    assert schedule.next_occurrence(utc(2026, 8, 3, 8, 0)) == utc(2026, 8, 3, 9, 0)
    assert schedule.next_occurrence(utc(2026, 8, 3, 10, 0)) == utc(2026, 8, 4, 9, 0)


def test_next_occurrence_weekly(tmp_path):
    # 2026-08-03 is a Monday.
    schedule, _, _ = make(tmp_path, "- [ ] weekly @ Monday 09:00: Timesheet\n")
    assert schedule.next_occurrence(utc(2026, 8, 3, 8, 0)) == utc(2026, 8, 3, 9, 0)
    assert schedule.next_occurrence(utc(2026, 8, 3, 10, 0)) == utc(2026, 8, 10, 9, 0)


def test_next_occurrence_monthly_rolls_to_next_month(tmp_path):
    schedule, _, _ = make(tmp_path, "- [ ] monthly @ 1 09:00: Pay rent\n")
    assert schedule.next_occurrence(utc(2026, 8, 3, 10, 0)) == utc(2026, 9, 1, 9, 0)
    assert schedule.next_occurrence(utc(2026, 8, 1, 8, 0)) == utc(2026, 8, 1, 9, 0)


def test_next_occurrence_quarterly_rolls_to_next_quarter(tmp_path):
    schedule, _, _ = make(tmp_path, "- [ ] quarterly @ 1 09:00: Report\n")
    assert schedule.next_occurrence(utc(2026, 8, 3, 10, 0)) == utc(2026, 10, 1, 9, 0)
    assert schedule.next_occurrence(utc(2026, 7, 1, 8, 0)) == utc(2026, 7, 1, 9, 0)


def test_next_occurrence_annually_rolls_to_next_year(tmp_path):
    schedule, _, _ = make(tmp_path, "- [ ] annually @ 12-25 09:00: Cards\n")
    assert schedule.next_occurrence(utc(2026, 8, 3, 10, 0)) == utc(2026, 12, 25, 9, 0)
    assert schedule.next_occurrence(utc(2026, 12, 26, 0, 0)) == utc(2027, 12, 25, 9, 0)


def test_next_occurrence_every_never_fired_is_due_now(tmp_path):
    schedule, _, _ = make(tmp_path, "- [ ] every 30 minutes: Check queue\n")
    now = utc(2026, 8, 3, 10, 0)
    assert schedule.next_occurrence(now) == now


def test_next_occurrence_every_with_future_last_fired(tmp_path):
    schedule, _, _ = make(
        tmp_path,
        "- [ ] every 30 minutes: Check queue"
        " <!-- last-fired: 2026-08-03T09:50:00+00:00 -->\n",
    )
    assert schedule.next_occurrence(utc(2026, 8, 3, 10, 0)) == utc(2026, 8, 3, 10, 20)


def test_next_occurrence_every_with_stale_last_fired_is_due_now(tmp_path):
    schedule, _, _ = make(
        tmp_path,
        "- [ ] every 30 minutes: Check queue"
        " <!-- last-fired: 2026-08-03T08:00:00+00:00 -->\n",
    )
    now = utc(2026, 8, 3, 10, 0)
    assert schedule.next_occurrence(now) == now


def test_next_occurrence_once_future_and_checked(tmp_path):
    schedule, _, _ = make(
        tmp_path,
        "- [ ] once @ 2026-08-05 14:00: Water the plants\n"
        "- [x] once @ 2026-08-04 14:00: Already done\n",
    )
    assert schedule.next_occurrence(utc(2026, 8, 3, 10, 0)) == utc(2026, 8, 5, 14, 0)
    # A once task already in the past yields no candidate.
    assert schedule.next_occurrence(utc(2026, 8, 6, 10, 0)) is None


def test_next_occurrence_minimum_across_lines(tmp_path):
    schedule, _, _ = make(
        tmp_path,
        "- [ ] daily @ 09:00: Check email\n"
        "- [ ] once @ 2026-08-03 06:00: Early bird\n",
    )
    assert schedule.next_occurrence(utc(2026, 8, 3, 5, 0)) == utc(2026, 8, 3, 6, 0)


def test_next_occurrence_missing_or_empty_file_is_none(tmp_path):
    schedule, _, _ = make(tmp_path, schedule_text=None)
    assert schedule.next_occurrence(utc(2026, 8, 3, 10, 0)) is None
    schedule, _, _ = make(tmp_path, schedule_text="only prose here\n")
    assert schedule.next_occurrence(utc(2026, 8, 3, 10, 0)) is None


def test_next_occurrence_defaults_to_injected_now(tmp_path):
    schedule, _, _ = make(
        tmp_path,
        "- [ ] daily @ 09:00: Check email\n",
        now=lambda: utc(2026, 8, 3, 8, 0),
    )
    assert schedule.next_occurrence() == utc(2026, 8, 3, 9, 0)


# --- lint_lines ---


def test_lint_returns_broken_task_lines_only(tmp_path):
    schedule, _, _ = make(
        tmp_path,
        "# My schedule\n"
        "some prose about intent\n"
        "- [ ] every other tuesday: water plants\n"
        "- [ ] daily @ 09:00: Check email\n"
        "- [x] daily @ 8pm: checked off so ignored\n"
        "- a plain bullet, not a task\n",
    )
    assert schedule.lint_lines() == ["- [ ] every other tuesday: water plants"]


def test_lint_missing_file_is_empty(tmp_path):
    schedule, _, _ = make(tmp_path, schedule_text=None)
    assert schedule.lint_lines() == []
