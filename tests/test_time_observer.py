from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

from theseus.stimulus_log import StimulusLog
from theseus.time_observer import TimeObserver


def make(tmp_path, interval=60.0, schedule_text=None, now=None):
    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    try_orient = MagicMock(return_value=True)
    schedule_path = tmp_path / "SCHEDULE.md"
    if schedule_text is not None:
        schedule_path.write_text(schedule_text, encoding="utf-8")
    obs = TimeObserver(
        log,
        try_orient,
        self_actor="Tam",
        interval_seconds=interval,
        schedule_path=schedule_path,
        now=(now if now is not None else (lambda: datetime.now(timezone.utc))),
    )
    return obs, log, try_orient


def test_start_checkpoints_past_existing_backlog(tmp_path):
    obs, log, try_orient = make(tmp_path)
    log.append(actor="user", type="chat_message", content={"message": "old"})

    obs.start()          # checkpoint moves to the current tip (the backlog entry)
    obs.stop(timeout=1)
    obs._tick()          # nothing newer than the backlog

    try_orient.assert_not_called()


def test_new_external_event_triggers_orient(tmp_path):
    obs, log, try_orient = make(tmp_path)
    obs.start()
    obs.stop(timeout=1)  # checkpoint = empty-log tip (None)

    log.append(actor="user", type="chat_message", content={"message": "hi"})
    obs._tick()

    try_orient.assert_called_once_with()


def test_core_authored_events_do_not_trigger(tmp_path):
    obs, log, try_orient = make(tmp_path)
    obs.start()
    obs.stop(timeout=1)

    # The agent's own decision output is not a stimulus to wake for.
    log.append(actor="Tam", type="decision", content={"text": "thinking"})
    obs._tick()

    try_orient.assert_not_called()


def test_same_event_triggers_only_once(tmp_path):
    obs, log, try_orient = make(tmp_path)
    obs.start()
    obs.stop(timeout=1)

    log.append(actor="user", type="chat_message", content={"message": "hi"})
    obs._tick()
    obs._tick()  # second wake, nothing new since the checkpoint advanced

    assert try_orient.call_count == 1


def test_stop_interrupts_the_interval_promptly(tmp_path):
    # A 1000s interval would never wake on its own; stop() must not wait it out.
    obs, log, try_orient = make(tmp_path, interval=1000.0)
    obs.start()

    t0 = time.monotonic()
    obs.stop(timeout=2)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0
    assert obs._thread is not None and not obs._thread.is_alive()


def test_contention_leaves_the_event_for_the_next_wake(tmp_path):
    # try_orient returns False (a cycle was already in flight): the event must NOT be
    # consumed — the next wake retries it rather than dropping it.
    obs, log, try_orient = make(tmp_path)
    try_orient.return_value = False
    obs.start()
    obs.stop(timeout=1)

    log.append(actor="user", type="chat_message", content={"message": "hi"})
    obs._tick()  # contention: try_orient called but returned False
    assert try_orient.call_count == 1

    # Cycle finished; the Core is free now. The same event must still trigger.
    try_orient.return_value = True
    obs._tick()
    assert try_orient.call_count == 2


def test_no_schedule_file_is_not_an_error(tmp_path):
    obs, log, try_orient = make(tmp_path)  # no schedule_text -> SCHEDULE.md absent
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_not_called()
    assert log.read_all() == []


def test_due_daily_task_fires_and_stops_refiring(tmp_path):
    now = datetime(2026, 8, 3, 9, 5, tzinfo=timezone.utc)
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] daily @ 09:00: Check email\n",
        now=lambda: now,
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_called_once_with()
    events = log.read_all()
    assert len(events) == 1
    assert events[0].actor == "schedule"
    assert events[0].type == "scheduled_task"
    assert events[0].content == {"message": "Time to Check email"}
    assert "last-fired: 2026-08-03T09:05:00+00:00" in obs.schedule_path.read_text()

    obs._tick()  # same day, already fired -> must not refire

    assert try_orient.call_count == 1
    assert len(log.read_all()) == 1


def test_due_weekly_task_fires_and_stops_refiring(tmp_path):
    now = datetime(2026, 8, 3, 9, 5, tzinfo=timezone.utc)
    weekday_name = now.strftime("%A")
    obs, log, try_orient = make(
        tmp_path,
        schedule_text=f"- [ ] weekly @ {weekday_name} 09:00: Submit timesheet\n",
        now=lambda: now,
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_called_once_with()
    events = log.read_all()
    assert events[0].content == {"message": "Time to Submit timesheet"}

    obs._tick()  # same week, already fired -> must not refire

    assert try_orient.call_count == 1
    assert len(log.read_all()) == 1


def test_due_once_task_fires_and_is_checked_off(tmp_path):
    now = datetime(2026, 8, 5, 14, 1, tzinfo=timezone.utc)
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] once @ 2026-08-05 14:00: Water the plants\n",
        now=lambda: now,
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_called_once_with()
    events = log.read_all()
    assert events[0].content == {"message": "Time to Water the plants"}
    assert obs.schedule_path.read_text().strip() == (
        "- [x] once @ 2026-08-05 14:00: Water the plants"
    )

    obs._tick()  # checked off -> must not refire

    assert try_orient.call_count == 1
    assert len(log.read_all()) == 1


def test_future_scheduled_task_does_not_fire(tmp_path):
    now = datetime(2026, 8, 3, 9, 5, tzinfo=timezone.utc)
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] daily @ 10:00: Check email\n",
        now=lambda: now,
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_not_called()
    assert log.read_all() == []


def test_checked_off_line_is_never_evaluated(tmp_path):
    now = datetime(2026, 8, 3, 9, 5, tzinfo=timezone.utc)
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [x] daily @ 00:00: Old task\n",
        now=lambda: now,
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_not_called()
    assert log.read_all() == []


def test_malformed_line_is_skipped_without_blocking_valid_lines(tmp_path):
    now = datetime(2026, 8, 3, 9, 5, tzinfo=timezone.utc)
    schedule_text = (
        "- [ ] this is not a valid schedule line\n"
        "- [ ] daily @ 09:00: Check email\n"
    )
    obs, log, try_orient = make(tmp_path, schedule_text=schedule_text, now=lambda: now)
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_called_once_with()
    events = log.read_all()
    assert len(events) == 1
    assert events[0].content == {"message": "Time to Check email"}

    contents = obs.schedule_path.read_text()
    assert "- [ ] this is not a valid schedule line" in contents


def test_every_seconds_task_fires_immediately_then_waits_for_interval(tmp_path):
    t = {"now": datetime(2026, 8, 3, 9, 5, 0, tzinfo=timezone.utc)}
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] every 10 seconds: Poll sensor\n",
        now=lambda: t["now"],
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()  # never fired -> due immediately

    try_orient.assert_called_once_with()
    events = log.read_all()
    assert len(events) == 1
    assert events[0].content == {"message": "Time to Poll sensor"}
    assert "last-fired: 2026-08-03T09:05:00+00:00" in obs.schedule_path.read_text()

    obs._tick()  # same instant, interval hasn't elapsed -> must not refire
    assert try_orient.call_count == 1

    t["now"] = datetime(2026, 8, 3, 9, 5, 10, tzinfo=timezone.utc)
    obs._tick()  # interval elapsed -> refires

    assert try_orient.call_count == 2
    assert len(log.read_all()) == 2


def test_every_minutes_and_hours_units_are_parsed(tmp_path):
    now = datetime(2026, 8, 3, 9, 5, tzinfo=timezone.utc)
    schedule_text = (
        "- [ ] every 30 minutes: Check queue depth\n"
        "- [ ] every 4 hours: Rotate logs\n"
    )
    obs, log, try_orient = make(tmp_path, schedule_text=schedule_text, now=lambda: now)
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_called_once_with()  # one tick, two due tasks -> one orient call
    events = log.read_all()
    assert {e.content["message"] for e in events} == {
        "Time to Check queue depth",
        "Time to Rotate logs",
    }


def test_monthly_task_fires_and_refires_next_month(tmp_path):
    t = {"now": datetime(2026, 8, 5, 9, 5, tzinfo=timezone.utc)}
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] monthly @ 5 09:00: Pay rent\n",
        now=lambda: t["now"],
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()
    try_orient.assert_called_once_with()

    t["now"] = datetime(2026, 8, 20, 9, 5, tzinfo=timezone.utc)
    obs._tick()  # same month, already fired -> must not refire
    assert try_orient.call_count == 1

    t["now"] = datetime(2026, 9, 5, 9, 5, tzinfo=timezone.utc)
    obs._tick()  # next month -> refires
    assert try_orient.call_count == 2
    assert len(log.read_all()) == 2


def test_monthly_day_of_month_overflow_clamps(tmp_path):
    now = datetime(2026, 4, 30, 9, 5, tzinfo=timezone.utc)  # April has 30 days
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] monthly @ 31 09:00: Rotate logs\n",
        now=lambda: now,
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_called_once_with()
    assert log.read_all()[0].content == {"message": "Time to Rotate logs"}


def test_monthly_day_of_month_overflow_clamps_in_february(tmp_path):
    now = datetime(2026, 2, 28, 9, 5, tzinfo=timezone.utc)  # 2026 is not a leap year
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] monthly @ 30 09:00: Rotate logs\n",
        now=lambda: now,
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_called_once_with()
    assert log.read_all()[0].content == {"message": "Time to Rotate logs"}


def test_quarterly_fires_only_at_quarter_start_and_not_other_months(tmp_path):
    t = {"now": datetime(2026, 1, 1, 9, 5, tzinfo=timezone.utc)}
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] quarterly @ 1 09:00: File quarterly report\n",
        now=lambda: t["now"],
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()  # Q1 start
    try_orient.assert_called_once_with()

    t["now"] = datetime(2026, 2, 1, 9, 5, tzinfo=timezone.utc)
    obs._tick()  # still Q1 -> must not refire
    assert try_orient.call_count == 1

    t["now"] = datetime(2026, 3, 15, 9, 5, tzinfo=timezone.utc)
    obs._tick()  # still Q1 -> must not refire
    assert try_orient.call_count == 1

    t["now"] = datetime(2026, 4, 1, 9, 5, tzinfo=timezone.utc)
    obs._tick()  # Q2 start -> refires
    assert try_orient.call_count == 2
    assert len(log.read_all()) == 2


def test_quarterly_day_of_month_overflow_clamps_in_april(tmp_path):
    now = datetime(2026, 4, 30, 9, 5, tzinfo=timezone.utc)  # April is a quarter-start
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] quarterly @ 31 09:00: File quarterly report\n",
        now=lambda: now,
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_called_once_with()
    assert log.read_all()[0].content == {"message": "Time to File quarterly report"}


def test_annually_fires_on_right_date_and_not_again_same_year(tmp_path):
    t = {"now": datetime(2026, 12, 25, 9, 5, tzinfo=timezone.utc)}
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] annually @ 12-25 09:00: Send holiday cards\n",
        now=lambda: t["now"],
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()
    try_orient.assert_called_once_with()

    t["now"] = datetime(2026, 12, 26, 9, 5, tzinfo=timezone.utc)
    obs._tick()  # same year, already fired -> must not refire
    assert try_orient.call_count == 1

    t["now"] = datetime(2027, 12, 25, 9, 5, tzinfo=timezone.utc)
    obs._tick()  # next year -> refires
    assert try_orient.call_count == 2
    assert len(log.read_all()) == 2


def test_annually_looks_back_to_last_years_date_when_this_years_hasnt_happened(tmp_path):
    now = datetime(2026, 1, 5, 9, 5, tzinfo=timezone.utc)  # before this year's Jan 10
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] annually @ 01-10 09:00: Renew domain\n",
        now=lambda: now,
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_called_once_with()
    assert log.read_all()[0].content == {"message": "Time to Renew domain"}


def test_annually_day_of_month_overflow_clamps(tmp_path):
    now = datetime(2026, 2, 28, 9, 5, tzinfo=timezone.utc)  # 2026 is not a leap year
    obs, log, try_orient = make(
        tmp_path,
        schedule_text="- [ ] annually @ 02-30 09:00: Weird anniversary\n",
        now=lambda: now,
    )
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_called_once_with()
    assert log.read_all()[0].content == {"message": "Time to Weird anniversary"}


def test_malformed_every_line_is_skipped_without_blocking_valid_lines(tmp_path):
    now = datetime(2026, 8, 3, 9, 5, tzinfo=timezone.utc)
    schedule_text = (
        "- [ ] every ten minutes: Bad interval\n"
        "- [ ] every 30 minutes: Check queue depth\n"
    )
    obs, log, try_orient = make(tmp_path, schedule_text=schedule_text, now=lambda: now)
    obs.start()
    obs.stop(timeout=1)

    obs._tick()

    try_orient.assert_called_once_with()
    events = log.read_all()
    assert len(events) == 1
    assert events[0].content == {"message": "Time to Check queue depth"}

    contents = obs.schedule_path.read_text()
    assert "- [ ] every ten minutes: Bad interval" in contents
