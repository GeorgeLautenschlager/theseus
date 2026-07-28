from __future__ import annotations

import time
from unittest.mock import MagicMock

from theseus.stimulus_log import StimulusLog
from theseus.time_observer import TimeObserver


def make(tmp_path, interval=60.0):
    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    try_orient = MagicMock(return_value=True)
    obs = TimeObserver(log, try_orient, self_actor="Tam", interval_seconds=interval)
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

