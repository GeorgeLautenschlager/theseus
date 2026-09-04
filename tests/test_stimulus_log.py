from __future__ import annotations

import threading
from datetime import datetime, timezone

from theseus.stimulus_log import DEFAULT_ORIGIN, StimulusEvent, StimulusLog


def make_log(tmp_path) -> StimulusLog:
    return StimulusLog(path=tmp_path / "stimulus_log.jsonl")


def test_subscribe_hands_each_appended_event_to_the_listener(tmp_path):
    log = make_log(tmp_path)
    seen = []
    log.subscribe(seen.append)

    event = log.append(actor="user", type="chat_message", content={"message": "hi"})

    assert seen == [event]


def test_listener_only_sees_events_appended_after_it_subscribed(tmp_path):
    log = make_log(tmp_path)
    log.append(actor="user", type="chat_message", content={"message": "before"})
    seen = []
    log.subscribe(seen.append)
    log.append(actor="user", type="chat_message", content={"message": "after"})

    assert [e.content["message"] for e in seen] == ["after"]


def test_unsubscribe_stops_the_notifications(tmp_path):
    log = make_log(tmp_path)
    seen = []
    unsubscribe = log.subscribe(seen.append)
    log.append(actor="user", type="chat_message", content={"message": "one"})
    unsubscribe()
    log.append(actor="user", type="chat_message", content={"message": "two"})

    assert len(seen) == 1
    unsubscribe()  # idempotent — a second call is not an error


def test_every_listener_hears_even_when_one_raises(tmp_path, capsys):
    """The log is the bedrock: a buggy listener must not turn a durable append into a
    raise, lose the record, or starve the listeners registered after it."""
    log = make_log(tmp_path)
    seen = []

    def boom(event):
        raise RuntimeError("listener bug")

    log.subscribe(boom)
    log.subscribe(seen.append)

    event = log.append(actor="user", type="chat_message", content={"message": "hi"})

    assert seen == [event]
    assert log.read_all() == [event]
    assert "listener bug" in capsys.readouterr().err


def test_listener_runs_after_the_record_is_durable(tmp_path):
    """The notification is a promise the event survives a crash, so the file must
    already hold it by the time a listener is told about it."""
    log = make_log(tmp_path)
    on_disk = []
    log.subscribe(lambda event: on_disk.append(log.read_all()))

    event = log.append(actor="user", type="chat_message", content={"message": "hi"})

    assert on_disk == [[event]]


def test_listener_fires_on_the_appending_thread(tmp_path):
    """Autocore's listener sets a flag another thread is waiting on; that only works
    if notification is synchronous with the append rather than queued somewhere."""
    log = make_log(tmp_path)
    threads = []
    log.subscribe(lambda event: threads.append(threading.current_thread()))

    appender = threading.Thread(
        target=lambda: log.append(actor="user", type="chat_message", content={})
    )
    appender.start()
    appender.join()

    assert threads == [appender]


def _event(**overrides) -> StimulusEvent:
    fields = dict(
        id="01ABCDEFGHJKMNPQRSTVWXYZ0",
        ts=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        actor="user",
        type="chat_message",
        content={"message": "hi"},
    )
    fields.update(overrides)
    return StimulusEvent(**fields)


def test_to_json_round_trips_every_envelope_field():
    event = _event(
        origin="kitchen-surrogate",
        seq=7,
        appended_ts=datetime(2026, 1, 1, 13, 30, tzinfo=timezone.utc),
    )

    assert StimulusEvent.from_json(event.to_json()) == event


def test_a_pre_change_log_line_parses_with_the_documented_defaults():
    """Every line written before this change lacks the envelope. They stay readable in
    place — no migration script — so the defaults are part of the contract."""
    legacy = (
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ0","ts":"2026-01-01T12:00:00+00:00",'
        '"actor":"user","type":"chat_message","content":{"message":"hi"}}'
    )

    event = StimulusEvent.from_json(legacy, default_origin="kitchen-surrogate")

    assert event.origin == "kitchen-surrogate"
    assert event.seq is None
    assert event.appended_ts == event.ts


def test_from_json_falls_back_to_the_module_default_origin():
    legacy = (
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ0","ts":"2026-01-01T12:00:00+00:00",'
        '"actor":"user","type":"chat_message","content":{}}'
    )

    assert StimulusEvent.from_json(legacy).origin == DEFAULT_ORIGIN


def test_appended_ts_defaults_to_event_ts_when_not_supplied():
    """An event that was never appended by a log still has a usable arrival timestamp,
    so downstream ordering never has to special-case None."""
    event = _event()

    assert event.appended_ts == event.ts


def test_envelope_fields_are_optional_so_existing_construction_sites_still_work():
    event = _event()

    assert event.origin == DEFAULT_ORIGIN
    assert event.seq is None
