from __future__ import annotations

import threading

from theseus.stimulus_log import StimulusLog


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
