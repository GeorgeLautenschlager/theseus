from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from theseus.stimulus_log import DEFAULT_ORIGIN, StimulusEvent, StimulusLog, new_id


def make_log(tmp_path) -> StimulusLog:
    return StimulusLog(path=tmp_path / "stimulus_log.jsonl")


# --- Listeners ------------------------------------------------------------------
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


def test_a_listener_may_append_without_deadlocking(tmp_path):
    """The append lock is released before listeners are notified, precisely so a listener
    can append. This pins that boundary: move `_notify` inside the lock and this hangs."""
    log = make_log(tmp_path)
    echoed = []

    def echo_once(event):
        if event.type == "chat_message":
            echoed.append(log.append(actor="tam", type="echo", content={}))

    log.subscribe(echo_once)
    log.append(actor="user", type="chat_message", content={})

    assert [e.seq for e in log.read_all()] == [1, 2]
    assert [e.type for e in log.read_all()] == ["chat_message", "echo"]
    assert len(echoed) == 1


# --- Event envelope -------------------------------------------------------------
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


def test_to_json_writes_the_wire_format_key_names():
    """The round-trip test above is symmetric — renaming a key in both directions keeps it
    green. These bytes are a cross-process contract, so the names themselves are pinned here."""
    event = _event(
        origin="kitchen-surrogate",
        seq=7,
        appended_ts=datetime(2026, 1, 1, 13, 30, tzinfo=timezone.utc),
    )

    wire = json.loads(event.to_json())

    assert set(wire) == {
        "id",
        "ts",
        "actor",
        "type",
        "content",
        "origin",
        "seq",
        "appended_ts",
    }
    assert wire["origin"] == "kitchen-surrogate"
    assert wire["seq"] == 7
    assert wire["ts"] == "2026-01-01T12:00:00+00:00"
    assert wire["appended_ts"] == "2026-01-01T13:30:00+00:00"


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
    """The subject here is the construction call itself: `_event()` passes only the five
    original keyword arguments, which is exactly how `tests/test_debug_pagination.py` and
    `tests/test_debug_row_rendering.py` build events. If the envelope fields ever lose their
    defaults, this stops constructing before it reaches an assertion."""
    event = _event()

    assert event.origin == DEFAULT_ORIGIN
    assert event.seq is None


# --- Origin and seq allocation --------------------------------------------------
def test_local_appends_get_a_monotonic_seq_starting_at_one(tmp_path):
    log = make_log(tmp_path)

    events = [
        log.append(actor="user", type="chat_message", content={"n": n}) for n in range(3)
    ]

    assert [e.seq for e in events] == [1, 2, 3]
    assert {e.origin for e in events} == {DEFAULT_ORIGIN}


def test_seq_is_monotonic_across_a_restart(tmp_path):
    """The log file is the only durable state. A sidecar counter that disagreed with it
    after a crash would either drop real events or issue the same seq twice."""
    path = tmp_path / "stimulus_log.jsonl"
    first = StimulusLog(path=path)
    first.append(actor="user", type="chat_message", content={})
    first.append(actor="user", type="chat_message", content={})

    reopened = StimulusLog(path=path)
    event = reopened.append(actor="user", type="chat_message", content={})

    assert event.seq == 3


def test_the_log_stamps_its_own_origin_on_local_appends(tmp_path):
    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl", origin="kitchen-surrogate")

    event = log.append(actor="user", type="chat_message", content={})

    assert event.origin == "kitchen-surrogate"


def test_id_order_matches_append_order_when_event_ts_is_backdated(tmp_path):
    """A replicated event can carry a skewed or hours-old event_ts. The id is minted from
    appended_ts so it never sorts into the middle of the log. The consumer that actually
    depends on this is `older_batch` in `theseus/web/debug_pagination.py`, which bisects a
    list of ids and therefore requires it to be sorted ascending; `read_range` here and the
    debug tail cursor read id order too."""
    log = make_log(tmp_path)
    backdated = datetime.now(timezone.utc) - timedelta(hours=1)

    old = log.append(actor="user", type="chat_message", content={"n": 1}, ts=backdated)
    time.sleep(0.002)  # ULIDs are millisecond-resolution; keep the two ids distinguishable
    new = log.append(actor="user", type="chat_message", content={"n": 2})

    assert old.id < new.id
    # Minted from arrival, not from the backdated event clock: an id minted an hour ago
    # would sort below this one and land in the middle of the log.
    assert old.id > new_id(int(backdated.timestamp() * 1000))
    assert log.read_range(old.id, new.id) == [old, new]


def test_appended_ts_is_minted_by_the_log_not_taken_from_the_caller(tmp_path):
    log = make_log(tmp_path)
    backdated = datetime.now(timezone.utc) - timedelta(hours=1)

    event = log.append(actor="user", type="chat_message", content={}, ts=backdated)

    assert event.ts == backdated
    assert event.appended_ts > event.ts


def test_a_replicated_append_keeps_the_origin_and_seq_its_producer_assigned(tmp_path):
    log = make_log(tmp_path)

    event = log.append(
        actor="user",
        type="chat_message",
        content={},
        origin="kitchen-surrogate",
        seq=41,
    )

    assert (event.origin, event.seq) == ("kitchen-surrogate", 41)
    assert log.read_all() == [event]


def test_a_replicated_append_must_carry_a_seq(tmp_path):
    """This log can only allocate for its own origin — inventing a seq for someone else's
    would collide with the one the producer already assigned."""
    log = make_log(tmp_path)

    with pytest.raises(ValueError):
        log.append(
            actor="user", type="chat_message", content={}, origin="kitchen-surrogate"
        )


def test_a_replicated_seq_does_not_disturb_the_local_counter(tmp_path):
    log = make_log(tmp_path)
    log.append(actor="user", type="chat_message", content={})
    log.append(
        actor="user", type="chat_message", content={}, origin="android-01", seq=900
    )

    event = log.append(actor="user", type="chat_message", content={})

    assert event.seq == 2


def test_seq_recovery_ignores_other_origins(tmp_path):
    """Recovery counts only this log's own origin — a surrogate's seq 900 must not push
    the host's own counter into the nine-hundreds."""
    path = tmp_path / "stimulus_log.jsonl"
    first = StimulusLog(path=path)
    first.append(actor="user", type="chat_message", content={})
    first.append(
        actor="user", type="chat_message", content={}, origin="android-01", seq=900
    )

    event = StimulusLog(path=path).append(actor="user", type="chat_message", content={})

    assert event.seq == 2


def test_legacy_lines_read_back_with_the_logs_own_origin(tmp_path):
    path = tmp_path / "stimulus_log.jsonl"
    path.write_text(
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ0","ts":"2026-01-01T12:00:00+00:00",'
        '"actor":"user","type":"chat_message","content":{}}\n',
        encoding="utf-8",
    )
    log = StimulusLog(path=path, origin="kitchen-surrogate")

    (event,) = log.read_all()

    assert event.origin == "kitchen-surrogate"
    assert event.seq is None
    assert event.appended_ts == event.ts


def test_appending_to_a_log_of_pre_envelope_lines_starts_seq_at_one(tmp_path):
    """The upgrade path every deployed agent takes: a log full of lines written before the
    envelope existed, then a restart on this code. Those lines carry no seq, recovery skips
    them, and numbering starts at 1 behind them — a gap, which the protocol permits."""
    path = tmp_path / "stimulus_log.jsonl"
    path.write_text(
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ0","ts":"2026-01-01T12:00:00+00:00",'
        '"actor":"user","type":"chat_message","content":{}}\n'
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ1","ts":"2026-01-01T12:01:00+00:00",'
        '"actor":"tam","type":"chat_message","content":{}}\n',
        encoding="utf-8",
    )
    log = StimulusLog(path=path)

    event = log.append(actor="user", type="chat_message", content={})

    assert event.seq == 1
    assert [e.seq for e in log.read_all()] == [None, None, 1]


def test_concurrent_appends_never_reuse_a_seq(tmp_path):
    log = make_log(tmp_path)
    events: list[StimulusEvent] = []
    barrier = threading.Barrier(8)

    def appender() -> None:
        barrier.wait()
        events.append(log.append(actor="user", type="chat_message", content={}))

    threads = [threading.Thread(target=appender) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(e.seq for e in events) == [1, 2, 3, 4, 5, 6, 7, 8]
    # The returned objects alone would pass with a torn or short file; only reading it back
    # proves the lock covered the write as well as the allocation.
    assert sorted(e.seq for e in log.read_all()) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_an_own_origin_append_may_not_carry_a_seq(tmp_path):
    """Two numbering authorities on one origin name is exactly what breaks duplicate
    suppression downstream — and since host and surrogate both default to the same origin
    name, it is what an unconfigured deployment would otherwise produce silently."""
    log = make_log(tmp_path)

    with pytest.raises(ValueError):
        log.append(
            actor="user", type="chat_message", content={}, origin=DEFAULT_ORIGIN, seq=1
        )


def test_a_log_cannot_be_configured_with_an_empty_origin(tmp_path):
    """An origin is configuration, so the error belongs at construction rather than on the
    first append."""
    with pytest.raises(ValueError):
        StimulusLog(path=tmp_path / "stimulus_log.jsonl", origin="")


def test_an_empty_origin_is_rejected(tmp_path):
    log = make_log(tmp_path)

    with pytest.raises(ValueError):
        log.append(actor="user", type="chat_message", content={}, origin="", seq=1)


def test_a_replicated_seq_below_one_is_rejected(tmp_path):
    """Starting at 1 keeps 0 below every real seq, so a reader tracking what it has
    accepted has a safe comparison floor. ("Nothing seen yet" is its own answer, distinct
    from 0 — see `HighWaterMarks.high_water`.)"""
    log = make_log(tmp_path)

    with pytest.raises(ValueError):
        log.append(
            actor="user", type="chat_message", content={}, origin="android-01", seq=0
        )


def test_a_locally_minted_ts_is_issued_in_lock_order(tmp_path):
    """`ts` is minted under the append lock, so arrival order and chronology agree for a
    single producer. Minted before the lock, a thread that blocked on another's fsync
    would land after an event carrying a later clock — an inversion with no surrogate
    anywhere in sight."""
    log = make_log(tmp_path)
    barrier = threading.Barrier(8)

    def appender() -> None:
        barrier.wait()
        for _ in range(40):
            log.append(actor="user", type="chat_message", content={})

    threads = [threading.Thread(target=appender) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    events = sorted(log.read_all(), key=lambda e: e.seq)
    assert [e.ts for e in events] == sorted(e.ts for e in events)


def test_a_naive_timestamp_from_a_foreign_line_is_read_as_aware(tmp_path):
    """A line this log wrote always carries an offset. One from somewhere else need not,
    and a naive datetime mixed with an aware one raises inside any sort — which since #28
    means every context assembly, and so the whole cognitive loop."""
    path = tmp_path / "stimulus_log.jsonl"
    path.write_text(
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ0","ts":"2026-01-01T12:00:00",'
        '"actor":"peer","type":"observation","content":{},'
        '"origin":"android-01","seq":1}\n',
        encoding="utf-8",
    )
    log = StimulusLog(path=path)

    (event,) = log.read_all()

    assert event.ts.tzinfo is not None
    assert event.appended_ts.tzinfo is not None
    # And the pair a sort would compare is now comparable.
    assert event.ts <= datetime.now(timezone.utc)


def _replicated(seq: int, message: str = "hi", origin: str = "kitchen-surrogate"):
    """One event shaped as it arrives off the wire: the producer's id and appended_ts are
    placeholders, because this log re-mints both."""
    return StimulusEvent(
        id="01PRODUCERSIDWILLBEDROPPED",
        # Midnight, not a wall-clock "now": the re-mint test asserts appended_ts (minted from
        # now) sorts after this producer ts, so the fixture must sit in the past.
        ts=datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=seq),
        actor="sensor",
        type="observation",
        content={"message": message},
        origin=origin,
        seq=seq,
    )


def test_append_many_writes_the_whole_batch(tmp_path):
    log = make_log(tmp_path)

    appended = log.append_many([_replicated(1), _replicated(2), _replicated(3)])

    assert [e.seq for e in appended] == [1, 2, 3]
    assert [e.seq for e in log.read_all()] == [1, 2, 3]
    assert {e.origin for e in log.read_all()} == {"kitchen-surrogate"}


def test_append_many_re_mints_id_and_appended_ts(tmp_path):
    """Identity across nodes is (origin, seq), never id. The producer's id is its own."""
    log = make_log(tmp_path)

    (appended,) = log.append_many([_replicated(1)])

    assert appended.id != "01PRODUCERSIDWILLBEDROPPED"
    assert appended.appended_ts > appended.ts
    assert appended.ts == datetime(2026, 9, 4, 0, 0, 1, tzinfo=timezone.utc)


def test_append_many_ids_are_strictly_increasing(tmp_path):
    """A whole batch lands inside one millisecond, and `older_batch` bisects a list of ids
    expecting it sorted. Independent random suffixes would collide with that; a batch
    mints a monotonic run instead."""
    log = make_log(tmp_path)

    appended = log.append_many([_replicated(n) for n in range(1, 51)])

    ids = [e.id for e in appended]
    assert ids == sorted(ids)
    assert len(set(ids)) == 50


def test_append_many_ids_sort_above_everything_already_on_the_log(tmp_path):
    log = make_log(tmp_path)
    earlier = log.append(actor="george", type="exchange", content={})

    appended = log.append_many([_replicated(1), _replicated(2)])

    assert earlier.id < appended[0].id


def test_append_many_is_one_fsync_for_the_whole_batch(tmp_path, monkeypatch):
    """All-or-nothing application: the spec forbids a partial state the surrogate would
    have to reason about. One open, one write, one fsync."""
    log = make_log(tmp_path)
    fsyncs = []
    real_fsync = os.fsync
    monkeypatch.setattr(
        os, "fsync", lambda fd: (fsyncs.append(fd), real_fsync(fd))[1]
    )

    log.append_many([_replicated(n) for n in range(1, 11)])

    assert len(fsyncs) == 1


def test_append_many_notifies_listeners_for_every_event(tmp_path):
    log = make_log(tmp_path)
    seen = []
    log.subscribe(seen.append)

    appended = log.append_many([_replicated(1), _replicated(2)])

    assert seen == appended


def test_append_many_notifies_after_the_whole_batch_is_durable(tmp_path):
    """A listener must never see event 1 while event 2 of the same batch could still be
    lost — the batch is the unit that is all-or-nothing."""
    log = make_log(tmp_path)
    on_disk = []
    log.subscribe(lambda event: on_disk.append(len(log.read_all())))

    log.append_many([_replicated(1), _replicated(2)])

    assert on_disk == [2, 2]


def test_append_many_rejects_a_batch_before_writing_any_of_it(tmp_path):
    """Validation ahead of the lock, exactly as `append` does it: a rejected batch leaves
    nothing on disk, so the surrogate has no partial state to reason about."""
    log = make_log(tmp_path)

    with pytest.raises(ValueError):
        log.append_many([_replicated(1), _replicated(2, origin=log.origin)])

    assert log.read_all() == []


def test_append_many_rejects_a_batch_spanning_two_origins(tmp_path):
    """The spec's batch is a contiguous seq range from exactly one origin. Two origins in
    one batch would make the all-or-nothing guarantee span two dedupe streams."""
    log = make_log(tmp_path)

    with pytest.raises(ValueError, match="exactly one origin"):
        log.append_many([_replicated(1), _replicated(1, origin="android-01")])


def test_append_many_of_nothing_is_a_no_op(tmp_path):
    log = make_log(tmp_path)

    assert log.append_many([]) == []
    assert log.read_all() == []


def test_append_many_does_not_disturb_the_local_counter(tmp_path):
    log = make_log(tmp_path)
    log.append(actor="george", type="exchange", content={})

    log.append_many([_replicated(90), _replicated(91)])

    assert log.append(actor="george", type="exchange", content={}).seq == 2
