"""BufferPolicy's contract, and BufferedStimulusLog's: evict oldest-first under pressure."""

from __future__ import annotations

import dataclasses
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from theseus.replication_events import GAP
from theseus.surrogates.buffer import BufferedStimulusLog, BufferPolicy
from theseus.stimulus_log import DEFAULT_ORIGIN, StimulusEvent, StimulusLog


def test_defaults_are_the_documented_ones():
    policy = BufferPolicy()
    assert policy.max_bytes == 256 * 1024 * 1024
    assert policy.low_water == 0.8


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_max_bytes_refused(value):
    with pytest.raises(ValueError, match="max_bytes"):
        BufferPolicy(max_bytes=value)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.5, 1.5])
def test_low_water_outside_open_interval_refused(value):
    with pytest.raises(ValueError, match="low_water"):
        BufferPolicy(low_water=value)


@pytest.mark.parametrize("value", [0.99, 0.01])
def test_legal_boundary_values_accepted(value):
    assert BufferPolicy(low_water=value).low_water == value


def test_policy_is_frozen():
    policy = BufferPolicy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_bytes = 1  # type: ignore[misc]


def _event(n: int, fill: str = "x") -> dict:
    return {"n": n, "fill": fill}


def _replicated(n: int, origin: str = "host", size: int = 0) -> StimulusEvent:
    return StimulusEvent(
        id=f"REP{n:026d}",
        ts=datetime.now(timezone.utc),
        actor="host",
        type="observation",
        content={"n": n, "fill": "y" * size},
        origin=origin,
        seq=n,
    )


def test_under_budget_never_rewrites(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=100_000))
    inode = path.stat().st_ino  # captured before: comparing to a later stat of the same path is the only way a rewrite can show
    appended = [log.append("env", "observation", _event(i)) for i in range(10)]
    assert [e.seq for e in log.read_all()] == [e.seq for e in appended]
    assert path.stat().st_ino == inode  # never rewritten


def test_crossing_threshold_evicts_oldest_first(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    # One local line is ~120 bytes; 12 events clears a 1000-byte cap.
    policy = BufferPolicy(max_bytes=1000, low_water=0.5)
    log = BufferedStimulusLog(path, policy=policy)
    appended = [log.append("env", "observation", _event(i)) for i in range(12)]
    survivors = log.read_all()
    seqs = [e.seq for e in survivors]
    # The survivors are a contiguous suffix — the newest events — and the evicted
    # ones are the oldest, identified by seq, not by count.
    assert seqs == list(range(seqs[0], seqs[-1] + 1))
    assert seqs[-1] == appended[-1].seq
    assert set(seqs).isdisjoint({appended[0].seq, appended[1].seq})


def test_eviction_lands_under_low_water_not_just_cap(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    policy = BufferPolicy(max_bytes=4000, low_water=0.5)
    log = BufferedStimulusLog(path, policy=policy)
    sizes = []
    for i in range(12):
        log.append("env", "observation", _event(i, fill="x" * 400))
        sizes.append(path.stat().st_size)
    # A shrink step is an eviction; the moment it matters is right after it, where a
    # rule that stopped at max_bytes instead of low_water would re-evict on nearly
    # every subsequent append.
    shrank = [(a, b) for a, b in zip(sizes, sizes[1:]) if b < a]
    assert shrank
    assert all(b <= policy.max_bytes * policy.low_water for _, b in shrank)
    assert all(a <= policy.max_bytes for a in sizes)


def test_log_readable_and_well_formed_after_eviction(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    for i in range(12):
        log.append("env", "observation", _event(i))
    survivors = [e for e in log.read_all() if e.type != GAP]
    assert survivors  # raises on any torn or interior-corrupt line
    assert all(e.content["n"] >= survivors[0].content["n"] for e in survivors)
    assert [e.content["n"] for e in survivors] == sorted(
        e.content["n"] for e in survivors
    )


def test_seq_counter_survives_eviction(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    policy = BufferPolicy(max_bytes=2000, low_water=0.5)
    # Own seqs 1..5 written by a previous incarnation, then a fresh log on the same
    # file that has not yet allocated a seq of its own: a wall of big replicated
    # events pushes the buffer over the cap and evicts every own-origin event. A lazy
    # recovery *after* the truncation would scan the shrunken file, find no own seq,
    # and reissue seq 1 — a duplicate under an identity the protocol assumes is
    # unique. It only passes because the counter was recovered before the file shrank.
    warm = BufferedStimulusLog(path, policy=policy)
    for i in range(5):
        warm.append("env", "observation", _event(i))
    log = BufferedStimulusLog(path, policy=policy)
    log.append_many([_replicated(i, size=200) for i in range(1, 21)])
    # Own observations are evicted; the only own-origin event left is the gap marker
    # declaring their range (seq 6, minted from the counter recovered pre-shrink).
    assert not any(
        e.origin == DEFAULT_ORIGIN and e.type != GAP for e in log.read_all()
    )
    marker = [e for e in log.read_all() if e.type == GAP][-1]
    assert marker.seq == 6
    assert log._recover_next_seq() == 7  # the marker's seq now sits on the tape
    assert log.append("env", "observation", _event(99)).seq == 7


def test_appends_continue_during_pressure(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    last = None
    for i in range(200):  # crosses the threshold many times over
        last = log.append("env", "observation", _event(i))
    assert log.read_all()[-1].seq == last.seq


def test_byte_budget_measured_in_bytes_not_characters(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    # CJK payloads: 3 UTF-8 bytes per character, so a character-counting
    # implementation measures each line at roughly a third of its on-disk size.
    # The payload must dominate the ASCII overhead or the two measurements don't
    # separate: a 1000-char CJK fill is ~1150 chars but ~3150 bytes per line.
    # With max_bytes=12000 and low_water=0.5, a char-counting impl never evicts
    # at all (6 lines total only ~6900 chars, under the 6000-char floor) while
    # the real file sits at ~18900 bytes — past the 12000-byte cap. Only byte
    # counting keeps the file within budget. Assert on st_size because that is
    # exactly what character counting gets wrong.
    policy = BufferPolicy(max_bytes=12_000, low_water=0.5)
    log = BufferedStimulusLog(path, policy=policy)
    for i in range(6):
        log.append("env", "observation", _event(i, fill="語" * 1000))
    assert path.stat().st_size <= policy.max_bytes  # the thing character counting gets wrong
    log.read_all()  # survivors parse


def test_lone_oversized_event_is_kept(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=10, low_water=0.5))
    event = log.append("env", "observation", _event(1, fill="y" * 500))
    assert log.read_all() == [event]
    assert path.stat().st_size > 10  # over its cap rather than empty


def test_eviction_safe_under_concurrent_appends(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=2000, low_water=0.5))
    errors: list[BaseException] = []

    def worker(base: int) -> None:
        try:
            for i in range(60):
                log.append("env", "observation", _event(base + i))
        except BaseException as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k * 1000,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    events = log.read_all()  # raises on any torn line
    own_seqs = [e.seq for e in events if e.origin == DEFAULT_ORIGIN]
    assert len(own_seqs) == len(set(own_seqs))  # no duplicate seqs


def test_eviction_preserves_file_permissions(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    policy = BufferPolicy(max_bytes=1000, low_water=0.5)
    log = BufferedStimulusLog(path, policy=policy)
    log.append("env", "observation", _event(0))
    # Distinctive mode, not the umask default — otherwise the test could pass by
    # coincidence of the temp file happening to match.
    os.chmod(path, 0o640)
    for i in range(1, 12):  # crosses the threshold, forcing an eviction
        log.append("env", "observation", _event(i))
    assert path.stat().st_mode & 0o777 == 0o640


def test_plain_stimulus_log_never_evicts(tmp_path: Path):
    path = tmp_path / "plain.jsonl"
    log = StimulusLog(path)  # no policy argument exists to pass it
    for i in range(50):
        log.append("env", "observation", _event(i, fill="y" * 100))
    assert path.stat().st_size > 2000  # grew without any cap


# --- Task 3: eviction declares what storage pressure took -------------------


def _gap_markers(events: list[StimulusEvent]) -> list[StimulusEvent]:
    return [e for e in events if e.type == GAP]



def _drive_one_eviction(log: BufferedStimulusLog, start: int) -> list[StimulusEvent]:
    """Append small events until exactly one eviction (one new gap marker) has
    landed, returning the events appended since the call began. Detects the eviction
    via a marker listener rather than re-reading the file each iteration — a scan of
    the whole growing file per append made the ceiling unreachable in any real time.
    Robust to being called on a log that already holds markers."""
    heard: list[StimulusEvent] = []
    # Unsubscribed on the way out: the helper is called more than once against one log,
    # and a listener left behind keeps firing into a list nobody reads again.
    unsubscribe = log.subscribe(lambda e: e.type == GAP and heard.append(e))
    appended: list[StimulusEvent] = []
    i = start
    try:
        while not heard:
            appended.append(log.append("env", "observation", _event(i)))
            i += 1
            # Ceiling is 2000 appends (~240 KB here). A mutation that kills marker
            # emission must fail here, not hang.
            if len(appended) > 2000:
                raise AssertionError(f"no new gap marker after {len(appended)} appends")
    finally:
        unsubscribe()
    return appended


def test_eviction_declares_the_range_it_dropped(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    appended = _drive_one_eviction(log, 0)
    survivors = log.read_all()
    assert len(_gap_markers(survivors)) == 1
    marker = _gap_markers(survivors)[-1]
    kept_seqs = {e.seq for e in survivors if e.type != GAP}
    evicted = [e.seq for e in appended if e.seq not in kept_seqs]
    assert evicted, "the test must actually have evicted something"
    assert marker.content["from_seq"] == min(evicted)
    assert marker.content["to_seq"] == max(evicted)
    assert marker.content["reason"] == "storage_pressure"


def test_marker_span_is_the_evicted_events_own_clock(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    # Explicit ts values far from now, so a wall-clock implementation cannot pass.
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    appended = []
    for i in range(2000):  # bounded for the same reason _drive_one_eviction is
        if _gap_markers(log.read_all()):
            break
        appended.append(
            log.append(
                "env", "observation", _event(i), ts=base + timedelta(minutes=i)
            )
        )
    else:
        raise AssertionError("no gap marker after 2000 appends")
    survivors = log.read_all()
    marker = _gap_markers(survivors)[-1]
    kept_seqs = {e.seq for e in survivors if e.type != GAP}
    evicted = [e for e in appended if e.seq not in kept_seqs]
    assert datetime.fromisoformat(marker.content["span_start"]) == min(e.ts for e in evicted)
    assert datetime.fromisoformat(marker.content["span_end"]) == max(e.ts for e in evicted)
    assert datetime.fromisoformat(marker.content["span_start"]) < datetime.now(timezone.utc) - timedelta(days=1000)


def test_marker_is_declared_not_inferred(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    _drive_one_eviction(log, 0)
    marker = _gap_markers(log.read_all())[-1]
    assert marker.content["declared"] is True
    assert marker.content["origin"] == DEFAULT_ORIGIN


def test_marker_seq_is_above_every_survivor(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    _drive_one_eviction(log, 0)
    survivors = log.read_all()
    marker = _gap_markers(survivors)[-1]
    other = [e.seq for e in survivors if e.type != GAP]
    assert all(marker.seq > s for s in other)


def test_truncation_and_marker_are_one_act(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    replaces: list[object] = []
    real = os.replace

    def counting(src, dst):
        replaces.append(dst)
        return real(src, dst)

    monkeypatch.setattr(os, "replace", counting)
    _drive_one_eviction(log, 0)  # stops at the first eviction
    assert len(replaces) == 1  # one eviction, one replace
    events = log.read_all()  # one file read shows both survivors and marker
    kept = [e for e in events if e.type != GAP]
    assert kept and _gap_markers(events)
    assert kept[-1].seq < _gap_markers(events)[-1].seq


def test_eviction_of_only_foreign_events_emits_no_marker(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=2000, low_water=0.5))
    log.append_many([_replicated(i, size=200) for i in range(1, 21)])
    assert path.stat().st_size <= log._policy.max_bytes  # rewritten, i.e. evicted
    assert not _gap_markers(log.read_all())


def test_unsequenced_events_evicted_but_not_described(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    # Hand-write an own-origin event with seq null: it predates the envelope and has
    # no range to name in a marker.
    ghost = StimulusEvent(
        id="GHOST000000000000000000000",
        ts=datetime(2020, 1, 1, tzinfo=timezone.utc),
        actor="env",
        type="observation",
        content={"n": -1},
        origin=DEFAULT_ORIGIN,
        seq=None,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(ghost.to_json() + "\n")
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=2000, low_water=0.5))
    log.append_many([_replicated(i, size=200) for i in range(1, 21)])
    events = log.read_all()
    assert ghost.id not in {e.id for e in events}  # evicted
    assert not _gap_markers(events)  # and never described


def test_listener_hears_the_marker(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    heard: list[StimulusEvent] = []
    log.subscribe(heard.append)
    # Subscribed before any append: the eviction must not be able to land before the
    # listener exists, or the test watches a window in which the behaviour cannot fire.
    appended = _drive_one_eviction(log, 0)
    gaps = [e for e in heard if e.type == GAP]
    assert len(gaps) == 1
    assert any(e.type != GAP for e in heard)  # survivors also notified
    assert gaps[0] in log.read_all()  # durable by the time it was announced


def test_listener_that_appends_does_not_deadlock(tmp_path: Path):
    # The house pattern: a listener that appends on seeing an event. Holding
    # `_append_lock` across the callback would self-deadlock the same thread on
    # the plain Lock; run the drive on a worker thread so a regression fails
    # with a join timeout instead of hanging the suite.
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    acked = threading.Event()

    def react(event: StimulusEvent) -> None:
        if event.type == GAP:
            log.append("env", "observation", _event(10 ** 6))
            acked.set()

    log.subscribe(react)
    # daemon=True is load-bearing: if a regression deadlocks the worker it will never be
    # joinable, so the interpreter must be allowed to exit and surface the failed assertion
    # instead of hanging at shutdown waiting for a wedged thread.
    worker = threading.Thread(target=_drive_one_eviction, args=(log, 0), daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "listener append deadlocked the eviction"
    assert acked.is_set()
    assert any(e.content.get("n") == 10 ** 6 for e in log.read_all())  # the listener's append landed


def test_successive_evictions_declare_successive_ranges(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    next_i = 0
    seen_seq: int | None = None
    ranges: list[tuple[int, int]] = []
    while len(ranges) < 2:
        # An eviction can take events that were already on file before this round's
        # appends, so the evicted set is everything seen before minus the survivors.
        before = {e.seq for e in log.read_all() if e.type != GAP}
        appended = _drive_one_eviction(log, next_i)
        next_i += len(appended)
        marker = _gap_markers(log.read_all())[-1]
        assert marker.seq != seen_seq  # a genuinely new eviction, not the old marker
        seen_seq = marker.seq
        kept = {e.seq for e in log.read_all() if e.type != GAP}
        evicted = before | {e.seq for e in appended}
        evicted -= kept
        assert marker.content["from_seq"] == min(evicted)
        assert marker.content["to_seq"] == max(evicted)
        ranges.append((marker.content["from_seq"], marker.content["to_seq"]))
    (a, b) = ranges
    assert a[1] < b[0]  # disjoint and ascending


# --- review findings: eviction failure is disk pressure, not a crash ---------


def test_failed_eviction_does_not_break_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    for i in range(12):  # well over the cap; appends have been evicting
        log.append("env", "observation", _event(i))

    def no_space(*args: object, **kwargs: object) -> object:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tempfile, "mkstemp", no_space)
    second = log.append("env", "observation", _event(99))  # triggers a failing eviction
    assert second.seq > 0  # the append itself succeeded
    assert {e.content["n"] for e in log.read_all() if e.type != GAP} >= {11, 99}


def test_appends_keep_landing_while_eviction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))

    def no_space(*args: object, **kwargs: object) -> object:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tempfile, "mkstemp", no_space)
    appended = [log.append("env", "observation", _event(i)) for i in range(50)]
    assert path.stat().st_size > 1000  # sits over its cap, as the failing eviction implies
    on_disk = [e.seq for e in log.read_all() if e.type != GAP]
    assert on_disk == [e.seq for e in appended]  # every event landed, none lost


def test_non_oserror_from_eviction_still_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=1000, low_water=0.5))
    for i in range(12):  # well over the cap; each append has been evicting
        log.append("env", "observation", _event(i))

    def bug(*args: object, **kwargs: object) -> object:
        raise RuntimeError("a real programming error, not disk pressure")

    monkeypatch.setattr(tempfile, "mkstemp", bug)
    with pytest.raises(RuntimeError):
        log.append("env", "observation", _event(1))


def test_failed_eviction_does_not_burn_a_seq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "log.jsonl"
    log = BufferedStimulusLog(path, policy=BufferPolicy(max_bytes=5000, low_water=0.5))
    for i in range(5):
        log.append("env", "observation", _event(i))
    seqs_before = [e.seq for e in log.read_all() if e.origin == DEFAULT_ORIGIN]
    assert seqs_before == [1, 2, 3, 4, 5]

    def no_space(*args: object, **kwargs: object) -> object:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tempfile, "mkstemp", no_space)
    for i in range(10, 30):  # crosses the threshold; eviction fails every time
        log.append("env", "observation", _event(i))
    monkeypatch.undo()
    # The next successful append must take the seq the failed evictions peeked but
    # never used — no hole in the own-origin sequence on the tape.
    nxt = log.append("env", "observation", _event(99))
    all_own = [e.seq for e in log.read_all() if e.origin == DEFAULT_ORIGIN]
    # Appends 6..25 each took one seq, the successful eviction's marker took exactly
    # the next one (26), and append(99) the one after — a failed eviction consumed
    # nothing, so the own-origin sequence on the tape has no hole. The oldest end is
    # itself evicted (by that successful eviction), so assert contiguity, not extent.
    assert all_own == list(range(all_own[0], all_own[-1] + 1))
    # append(99) itself took 26 (the first seq after the 25 appends — failed evictions
    # consumed nothing); the successful eviction's marker then took the next one, 27.
    assert nxt.seq == 26
    assert all_own[-1] == 27
    assert any(e.seq == 27 and e.type == GAP for e in log.read_all())  # marker took 27


def test_orphan_temp_files_swept_on_open(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    orphan = tmp_path / ".buffer-evict-ab12cd"
    orphan.write_text("{\"orphan\": true}\n")
    bystander = tmp_path / "unrelated.txt"
    bystander.write_text("keep me")
    BufferedStimulusLog(path)
    assert not orphan.exists()
    assert bystander.read_text() == "keep me"
    assert path.exists()  # the log opened normally despite the litter


def test_unsweepable_orphan_does_not_stop_boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "log.jsonl"
    orphan = tmp_path / ".buffer-evict-ab12cd"
    orphan.write_text("{}\n")

    def refuse(*args: object, **kwargs: object) -> object:
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(os, "unlink", refuse)
    log = BufferedStimulusLog(path)  # must not raise
    monkeypatch.undo()
    assert log.append("env", "observation", _event(0))
