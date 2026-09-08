"""BufferPolicy's contract, and BufferedStimulusLog's: evict oldest-first under pressure."""

from __future__ import annotations

import dataclasses
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

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
    appended = [log.append("env", "observation", _event(i)) for i in range(10)]
    assert [e.seq for e in log.read_all()] == [e.seq for e in appended]
    assert path.stat().st_ino == log.path.stat().st_ino  # same file — never rewritten


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
    policy = BufferPolicy(max_bytes=1000, low_water=0.5)
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
    survivors = log.read_all()  # raises on any torn or interior-corrupt line
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
    assert not any(e.origin == DEFAULT_ORIGIN for e in log.read_all())  # own seqs evicted
    assert log._recover_next_seq() == 1  # what a post-eviction lazy recovery would give
    assert log.append("env", "observation", _event(99)).seq == 6


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
