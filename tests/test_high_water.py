from __future__ import annotations

import threading

import pytest

from theseus.high_water import HighWaterMarks
from theseus.stimulus_log import StimulusLog


def make_log(tmp_path) -> StimulusLog:
    return StimulusLog(path=tmp_path / "stimulus_log.jsonl")


def replicate(log: StimulusLog, origin: str, seq: int) -> None:
    log.append(
        actor="sensor", type="observation", content={}, origin=origin, seq=seq
    )


def test_marks_are_recovered_from_several_origins_interleaved(tmp_path):
    log = make_log(tmp_path)
    replicate(log, "kitchen-surrogate", 1)
    replicate(log, "android-01", 7)
    replicate(log, "kitchen-surrogate", 2)
    replicate(log, "android-01", 5)  # out of order on the wire; the mark is a maximum

    marks = HighWaterMarks(log)

    assert marks.high_water("kitchen-surrogate") == 2
    assert marks.high_water("android-01") == 7


def test_an_origin_never_seen_returns_none(tmp_path):
    """None is not 0. Seqs start at 1, so 0 would claim a seq had been seen; an origin
    whose first event is still in flight has no mark at all."""
    marks = HighWaterMarks(make_log(tmp_path))

    assert marks.high_water("kitchen-surrogate") is None


def test_a_restart_mid_stream_recovers_the_same_marks(tmp_path):
    path = tmp_path / "stimulus_log.jsonl"
    log = StimulusLog(path=path)
    replicate(log, "kitchen-surrogate", 4)
    before = HighWaterMarks(log).high_water("kitchen-surrogate")

    after = HighWaterMarks(StimulusLog(path=path)).high_water("kitchen-surrogate")

    assert before == after == 4


def test_local_host_events_do_not_pollute_a_surrogates_mark(tmp_path):
    """The host's own log entries carry its own origin and its own seq counter. If those
    leaked into a surrogate's mark, dedupe would discard real events as duplicates."""
    log = make_log(tmp_path)
    replicate(log, "kitchen-surrogate", 2)
    for _ in range(9):
        log.append(actor="george", type="exchange", content={})

    marks = HighWaterMarks(log)

    assert marks.high_water("kitchen-surrogate") == 2
    assert marks.high_water(log.origin) == 9


def test_advance_records_a_commit(tmp_path):
    marks = HighWaterMarks(make_log(tmp_path))

    marks.advance("kitchen-surrogate", 12)

    assert marks.high_water("kitchen-surrogate") == 12


def test_a_mark_never_moves_backwards(tmp_path):
    """A batch straddling the mark commits only its tail and a duplicate commits nothing,
    so the mark is a maximum rather than a last-write."""
    marks = HighWaterMarks(make_log(tmp_path))
    marks.advance("kitchen-surrogate", 12)

    marks.advance("kitchen-surrogate", 5)

    assert marks.high_water("kitchen-surrogate") == 12


def test_pre_envelope_lines_leave_no_mark(tmp_path):
    """A log written before the envelope existed has no seqs at all. Those lines are
    history, not deliveries, and must not invent a mark for the origin reading them."""
    path = tmp_path / "stimulus_log.jsonl"
    path.write_text(
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ0","ts":"2026-01-01T12:00:00+00:00",'
        '"actor":"user","type":"chat_message","content":{}}\n',
        encoding="utf-8",
    )
    log = StimulusLog(path=path)

    marks = HighWaterMarks(log)

    assert marks.high_water(log.origin) is None


def test_a_concurrent_advance_cannot_clobber_a_higher_mark(tmp_path):
    """The defect this guards is a lost update: one caller reads a mark, is descheduled
    while a higher mark is committed, then writes its own lower value over the top —
    re-appending a batch the host already had.

    A test that merely runs threads and hopes for that interleaving is blind: the critical
    section is about a microsecond and the interpreter switches every five milliseconds, so
    it never lands. This forces the interleaving instead. The read inside `advance` is
    interposed, a competing higher `advance` runs at exactly that moment, and the lock is
    the only thing that can make it wait. The timeout is what makes holding the lock a pass
    rather than a hang.
    """
    marks = HighWaterMarks(make_log(tmp_path))
    competitor: list[threading.Thread] = []
    interposed = threading.Event()

    class InterposedMarks(dict):
        def get(self, key, default=None):
            value = super().get(key, default)
            if not interposed.is_set():
                interposed.set()
                done = threading.Event()

                def compete() -> None:
                    marks.advance("kitchen-surrogate", 900)
                    done.set()

                thread = threading.Thread(target=compete)
                competitor.append(thread)
                thread.start()
                # Blocked by the lock, this times out and the outer write wins the race
                # it was always going to win. Unlocked, it completes here and the outer
                # write silently reverts it.
                done.wait(timeout=0.5)
            return value

    marks._marks = InterposedMarks(marks._marks)

    marks.advance("kitchen-surrogate", 100)
    competitor[0].join(timeout=5)

    assert marks.high_water("kitchen-surrogate") == 900


def test_recovery_and_advance_compose(tmp_path):
    """The sequence every real ingress executes: boot with history on disk, commit
    something new, then see a duplicate arrive."""
    log = make_log(tmp_path)
    replicate(log, "kitchen-surrogate", 4)
    marks = HighWaterMarks(log)

    marks.advance("kitchen-surrogate", 6)
    assert marks.high_water("kitchen-surrogate") == 6

    marks.advance("kitchen-surrogate", 3)
    assert marks.high_water("kitchen-surrogate") == 6


def test_advancing_one_origin_leaves_the_others_alone(tmp_path):
    marks = HighWaterMarks(make_log(tmp_path))
    marks.advance("kitchen-surrogate", 5)

    marks.advance("android-01", 90)

    assert marks.high_water("kitchen-surrogate") == 5
    assert marks.high_water("android-01") == 90


def test_marks_are_a_snapshot_and_do_not_see_later_appends(tmp_path):
    """Pinning the precondition rather than endorsing it: an instance never re-reads the
    log, so an ingress that appends a replicated event without calling `advance` arms the
    duplicate this store exists to suppress — until a restart, which repairs it because the
    log is the truth."""
    log = make_log(tmp_path)
    replicate(log, "kitchen-surrogate", 1)
    marks = HighWaterMarks(log)

    replicate(log, "kitchen-surrogate", 2)

    assert marks.high_water("kitchen-surrogate") == 1
    assert HighWaterMarks(log).high_water("kitchen-surrogate") == 2


def test_a_seq_that_is_not_a_number_fails_the_boot_loudly(tmp_path):
    """Only a foreign writer can produce this — `append` refuses a non-integer seq. The
    contract is that recovery raises rather than silently under-counting, because an
    under-counted mark is the double-append direction."""
    path = tmp_path / "stimulus_log.jsonl"
    # The second line is what makes boot compare: first contact with an origin takes no
    # comparison, so a lone string seq would sail through and only the int that follows it
    # forces the str/int clash.
    path.write_text(
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ0","ts":"2026-01-01T12:00:00+00:00",'
        '"actor":"peer","type":"observation","content":{},'
        '"origin":"android-01","seq":"7"}\n'
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ02","ts":"2026-01-01T12:00:01+00:00",'
        '"actor":"peer","type":"observation","content":{},'
        '"origin":"android-01","seq":8}\n',
        encoding="utf-8",
    )

    with pytest.raises(TypeError):
        HighWaterMarks(StimulusLog(path=path))
