from __future__ import annotations

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
