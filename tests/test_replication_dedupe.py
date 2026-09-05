from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from theseus.replication_dedupe import BatchPlan, HOST_ACTOR, plan_batch
from theseus.replication_events import GAP, declared_gap
from theseus.stimulus_log import StimulusEvent

HOST = "local"
SURROGATE = "kitchen-surrogate"
BASE = datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc)
NOW = BASE + timedelta(hours=1)


def event(seq: int, *, origin: str = SURROGATE, ts: datetime | None = None) -> StimulusEvent:
    return StimulusEvent(
        id=f"01PRODUCERID{seq:014d}",
        ts=BASE + timedelta(seconds=seq) if ts is None else ts,
        actor="sensor",
        type="observation",
        content={"n": seq},
        origin=origin,
        seq=seq,
    )


def gap_event(seq: int, from_seq: int, to_seq: int, *, origin: str = SURROGATE):
    """A gap the surrogate declared about its own stream, as it arrives in a batch."""
    return StimulusEvent(
        id=f"01PRODUCERID{seq:014d}",
        ts=BASE + timedelta(seconds=seq),
        actor="sensor",
        type=GAP,
        content=declared_gap(
            origin=origin,
            from_seq=from_seq,
            to_seq=to_seq,
            reason="storage_pressure",
            span_start=BASE,
            span_end=BASE + timedelta(minutes=1),
        ),
        origin=origin,
        seq=seq,
    )


def plan(events, high_water=None, **kwargs):
    return plan_batch(
        events, high_water=high_water, host_origin=HOST, now=NOW, **kwargs
    )


# --- The three cases ------------------------------------------------------------
def test_a_batch_entirely_below_the_mark_adds_nothing():
    """A lost ack: the host committed, the response died, the surrogate resent. Its job on
    retry is to stop worrying, not to find out it was wrong."""
    result = plan([event(1), event(2)], high_water=5)

    assert result == BatchPlan(to_append=(), new_high_water=None, inferred_hole=None)


def test_a_batch_exactly_at_the_mark_adds_nothing():
    result = plan([event(4), event(5)], high_water=5)

    assert result.to_append == ()
    assert result.new_high_water is None


def test_a_batch_straddling_the_mark_commits_only_the_tail():
    result = plan([event(4), event(5), event(6), event(7)], high_water=5)

    assert [e.seq for e in result.to_append] == [6, 7]
    assert result.new_high_water == 7
    assert result.inferred_hole is None


def test_a_contiguous_batch_commits_whole_and_infers_nothing():
    result = plan([event(6), event(7)], high_water=5)

    assert [e.seq for e in result.to_append] == [6, 7]
    assert result.new_high_water == 7
    assert result.inferred_hole is None


def test_a_jump_past_the_mark_is_committed_with_a_marker_not_rejected():
    """The spec's third case: append it and carry on. Never reject — the events in hand are
    not the ones that went missing."""
    result = plan([event(10), event(11)], high_water=5)

    assert result.inferred_hole == (6, 9)
    assert [e.seq for e in result.to_append] == [None, 10, 11]
    assert result.new_high_water == 11


def test_the_marker_is_written_before_the_events_that_revealed_the_hole():
    """One write, marker first. Split across two, a crash commits the hole and loses the
    explanation — the one outcome this vocabulary exists to prevent."""
    result = plan([event(10)], high_water=5)

    marker = result.to_append[0]
    assert marker.type == GAP
    assert marker.origin == HOST
    assert marker.seq is None
    assert marker.actor == HOST_ACTOR
    assert marker.content["origin"] == SURROGATE
    assert (marker.content["from_seq"], marker.content["to_seq"]) == (6, 9)
    assert marker.content["declared"] is False


def test_a_one_event_hole_is_an_inclusive_range():
    result = plan([event(7)], high_water=5)

    assert result.inferred_hole == (6, 6)
    assert result.to_append[0].content["from_seq"] == 6
    assert result.to_append[0].content["to_seq"] == 6


# --- First contact --------------------------------------------------------------
def test_first_contact_at_seq_1_is_not_a_gap():
    result = plan([event(1), event(2)], high_water=None)

    assert result.inferred_hole is None
    assert [e.seq for e in result.to_append] == [1, 2]
    assert result.new_high_water == 2


def test_first_contact_above_seq_1_is_a_gap_from_1():
    """An origin whose first batch starts at 5 means seqs 1-4 never arrived. Recording that
    beats discarding the information that four events are missing."""
    result = plan([event(5)], high_water=None)

    assert result.inferred_hole == (1, 4)


# --- Declared gaps --------------------------------------------------------------
def test_a_declared_gap_covering_the_hole_stops_the_host_minting_one():
    """The surrogate evicted 6-9 and said so; its marker rides in this batch as seq 10. The
    host adds nothing, which is the whole distinction between declared and inferred."""
    result = plan([gap_event(10, 6, 9), event(11)], high_water=5)

    assert result.inferred_hole is None
    assert [e.seq for e in result.to_append] == [10, 11]
    assert result.to_append[0].content["declared"] is True


def test_a_declared_gap_covering_more_than_the_hole_still_counts():
    result = plan([gap_event(10, 1, 9), event(11)], high_water=5)

    assert result.inferred_hole is None


def test_a_declared_gap_covering_only_part_of_the_hole_does_not_silence_the_host():
    """6-7 explained of a 6-9 hole leaves 8-9 unaccounted for. Two overlapping markers is a
    legible tape; a half-explained hole is not."""
    result = plan([gap_event(10, 6, 7), event(11)], high_water=5)

    assert result.inferred_hole == (6, 9)
    assert result.to_append[0].origin == HOST


def test_a_declared_gap_about_another_origin_does_not_silence_the_host():
    result = plan([gap_event(10, 6, 9, origin="android-01"), event(11)], high_water=5)

    assert result.inferred_hole == (6, 9)


@pytest.mark.parametrize("bound", [True, "6", None, 6.0])
def test_a_declared_gap_whose_range_is_not_an_integer_explains_nothing(bound):
    """`bool` is an `int` in Python, so `from_seq: true` would compare as 1 and let a
    malformed marker silence a real hole. Nothing validates a marker off the wire."""
    marker = gap_event(10, 6, 9)
    marker = StimulusEvent(
        id=marker.id, ts=marker.ts, actor=marker.actor, type=GAP,
        content={**marker.content, "from_seq": bound},
        origin=marker.origin, seq=marker.seq,
    )

    assert plan([marker, event(11)], high_water=5).inferred_hole == (6, 9)


# --- The span -------------------------------------------------------------------
def test_the_span_runs_from_the_supplied_bound_to_the_first_event_that_arrived():
    previous = BASE - timedelta(minutes=30)
    result = plan([event(10)], high_water=5, previous_ts=previous)

    content = result.to_append[0].content
    assert content["span_start"] == previous.isoformat()
    assert content["span_end"] == event(10).ts.isoformat()


def test_with_no_lower_bound_the_span_collapses_onto_the_first_event():
    """Zero width reads as 'noticed here, no lower bound known' — the truth on first
    contact, rather than a fabricated interval."""
    result = plan([event(10)], high_water=5)

    content = result.to_append[0].content
    assert content["span_start"] == content["span_end"] == event(10).ts.isoformat()


def test_a_lower_bound_ahead_of_the_first_event_does_not_invert_the_span():
    """A surrogate's clock can sit ahead of the host's. Raising here would lose the marker
    over exactly the skew the marker exists to make visible."""
    result = plan([event(10)], high_water=5, previous_ts=BASE + timedelta(days=1))

    content = result.to_append[0].content
    assert content["span_start"] == content["span_end"]


def test_a_naive_lower_bound_is_read_as_host_local_rather_than_raising():
    result = plan([event(10)], high_water=5, previous_ts=datetime(2026, 9, 4, 0, 0))

    assert result.to_append[0].content["span_start"] <= result.to_append[0].content["span_end"]


def test_the_markers_own_ts_is_the_hosts_clock_not_the_producers():
    """`ts` answers when it happened: the host noticed the hole now, it did not happen when
    the surrogate's next event did."""
    result = plan([event(10)], high_water=5)

    assert result.to_append[0].ts == NOW


# --- Preconditions --------------------------------------------------------------
def test_an_empty_batch_is_a_caller_bug():
    with pytest.raises(ValueError, match="at least one event"):
        plan([])


def test_two_origins_in_one_batch_is_a_caller_bug():
    with pytest.raises(ValueError, match="one origin"):
        plan([event(1), event(2, origin="android-01")])


def test_a_batch_claiming_the_hosts_own_origin_is_a_caller_bug():
    with pytest.raises(ValueError, match="host's own origin"):
        plan([event(1, origin=HOST)])


def test_seqs_that_do_not_ascend_are_a_caller_bug():
    """The tail slice and the hole arithmetic are only correct on an ascending sequence, so
    this is this function's own precondition, not a restatement of the wire's rule."""
    with pytest.raises(ValueError, match="ascend"):
        plan([event(7), event(6)])


def test_a_repeated_seq_is_not_ascending():
    with pytest.raises(ValueError, match="ascend"):
        plan([event(6), event(6)])


@pytest.mark.parametrize("seq", [None, True, "6", 6.0])
def test_a_non_integer_seq_is_a_caller_bug(seq):
    bad = StimulusEvent(
        id="01X", ts=BASE, actor="sensor", type="observation",
        content={}, origin=SURROGATE, seq=seq,
    )

    with pytest.raises(ValueError, match="non-integer seq"):
        plan([bad])


# --- The plan is usable by the thing that will use it ---------------------------
def test_the_plan_can_be_appended_as_one_batch(tmp_path):
    """The end the whole module serves: marker and events in one `append_many`, which is the
    write the narrowed origin invariant was kept to allow."""
    from theseus.stimulus_log import StimulusLog

    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    result = plan([event(10), event(11)], high_water=5)

    appended = log.append_many(result.to_append)

    assert [(e.origin, e.seq, e.type) for e in appended] == [
        (HOST, 1, GAP),
        (SURROGATE, 10, "observation"),
        (SURROGATE, 11, "observation"),
    ]
