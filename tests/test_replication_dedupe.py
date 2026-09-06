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


def gap_event(seq: int, from_seq: int, to_seq: int, *, about: str = SURROGATE):
    """A gap the surrogate declared, as it arrives in a batch.

    `about` is the origin named in the marker's *content* — whose stream it claims to
    describe. That is a separate thing from the envelope origin, which is always the
    surrogate that sent the marker, and the two must not be conflated: a batch's events all
    carry one envelope origin by the time they reach the planner, while a marker inside it
    can claim to be about anybody. Telling those apart is exactly what `_declared_in`'s
    origin comparison is for.
    """
    return StimulusEvent(
        id=f"01PRODUCERID{seq:014d}",
        ts=BASE + timedelta(seconds=seq),
        actor="sensor",
        type=GAP,
        content=declared_gap(
            origin=about,
            from_seq=from_seq,
            to_seq=to_seq,
            reason="storage_pressure",
            span_start=BASE,
            span_end=BASE + timedelta(minutes=1),
        ),
        origin=SURROGATE,
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

    assert result == BatchPlan(to_append=(), new_high_water=None, inferred_holes=())


def test_a_batch_exactly_at_the_mark_adds_nothing():
    result = plan([event(4), event(5)], high_water=5)

    assert result.to_append == ()
    assert result.new_high_water is None


def test_a_batch_straddling_the_mark_commits_only_the_tail():
    result = plan([event(4), event(5), event(6), event(7)], high_water=5)

    assert [e.seq for e in result.to_append] == [6, 7]
    assert result.new_high_water == 7
    assert result.inferred_holes == ()


def test_a_contiguous_batch_commits_whole_and_infers_nothing():
    result = plan([event(6), event(7)], high_water=5)

    assert [e.seq for e in result.to_append] == [6, 7]
    assert result.new_high_water == 7
    assert result.inferred_holes == ()


def test_a_jump_past_the_mark_is_committed_with_a_marker_not_rejected():
    """The spec's third case: append it and carry on. Never reject — the events in hand are
    not the ones that went missing."""
    result = plan([event(10), event(11)], high_water=5)

    assert result.inferred_holes == ((6, 9),)
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

    assert result.inferred_holes == ((6, 6),)
    assert result.to_append[0].content["from_seq"] == 6
    assert result.to_append[0].content["to_seq"] == 6


# --- First contact --------------------------------------------------------------
def test_first_contact_at_seq_1_is_not_a_gap():
    result = plan([event(1), event(2)], high_water=None)

    assert result.inferred_holes == ()
    assert [e.seq for e in result.to_append] == [1, 2]
    assert result.new_high_water == 2


def test_first_contact_above_seq_1_is_a_gap_from_1():
    """An origin whose first batch starts at 5 means seqs 1-4 never arrived. Recording that
    beats discarding the information that four events are missing."""
    result = plan([event(5)], high_water=None)

    assert result.inferred_holes == ((1, 4),)


# --- Declared gaps --------------------------------------------------------------
def test_a_declared_gap_covering_the_hole_stops_the_host_minting_one():
    """The surrogate evicted 6-9 and said so; its marker rides in this batch as seq 10. The
    host adds nothing, which is the whole distinction between declared and inferred."""
    result = plan([gap_event(10, 6, 9), event(11)], high_water=5)

    assert result.inferred_holes == ()
    assert [e.seq for e in result.to_append] == [10, 11]
    assert result.to_append[0].content["declared"] is True


def test_a_declared_gap_covering_more_than_the_hole_still_counts():
    result = plan([gap_event(10, 1, 9), event(11)], high_water=5)

    assert result.inferred_holes == ()


def test_a_declared_gap_covering_only_part_of_the_hole_leaves_the_residue():
    """6-7 explained of a 6-9 hole leaves 8-9 unaccounted for — and 8-9 is exactly what the
    host mints for. Marking the whole 6-9 again would contradict a marker sitting beside it
    in the same write, and say the host had to guess about events the surrogate just
    explained."""
    result = plan([gap_event(10, 6, 7), event(11)], high_water=5)

    assert result.inferred_holes == ((8, 9),)
    assert result.to_append[0].origin == HOST
    assert (
        result.to_append[0].content["from_seq"],
        result.to_append[0].content["to_seq"],
    ) == (8, 9)


def test_two_declared_gaps_covering_a_hole_between_them_silence_the_host():
    """Ordinary surrogate behaviour: evict 6-7 under storage pressure, then lose 8-9 to a
    dead link. Nothing here was undiagnosed, so a third marker claiming the host had to
    guess would erase the distinction the vocabulary exists to carry."""
    result = plan([gap_event(10, 6, 7), gap_event(11, 8, 9), event(12)], high_water=5)

    assert result.inferred_holes == ()
    assert [e.seq for e in result.to_append] == [10, 11, 12]


def test_a_declared_gap_about_another_origin_does_not_silence_the_host():
    result = plan([gap_event(10, 6, 9, about="android-01"), event(11)], high_water=5)

    assert result.inferred_holes == ((6, 9),)


@pytest.mark.parametrize("bound", [True, "6", None, 6.0])
def test_a_declared_gap_whose_range_is_not_an_integer_explains_nothing(bound):
    """`bool` is an `int` in Python, so `from_seq: true` would compare as 1 and let a
    malformed marker silence a real hole. `parse_batch` now rejects such a marker at the
    door, but this function is reachable from callers that never went through it."""
    marker = gap_event(10, 6, 9)
    marker = StimulusEvent(
        id=marker.id, ts=marker.ts, actor=marker.actor, type=GAP,
        content={**marker.content, "from_seq": bound},
        origin=marker.origin, seq=marker.seq,
    )

    assert plan([marker, event(11)], high_water=5).inferred_holes == ((6, 9),)


# --- Holes inside a batch -------------------------------------------------------
def test_a_hole_between_two_events_of_one_batch_is_recorded():
    """`parse_batch` accepts ascending-but-not-contiguous batches on purpose, so a surrogate
    whose buffer has real holes is not told to throw away what it still has. That makes an
    internal hole an ordinary shape, and the commit steps over it exactly as it steps over a
    leading one."""
    result = plan([event(5), event(9)], high_water=4)

    assert result.inferred_holes == ((6, 8),)
    assert result.new_high_water == 9


def test_the_tape_does_not_depend_on_how_the_backlog_was_chunked():
    """The same hole, delivered two ways. Before internal holes were counted, `[5, 9]` in one
    batch advanced the mark to 9 and recorded nothing — 6-8 permanently undeliverable *and*
    unaccounted for — while `[5]` then `[9]` recorded it properly."""
    together = plan([event(5), event(9)], high_water=4)

    apart = plan([event(9)], high_water=5)

    assert together.inferred_holes == apart.inferred_holes == ((6, 8),)


def test_several_holes_in_one_batch_each_get_a_marker():
    result = plan([event(3), event(7), event(12)], high_water=1)

    assert result.inferred_holes == ((2, 2), (4, 6), (8, 11))
    assert [e.type for e in result.to_append] == [GAP, GAP, GAP] + ["observation"] * 3
    assert [
        (e.content["from_seq"], e.content["to_seq"]) for e in result.to_append[:3]
    ] == [(2, 2), (4, 6), (8, 11)]


def test_a_declared_marker_can_explain_an_internal_hole_too():
    result = plan([event(5), gap_event(9, 6, 8), event(10)], high_water=4)

    assert result.inferred_holes == ()


def test_a_marker_below_the_mark_explains_nothing():
    """It is a duplicate — the host already has it — and it is discarded with every other
    duplicate. Counting it would let an already-committed marker silence a hole that nothing
    in this batch accounts for."""
    result = plan([gap_event(6, 11, 19), event(20)], high_water=10)

    assert [e.seq for e in result.to_append if e.seq is not None] == [20]
    assert result.inferred_holes == ((11, 19),)


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


# --- Per-hole spans -------------------------------------------------------------
def test_an_internal_hole_spans_the_events_it_sits_between():
    """The hole is between seq 3 and seq 7, so its honest bounds are their timestamps —
    not the mark's last event and the batch's first. The once-per-batch bounds ended
    before the events they claimed to cover."""
    result = plan([event(3), event(7)], high_water=2, previous_ts=event(2).ts)

    assert result.inferred_holes == ((4, 6),)
    content = result.to_append[0].content
    assert content["span_start"] == event(3).ts.isoformat()
    assert content["span_end"] == event(7).ts.isoformat()


def test_a_leading_hole_still_spans_from_the_mark_to_the_first_event():
    """The leading hole was the one case the old bounds were right for; per-hole
    arithmetic must not move it."""
    result = plan([event(5), event(6)], high_water=2, previous_ts=event(2).ts)

    assert result.inferred_holes == ((3, 4),)
    content = result.to_append[0].content
    assert content["span_start"] == event(2).ts.isoformat()
    assert content["span_end"] == event(5).ts.isoformat()


def test_a_batch_with_both_kinds_of_hole_bounds_each_one_its_own_way():
    """One batch, two holes: leading (3, 4) and internal (6, 6). Each marker carries the
    bounds of its own hole — which differ, so a single pair passed to both cannot be right."""
    result = plan([event(5), event(7)], high_water=2, previous_ts=event(2).ts)

    assert result.inferred_holes == ((3, 4), (6, 6))
    leading, internal = result.to_append[:2]
    assert leading.content["span_start"] == event(2).ts.isoformat()
    assert leading.content["span_end"] == event(5).ts.isoformat()
    assert internal.content["span_start"] == event(5).ts.isoformat()
    assert internal.content["span_end"] == event(7).ts.isoformat()


def test_a_fragment_left_by_a_partial_declaration_keeps_honest_bounds():
    """A declared marker covering 6-7 of a 6-8 hole leaves fragment (8, 8) — and its
    bounds come from the events around *that* fragment: seq 5 behind it, the surrogate's
    own marker ahead of it. Carrying the original hole's bounds across the subtraction
    would span past what was explained."""
    result = plan([event(5), gap_event(9, 6, 7), event(10)], high_water=4)

    assert result.inferred_holes == ((8, 8),)
    content = result.to_append[0].content
    assert (content["from_seq"], content["to_seq"]) == (8, 8)
    assert content["span_start"] == event(5).ts.isoformat()
    assert content["span_end"] == gap_event(9, 6, 7).ts.isoformat()


def test_first_contact_above_seq_1_still_collapses_to_zero_width():
    """Nothing committed for the origin means no lower bound is known anywhere — not even
    per hole. The leading hole's span says exactly that."""
    result = plan([event(5)], high_water=None)

    assert result.inferred_holes == ((1, 4),)
    content = result.to_append[0].content
    assert content["span_start"] == content["span_end"] == event(5).ts.isoformat()


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
