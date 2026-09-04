from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from theseus.replication_events import (
    BATCH_REJECTED,
    DECLARED_REASONS,
    GAP,
    INFERRED_REASON,
    batch_rejected,
    declared_gap,
    inferred_gap,
)
from theseus.stimulus_log import StimulusEvent

SPAN_START = datetime(2026, 9, 4, 16, 2, tzinfo=timezone.utc)
SPAN_END = datetime(2026, 9, 4, 16, 40, tzinfo=timezone.utc)


def test_a_declared_gap_carries_the_range_the_surrogate_abandoned():
    content = declared_gap(
        origin="kitchen-surrogate",
        from_seq=12,
        to_seq=48,
        reason="link_down",
        span_start=SPAN_START,
        span_end=SPAN_END,
    )

    assert content == {
        "origin": "kitchen-surrogate",
        "from_seq": 12,
        "to_seq": 48,
        "reason": "link_down",
        "span_start": "2026-09-04T16:02:00+00:00",
        "span_end": "2026-09-04T16:40:00+00:00",
        "declared": True,
    }


def test_an_inferred_gap_is_marked_undeclared_and_reasonless():
    """A seq jump with no marker: the surrogate died mid-buffer, or something is broken.
    Same hole as a declared gap, different diagnosis — so the host says which it is."""
    content = inferred_gap(
        origin="android-01",
        from_seq=5,
        to_seq=9,
        span_start=SPAN_START,
        span_end=SPAN_END,
    )

    assert content["declared"] is False
    assert content["reason"] == INFERRED_REASON


@pytest.mark.parametrize("reason", DECLARED_REASONS)
def test_every_declared_reason_is_accepted(reason):
    content = declared_gap(
        origin="kitchen-surrogate",
        from_seq=1,
        to_seq=2,
        reason=reason,
        span_start=SPAN_START,
        span_end=SPAN_END,
    )

    assert content["reason"] == reason


def test_an_unknown_reason_raises_rather_than_serialising():
    with pytest.raises(ValueError):
        declared_gap(
            origin="kitchen-surrogate",
            from_seq=1,
            to_seq=2,
            reason="gremlins",
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_the_inferred_reason_is_not_declarable():
    """`inferred` is what the host writes when nobody declared anything. A surrogate
    claiming it would erase the one distinction these events exist to carry."""
    with pytest.raises(ValueError):
        declared_gap(
            origin="kitchen-surrogate",
            from_seq=1,
            to_seq=2,
            reason=INFERRED_REASON,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_an_inverted_seq_range_raises():
    with pytest.raises(ValueError):
        inferred_gap(
            origin="android-01",
            from_seq=9,
            to_seq=5,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_a_seq_below_one_raises():
    """Seqs start at 1, so 0 in a range is a bug, not a boundary."""
    with pytest.raises(ValueError):
        inferred_gap(
            origin="android-01",
            from_seq=0,
            to_seq=5,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_an_inverted_span_raises():
    with pytest.raises(ValueError):
        inferred_gap(
            origin="android-01",
            from_seq=1,
            to_seq=2,
            span_start=SPAN_END,
            span_end=SPAN_START,
        )


def test_a_single_event_range_is_legal():
    """A one-event hole is a hole."""
    content = inferred_gap(
        origin="android-01",
        from_seq=7,
        to_seq=7,
        span_start=SPAN_START,
        span_end=SPAN_START,
    )

    assert (content["from_seq"], content["to_seq"]) == (7, 7)


def test_a_batch_rejection_records_what_the_host_said():
    content = batch_rejected(
        origin="kitchen-surrogate",
        from_seq=100,
        to_seq=120,
        status=413,
        reason="batch exceeds max byte size",
    )

    assert content == {
        "origin": "kitchen-surrogate",
        "from_seq": 100,
        "to_seq": 120,
        "status": 413,
        "reason": "batch exceeds max byte size",
    }


def test_a_non_4xx_rejection_status_raises():
    """This event exists for the permanently-unacceptable class. A 5xx is retried, not
    rejected, and recording one here would make the tape claim a batch was abandoned
    when the surrogate is still trying to send it."""
    with pytest.raises(ValueError):
        batch_rejected(
            origin="kitchen-surrogate",
            from_seq=1,
            to_seq=2,
            status=503,
            reason="upstream down",
        )


def test_an_empty_origin_raises():
    with pytest.raises(ValueError):
        inferred_gap(
            origin="",
            from_seq=1,
            to_seq=2,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_a_gap_round_trips_through_the_event_envelope():
    """These are ordinary events on the tape — the whole point is that a hole is
    something the agent reads, not an error channel beside the log."""
    event = StimulusEvent(
        id="01ABCDEFGHJKMNPQRSTVWXYZ0",
        ts=SPAN_END,
        actor="kitchen-surrogate",
        type=GAP,
        content=declared_gap(
            origin="kitchen-surrogate",
            from_seq=12,
            to_seq=48,
            reason="retry_exhausted",
            span_start=SPAN_START,
            span_end=SPAN_END,
        ),
    )

    assert StimulusEvent.from_json(event.to_json()) == event


def test_a_rejection_round_trips_through_the_event_envelope():
    event = StimulusEvent(
        id="01ABCDEFGHJKMNPQRSTVWXYZ1",
        ts=SPAN_END,
        actor="kitchen-surrogate",
        type=BATCH_REJECTED,
        content=batch_rejected(
            origin="kitchen-surrogate",
            from_seq=100,
            to_seq=120,
            status=400,
            reason="malformed line 3",
        ),
    )

    assert StimulusEvent.from_json(event.to_json()) == event


def test_spans_are_normalised_to_utc():
    """A gap is evidence about time, so it is stored in the one zone every node agrees on."""
    tokyo = timezone(timedelta(hours=9))
    content = inferred_gap(
        origin="android-01",
        from_seq=1,
        to_seq=2,
        span_start=datetime(2026, 9, 5, 1, 2, tzinfo=tokyo),
        span_end=datetime(2026, 9, 5, 1, 40, tzinfo=tokyo),
    )

    assert content["span_start"] == "2026-09-04T16:02:00+00:00"
    assert content["span_end"] == "2026-09-04T16:40:00+00:00"


def test_content_is_json_native():
    """`content` goes through `json.dumps` inside `to_json`; a datetime left in it would
    raise there rather than here, a long way from the mistake."""
    content = declared_gap(
        origin="kitchen-surrogate",
        from_seq=1,
        to_seq=2,
        reason="storage_pressure",
        span_start=SPAN_START,
        span_end=SPAN_END,
    )

    assert json.loads(json.dumps(content)) == content
