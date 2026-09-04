from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from theseus.replication_events import (
    BATCH_REJECTED,
    DECLARED_REASONS,
    GAP,
    INFERRED_REASON,
    MAX_REASON_CHARS,
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


def test_an_inferred_gap_is_marked_undeclared():
    """A seq jump with no marker: the surrogate died mid-buffer, or something is broken.
    Same hole as a declared gap, different diagnosis — so the host says which it is."""
    content = inferred_gap(
        origin="android-01",
        from_seq=5,
        to_seq=9,
        span_start=SPAN_START,
        span_end=SPAN_END,
    )

    assert content == {
        "origin": "android-01",
        "from_seq": 5,
        "to_seq": 9,
        "reason": INFERRED_REASON,
        "span_start": "2026-09-04T16:02:00+00:00",
        "span_end": "2026-09-04T16:40:00+00:00",
        "declared": False,
    }


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
    with pytest.raises(ValueError, match="unknown declared reason"):
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
    with pytest.raises(ValueError, match="unknown declared reason"):
        declared_gap(
            origin="kitchen-surrogate",
            from_seq=1,
            to_seq=2,
            reason=INFERRED_REASON,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_an_inverted_seq_range_raises():
    with pytest.raises(ValueError, match="inverted seq range"):
        inferred_gap(
            origin="android-01",
            from_seq=9,
            to_seq=5,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_a_seq_below_one_raises():
    """Seqs start at 1, so 0 in a range is a bug, not a boundary."""
    with pytest.raises(ValueError, match="from_seq must be 1 or greater"):
        inferred_gap(
            origin="android-01",
            from_seq=0,
            to_seq=5,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_an_inverted_span_raises():
    with pytest.raises(ValueError, match="inverted span"):
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


def test_the_wire_vocabulary_is_fixed():
    """These four constants are the module's entire compatibility surface with a future
    non-Python surrogate. Every other test uses them as symbols, so a rename would stay
    green everywhere except here — which is the point."""
    assert GAP == "stimulus.gap"
    assert BATCH_REJECTED == "replication.batch_rejected"
    assert DECLARED_REASONS == ("link_down", "retry_exhausted", "storage_pressure")
    assert INFERRED_REASON == "inferred"


def test_a_naive_span_is_read_as_host_local():
    """The locked-in decision: naive datetimes are host-local, exactly as
    `StimulusEvent.to_json` already treats them. Pinned so a future "require tz-aware"
    patch has to be a deliberate change to the contract rather than a silent one."""
    naive_start = datetime(2026, 9, 4, 12, 0)
    naive_end = datetime(2026, 9, 4, 12, 30)

    content = inferred_gap(
        origin="android-01",
        from_seq=1,
        to_seq=2,
        span_start=naive_start,
        span_end=naive_end,
    )

    assert content["span_start"] == naive_start.astimezone(timezone.utc).isoformat()
    assert content["span_end"] == naive_end.astimezone(timezone.utc).isoformat()


def test_a_span_mixing_naive_and_aware_is_normalised_before_it_is_compared():
    """An ingress holding `datetime.now(timezone.utc)` beside a naive timestamp parsed from
    a surrogate's payload is the realistic case. Comparing the two raw raises an opaque
    TypeError naming neither field."""
    naive_start = datetime(2026, 9, 4, 12, 0)
    aware_end = naive_start.astimezone(timezone.utc) + timedelta(minutes=30)

    content = inferred_gap(
        origin="android-01",
        from_seq=1,
        to_seq=2,
        span_start=naive_start,
        span_end=aware_end,
    )

    assert content["span_end"] > content["span_start"]


@pytest.mark.parametrize("status", [400, 404, 413, 499])
def test_the_whole_4xx_range_is_accepted(status):
    content = batch_rejected(
        origin="kitchen-surrogate", from_seq=1, to_seq=2, status=status, reason="no"
    )

    assert content["status"] == status


@pytest.mark.parametrize("status", [399, 500, 200, 302])
def test_a_status_outside_4xx_raises(status):
    with pytest.raises(ValueError, match="status must be 4xx"):
        batch_rejected(
            origin="kitchen-surrogate",
            from_seq=1,
            to_seq=2,
            status=status,
            reason="no",
        )


@pytest.mark.parametrize("reason", [None, 42, object(), "", "   "])
def test_a_rejection_reason_that_is_not_real_text_raises(reason):
    """Unvalidated, a non-string reason builds cleanly here and then explodes inside
    `to_json` — a long way from the mistake. #33 copies a remote host's response body into
    this field."""
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        batch_rejected(
            origin="kitchen-surrogate",
            from_seq=1,
            to_seq=2,
            status=400,
            reason=reason,
        )


def test_an_over_long_rejection_reason_is_truncated_not_refused():
    """A rejection that cannot be recorded because the host was verbose is a silence on the
    tape, which is the failure this event type exists to prevent. So it is bounded, not
    rejected."""
    content = batch_rejected(
        origin="kitchen-surrogate",
        from_seq=1,
        to_seq=2,
        status=400,
        reason="x" * (MAX_REASON_CHARS + 50),
    )

    assert content["reason"] == "x" * MAX_REASON_CHARS


@pytest.mark.parametrize("seq", [True, False, 1.5, "3", None])
def test_a_non_integer_seq_raises(seq):
    """`bool` is an `int` in Python, so `from_seq=True` slips past a `< 1` guard and
    serialises as `true` onto a wire field a non-Python surrogate parses as a number."""
    with pytest.raises(ValueError, match="must be an integer"):
        inferred_gap(
            origin="android-01",
            from_seq=seq,
            to_seq=9,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_a_non_integer_status_raises():
    with pytest.raises(ValueError, match="status must be an integer"):
        batch_rejected(
            origin="kitchen-surrogate",
            from_seq=1,
            to_seq=2,
            status=400.5,
            reason="no",
        )


@pytest.mark.parametrize("origin", [None, 42, ["kitchen"], "", "   "])
def test_an_origin_that_is_not_a_real_name_raises(origin):
    with pytest.raises(ValueError, match="origin must be a non-empty name"):
        inferred_gap(
            origin=origin,
            from_seq=1,
            to_seq=2,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_a_to_seq_below_one_names_to_seq():
    """It is also an inverted range, but the fault is `to_seq`, and an error naming
    `from_seq` sends the reader to the wrong argument."""
    with pytest.raises(ValueError, match="to_seq must be 1 or greater"):
        inferred_gap(
            origin="android-01",
            from_seq=1,
            to_seq=0,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_a_rejection_is_json_native():
    """`declared_gap` has this covered; `batch_rejected` is the one carrying a remote
    string, so it is the one that most needs it."""
    content = batch_rejected(
        origin="kitchen-surrogate",
        from_seq=1,
        to_seq=2,
        status=400,
        reason="malformed line 3",
    )

    assert json.loads(json.dumps(content)) == content
