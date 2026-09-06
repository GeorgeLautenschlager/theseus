"""chunk_events: split a backlog into batches by whichever of the host's limits binds first.

The limits are the host's — `parse_batch` rejects on either one — so the chunker must
measure what the host measures: the serialised body, UTF-8 encoded bytes, not characters.
"""

from __future__ import annotations

from datetime import datetime, timezone

from theseus.replication_batch import DEFAULT_MAX_BATCH_BYTES, DEFAULT_MAX_BATCH_EVENTS
from theseus.stimulus_log import StimulusEvent
from theseus.surrogates.replicator import chunk_events


def make_event(seq: int, filler: str = "x" * 8) -> StimulusEvent:
    return StimulusEvent(
        id=f"01CHUNKTEST{seq:017d}",
        ts=datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc),
        actor="george",
        type="exchange",
        content={"message": filler},
        origin="kitchen-surrogate",
        seq=seq,
    )


def line_bytes(event: StimulusEvent) -> int:
    """What the host counts for one event's line: its `to_json()` plus the newline."""
    return len(event.to_json().encode("utf-8")) + 1


def body_bytes(batch: list[StimulusEvent]) -> int:
    """The host's measurement of a whole batch body (`parse_batch` checks `len(raw)`)."""
    return len(("\n".join(e.to_json() for e in batch) + "\n").encode("utf-8"))


def test_a_small_backlog_is_one_batch():
    events = [make_event(i) for i in range(1, 4)]
    batches = chunk_events(events, max_events=DEFAULT_MAX_BATCH_EVENTS, max_bytes=DEFAULT_MAX_BATCH_BYTES)
    assert len(batches) == 1
    assert [e.seq for e in batches[0]] == [1, 2, 3]


def test_the_count_limit_closes_a_batch():
    events = [make_event(i) for i in range(1, 6)]
    batches = chunk_events(events, max_events=2, max_bytes=DEFAULT_MAX_BATCH_BYTES)
    assert [[e.seq for e in b] for b in batches] == [[1, 2], [3, 4], [5]]


def test_the_byte_limit_closes_a_batch():
    events = [make_event(i, filler="x" * 64) for i in range(1, 6)]
    sizes = [line_bytes(e) for e in events]
    # Two lines fit under the cap, three do not; the count limit is slack.
    max_bytes = sizes[0] + sizes[1]
    batches = chunk_events(events, max_events=50, max_bytes=max_bytes)
    assert len(batches) == 3  # more than the one batch the count limit alone would give
    assert all(body_bytes(b) <= max_bytes for b in batches)


def test_whichever_limit_binds_first_is_the_one_that_binds():
    events = [make_event(i, filler="x" * 64) for i in range(1, 6)]
    # Count binds: bytes are slack, so the batch closes at two events.
    by_count = chunk_events(events, max_events=2, max_bytes=DEFAULT_MAX_BATCH_BYTES)
    assert [len(b) for b in by_count] == [2, 2, 1]
    # Bytes bind: the count limit (50) would take all five in one batch.
    sizes = [line_bytes(e) for e in events]
    by_bytes = chunk_events(events, max_events=50, max_bytes=sizes[0] + sizes[1])
    assert [len(b) for b in by_bytes] == [2, 2, 1]


def test_every_batch_fits_what_the_host_accepts():
    # Non-ASCII filler: characters and bytes disagree (é is 1 char, 2 bytes), so a
    # chunker that counts `len(str)` instead of encoded bytes overpacks — and this
    # assertion is the one that catches it.
    events = [make_event(i, filler="é" * 300) for i in range(1, 6)]
    sizes = [line_bytes(e) for e in events]
    max_bytes = sizes[0] + sizes[1]
    batches = chunk_events(events, max_events=50, max_bytes=max_bytes)
    assert len(batches) > 1
    for batch in batches:
        assert body_bytes(batch) <= max_bytes


def test_no_event_is_lost_or_duplicated_across_batches():
    events = [make_event(i) for i in range(1, 8)]
    sizes = [line_bytes(e) for e in events]
    batches = chunk_events(events, max_events=3, max_bytes=sizes[0] + sizes[1])
    flat = [e for b in batches for e in b]
    assert len(flat) == len(events)
    assert [e.seq for e in flat] == [e.seq for e in events]


def test_a_single_oversized_event_becomes_a_lone_batch():
    small = make_event(1, filler="x" * 8)
    big = make_event(2, filler="x" * 5000)
    last = make_event(3, filler="x" * 8)
    # Two smalls fit under the cap; the big one's line alone does not.
    max_bytes = line_bytes(small) * 2
    batches = chunk_events([small, big, last], max_events=50, max_bytes=max_bytes)
    assert [[e.seq for e in b] for b in batches] == [[1], [2], [3]]


def test_an_empty_input_yields_no_batches():
    assert chunk_events([], max_events=DEFAULT_MAX_BATCH_EVENTS, max_bytes=DEFAULT_MAX_BATCH_BYTES) == []
