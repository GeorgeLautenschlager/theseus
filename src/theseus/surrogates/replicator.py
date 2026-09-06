"""Backlog chunking for the surrogate's replication path.

The `Replicator` (Task 3) reads from the acked cursor and ships one batch at a time;
this module holds the pure helper it chunks with, so the byte accounting is testable
without a transport.
"""

from __future__ import annotations

from collections.abc import Sequence

from theseus.stimulus_log import StimulusEvent


def chunk_events(
    events: Sequence[StimulusEvent],
    *,
    max_events: int,
    max_bytes: int,
) -> list[list[StimulusEvent]]:
    """Split into batches no larger than either limit, preserving order.

    Bytes are measured the way the host measures them — each event's `to_json()` line
    plus its newline, UTF-8 encoded — because that is what `parse_batch` checks. A
    character count would under-measure any non-ASCII payload and build a batch the
    host rejects at exactly the boundary.

    An event whose own line exceeds `max_bytes` becomes a lone batch: it cannot be
    dropped or merged, and with no abandon rule yet (#32) the replicator will stall on
    it — the honest behaviour for this issue, named here so #32 has something to change.
    """
    batches: list[list[StimulusEvent]] = []
    current: list[StimulusEvent] = []
    current_bytes = 0
    for event in events:
        line_bytes = len(event.to_json().encode("utf-8")) + 1
        if current and (len(current) >= max_events or current_bytes + line_bytes > max_bytes):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(event)
        current_bytes += line_bytes
    if current:
        batches.append(current)
    return batches
