"""The surrogate's replication path: backlog chunking and the drain loop.

`chunk_events` is a pure helper so the byte accounting is testable without a transport;
`Replicator` reads from the acked cursor and ships one batch at a time through a
`StimulusTransport`.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

from theseus.replication_batch import (
    DEFAULT_MAX_BATCH_BYTES,
    DEFAULT_MAX_BATCH_EVENTS,
)
from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.transport import StimulusTransport


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


@dataclass(frozen=True, slots=True)
class DrainResult:
    """Where a drain ended up: what shipped, where the cursor landed, why it stopped."""

    batches_sent: int
    events_sent: int
    acked_seq: int | None      # the cursor after this drain
    stopped_on: int | None     # the non-2xx status that ended it, or None if it drained fully


class Replicator:
    """Drains a surrogate's backlog to its host, one batch at a time, in seq order.

    The lock is not a courtesy — it is the protocol. Two interleaved drains would put two
    requests in flight and reorder the stream; #30 shipped a deadlock for exactly this rule
    living in a docstring instead of code. Transport failures propagate: deciding what they
    mean (retry, backoff, abandonment) is #32's job, not this loop's.
    """

    def __init__(
        self,
        log: StimulusLog,
        transport: StimulusTransport,
        cursor: AckedCursor,
        *,
        max_events: int = DEFAULT_MAX_BATCH_EVENTS,
        max_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    ) -> None:
        self._log = log
        self._transport = transport
        self._cursor = cursor
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    def drain(self) -> DrainResult:
        """Ship everything above the cursor, one batch at a time, in seq order."""
        with self._lock:
            acked = self._cursor.acked_seq
            pending: list[StimulusEvent] = []
            for event in self._log.read_all():
                # Only this log's own origin — shipping host-origin events back would put two
                # numbering authorities on one seq stream. `seq is None` lines predate the
                # envelope and carry no dedupe identity, so they are not part of the stream.
                if event.origin != self._log.origin or event.seq is None:
                    continue
                if acked is not None and event.seq <= acked:
                    continue
                pending.append(event)
            # The log is arrival-ordered, but seq order is what the host's dedupe actually
            # depends on — it should not rest on a coincidence.
            pending.sort(key=lambda e: e.seq)
            batches = chunk_events(
                pending, max_events=self._max_events, max_bytes=self._max_bytes
            )

            sent_batches = 0
            sent_events = 0
            stopped_on: int | None = None
            for batch in batches:
                # The wire body is exactly what the host's `parse_batch` measures: each
                # event's line plus its newline.
                body = "".join(e.to_json() + "\n" for e in batch)
                result = self._transport.send(body)
                sent_batches += 1
                sent_events += len(batch)
                if not 200 <= result.status < 300:
                    stopped_on = result.status
                    break
                # Advance only after the ack, never before: behind is recoverable (the host
                # dedupes the re-send), ahead is not.
                self._cursor.advance(batch[-1].seq)

            return DrainResult(
                batches_sent=sent_batches,
                events_sent=sent_events,
                acked_seq=self._cursor.acked_seq,
                stopped_on=stopped_on,
            )
