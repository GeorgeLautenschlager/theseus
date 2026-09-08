"""The surrogate's replication path: backlog chunking and the drain loop.

`chunk_events` is a pure helper so the byte accounting is testable without a transport;
`Replicator` reads from the acked cursor and ships one batch at a time through a
`StimulusTransport`.
"""

from __future__ import annotations

import logging
import random
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

from theseus.replication_batch import (
    DEFAULT_MAX_BATCH_BYTES,
    DEFAULT_MAX_BATCH_EVENTS,
)
from theseus.replication_events import BATCH_REJECTED, GAP, batch_rejected, declared_gap
from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.surrogates.clock import Clock, SystemClock
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.retry import RetryBudget, backoff_delay, is_too_old
from theseus.surrogates.transport import StimulusTransport

logger = logging.getLogger(__name__)


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

    batches_attempted: int   # sent to the transport, acked or not
    events_attempted: int    # ditto; `seq` is not contiguous, so the acked_seq
                             # delta is not a count and cannot stand in for this
    acked_seq: int | None      # the cursor after this drain
    stopped_on: int | None     # the non-2xx status that ended it, or None if it drained fully
    skipped_unsequenced: int = 0  # own-origin lines with no seq (pre-envelope): unreplicable by design
    skipped_duplicate: int = 0    # repeat seqs dropped so the host's strict-ascending check cannot 400
    rejected_batches: int = 0     # 4xx: stepped over, batch_rejected recorded
    abandoned_batches: int = 0    # retry budget exhausted: stepped over, retry_exhausted gap
    unreachable: bool = False     # the drain stopped because nothing answered


class Replicator:
    """Drains a surrogate's backlog to its host, one batch at a time, in seq order.

    Transport failures are not all the same failure. A status is the host speaking: `2xx`
    acks, `4xx` is a permanent rejection (stepped over, marked `batch_rejected`), `5xx` is
    transient (retried with backoff on the injected clock). A raise is the link being down:
    the drain stops cleanly, abandons nothing, and spends no budget — the next drain retries
    the same range.
    """

    def __init__(
        self,
        log: StimulusLog,
        transport: StimulusTransport,
        cursor: AckedCursor,
        *,
        max_events: int = DEFAULT_MAX_BATCH_EVENTS,
        max_bytes: int = DEFAULT_MAX_BATCH_BYTES,
        budget: RetryBudget = RetryBudget(),
        clock: Clock = SystemClock(),
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self._log = log
        self._transport = transport
        self._cursor = cursor
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._budget = budget
        self._clock = clock
        self._random_fn = random_fn
        self._lock = threading.Lock()

    def drain(self) -> DrainResult:
        """Ship everything above the cursor, one batch at a time, in seq order."""
        with self._lock:
            acked = self._cursor.acked_seq
            pending: list[StimulusEvent] = []
            skipped_unsequenced = 0
            skipped_duplicate = 0
            seen_seqs: set[int] = set()
            for event in self._log.read_all():
                # Only this log's own origin — shipping host-origin events back would put two
                # numbering authorities on one seq stream.
                if event.origin != self._log.origin:
                    continue
                if event.seq is None:
                    # Predates the envelope: no dedupe identity, so it cannot be replicated —
                    # shipping it earns a 400. Skip, but count; silence here is data loss.
                    skipped_unsequenced += 1
                    continue
                if acked is not None and event.seq <= acked:
                    continue
                if event.seq in seen_seqs:
                    # A repeat seq means two processes wrote this log (forbidden by the log's
                    # contract, unenforceable across processes) or a restored backup. The host
                    # 400s any batch whose seqs do not strictly ascend, so without this drop the
                    # whole channel wedges — and the host would dedupe it anyway.
                    skipped_duplicate += 1
                    continue
                seen_seqs.add(event.seq)
                pending.append(event)
            if skipped_unsequenced or skipped_duplicate:
                # One line per drain, not per event: a 468-line legacy log must not produce
                # 468 log lines.
                logger.warning(
                    "drain skipped %d event(s) that predate the seq envelope and cannot be "
                    "replicated, and %d duplicate seq(s)",
                    skipped_unsequenced,
                    skipped_duplicate,
                )
            # The log is arrival-ordered, but seq order is what the host's dedupe actually
            # depends on — it should not rest on a coincidence.
            pending.sort(key=lambda e: e.seq)
            batches = chunk_events(
                pending, max_events=self._max_events, max_bytes=self._max_bytes
            )

            attempted_batches = 0
            attempted_events = 0
            stopped_on: int | None = None
            rejected_batches = 0
            abandoned_batches = 0
            unreachable = False

            def abandon(batch: list[StimulusEvent]) -> None:
                # Both exhaustion paths (age, attempts) converge here: declare the hole on
                # our own log, step the cursor past it, and let the drain continue. The
                # span is the batch's own oldest and newest event_ts — we know exactly
                # what we dropped, which is why a declared gap beats the host's inferred
                # one. The marker's seq sits above the range it describes, so on a large
                # backlog the host mints its own inferred marker first and our declared
                # one arrives later; both are on the tape, reason is authoritative.
                nonlocal abandoned_batches
                self._log.append(
                    "replicator",
                    GAP,
                    declared_gap(
                        origin=self._log.origin,
                        from_seq=batch[0].seq,
                        to_seq=batch[-1].seq,
                        reason="retry_exhausted",
                        span_start=min(e.ts for e in batch),
                        span_end=max(e.ts for e in batch),
                    ),
                )
                self._cursor.advance(batch[-1].seq)
                abandoned_batches += 1

            for batch in batches:
                # Oldest event first: ts need not ascend with seq when a producer's clock
                # drifts, so `batch[0].ts` would flatter the age and over-retry.
                oldest_ts = min(e.ts for e in batch)
                body = "".join(e.to_json() + "\n" for e in batch)
                attempted_batches += 1
                attempted_events += len(batch)
                for attempt in range(1, self._budget.max_attempts + 1):
                    # Checked before the first attempt and before every retry: a batch that
                    # ages out mid-backoff is abandoned, not retried into staleness.
                    if is_too_old(oldest_ts, self._clock.now(), self._budget):
                        abandon(batch)
                        break
                    try:
                        result = self._transport.send(body)
                    except Exception:
                        # Nothing answered: the link is down, not the batch bad. Stop the
                        # drain — abandon nothing, spend no budget — and retry next drain.
                        unreachable = True
                        stopped_on = None
                        break
                    if 200 <= result.status < 300:
                        # Advance only after the ack, never before: behind is recoverable (the host
                        # dedupes the re-send), ahead is not.
                        self._cursor.advance(batch[-1].seq)
                        break
                    if 400 <= result.status < 500:
                        # Permanent: retrying is how one poison batch wedges a channel. Record
                        # the hole on our own log and step over it.
                        self._log.append(
                            "replicator",
                            BATCH_REJECTED,
                            batch_rejected(
                                origin=self._log.origin,
                                from_seq=batch[0].seq,
                                to_seq=batch[-1].seq,
                                status=result.status,
                                # The constructor demands a non-empty reason; a host that
                                # sends none still gets a truthful one.
                                reason=result.reason.strip()
                                or f"host returned {result.status} with no reason",
                            ),
                        )
                        rejected_batches += 1
                        # Advance past it too: a 4xx will never succeed, so leaving the
                        # cursor behind would re-send a poison batch every drain.
                        self._cursor.advance(batch[-1].seq)
                        break
                    if not 500 <= result.status < 600:
                        # A 3xx is neither permanent rejection nor transient failure — guessing
                        # is worse than stopping.
                        stopped_on = result.status
                        break
                    if attempt == self._budget.max_attempts:
                        # The host answered but never accepted: budget spent, so abandon and
                        # move on — one undeliverable batch must not hold the channel.
                        abandon(batch)
                        break
                    # `attempt + 1`: `attempt` is the try that just failed, and
                    # `backoff_delay(n)` is the wait *before* try n. Passing `attempt` here
                    # sleeps 2, 6, 18, 54 = 80s across five tries and never reaches the
                    # 120s ceiling at all — 40% of the window #39 decided on for a link
                    # that may be cellular.
                    self._clock.sleep(
                        backoff_delay(attempt + 1, self._budget, random_fn=self._random_fn)
                    )
                if unreachable or stopped_on is not None:
                    break

            return DrainResult(
                batches_attempted=attempted_batches,
                events_attempted=attempted_events,
                acked_seq=self._cursor.acked_seq,
                stopped_on=stopped_on,
                skipped_unsequenced=skipped_unsequenced,
                skipped_duplicate=skipped_duplicate,
                rejected_batches=rejected_batches,
                abandoned_batches=abandoned_batches,
                unreachable=unreachable,
            )
