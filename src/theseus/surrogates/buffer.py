"""How large the surrogate's local buffer may grow, and how far back eviction cuts.

The surrogate's log is a buffer, not a tape: under storage pressure it may evict
oldest-first ahead of the acked cursor, declaring the hole with a `storage_pressure`
gap marker. This policy is the knob for that eviction; the evicting log itself
consumes it. A plain `StimulusLog` never enters this mode and has no knob that
would let it.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from theseus.replication_events import GAP, declared_gap
from theseus.stimulus_log import DEFAULT_ORIGIN, StimulusEvent, StimulusLog, new_id


@dataclass(frozen=True, slots=True)
class BufferPolicy:
    """How large the surrogate's buffer may grow, and how far back eviction cuts."""

    # 256 MB is a buffer a modest edge device can spare, and holds days of
    # text-rate observation. A deployment override, not a law — a starting
    # point chosen so the default never evicts in normal operation.
    max_bytes: int = 256 * 1024 * 1024
    # Where eviction cuts back to, as a fraction of max_bytes — not where it
    # starts. Eviction triggers at max_bytes and then keeps going until the buffer
    # is at or under max_bytes * low_water, so the two lines are a hysteresis band.
    # Strictly inside (0, 1): at 1.0 the band is empty and the buffer rewrites
    # itself on every append past the line; at 0.0 one eviction empties it.
    low_water: float = 0.8

    def __post_init__(self) -> None:
        """Fail at composition, not under pressure: bad config must never reach a live surrogate."""
        if self.max_bytes < 1:
            raise ValueError(
                f"max_bytes must be at least 1 (got {self.max_bytes!r}); a non-positive "
                f"buffer can hold nothing and evicts on every append"
            )
        if not 0 < self.low_water < 1:
            raise ValueError(
                f"low_water must be strictly between 0 and 1 (got {self.low_water!r}); at "
                f"1.0 there is no hysteresis and at 0.0 eviction empties the buffer"
            )


class BufferedStimulusLog(StimulusLog):
    """A surrogate's local log: an append-only tape that is allowed to forget its oldest end.

    Everything else about `StimulusLog`'s contract holds — appends are durable before
    they return, `read_all` still parses the whole tape — except that under storage
    pressure the oldest events are dropped and the file rewritten. Observation never
    pauses for eviction: an append is never rejected, delayed or altered by pressure,
    it just may not be remembered forever.

    Forgetting is never silent. Each eviction writes a `stimulus.gap` marker
    (`reason="storage_pressure"`) naming the inclusive own-origin seq range it took,
    into the same atomic rewrite as the survivors — the truncation and its declaration
    land together or not at all.

    The marker is itself an ordinary event on this buffer, evictable like any other.
    A surrogate that evicts a gap marker before it replicates has forgotten that it
    forgot — but the eviction that removed it declares a range covering the marker's
    own seq, so the tape stays honest even then.

    Known cost (deliberate, do not rediscover as a bug): eviction rewrites the whole
    surviving buffer. At the defaults that is a ~205 MB copy roughly every 51 MB
    appended (256 MB cap, cut back to the 0.8 low water). Correctness was chosen over
    a segmented log; revisit only if a real deployment measures this.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        origin: str = DEFAULT_ORIGIN,
        *,
        policy: BufferPolicy = BufferPolicy(),
    ) -> None:
        super().__init__(path, origin)
        self._policy = policy

    def append(self, *args: Any, **kwargs: Any) -> StimulusEvent:
        event = super().append(*args, **kwargs)
        self._evict_if_needed()
        return event

    def append_many(self, events: Iterable[StimulusEvent]) -> list[StimulusEvent]:
        minted = super().append_many(events)
        self._evict_if_needed()
        return minted

    def _evict_if_needed(self) -> StimulusEvent | None:
        """Evict oldest-first if the file is over budget. Returns the gap marker, if one
        was emitted (Task 3); `None` when nothing was evicted."""
        # Bare stat first: the common case is a buffer within budget, and it must not
        # pay for a full read to learn that.
        if self.path.stat().st_size <= self._policy.max_bytes:
            return None
        with self._append_lock:
            # Re-check: another thread may have evicted while we waited, and rewriting
            # a buffer that is already under budget would discard events for nothing.
            if self.path.stat().st_size <= self._policy.max_bytes:
                return None
            # Before the file shrinks: `_recover_next_seq` derives the next seq from the
            # highest own-origin seq in the file, and eviction is about to remove the
            # oldest of those. A lazy recovery after the truncation would reissue seqs
            # the host has already accepted, so the counter is materialised now, while
            # the tape is still whole.
            if self._next_seq is None:
                self._next_seq = self._recover_next_seq()

            events = self.read_all()
            # Byte lengths, not character counts — the budget is a byte budget and the
            # file is UTF-8, so any non-ASCII payload would be under-measured otherwise.
            sizes = [len((e.to_json() + "\n").encode("utf-8")) for e in events]
            floor = self._policy.max_bytes * self._policy.low_water
            total = sum(sizes)
            keep_from = 0
            while total > floor and keep_from < len(events) - 1:
                # `keep_from < len - 1` is the never-empty rule: a single event larger
                # than the whole budget is kept, and the buffer sits over its cap.
                total -= sizes[keep_from]
                keep_from += 1
            if keep_from == 0:
                return None
            evicted = events[:keep_from]
            # Only own-origin sequenced events are described: `declared_gap` names one
            # origin's seq space, and that is what the host dedupes on. Foreign events
            # were authored by the host and not lost; seq-less events predate the
            # envelope and were already unreplicable.
            described = [
                e for e in evicted if e.origin == self.origin and e.seq is not None
            ]
            if not described:
                survivors = events[keep_from:]
                marker = None
            else:
                # Two-pass keep computation: the marker rides in the same rewrite, so
                # its bytes come out of the same low-water budget. Pass one finds the
                # eviction extent; the marker line for that extent gives the exact
                # budget for pass two. Pass two only ever evicts *fewer* events, and
                # if that shrinks the described set away, we fall back to pass one's
                # larger eviction so nothing is dropped unmarked — the two-pass result
                # is only taken when it can still be declared.
                keep1 = keep_from
                described1 = described
                bounds = (min(e.seq for e in described1), max(e.seq for e in described1))
                span = (min(e.ts for e in described1), max(e.ts for e in described1))
                # Mint the marker the way `append` would: seq under the recovered
                # counter, id from the appended timestamp. The seq is peeked, not
                # consumed, until we know a rewrite will actually happen.
                marker_seq = self._next_seq
                marker_json = self._mint_gap(
                    marker_seq, bounds, span
                ).to_json() + "\n"
                keep2 = self._keep_index(sizes, floor - len(marker_json.encode("utf-8")))
                described2 = [
                    e for e in events[:keep2] if e.origin == self.origin and e.seq is not None
                ]
                if described2:
                    keep_from = keep2
                    bounds = (min(e.seq for e in described2), max(e.seq for e in described2))
                    span = (min(e.ts for e in described2), max(e.ts for e in described2))
                marker = self._mint_gap(marker_seq, bounds, span)
                self._next_seq = marker_seq + 1
                survivors = events[keep_from:] + [marker]

            # Same atomic-replace dance as the cursor sidecar: write the survivors to a
            # temp file in this log's own directory (so os.replace is a rename on one
            # filesystem), fsync it, then replace. A crash leaves either the whole old
            # file or the whole new one, never a half-truncated tape. The lock is held
            # across the whole rewrite, so no append can land between the read and the
            # replace and be lost by it.
            directory = os.path.dirname(os.fspath(self.path)) or "."
            fd, tmp = tempfile.mkstemp(prefix=".buffer-evict-", dir=directory)
            try:
                # mkstemp creates the temp file at 0600; os.replace would move that
                # mode into place and silently narrow the log's permissions the first
                # time eviction fires. Copy the existing file's mode so eviction
                # preserves whatever the operator set (group-readable log shippers,
                # debug UIs) instead of imposing the temp file's default.
                os.chmod(tmp, os.stat(self.path).st_mode & 0o777)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.writelines(e.to_json() + "\n" for e in survivors)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            if marker is not None:
                # Outside the lock, after the replace is durable — the same ordering
                # `append` uses: a listener is free to append, and holding the lock
                # across a callback would deadlock it.
                self._notify(marker)
            return marker

    def _mint_gap(
        self,
        seq: int,
        bounds: tuple[int, int],
        span: tuple[Any, Any],
    ) -> StimulusEvent:
        """Build the `storage_pressure` marker in memory, exactly as `append` mints an
        event: id from `appended_ts`, ts = appended_ts. Caller holds `_append_lock`
        and has already recovered `_next_seq`; `seq` is peeked, not consumed."""
        now = datetime.now(timezone.utc)
        return StimulusEvent(
            id=new_id(int(now.timestamp() * 1000)),
            ts=now,
            # Same actor convention as the Replicator's declared gaps: the declaring
            # component's own name.
            actor="buffer",
            type=GAP,
            content=declared_gap(
                origin=self.origin,
                from_seq=bounds[0],
                to_seq=bounds[1],
                reason="storage_pressure",
                span_start=span[0],
                span_end=span[1],
            ),
            origin=self.origin,
            seq=seq,
            appended_ts=now,
        )

    def _keep_index(self, sizes: list[int], floor: float) -> int:
        """How many oldest events to drop so the rest fits under `floor`, never
        emptying the buffer (a single oversized event is kept and the buffer sits
        over its cap)."""
        total = sum(sizes)
        keep = 0
        while total > floor and keep < len(sizes) - 1:
            total -= sizes[keep]
            keep += 1
        return keep
