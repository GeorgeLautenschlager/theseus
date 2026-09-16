"""Optional application policy for bounded, restartable memory formation."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from typing import Callable

from theseus.layer_store import atomic_json, store_lock
from theseus.memory_module import Episode, MemoryModule, ConsolidationResult

# A decision realistically produces a handful of tool calls; capped so one
# unusual run (or an adversarial log) can't make a single interaction unit
# swallow the rest of the pending backlog while hunting for its boundary.
_MAX_UNIT_EVENTS = 50


def _interaction_units(events: list) -> list[list]:
    """Group consecutive events into observable interaction units.

    A `decision` event and every `tool_result` immediately following it (up to
    the next `decision`, `_MAX_UNIT_EVENTS`, or the end of the run) form one
    unit — an action and its own outcome are never split across episodes by
    the batching below. Every other event stands alone.

    This reads only `type`, which every event already carries, so it is also
    the deterministic fallback for logs with no richer interaction metadata:
    an egocentric capture or replicated-surrogate stream with no `decision`/
    `tool_result` events at all just produces one unit per event, identical to
    plain per-event batching.
    """
    units = []
    i = 0
    n = len(events)
    while i < n:
        unit = [events[i]]
        if events[i].type == "decision":
            j = i + 1
            while j < n and events[j].type == "tool_result" and len(unit) < _MAX_UNIT_EVENTS:
                unit.append(events[j])
                j += 1
            i = j
        else:
            i += 1
        units.append(unit)
    return units


class MemoryConsolidator:
    """At most one episode per due tick, with boundaries persisted before inference.

    The cursor advances only after a successful consolidation. An interrupted or
    failed attempt retries the same range, even if new stimuli have arrived.
    """

    def __init__(self, memory: MemoryModule, *, every_seconds: float = 300,
                 max_events: int = 20, max_chars: int = 24000, context_events: int = 2,
                 now: Callable[[], float] = time.monotonic):
        if not math.isfinite(every_seconds) or every_seconds <= 0 or type(max_events) is not int or max_events < 1:
            raise ValueError("consolidation interval and event count must be positive")
        if type(max_chars) is not int or max_chars < 4096:
            raise ValueError("max_chars must be at least 4096")
        if type(context_events) is not int or context_events < 0:
            raise ValueError("context_events must be a nonnegative integer")
        self.memory = memory
        self.every_seconds = every_seconds
        self.max_events = max_events
        self.max_chars = max_chars
        self.context_events = context_events
        self._now = now
        self._next_due = 0.0
        self._lock = threading.Lock()
        self.pending_events = 0
        self.last_error: str | None = None

    def tick(self) -> ConsolidationResult | None:
        with self._lock:
            if self._now() < self._next_due:
                return None
            self._next_due = self._now() + self.every_seconds
            try:
                with store_lock(self.memory.memory_dir / "formation"):
                    result = self._run()
                # Repair a small amount of derived vector state at the same
                # explicit maintenance cadence, including after an outage.
                self.memory.repair_embeddings(limit=5)
                self.last_error = None
                return result
            except Exception as exc:
                self.last_error = str(exc)
                raise

    def _run(self) -> ConsolidationResult | None:
        path = self.memory.memory_dir / "formation" / "cursor.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        all_events = self.memory._stimulus_log.read_all()
        cursor_pos = -1
        if state.get("last_id"):
            positions = [i for i, event in enumerate(all_events) if event.id == state["last_id"]]
            if not positions:
                raise ValueError("memory cursor is absent from the stimulus log")
            cursor_pos = positions[0]
        events = all_events[cursor_pos + 1:]
        self.pending_events = len(events)
        if not events:
            return None
        pending = state.get("pending")
        if pending is None:
            batch = []
            size = 0
            # O(n) over the pending backlog; fine at human rates, same tradeoff
            # MemoryModule's own event lookups already make.
            for unit in _interaction_units(events):
                unit_cost = sum(len(event.to_json()) + 1 for event in unit)
                if batch and (len(batch) + len(unit) > self.max_events or size + unit_cost > self.max_chars):
                    break
                batch.extend(unit)
                size += unit_cost
            first, last = batch[0].id, batch[-1].id
            episode_id = "auto-" + hashlib.sha256(f"{first}:{last}".encode()).hexdigest()
            # Bounded prior context: whatever immediately preceded this batch (the
            # tail of an already-consolidated episode, most recently its own
            # action/result pair) — so a continuation ("...it failed") is
            # interpretable without re-offering that prior material as evidence
            # this episode could cite as its own support.
            context_start = max(0, cursor_pos + 1 - self.context_events)
            context_ids = [event.id for event in all_events[context_start:cursor_pos + 1]]
            pending = {
                "episode_id": episode_id, "start_id": first, "end_id": last,
                "context_event_ids": context_ids,
            }
            atomic_json(path, {**state, "pending": pending})
        episode = Episode(
            pending["episode_id"], pending["start_id"], pending["end_id"],
            context_event_ids=tuple(pending.get("context_event_ids", ())),
        )
        result = self.memory.consolidate(episode)
        atomic_json(path, {"last_id": episode.end_id})
        end = next(i for i, event in enumerate(events) if event.id == episode.end_id)
        self.pending_events = len(events) - end - 1
        return result
