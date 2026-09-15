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


class MemoryConsolidator:
    """At most one episode per due tick, with boundaries persisted before inference.

    The cursor advances only after a successful consolidation. An interrupted or
    failed attempt retries the same range, even if new stimuli have arrived.
    """

    def __init__(self, memory: MemoryModule, *, every_seconds: float = 300,
                 max_events: int = 20, max_chars: int = 24000,
                 now: Callable[[], float] = time.monotonic):
        if not math.isfinite(every_seconds) or every_seconds <= 0 or type(max_events) is not int or max_events < 1:
            raise ValueError("consolidation interval and event count must be positive")
        if type(max_chars) is not int or max_chars < 4096:
            raise ValueError("max_chars must be at least 4096")
        self.memory = memory
        self.every_seconds = every_seconds
        self.max_events = max_events
        self.max_chars = max_chars
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
        events = self.memory._stimulus_log.read_all()
        if state.get("last_id"):
            positions = [i for i, event in enumerate(events) if event.id == state["last_id"]]
            if not positions:
                raise ValueError("memory cursor is absent from the stimulus log")
            events = events[positions[0] + 1:]
        self.pending_events = len(events)
        if not events:
            return None
        pending = state.get("pending")
        if pending is None:
            batch = []
            size = 0
            for event in events[:self.max_events]:
                cost = len(event.to_json()) + 1
                if batch and size + cost > self.max_chars:
                    break
                batch.append(event)
                size += cost
            first, last = batch[0].id, batch[-1].id
            episode_id = "auto-" + hashlib.sha256(f"{first}:{last}".encode()).hexdigest()
            pending = {"episode_id": episode_id, "start_id": first, "end_id": last}
            atomic_json(path, {**state, "pending": pending})
        episode = Episode(**pending)
        result = self.memory.consolidate(episode)
        atomic_json(path, {"last_id": episode.end_id})
        end = next(i for i, event in enumerate(events) if event.id == episode.end_id)
        self.pending_events = len(events) - end - 1
        return result
