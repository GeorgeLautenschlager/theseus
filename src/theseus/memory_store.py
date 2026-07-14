"""MemoryStore — append-only JSONL persistence for MemoryNotes.

Same write discipline as StimulusLog: one record per line, append, flush,
fsync. Reads tolerate a torn trailing line (crash mid-write); interior
corruption raises. Notes are immutable in v1 (memory evolution is a deferred
follow-up), which is what lets the store stay append-only.

Retrieval is in-process cosine similarity over all notes — pure Python, no
vector-store dependency. At human rates (thousands of notes, sub-thousand
dims) this is comfortably fast.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from theseus.memory_note import MemoryNote


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class MemoryStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._notes: list[MemoryNote] = self._load()

    def _load(self) -> list[MemoryNote]:
        notes: list[MemoryNote] = []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            try:
                notes.append(MemoryNote.from_json(stripped))
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                is_last = i == len(lines) - 1
                if is_last and not line.endswith("\n"):
                    break  # torn final write — recover by dropping it
                raise ValueError(f"corrupt interior record at line {i}: {exc}") from exc
        return notes

    def add(self, note: MemoryNote) -> MemoryNote:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(note.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._notes.append(note)
        return note

    def read_all(self) -> list[MemoryNote]:
        return list(self._notes)

    def get(self, note_id: str) -> MemoryNote | None:
        return next((n for n in self._notes if n.id == note_id), None)

    def top_k(self, query_embedding: list[float], k: int) -> list[MemoryNote]:
        scored = sorted(
            self._notes,
            key=lambda n: _cosine(query_embedding, n.embedding),
            reverse=True,
        )
        return scored[:k]

    def last_consolidated_id(self) -> str | None:
        """High-water mark of memory formation: the greatest stimulus-event id
        any note was formed from. Derived rather than stored, so deleting the
        notes file resets formation to the start of the log (replayability)."""
        ends = [n.source_span[1] for n in self._notes if n.source_span[1]]
        return max(ends) if ends else None

    def __len__(self) -> int:
        return len(self._notes)
