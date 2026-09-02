"""Shared JSONL persistence discipline for the layered memory stores.

One write contract, shared by every file-backed layer (knowledge, memory, wisdom)
and by the module's ledger and dead-letter files: one record per line, append,
flush, fsync. Reads tolerate a torn trailing line (crash mid-write); corruption
of an *interior* line is a real error and raises — same contract as StimulusLog
and MemoryStore, so a crash in any of these stores recovers the same way.

Layers are separate collaborators behind this common interface; nothing here
knows what a record means. `LayerHit` is the one shared retrieval shape: a
ranked (id, text, score) triple, so the module can fuse layers without knowing
their record types.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LayerHit:
    """One ranked retrieval hit from a layer. `text` is what an agent would read;
    `score` is layer-internal (higher is better); fusion only ever sees ranks."""

    id: str
    text: str
    score: float


def ensure_store(path: str | os.PathLike[str]) -> Path:
    """Create the store's parent dir and an empty file if absent; return a Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)
    return p


def append_record(path: str | os.PathLike[str], line: str) -> None:
    """Append one serialized record, durable before return. Creates parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_lines(path: str | os.PathLike[str]) -> list[str]:
    """All complete lines. A torn final line (crash mid-write) is dropped, never
    raised; a corrupt interior line raises ValueError."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if not stripped:
            continue
        try:
            json.loads(stripped)  # interior corruption check; layers parse for real
        except json.JSONDecodeError as exc:
            is_last = i == len(lines) - 1
            if is_last and not line.endswith("\n"):
                break  # torn final write — recover by dropping it
            raise ValueError(f"corrupt interior record at line {i}: {exc}") from exc
        out.append(stripped)
    return out
