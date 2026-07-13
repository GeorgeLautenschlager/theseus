"""StimulusLog — the immutable, append-only event log.

This is the bedrock. Everything downstream (segments, episodes) is a disposable
projection over this; if a transform improves, we replay the log and rebuild.
So the log's one job is to be the dumbest, most bulletproof link in the chain:
append, fsync, survive crashes, and bake in *no* substrate assumptions.

A record is a typed StimulusEvent — NOT a "turn", NOT a prompt/response pair.
A conversational exchange is just one `type` whose payload lives in `content`.
An egocentric capture or a game observation is another. All three agents can
therefore share one log.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

# --- Sortable IDs (minimal ULID) ------------------------------------------------
# 48-bit ms timestamp + 80-bit randomness, Crockford base32. Lexically sortable
# by creation time, no coordination needed. Good enough for a v1 WAL at human
# rates; strict intra-millisecond monotonicity is not guaranteed (file order is
# the tiebreaker, and the log is append-only so file order is stable).
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _b32(n: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


def new_id(ms: int | None = None) -> str:
    ms = int(time.time() * 1000) if ms is None else ms
    rand = int.from_bytes(os.urandom(10), "big")
    return _b32(ms, 10) + _b32(rand, 16)  # 26 chars


# --- Event ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StimulusEvent:
    id: str
    ts: datetime
    actor: str           # who/what produced it ("george", "tam", "env", "sensor")
    type: str            # "exchange" | "capture" | "observation" | ...
    content: dict[str, Any]  # type-specific payload; e.g. {"prompt":..,"response":..}

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "ts": self.ts.astimezone(timezone.utc).isoformat(),
                "actor": self.actor,
                "type": self.type,
                "content": self.content,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> "StimulusEvent":
        d = json.loads(line)
        return cls(
            id=d["id"],
            ts=datetime.fromisoformat(d["ts"]),
            actor=d["actor"],
            type=d["type"],
            content=d["content"],
        )


# --- Log ------------------------------------------------------------------------
class StimulusLog:
    """Append-only JSONL. One event per line. fsync per append.

    Reads tolerate a torn trailing line (crash mid-write): the partial final
    line is dropped on read, never raised. Corruption of an *interior* line is
    a real error and is raised, because that should never happen to an
    append-only file and silently skipping it would hide data loss.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(
        self,
        actor: str,
        type: str,
        content: dict[str, Any],
        ts: datetime | None = None,
    ) -> StimulusEvent:
        ts = ts or datetime.now(timezone.utc)
        event = StimulusEvent(id=new_id(int(ts.timestamp() * 1000)),
                              ts=ts, actor=actor, type=type, content=content)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
        return event

    def read_all(self) -> list[StimulusEvent]:
        events: list[StimulusEvent] = []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            try:
                events.append(StimulusEvent.from_json(stripped))
            except (json.JSONDecodeError, KeyError) as exc:
                is_last = i == len(lines) - 1
                if is_last and not line.endswith("\n"):
                    break  # torn final write — recover by dropping it
                raise ValueError(f"corrupt interior record at line {i}: {exc}") from exc
        return events

    def read_range(self, start_id: str, end_id: str) -> list[StimulusEvent]:
        """Inclusive [start_id, end_id]. IDs are lexically sortable, so a span
        is just a string range — robust to re-encoding, unlike line offsets."""
        return [e for e in self.read_all() if start_id <= e.id <= end_id]

    def __iter__(self) -> Iterator[StimulusEvent]:
        return iter(self.read_all())
