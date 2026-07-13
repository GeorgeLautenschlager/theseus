"""MemoryNote — one atomic, immutable long-term memory.

A note is an A-MEM-style enriched record: the raw content it was distilled
from, LLM-generated context/keywords/tags, agentically chosen links to other
notes, and an embedding for similarity retrieval. Notes are a disposable
projection over the StimulusLog: `source_span` records the inclusive event-ID
range a note was formed from, so wiping the note store and replaying the log
rebuilds memory from scratch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class MemoryNote:
    id: str
    ts: datetime
    content: str                      # raw material the note was distilled from
    context: str                      # LLM-generated situating description
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)   # ids of related notes
    source_span: tuple[str, str] = ("", "")          # inclusive [start_id, end_id] of stimulus events
    embedding: list[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "ts": self.ts.astimezone(timezone.utc).isoformat(),
                "content": self.content,
                "context": self.context,
                "keywords": self.keywords,
                "tags": self.tags,
                "links": self.links,
                "source_span": list(self.source_span),
                "embedding": self.embedding,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> "MemoryNote":
        d: dict[str, Any] = json.loads(line)
        return cls(
            id=d["id"],
            ts=datetime.fromisoformat(d["ts"]),
            content=d["content"],
            context=d["context"],
            keywords=d["keywords"],
            tags=d["tags"],
            links=d["links"],
            source_span=(d["source_span"][0], d["source_span"][1]),
            embedding=d["embedding"],
        )

    def render(self) -> str:
        """Human/LLM-readable one-note rendering used in prompt sections."""
        parts = [f"[{self.id}] {self.context}"]
        if self.keywords:
            parts.append(f"keywords: {', '.join(self.keywords)}")
        if self.tags:
            parts.append(f"tags: {', '.join(self.tags)}")
        parts.append(f"content: {self.content}")
        return "\n".join(parts)
