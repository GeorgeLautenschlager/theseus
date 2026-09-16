"""MemoryLayer — append-only episode records with bi-temporal timestamps.

Each record is one consolidated episode: the rendered evidence text it was
formed from, an LLM summary of what happened, and the embedding retrieval
runs against. Records are never mutated after write — `ts` is when the
episode happened (the event range's time), the file order is when it was
recorded, and nothing here decays or forgets.

Recency weighting happens at *query* time only: score = cosine × a half-life
decay on record age, computed against "now" passed in by the caller. The
stored records are identical whether queried today or in a year; what shifts
is the weight, not the data.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from theseus.assertion_metadata import render_metadata
from theseus.layer_store import LayerHit, append_record, ensure_store, load_lines, lexical_score, valid_vector, terms


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    ts: datetime
    content: str                      # rendered evidence the episode was formed from
    summary: str                      # LLM one-paragraph rendering — what retrieval embeds and agents read back
    embedding: list[float] = field(default_factory=list)
    source_episode_id: str = ""
    embedding_model: str = ""
    support_event_ids: tuple[str, ...] | None = None
    attribution: str | None = None
    reported_by: str | None = None
    action_status: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "ts": self.ts.astimezone(timezone.utc).isoformat(),
                "content": self.content,
                "summary": self.summary,
                "embedding": self.embedding,
                "source_episode_id": self.source_episode_id,
                "embedding_model": self.embedding_model,
                "support_event_ids": self.support_event_ids,
                "attribution": self.attribution,
                "reported_by": self.reported_by,
                "action_status": self.action_status,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> "MemoryRecord":
        d: dict[str, Any] = json.loads(line)
        return cls(
            id=d["id"],
            ts=datetime.fromisoformat(d["ts"]),
            content=d["content"],
            summary=d["summary"],
            embedding=d.get("embedding", []),
            source_episode_id=d.get("source_episode_id", ""),
            embedding_model=d.get("embedding_model", ""),
            support_event_ids=tuple(d["support_event_ids"]) if d.get("support_event_ids") is not None else None,
            attribution=d.get("attribution"),
            reported_by=d.get("reported_by"),
            action_status=d.get("action_status"),
        )

    def render(self) -> str:
        """What an agent reads back. `content` is provenance (raw evidence);
        rendering it would echo log text the agent may already hold — same
        discipline as MemoryNote.render."""
        return (f"[{self.id}] Episode {self.ts.isoformat()}: {self.summary}"
                + render_metadata(self.support_event_ids, self.attribution,
                                  self.reported_by, self.action_status))


def _recency_weight(ts: datetime, now: datetime, half_life_days: float) -> float:
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life_days) if half_life_days > 0 else 1.0


class MemoryLayer:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = ensure_store(path)
        self._records: list[MemoryRecord] = []
        for line in load_lines(self.path):
            self._records.append(MemoryRecord.from_json(line))

    def add(self, record: MemoryRecord) -> MemoryRecord:
        existing = self.get(record.id)
        if existing is not None:
            return existing
        append_record(self.path, record.to_json())
        self._records.append(record)
        return record

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        now: datetime | None = None,
        half_life_days: float = 30.0,
        embedding_model: str | None = None,
        embeddings: dict[str, list[float]] | None = None,
    ) -> list[LayerHit]:
        """Top-k by cosine × recency weight. Records are never touched; the
        weight is computed here, against `now` (defaults to wall clock)."""
        if not self._records or not valid_vector(embedding):
            return []
        overrides = embeddings or {}
        records = [r for r in self._records if valid_vector(overrides.get(r.id, r.embedding), len(embedding))
                   and (r.id in overrides or embedding_model is None or r.embedding_model == embedding_model)]
        if not records:
            return []
        now = now or datetime.now(timezone.utc)
        q = np.asarray(embedding, dtype=np.float64)
        matrix = np.asarray([overrides.get(r.id, r.embedding) for r in records], dtype=np.float64)
        scores = _cosine_scores(q, matrix)
        weighted = [
            s * _recency_weight(r.ts, now, half_life_days) for s, r in zip(scores, records)
        ]
        order = sorted(range(len(records)), key=lambda i: weighted[i], reverse=True)[:k]
        return [
            LayerHit(id=records[i].id, text=records[i].render(), score=weighted[i])
            for i in order
            if weighted[i] > 0.0
        ]

    def search(self, query: str, k: int = 5) -> list[LayerHit]:
        scored = [(max(lexical_score(query, r.summary), lexical_score(query, r.content)), r)
                  for r in self._records]
        scored.sort(key=lambda pair: (pair[0], pair[1].ts), reverse=True)
        hits = []
        for score, record in scored:
            if score <= 0 or len(hits) >= k:
                break
            text = record.render()
            if lexical_score(query, record.content) > lexical_score(query, record.summary):
                positions = [record.content.casefold().find(term) for term in terms(query)]
                start = max(0, min((p for p in positions if p >= 0), default=0) - 200)
                text += "\nHistorical evidence excerpt: " + record.content[start:start + 1200]
            hits.append(LayerHit(record.id, text, score))
        return hits

    def get(self, record_id: str) -> MemoryRecord | None:
        return next((r for r in self._records if r.id == record_id), None)

    def read_all(self) -> list[MemoryRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)


def _cosine_scores(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Brute-force cosine over every stored embedding. At these scales (thousands
    of 768-d vectors) this is comfortably fast and needs no vector database."""
    if matrix.size == 0:
        return np.zeros(0)
    q_norm = np.linalg.norm(q)
    m_norms = np.linalg.norm(matrix, axis=1)
    denom = m_norms * q_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = matrix @ q / denom
    return np.where(denom > 0, sims, 0.0)
