"""WisdomLayer — generalized principles, append-only. A v0 stub on purpose.

Records carry an `evidence_count` (how many distinct episodes supported the
principle when it was written) and retrieval filters on it: a principle backed
by five episodes outranks one backed by one, at equal similarity. What this
layer deliberately does NOT have is promotion or revision logic — nothing here
decides that a memory *becomes* wisdom over time, and no record is ever edited
after write. That machinery is a later module; the store just needs to be able
to hold and serve what consolidation routes here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from theseus.layer_store import LayerHit, append_record, ensure_store, load_lines


@dataclass(frozen=True, slots=True)
class WisdomRecord:
    id: str
    ts: datetime
    statement: str
    embedding: list[float] = field(default_factory=list)
    evidence_count: int = 1
    source_episode_id: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "ts": self.ts.astimezone(timezone.utc).isoformat(),
                "statement": self.statement,
                "embedding": self.embedding,
                "evidence_count": self.evidence_count,
                "source_episode_id": self.source_episode_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> "WisdomRecord":
        d: dict[str, Any] = json.loads(line)
        return cls(
            id=d["id"],
            ts=datetime.fromisoformat(d["ts"]),
            statement=d["statement"],
            embedding=d.get("embedding", []),
            evidence_count=d.get("evidence_count", 1),
            source_episode_id=d.get("source_episode_id", ""),
        )

    def render(self) -> str:
        return f"[{self.id}] {self.statement}"


class WisdomLayer:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = ensure_store(path)
        self._records: list[WisdomRecord] = []
        for line in load_lines(self.path):
            self._records.append(WisdomRecord.from_json(line))

    def add(self, record: WisdomRecord) -> WisdomRecord:
        append_record(self.path, record.to_json())
        self._records.append(record)
        return record

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        min_evidence: int = 0,
    ) -> list[LayerHit]:
        """Top-k by cosine among records with evidence_count >= min_evidence."""
        eligible = [r for r in self._records if r.evidence_count >= min_evidence]
        if not eligible or not embedding:
            return []
        q = np.asarray(embedding, dtype=np.float64)
        matrix = np.asarray([r.embedding for r in eligible], dtype=np.float64)
        q_norm = float(np.linalg.norm(q))
        m_norms = np.linalg.norm(matrix, axis=1)
        denom = m_norms * q_norm
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = matrix @ q / denom
        scores = np.where(denom > 0, sims, 0.0)
        order = sorted(range(len(eligible)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            LayerHit(id=eligible[i].id, text=eligible[i].render(), score=float(scores[i]))
            for i in order
            if scores[i] > 0.0
        ]

    def get(self, record_id: str) -> WisdomRecord | None:
        return next((r for r in self._records if r.id == record_id), None)

    def read_all(self) -> list[WisdomRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)
