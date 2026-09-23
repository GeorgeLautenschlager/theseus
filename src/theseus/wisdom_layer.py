"""WisdomLayer — generalized principles, append-only.

Records carry `status` ("provisional" or "established") and traceable
`supporting_episode_ids` — see assertion_metadata.WISDOM_PROMOTION_THRESHOLD
for the promotion policy. `evidence_count` mirrors `len(supporting_episode_ids)`
for records written under that policy; it is kept as its own field because
older records (written before per-episode tracking existed) only ever had a
bare count. As with KnowledgeLayer (#66), this layer does not decide *how* a
new principle relates to what's already known — that is memory_module's
reconciliation step, which weighs wording and independence that a similarity
threshold alone can't. This layer only applies whatever `supersedes` the
caller names: a record that supersedes nothing stays current alongside
whatever else is current (an unresolved contradiction, or simply a second,
unrelated principle), and a record marked `reconciliation="historical"` is
kept in the append-only file but never enters the current set. No record is
ever edited after write — a promotion from provisional to established is a
new record superseding the old one, not a mutation.
"""

from __future__ import annotations

import json
import os
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from theseus.assertion_metadata import render_metadata
from theseus.layer_store import LayerHit, append_record, compact_vector, ensure_store, iter_lines, lexical_score, valid_vector


@dataclass(frozen=True, slots=True)
class WisdomRecord:
    id: str
    ts: datetime
    statement: str
    embedding: Sequence[float] = field(default_factory=list)  # array('d') once loaded
    evidence_count: int = 1
    source_episode_id: str = ""
    embedding_model: str = ""
    support_event_ids: tuple[str, ...] | None = None
    attribution: str | None = None
    reported_by: str | None = None
    action_status: str | None = None
    status: str | None = None                       # assertion_metadata.WISDOM_STATUSES
    supporting_episode_ids: tuple[str, ...] = ()
    supersedes: str | None = None
    reconciliation: str | None = None                # assertion_metadata.RECONCILIATION_DECISIONS
    contradicts: tuple[str, ...] | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "ts": self.ts.astimezone(timezone.utc).isoformat(),
                "statement": self.statement,
                "embedding": list(self.embedding) if isinstance(self.embedding, array) else self.embedding,
                "evidence_count": self.evidence_count,
                "source_episode_id": self.source_episode_id,
                "embedding_model": self.embedding_model,
                "support_event_ids": self.support_event_ids,
                "attribution": self.attribution,
                "reported_by": self.reported_by,
                "action_status": self.action_status,
                "status": self.status,
                "supporting_episode_ids": list(self.supporting_episode_ids),
                "supersedes": self.supersedes,
                "reconciliation": self.reconciliation,
                "contradicts": self.contradicts,
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
            embedding=compact_vector(d.get("embedding", [])),
            evidence_count=d.get("evidence_count", 1),
            source_episode_id=d.get("source_episode_id", ""),
            embedding_model=d.get("embedding_model", ""),
            support_event_ids=tuple(d["support_event_ids"]) if d.get("support_event_ids") is not None else None,
            attribution=d.get("attribution"),
            reported_by=d.get("reported_by"),
            action_status=d.get("action_status"),
            status=d.get("status"),
            supporting_episode_ids=tuple(d.get("supporting_episode_ids") or ()),
            supersedes=d.get("supersedes"),
            reconciliation=d.get("reconciliation"),
            contradicts=tuple(d["contradicts"]) if d.get("contradicts") is not None else None,
        )

    def render(self) -> str:
        text = f"[{self.id}] {self.statement}"
        if self.status == "provisional":
            text += " (provisional)"
        text += render_metadata(self.support_event_ids, self.attribution,
                                self.reported_by, self.action_status)
        if self.contradicts:
            text += f" [contradicts {', '.join(self.contradicts)}]"
        return text


class WisdomLayer:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = ensure_store(path)
        self._records: list[WisdomRecord] = []
        self._by_id: dict[str, WisdomRecord] = {}
        self._current_ids: set[str] = set()
        for line in iter_lines(self.path):
            self._apply(WisdomRecord.from_json(line))

    def _apply(self, record: WisdomRecord) -> None:
        """Fold one record into the in-memory projection. File order only: a
        record can supersede nothing but an id already on file by the time it
        was written, so one forward pass is enough."""
        self._records.append(record)
        self._by_id[record.id] = record
        if record.reconciliation != "historical":
            self._current_ids.add(record.id)
        if record.supersedes is not None:
            self._current_ids.discard(record.supersedes)

    def add(self, record: WisdomRecord) -> WisdomRecord:
        """Append `record`, durably, and apply whatever supersession it names.

        This layer does not decide whether a new principle reinforces,
        contradicts, or stands independent of an existing one, nor whether it
        starts current or merely historical — the caller (reconciliation)
        already decided that. Applying it here is pure bookkeeping: drop the
        named id from the current set, and skip adding this one if historical.
        """
        if record.id in self._by_id:
            return self._by_id[record.id]
        if record.supersedes is not None and record.supersedes not in self._by_id:
            raise ValueError(f"supersedes references unknown record {record.supersedes!r}")
        append_record(self.path, record.to_json())
        self._apply(record)
        return record

    def current(self) -> list[WisdomRecord]:
        """Current (non-superseded, non-historical) records. More than one
        can coexist — independent principles, or an unresolved contradiction
        that stays visible rather than picking a winner."""
        return sorted((self._by_id[rid] for rid in self._current_ids), key=lambda r: r.ts)

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        min_evidence: int = 0,
        embedding_model: str | None = None,
        embeddings: dict[str, list[float]] | None = None,
    ) -> list[LayerHit]:
        """Top-k by cosine among current records with evidence_count >= min_evidence."""
        if not valid_vector(embedding):
            return []
        overrides = embeddings or {}
        eligible = [r for r in self.current() if r.evidence_count >= min_evidence
                    and valid_vector(overrides.get(r.id, r.embedding), len(embedding))
                    and (r.id in overrides or embedding_model is None or r.embedding_model == embedding_model)]
        if not eligible:
            return []
        q = np.asarray(embedding, dtype=np.float64)
        matrix = np.asarray([overrides.get(r.id, r.embedding) for r in eligible], dtype=np.float64)
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

    def search(self, query: str, k: int = 5) -> list[LayerHit]:
        scored = [(lexical_score(query, r.statement), r) for r in self.current()]
        scored.sort(key=lambda pair: (pair[0], pair[1].evidence_count, pair[1].ts), reverse=True)
        return [LayerHit(r.id, r.render(), score) for score, r in scored if score > 0][:k]

    def get(self, record_id: str) -> WisdomRecord | None:
        return self._by_id.get(record_id)

    def read_all(self) -> list[WisdomRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)
