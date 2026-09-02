"""KnowledgeLayer — current-state facts, append-only with load-time projection.

The file holds every KnowledgeRecord ever written; the *projection* (latest
record per subject+predicate) is rebuilt in memory on load and maintained on
append. Supersession is explicit and logged: writing a new value for an existing
subject+predicate appends a record whose `supersedes` names the record it
replaces — the old record stays in the file, untouched, and drops out of the
projection. There is no decay and no mutation anywhere in this layer; what the
agent knows *now* is always derivable by replaying the file.

Retrieval is deterministic token overlap between query terms and each current
record's subject/predicate — no embedding, no LLM. Facts are looked up by
predicate and subject, not by vibes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseus.layer_store import LayerHit, append_record, ensure_store, load_lines


@dataclass(frozen=True, slots=True)
class KnowledgeRecord:
    id: str
    ts: datetime
    subject: str
    predicate: str
    value: str
    source_episode_id: str = ""
    supersedes: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "ts": self.ts.astimezone(timezone.utc).isoformat(),
                "subject": self.subject,
                "predicate": self.predicate,
                "value": self.value,
                "source_episode_id": self.source_episode_id,
                "supersedes": self.supersedes,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> "KnowledgeRecord":
        d: dict[str, Any] = json.loads(line)
        return cls(
            id=d["id"],
            ts=datetime.fromisoformat(d["ts"]),
            subject=d["subject"],
            predicate=d["predicate"],
            value=d["value"],
            source_episode_id=d.get("source_episode_id", ""),
            supersedes=d.get("supersedes"),
        )

    def render(self) -> str:
        return f"{self.subject} {self.predicate}: {self.value}"


def _key(subject: str, predicate: str) -> tuple[str, str]:
    return (subject.casefold(), predicate.casefold())


class KnowledgeLayer:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = ensure_store(path)
        self._records: list[KnowledgeRecord] = []
        self._projection: dict[tuple[str, str], KnowledgeRecord] = {}
        self._by_id: dict[str, KnowledgeRecord] = {}
        for line in load_lines(self.path):
            record = KnowledgeRecord.from_json(line)
            self._records.append(record)
            self._by_id[record.id] = record
            # Replay: later records for the same key supersede earlier ones.
            self._projection[_key(record.subject, record.predicate)] = record

    def add(self, record: KnowledgeRecord) -> KnowledgeRecord:
        """Append `record`, durably. If a current record exists for the same
        subject+predicate, the appended record is rewritten to name it in
        `supersedes` — supersession happens here and only here."""
        key = _key(record.subject, record.predicate)
        current = self._projection.get(key)
        if current is not None:
            record = replace(record, supersedes=current.id)
        append_record(self.path, record.to_json())
        self._records.append(record)
        self._by_id[record.id] = record
        self._projection[key] = record
        return record

    def current(self, subject: str | None = None, predicate: str | None = None) -> list[KnowledgeRecord]:
        """Current (non-superseded) records, optionally filtered by exact
        subject and/or predicate (None = wildcard)."""
        out = []
        for r in self._projection.values():
            if subject is not None and r.subject.casefold() != subject.casefold():
                continue
            if predicate is not None and r.predicate.casefold() != predicate.casefold():
                continue
            out.append(r)
        return sorted(out, key=lambda r: r.ts)

    def get(self, record_id: str) -> KnowledgeRecord | None:
        return self._by_id.get(record_id)

    def search(self, terms: set[str], k: int = 5) -> list[LayerHit]:
        """Current records ranked by how many distinct query terms hit their
        subject or predicate. Deterministic; no embedding involved."""
        if not terms:
            return []
        scored: list[tuple[float, datetime, KnowledgeRecord]] = []
        for record in self._projection.values():
            haystack = f"{record.subject} {record.predicate}".casefold()
            hits = sum(1 for t in terms if t.casefold() in haystack)
            if hits:
                scored.append((float(hits), record.ts, record))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [LayerHit(id=r.id, text=r.render(), score=s) for s, _, r in scored[:k]]

    def read_all(self) -> list[KnowledgeRecord]:
        """Every record ever written, file order — the append-only truth."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)
