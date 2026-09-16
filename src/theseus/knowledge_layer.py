"""KnowledgeLayer — current-state facts, append-only with load-time projection.

The file holds every KnowledgeRecord ever written; the *projection* (which
records are current) is rebuilt in memory on load and maintained on append.
Supersession is explicit and logged, but this layer does not decide it: a
record's `supersedes` names the record it replaces, chosen by the caller
(memory_module's reconciliation step, weighing wording, timing, and evidence
that mere subject+predicate key matching and "newest timestamp wins" cannot
capture — reconciliation exists precisely because those two are not reliable
proxies for "the same claim" or "the more current claim"). This layer's job is
mechanical: apply whatever supersession the caller decided, durably. A record
that supersedes nothing stays current alongside whatever else is current for
its subject+predicate — two records can legitimately share a key at once
(coexisting attributes under a broad predicate, or an unresolved contradiction
between two reports) since nothing here forces exclusivity. A record marked
`reconciliation="historical"` or `"unresolved"` is written for the append-only record but never
enters the current set at all. There is no decay and no mutation anywhere in
this layer; what the agent knows *now* is always derivable by replaying the
file.

Retrieval is deterministic token overlap between query terms and each current
record's subject, predicate, and value — no embedding, no LLM. Facts are looked up by
predicate and subject, not by vibes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseus.assertion_metadata import render_metadata
from theseus.layer_store import LayerHit, append_record, ensure_store, load_lines, terms as tokenize


@dataclass(frozen=True, slots=True)
class KnowledgeRecord:
    id: str
    ts: datetime
    subject: str
    predicate: str
    value: str
    source_episode_id: str = ""
    supersedes: str | None = None
    support_event_ids: tuple[str, ...] | None = None
    attribution: str | None = None
    reported_by: str | None = None
    action_status: str | None = None
    reconciliation: str | None = None       # model decisions, or internal "unresolved" on failure
    contradicts: tuple[str, ...] | None = None  # current records this one conflicts with

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
                "support_event_ids": self.support_event_ids,
                "attribution": self.attribution,
                "reported_by": self.reported_by,
                "action_status": self.action_status,
                "reconciliation": self.reconciliation,
                "contradicts": self.contradicts,
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
            support_event_ids=tuple(d["support_event_ids"]) if d.get("support_event_ids") is not None else None,
            attribution=d.get("attribution"),
            reported_by=d.get("reported_by"),
            action_status=d.get("action_status"),
            reconciliation=d.get("reconciliation"),
            contradicts=tuple(d["contradicts"]) if d.get("contradicts") is not None else None,
        )

    def render(self) -> str:
        label = "Unresolved claim" if self.reconciliation == "unresolved" else "Current fact"
        text = (f"[{self.id}] {label}: {self.subject} {self.predicate}: {self.value}"
                + render_metadata(self.support_event_ids, self.attribution,
                                  self.reported_by, self.action_status))
        if self.contradicts:
            text += f" [contradicts {', '.join(self.contradicts)}]"
        return text


def _key(subject: str, predicate: str) -> tuple[str, str]:
    return (" ".join(subject.casefold().split()), " ".join(predicate.casefold().split()))


class KnowledgeLayer:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = ensure_store(path)
        self._records: list[KnowledgeRecord] = []
        self._by_id: dict[str, KnowledgeRecord] = {}
        self._current_ids: set[str] = set()
        for line in load_lines(self.path):
            self._apply(KnowledgeRecord.from_json(line))

    def _apply(self, record: KnowledgeRecord) -> None:
        """Fold one record into the in-memory projection. File order only: a
        record can supersede nothing but an id already on file by the time it
        was written, so one forward pass is enough — no re-ranking by ts."""
        self._records.append(record)
        self._by_id[record.id] = record
        if record.reconciliation not in ("historical", "unresolved"):
            self._current_ids.add(record.id)
        if record.supersedes is not None:
            self._current_ids.discard(record.supersedes)

    def add(self, record: KnowledgeRecord) -> KnowledgeRecord:
        """Append `record`, durably, and apply whatever supersession it names.

        This layer does not decide *whether* one fact replaces, reinforces,
        coexists with, or contradicts another, nor whether it is current or
        merely historical — the caller (reconciliation) already decided that
        and encoded it in `supersedes`/`reconciliation`. Applying it here is
        pure bookkeeping: drop the named id from the current set, and skip
        adding this one if it is historical or unresolved.
        """
        if record.id in self._by_id:
            return self._by_id[record.id]
        if record.supersedes is not None and record.supersedes not in self._by_id:
            raise ValueError(f"supersedes references unknown record {record.supersedes!r}")
        append_record(self.path, record.to_json())
        self._apply(record)
        return record

    def current(self, subject: str | None = None, predicate: str | None = None) -> list[KnowledgeRecord]:
        """Current (non-superseded, non-historical, resolved) records, optionally
        filtered by exact subject and/or predicate (None = wildcard). More
        than one record can be current for the same subject+predicate at
        once — coexisting attributes or an unresolved contradiction are not
        collapsed to a single "latest" value."""
        out = []
        for rid in self._current_ids:
            r = self._by_id[rid]
            if subject is not None and _key(r.subject, "")[0] != _key(subject, "")[0]:
                continue
            if predicate is not None and _key("", r.predicate)[1] != _key("", predicate)[1]:
                continue
            out.append(r)
        return sorted(out, key=lambda r: r.ts)

    def get(self, record_id: str) -> KnowledgeRecord | None:
        return self._by_id.get(record_id)

    def search(self, terms: set[str], k: int = 5) -> list[LayerHit]:
        """Current records ranked by how many distinct query terms hit their
        subject, predicate, or value. Deterministic; no embedding involved."""
        if not terms:
            return []
        scored: list[tuple[float, datetime, KnowledgeRecord]] = []
        for rid in self._current_ids:
            record = self._by_id[rid]
            haystack = tokenize(f"{record.subject} {record.predicate} {record.value}")
            hits = len({term.casefold() for term in terms} & haystack)
            if hits:
                scored.append((float(hits), record.ts, record))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [LayerHit(id=r.id, text=r.render(), score=s) for s, _, r in scored[:k]]

    def read_all(self) -> list[KnowledgeRecord]:
        """Every record ever written, file order — the append-only truth."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)
