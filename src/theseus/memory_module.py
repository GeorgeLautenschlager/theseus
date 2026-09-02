"""MemoryModule — the single memory boundary an agent core programs against.

The core sees two capabilities and nothing else: `recall(query, budget_tokens)`
and `consolidate(episode)`. Everything below this line — which layers exist,
how a query fans out, how results fuse, what gets written where — is internal
and may change without the core noticing. That is the leak contract: no layer
name appears in a public signature or in a result field the core must
*interpret to act*; provenance and cost data carry layer names as opaque
strings a caller may log but never branch on.

Retrieval is deterministic fan-out + reciprocal rank fusion (RRF): every layer
answers the query in its own way, each returns a ranked list, and RRF merges
the rankings without comparing any layer's scores to another's (they live on
different scales). Misses are data — a layer that finds nothing contributes a
string to `RecallResult.misses`, never an exception.
"""

from __future__ import annotations

import json
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseus.intelligence_layer import IntelligenceLayer
from theseus.json_utils import parse_json_response
from theseus.knowledge_layer import KnowledgeLayer, KnowledgeRecord
from theseus.layer_store import LayerHit, append_record, load_lines
from theseus.memory_layer import MemoryLayer, MemoryRecord
from theseus.memory_prompts import build_extraction_prompt, extraction_json_schema
from theseus.stimulus_log import StimulusLog, new_id
from theseus.tools.recall import RECALL_TOOL_NAME
from theseus.wisdom_layer import WisdomLayer, WisdomRecord

# RRF constant: 60 is the standard choice; it flattens score-scale differences
# between layers so rank order, not raw scores, drives fusion.
_RRF_K = 60


def estimate_tokens(text: str) -> int:
    """Naive token estimate: ~4 chars/token. Good enough to budget a window."""
    return max(1, len(text) // 4) if text else 0


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where one recalled entry came from. `layer` is opaque to callers —
    log it, don't branch on it."""

    layer: str
    record_id: str
    rank: int          # pre-fusion rank within its layer (1-based)
    score: float       # pre-fusion, layer-internal score


@dataclass(frozen=True, slots=True)
class RecallEntry:
    text: str
    provenance: Provenance
    tokens: int        # estimated cost of this entry in the window


@dataclass(frozen=True, slots=True)
class RecallResult:
    query: str
    subqueries: tuple[str, ...]
    entries: tuple[RecallEntry, ...]
    misses: tuple[str, ...]      # explicit: what was looked for and not found
    total_tokens: int            # estimated tokens the entries cost
    costs: dict[str, float] = field(default_factory=dict)  # per-layer + "total" seconds


@dataclass(frozen=True, slots=True)
class Episode:
    """One consolidation unit, supplied by the caller. The module does no
    episode detection — the agent application decides when an episode ends."""

    episode_id: str
    start_id: str    # inclusive stimulus-event id
    end_id: str      # inclusive


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    episode_id: str
    extracted: int            # candidate assertions the LLM produced
    routed: dict[str, int]    # layer -> records written (opaque data)
    supersessions: int        # knowledge writes that replaced an earlier value
    schema_failures: int      # dead-lettered, counted
    tokens_in: int            # estimated prompt tokens consumed
    tokens_out: int           # estimated response tokens produced
    wall_time_s: float
    skipped: bool = False     # True when the episode was already consolidated


class MemoryModule:
    def __init__(
        self,
        memory_dir: str | Path,
        stimulus_log: StimulusLog,
        embedding_providers: list[Any] | None = None,
        model_providers: list[Any] | None = None,
        *,
        per_layer_k: int = 5,
        intelligence_tail: int = 20,
        recency_half_life_days: float = 30.0,
        lenient_fact_routing: bool = False,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.knowledge = KnowledgeLayer(self.memory_dir / "knowledge.jsonl")
        self.memory = MemoryLayer(self.memory_dir / "memory.jsonl")
        self.wisdom = WisdomLayer(self.memory_dir / "wisdom.jsonl")
        self.intelligence = IntelligenceLayer(stimulus_log, tail=intelligence_tail)
        self._stimulus_log = stimulus_log
        self._embedding_providers = embedding_providers or []
        self._model_providers = model_providers or []
        self._per_layer_k = per_layer_k
        self._recency_half_life_days = recency_half_life_days
        # Strict (default): a fact without a full triple is schema-invalid and
        # dead-letters. Lenient: it routes to Memory with its statement intact —
        # the model clearly meant "durable claim", and the statement is what
        # recall needs. A/B'd in the 2026-08-30 sonnet eval.
        self._lenient_fact_routing = lenient_fact_routing
        # Idempotency ledger: episode ids already consolidated. Re-consolidating
        # one is a no-op, so replaying a range never double-writes.
        self._processed_episodes: set[str] = {
            json.loads(line)["episode_id"]
            for line in load_lines(self.memory_dir / "consolidation_ledger.jsonl")
        }

    # -- recall ---------------------------------------------------------------

    def recall(self, query: str, budget_tokens: int) -> RecallResult:
        """Fan `query` out to every layer, fuse the rankings (RRF), and fill
        `budget_tokens` of estimated window with the fused entries. Misses come
        back as data; nothing here raises for an empty or unavailable layer."""
        started = time.monotonic()
        per_layer: dict[str, list[LayerHit]] = {}
        costs: dict[str, float] = {}

        t0 = time.monotonic()
        per_layer["knowledge"] = self._search_knowledge(query)
        costs["knowledge"] = time.monotonic() - t0

        embedding = self._embed(query)
        if embedding is not None:
            t0 = time.monotonic()
            per_layer["memory"] = self.memory.query(
                embedding, k=self._per_layer_k, half_life_days=self._recency_half_life_days
            )
            costs["memory"] = time.monotonic() - t0
            t0 = time.monotonic()
            per_layer["wisdom"] = self.wisdom.query(embedding, k=self._per_layer_k)
            costs["wisdom"] = time.monotonic() - t0
        else:
            per_layer["memory"] = []
            per_layer["wisdom"] = []

        t0 = time.monotonic()
        per_layer["intelligence"] = self.intelligence.read()[: self._per_layer_k]
        costs["intelligence"] = time.monotonic() - t0

        fused = _rrf_fuse(per_layer)
        misses = [f"{layer}: no matches" for layer, hits in per_layer.items() if not hits]
        if embedding is None:
            misses.append("embedding unavailable")

        entries: list[tuple[RecallEntry, int]] = []  # (entry, post-fusion rank)
        total = 0
        for post_rank, entry in enumerate(fused, start=1):
            if total + entry.tokens <= budget_tokens:
                entries.append((entry, post_rank))
                total += entry.tokens

        costs["total"] = time.monotonic() - started
        result = RecallResult(
            query=query,
            subqueries=(query,),
            entries=tuple(e for e, _ in entries),
            misses=tuple(misses),
            total_tokens=total,
            costs=costs,
        )
        self._trace_recall(result, per_layer, [(e, pr) for e, pr in entries])
        return result

    # -- instrumentation --------------------------------------------------------

    def _trace_recall(
        self,
        result: RecallResult,
        per_layer: dict[str, list[LayerHit]],
        returned: list[tuple[RecallEntry, int]],
    ) -> None:
        """One JSONL record per recall — the standing trace the eval harness and
        debugging live off of. `re_queried_within_n_turns` is derived offline
        from the StimulusLog (a future event can't be known at write time)."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "query": result.query,
            "subqueries": list(result.subqueries),
            "layers_queried": sorted(per_layer),
            "layer_hit_counts": {layer: len(hits) for layer, hits in per_layer.items()},
            "returned": [
                {
                    "record_id": entry.provenance.record_id,
                    "layer": entry.provenance.layer,
                    "pre_fusion_rank": entry.provenance.rank,
                    "post_fusion_rank": post_rank,
                }
                for entry, post_rank in returned
            ],
            "misses": list(result.misses),
            "total_tokens": result.total_tokens,
            "costs": dict(result.costs),
            "re_queried_within_n_turns": None,
        }
        append_record(self.memory_dir / "traces" / "recall.jsonl", json.dumps(record, ensure_ascii=False))

    def _search_knowledge(self, query: str) -> list[LayerHit]:
        # strip punctuation per token: "standup?" must match "standup"
        terms = {t.strip(string.punctuation) for t in query.lower().split() if len(t.strip(string.punctuation)) > 2}
        return self.knowledge.search(terms, k=self._per_layer_k)

    def _embed(self, text: str) -> list[float] | None:
        for provider in self._embedding_providers:
            if not getattr(provider, "is_available", lambda: True)():
                continue
            try:
                return list(provider.embed(text))
            except Exception:
                continue
        return None

    # -- consolidation ---------------------------------------------------------

    def consolidate(self, episode: Episode) -> ConsolidationResult:
        """Consolidate one caller-supplied episode: extract candidate assertions
        from its evidence, validate them, route each to a layer, and write. The
        recall-flagged stimuli in the range are readable context only — never
        evidence, never provenance. Idempotent per episode."""
        started = time.monotonic()
        if episode.episode_id in self._processed_episodes:
            return ConsolidationResult(
                episode_id=episode.episode_id, extracted=0, routed={},
                supersessions=0, schema_failures=0, tokens_in=0, tokens_out=0,
                wall_time_s=0.0, skipped=True,
            )

        events = self._episode_events(episode)
        evidence = [e for e in events if not _is_recall_flagged(e)]
        context_only = [e for e in events if _is_recall_flagged(e)]
        if not evidence:
            # Nothing to consolidate (empty range, or only recall output). No
            # ledger write: nothing was written, so a retry is safe.
            return ConsolidationResult(
                episode_id=episode.episode_id, extracted=0, routed={},
                supersessions=0, schema_failures=0, tokens_in=0, tokens_out=0,
                wall_time_s=time.monotonic() - started,
            )

        provider = self._first_model_provider()
        prompt = build_extraction_prompt(
            "\n".join(e.to_json() for e in evidence),
            "\n".join(e.to_json() for e in context_only),
        )
        raw = provider.chat(prompt, json_schema=extraction_json_schema())
        tokens_in, tokens_out = estimate_tokens(prompt), estimate_tokens(raw)

        parsed = parse_json_response(raw)
        summary = str(parsed.get("summary", ""))
        candidates = parsed.get("assertions", [])
        episode_ts = events[-1].ts

        routed: dict[str, int] = {}
        supersessions = 0
        schema_failures = 0
        for candidate in candidates:
            assertion_id = new_id()  # assigned before routing, per the brief
            reason = _validate_assertion(
                candidate, lenient=self._lenient_fact_routing
            )
            if reason is not None:
                schema_failures += 1
                append_record(
                    self.memory_dir / "dead_letter.jsonl",
                    json.dumps(
                        {
                            "episode_id": episode.episode_id,
                            "assertion_id": assertion_id,
                            "candidate": candidate,
                            "reason": reason,
                        },
                        ensure_ascii=False, default=str,
                    ),
                )
                continue
            layer = _route_write(candidate)
            self._write_assertion(
                layer, assertion_id, episode.episode_id, episode_ts, candidate
            )
            routed[layer] = routed.get(layer, 0) + 1
            if layer == "knowledge" and self.knowledge.get(assertion_id).supersedes:
                supersessions += 1

        # The episode record itself: what happened, formed from the evidence.
        self._write_episode_record(episode.episode_id, episode_ts, evidence, summary)
        routed["memory"] = routed.get("memory", 0) + 1

        append_record(
            self.memory_dir / "consolidation_ledger.jsonl",
            json.dumps({"episode_id": episode.episode_id}),
        )
        self._processed_episodes.add(episode.episode_id)
        wall_time_s = time.monotonic() - started
        self._trace_consolidation(
            {
                "episode_id": episode.episode_id,
                "candidates_extracted": len(candidates),
                "routed": routed,
                "supersessions": supersessions,
                "schema_failures": schema_failures,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "wall_time_s": wall_time_s,
            }
        )
        return ConsolidationResult(
            episode_id=episode.episode_id, extracted=len(candidates), routed=routed,
            supersessions=supersessions, schema_failures=schema_failures,
            tokens_in=tokens_in, tokens_out=tokens_out, wall_time_s=wall_time_s,
        )

    def _write_assertion(
        self, layer: str, assertion_id: str, episode_id: str, ts, candidate: dict
    ) -> None:
        if layer == "knowledge":
            self.knowledge.add(
                KnowledgeRecord(
                    id=assertion_id, ts=ts, subject=candidate["subject"].strip(),
                    predicate=candidate["predicate"].strip(), value=candidate["value"].strip(),
                    source_episode_id=episode_id,
                )
            )
        elif layer == "wisdom":
            self.wisdom.add(
                WisdomRecord(
                    id=assertion_id, ts=ts, statement=candidate["statement"].strip(),
                    embedding=self._embed(candidate["statement"]) or [],
                    evidence_count=1, source_episode_id=episode_id,
                )
            )
        else:  # memory: an event assertion is a small record of its own
            self.memory.add(
                MemoryRecord(
                    id=assertion_id, ts=ts, content=candidate["statement"].strip(),
                    summary=candidate["statement"].strip(),
                    embedding=self._embed(candidate["statement"]) or [],
                    source_episode_id=episode_id,
                )
            )

    def _write_episode_record(self, episode_id: str, ts, evidence, summary: str) -> None:
        self.memory.add(
            MemoryRecord(
                id=new_id(), ts=ts,
                content="\n".join(e.to_json() for e in evidence),
                summary=summary or "(no summary)",
                embedding=self._embed(summary) or [],
                source_episode_id=episode_id,
            )
        )

    def _episode_events(self, episode: Episode) -> list:
        """The episode's stimuli in file order, between its boundary ids.

        Not `read_range`: ULIDs are random within a millisecond, and real logs
        burst events faster than that — the log's own contract makes file order
        the tiebreaker, so the span is positional, not lexical. O(n) scan; fine
        at human rates (ponytail: windowed index if logs get huge).
        """
        events = self._stimulus_log.read_all()
        start = end = None
        for i, event in enumerate(events):
            if event.id == episode.start_id:
                start = i
            if event.id == episode.end_id:
                end = i
        if start is None or end is None:
            raise ValueError("episode boundary id not found in stimulus log")
        lo, hi = (start, end) if start <= end else (end, start)
        return events[lo : hi + 1]

    def _first_model_provider(self):
        for provider in self._model_providers:
            if getattr(provider, "is_available", lambda: True)():
                return provider
        raise RuntimeError("no model providers available for consolidation")

    def _trace_consolidation(self, record: dict[str, Any]) -> None:
        append_record(
            self.memory_dir / "traces" / "consolidation.jsonl",
            json.dumps({**record, "ts": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
        )


# -- fusion ---------------------------------------------------------------------

def _is_recall_flagged(event) -> bool:
    """A recall-flagged stimulus is the logged output of the agent's own recall
    act — readable as context during consolidation, never evidence."""
    return event.type == "tool_result" and event.content.get("tool") == RECALL_TOOL_NAME


def _route_write(candidate: dict) -> str:
    """Deterministic write routing from the extraction's own signals. No LLM in
    the loop for v0: facts are checkable claims, principles guide behavior,
    everything else is an event. A fact without a full triple (only possible
    under lenient routing) keeps its statement in Memory."""
    kind = candidate.get("kind")
    if kind == "fact" and _has_triple(candidate):
        return "knowledge"
    if kind == "principle":
        return "wisdom"
    return "memory"


def _has_triple(candidate: dict) -> bool:
    return all(
        isinstance(candidate.get(field_name), str) and candidate[field_name].strip()
        for field_name in ("subject", "predicate", "value")
    )


def _validate_assertion(candidate: Any, lenient: bool = False) -> str | None:
    """Routing contract per candidate. Returns a dead-letter reason, or None.

    With `lenient`, a fact missing its triple is still valid — it just cannot
    reach Knowledge, so `_route_write` sends it to Memory."""
    if not isinstance(candidate, dict):
        return "not an object"
    kind = candidate.get("kind")
    if kind not in ("fact", "principle", "event"):
        return f"unknown kind {kind!r}"
    statement = candidate.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        return "missing statement"
    if kind == "fact" and not lenient:
        for field_name in ("subject", "predicate", "value"):
            value = candidate.get(field_name)
            if not isinstance(value, str) or not value.strip():
                return f"fact missing {field_name}"
    return None


def _rrf_fuse(per_layer: dict[str, list[LayerHit]]) -> list[RecallEntry]:
    """Reciprocal rank fusion over the per-layer ranked lists. Layers never see
    each other's scores; only ranks combine."""
    fused: dict[tuple[str, str], tuple[float, Provenance, str]] = {}
    for layer, hits in per_layer.items():
        for rank, hit in enumerate(hits, start=1):
            key = (layer, hit.id)
            contribution = 1.0 / (_RRF_K + rank)
            if key in fused:
                score, provenance, text = fused[key]
                fused[key] = (score + contribution, provenance, text)
            else:
                fused[key] = (
                    contribution,
                    Provenance(layer=layer, record_id=hit.id, rank=rank, score=hit.score),
                    hit.text,
                )
    ordered = sorted(
        fused.items(), key=lambda kv: (-kv[1][0], kv[0][0], kv[0][1])
    )
    return [
        RecallEntry(text=text, provenance=prov, tokens=estimate_tokens(text))
        for (_, _), (_score, prov, text) in ordered
    ]
