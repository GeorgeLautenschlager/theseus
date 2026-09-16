"""MemoryModule — the single memory boundary an agent core programs against.

The core sees two capabilities and nothing else: `recall(query, budget_tokens)`
and `consolidate(episode)`; applications can also repair derived embeddings.
Everything below this line — which layers exist,
how a query fans out, how results fuse, what gets written where — is internal
and may change without the core noticing. That is the leak contract: no layer
name appears in a public signature or in a result field the core must
*interpret to act*; provenance and cost data carry layer names as opaque
strings a caller may log but never branch on.

Retrieval is deterministic fan-out + reciprocal rank fusion (RRF): every layer
answers the query in its own way, each returns a ranked list, and RRF merges
the rankings without comparing raw scores across layers. Within-layer lexical
and vector hits share IDs; across layers this is a weighted rank interleave,
not evidence of semantic agreement. Misses are data — a layer that finds nothing contributes a
string to `RecallResult.misses`, never an exception.
"""

from __future__ import annotations

import json
import threading
import logging
from functools import wraps
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseus.assertion_metadata import ACTION_STATUSES, ATTRIBUTIONS
from theseus.intelligence_layer import IntelligenceLayer
from theseus.json_utils import parse_json_response
from theseus.knowledge_layer import KnowledgeLayer, KnowledgeRecord
from theseus.layer_store import (LayerHit, append_record, load_lines, atomic_json,
                                 fsync_directory, store_lock, terms, valid_vector)
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


logger = logging.getLogger(__name__)


def _serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock, store_lock(self.memory_dir):
            self._reload()
            self._recover_pending()
            return method(self, *args, **kwargs)
    return wrapped


def _provider_identity(provider) -> str:
    model = getattr(provider, "model", "")
    return type(provider).__module__ + "." + type(provider).__qualname__ + ":" + (model if isinstance(model, str) else "")


def _combine(*rankings: list[LayerHit], k: int) -> list[LayerHit]:
    scores = {}
    hits = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1 / (_RRF_K + rank)
            hits[hit.id] = hit
    return [hits[key] for key in sorted(scores, key=lambda key: -scores[key])[:k]]


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
        max_input_chars: int = 48000,
        max_output_tokens: int = 4096,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self._lock = threading.RLock()
        self._last_embedding_model = ""
        self._embedding_calls = 0
        self._embedding_tokens = 0
        if type(max_input_chars) is not int or max_input_chars < 4096:
            raise ValueError("max_input_chars must be at least 4096")
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        self._max_input_chars = max_input_chars
        self._max_output_tokens = max_output_tokens
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
        with self._lock, store_lock(self.memory_dir):
            self._reload()
            self._recover_pending()

    def _reload(self) -> None:
        # Every operation reloads under the process lock: another owner or a failed
        # append cannot leave the in-memory projection ahead of/behind durable state.
        self.knowledge = KnowledgeLayer(self.memory_dir / "knowledge.jsonl")
        self.memory = MemoryLayer(self.memory_dir / "memory.jsonl")
        self.wisdom = WisdomLayer(self.memory_dir / "wisdom.jsonl")
        ledger = [json.loads(line) for line in load_lines(self.memory_dir / "consolidation_ledger.jsonl")]
        self._processed_episodes = {row["episode_id"] for row in ledger}
        self._episode_ranges = {row["episode_id"]: (row.get("start_id"), row.get("end_id")) for row in ledger}
        self._embedding_index = {}
        for line in load_lines(self.memory_dir / "embeddings.jsonl"):
            row = json.loads(line)
            self._embedding_index[(row["model"], row["layer"], row["id"])] = row["vector"]

    def _vectors(self, layer: str) -> dict[str, list[float]]:
        return {rid: vector for (model, name, rid), vector in self._embedding_index.items()
                if model == self._last_embedding_model and name == layer}

    def _recover_pending(self) -> None:
        path = self.memory_dir / "pending.json"
        if not path.exists():
            return
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan["episode_id"] not in self._processed_episodes:
            for name, record_type in (("knowledge", KnowledgeRecord), ("memory", MemoryRecord), ("wisdom", WisdomRecord)):
                for line in plan["records"][name]:
                    getattr(self, name).add(record_type.from_json(line))
            existing = {json.loads(line)["assertion_id"] for line in load_lines(self.memory_dir / "dead_letter.jsonl")}
            for row in plan["dead_letters"]:
                if row["assertion_id"] not in existing:
                    append_record(self.memory_dir / "dead_letter.jsonl", json.dumps(row))
            append_record(self.memory_dir / "consolidation_ledger.jsonl", json.dumps({
                "episode_id": plan["episode_id"], "start_id": plan["start_id"], "end_id": plan["end_id"],
            }))
            self._processed_episodes.add(plan["episode_id"])
            self._episode_ranges[plan["episode_id"]] = (plan["start_id"], plan["end_id"])
        traced = {json.loads(line)["episode_id"] for line in load_lines(self.memory_dir / "traces" / "consolidation.jsonl")}
        if plan["episode_id"] not in traced:
            self._trace_consolidation(plan["trace"])
        path.unlink()
        fsync_directory(self.memory_dir)

    # -- recall ---------------------------------------------------------------

    @_serialized
    def recall(self, query: str, budget_tokens: int) -> RecallResult:
        """Fan `query` out to every layer, fuse the rankings (RRF), and fill
        `budget_tokens` of estimated window with the fused entries. Misses come
        back as data; nothing here raises for an empty or unavailable layer."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty text")
        if type(budget_tokens) is not int or budget_tokens < 0:
            raise ValueError("budget_tokens must be a nonnegative integer")
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
                embedding, k=self._per_layer_k, half_life_days=self._recency_half_life_days,
                embedding_model=self._last_embedding_model, embeddings=self._vectors("memory")
            )
            costs["memory"] = time.monotonic() - t0
            t0 = time.monotonic()
            per_layer["wisdom"] = self.wisdom.query(
                embedding, k=self._per_layer_k, embedding_model=self._last_embedding_model,
                embeddings=self._vectors("wisdom"))
            costs["wisdom"] = time.monotonic() - t0
        else:
            per_layer["memory"] = []
            per_layer["wisdom"] = []

        for name in ("memory", "wisdom"):
            t0 = time.monotonic()
            per_layer[name] = _combine(getattr(self, name).search(query, self._per_layer_k),
                                       per_layer[name], k=self._per_layer_k)
            costs[name] = costs.get(name, 0.0) + time.monotonic() - t0
        t0 = time.monotonic()
        per_layer["intelligence"] = self.intelligence.search(query, self._per_layer_k)
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
        try:
            self._trace_recall(result, per_layer, [(e, pr) for e, pr in entries])
        except OSError:
            logger.exception("Could not persist recall trace")
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
        return self.knowledge.search(terms(query), k=self._per_layer_k)

    def _embed(self, text: str) -> list[float] | None:
        for provider in self._embedding_providers:
            try:
                if not getattr(provider, "is_available", lambda: True)():
                    continue
                self._embedding_calls += 1
                vector = list(provider.embed(text))
                usage = getattr(provider, "last_embedding_usage", None)
                if type(usage) is int:
                    self._embedding_tokens += usage
                if not valid_vector(vector):
                    continue
                self._last_embedding_model = _provider_identity(provider)
                return vector
            except Exception:
                continue
        self._last_embedding_model = ""
        return None

    @_serialized
    def repair_embeddings(self, limit: int = 20) -> int:
        """Rebuild missing/legacy/incompatible vectors without changing evidence.

        A bounded, explicit maintenance operation; recall itself never embeds records.
        """
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        probe = self._embed("memory index")
        if probe is None:
            return 0
        model = self._last_embedding_model
        repaired = 0
        for name in ("memory", "wisdom"):
            for record in getattr(self, name).read_all():
                key = (model, name, record.id)
                vector = self._embedding_index.get(key)
                if valid_vector(vector, len(probe)):
                    continue
                if record.embedding_model == model and valid_vector(record.embedding, len(probe)):
                    continue
                text = record.summary if name == "memory" else record.statement
                vector = self._embed(text)
                if vector is None or self._last_embedding_model != model or not valid_vector(vector, len(probe)):
                    return repaired
                append_record(self.memory_dir / "embeddings.jsonl", json.dumps({
                    "id": record.id, "layer": name, "model": model, "vector": vector,
                }))
                self._embedding_index[key] = vector
                repaired += 1
                if repaired >= limit:
                    return repaired
        return repaired

    # -- consolidation ---------------------------------------------------------

    @_serialized
    def consolidate(self, episode: Episode) -> ConsolidationResult:
        """Prepare then replay a durable episode transaction, idempotently."""
        started = time.monotonic()
        if not isinstance(episode.episode_id, str) or not episode.episode_id.strip():
            raise ValueError("episode_id must be nonempty text")
        if episode.episode_id in self._processed_episodes:
            previous = self._episode_ranges[episode.episode_id]
            if previous != (None, None) and previous != (episode.start_id, episode.end_id):
                raise ValueError("episode_id already belongs to a different event range")
            return ConsolidationResult(episode.episode_id, 0, {}, 0, 0, 0, 0, 0.0, skipped=True)
        events = self._episode_events(episode)
        evidence = [e for e in events if not _is_recall_flagged(e)]
        context_only = [e for e in events if _is_recall_flagged(e)]
        eligible_events = {e.id: e for e in evidence}
        context_only_ids = {e.id for e in context_only}
        if not evidence:
            return ConsolidationResult(episode.episode_id, 0, {}, 0, 0, 0, 0,
                                       time.monotonic() - started)
        overhead = len(build_extraction_prompt("", "context"))
        per_event = (self._max_input_chars - overhead - len(events)) // len(events)
        truncated = 0

        def render(event):
            nonlocal truncated
            line = event.to_json()
            if len(line) <= per_event:
                return line
            truncated += 1
            # Preserve the original in the episode record. Extraction sees an
            # explicitly marked excerpt; the full evidence remains searchable.
            content = json.dumps(event.content, ensure_ascii=False)
            size = max(0, per_event - len(replace(event, content={}).to_json()) - 100)
            while True:
                line = replace(event, content={"excerpt": content[:size], "truncated": True}).to_json()
                if len(line) <= per_event:
                    return line
                if size == 0:
                    raise ValueError("episode has too many events for the extraction budget")
                size //= 2

        prompt = build_extraction_prompt("\n".join(render(e) for e in evidence),
                                         "\n".join(render(e) for e in context_only))
        tokens_in = tokens_out = calls = 0
        reported_usage = []
        parsed = None
        for provider in self._model_providers:
            try:
                if not getattr(provider, "is_available", lambda: True)():
                    continue
                calls += 1
                tokens_in += estimate_tokens(prompt)
                raw = provider.chat(prompt, json_schema=extraction_json_schema(),
                                    max_tokens=self._max_output_tokens)
                usage = getattr(provider, "last_chat_usage", None)
                if isinstance(usage, dict):
                    reported_usage.append({"provider": _provider_identity(provider), **usage})
                tokens_out += estimate_tokens(raw)
                candidate_response = parse_json_response(raw)
                if (not isinstance(candidate_response, dict)
                    or not isinstance(candidate_response.get("summary"), str)
                    or not candidate_response["summary"].strip()
                    or not isinstance(candidate_response.get("assertions"), list)):
                    raise ValueError("extraction requires a nonempty summary and an assertions list")
                parsed = candidate_response
                break
            except Exception as exc:
                append_record(self.memory_dir / "extraction_failures.jsonl", json.dumps({
                    "episode_id": episode.episode_id, "provider": _provider_identity(provider),
                    "error": str(exc), "tokens_in_estimated": tokens_in,
                    "tokens_out_estimated": tokens_out,
                }))
        if parsed is None:
            raise RuntimeError("no model provider produced a valid extraction; episode remains pending")
        records = {name: [] for name in ("knowledge", "memory", "wisdom")}
        dead = []
        routed = {}
        supersessions = 0
        before_embeddings = self._embedding_calls
        before_embedding_tokens = self._embedding_tokens
        ts = max(e.ts for e in evidence)
        for candidate in parsed["assertions"]:
            rid = new_id()
            reason = _validate_assertion(candidate, eligible_events, context_only_ids,
                                         lenient=self._lenient_fact_routing)
            if reason:
                dead.append({"episode_id": episode.episode_id, "assertion_id": rid,
                             "candidate": candidate, "reason": reason})
                continue
            layer = _route_write(candidate)
            metadata = {
                "support_event_ids": tuple(candidate["support_event_ids"]),
                "attribution": candidate["attribution"],
                "reported_by": candidate.get("reported_by"),
                "action_status": candidate["action_status"],
            }
            if layer == "knowledge":
                record = KnowledgeRecord(rid, ts, candidate["subject"].strip(),
                                         candidate["predicate"].strip(), candidate["value"].strip(),
                                         source_episode_id=episode.episode_id, **metadata)
                current = self.knowledge.current(record.subject, record.predicate)
                supersessions += int(bool(current) and ts >= current[0].ts)
            else:
                vector = self._embed(candidate["statement"]) or []
                if layer == "wisdom":
                    record = WisdomRecord(rid, ts, candidate["statement"].strip(), vector,
                                          source_episode_id=episode.episode_id,
                                          embedding_model=self._last_embedding_model, **metadata)
                else:
                    record = MemoryRecord(rid, ts, candidate["statement"].strip(),
                                          candidate["statement"].strip(), vector,
                                          source_episode_id=episode.episode_id,
                                          embedding_model=self._last_embedding_model, **metadata)
            records[layer].append(record.to_json())
            routed[layer] = routed.get(layer, 0) + 1
        vector = self._embed(parsed["summary"]) or []
        record = MemoryRecord(new_id(), ts, "\n".join(e.to_json() for e in evidence),
                              parsed["summary"], vector, source_episode_id=episode.episode_id,
                              embedding_model=self._last_embedding_model,
                              support_event_ids=tuple(e.id for e in evidence))
        records["memory"].append(record.to_json())
        routed["memory"] = routed.get("memory", 0) + 1
        elapsed = time.monotonic() - started
        trace = {"episode_id": episode.episode_id, "candidates_extracted": len(parsed["assertions"]),
                 "routed": routed, "supersessions": supersessions, "schema_failures": len(dead),
                 "tokens_in": tokens_in, "tokens_out": tokens_out, "wall_time_s": elapsed,
                 "token_counts_estimated": True, "chat_calls": calls,
                 "embedding_calls": self._embedding_calls - before_embeddings}
        trace["truncated_events"] = truncated
        trace["reported_chat_usage"] = reported_usage
        trace["reported_embedding_tokens"] = self._embedding_tokens - before_embedding_tokens
        atomic_json(self.memory_dir / "pending.json", {
            "episode_id": episode.episode_id, "start_id": episode.start_id, "end_id": episode.end_id,
            "records": records, "dead_letters": dead, "trace": trace,
        })
        self._recover_pending()
        return ConsolidationResult(episode.episode_id, len(parsed["assertions"]), routed,
                                   supersessions, len(dead), tokens_in, tokens_out,
                                   time.monotonic() - started)

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


def _validate_assertion(candidate: Any, eligible_events: dict[str, Any],
                        context_only_ids: set[str], lenient: bool = False) -> str | None:
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
    support_ids = candidate.get("support_event_ids")
    if (not isinstance(support_ids, list) or not support_ids
        or any(not isinstance(event_id, str) or not event_id.strip() for event_id in support_ids)
        or len(set(support_ids)) != len(support_ids)):
        return "missing or invalid support_event_ids"
    for event_id in support_ids:
        if event_id in context_only_ids:
            return f"context-only support event {event_id}"
        if event_id not in eligible_events:
            return f"unknown support event {event_id}"
    attribution = candidate.get("attribution")
    if not isinstance(attribution, str) or attribution not in ATTRIBUTIONS:
        return "missing or invalid attribution"
    reported_by = candidate.get("reported_by")
    if attribution == "partner_report":
        if not isinstance(reported_by, str) or not reported_by.strip():
            return "partner report missing reported_by"
        if not any(eligible_events[event_id].actor.casefold() == reported_by.strip().casefold()
                   for event_id in support_ids):
            return "reported_by is not a supporting event actor"
    elif reported_by is not None:
        return "reported_by requires partner_report attribution"
    status = candidate.get("action_status")
    if not isinstance(status, str) or status not in ACTION_STATUSES:
        return "missing or invalid action_status"
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
            weight = {"knowledge": 1.2, "intelligence": 0.5}.get(layer, 1.0)
            contribution = weight / (_RRF_K + rank)
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
