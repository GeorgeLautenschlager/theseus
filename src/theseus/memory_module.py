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

from theseus.assertion_metadata import ACTION_STATUSES, ATTRIBUTIONS, RECONCILIATION_DECISIONS
from theseus.intelligence_layer import IntelligenceLayer
from theseus.json_utils import parse_json_response
from theseus.knowledge_layer import KnowledgeLayer, KnowledgeRecord
from theseus.layer_store import (LayerHit, append_record, load_lines, atomic_json,
                                 fsync_directory, store_lock, terms, valid_vector)
from theseus.memory_layer import MemoryLayer, MemoryRecord
from theseus.memory_prompts import (build_extraction_prompt, build_reconciliation_prompt,
                                    extraction_json_schema, reconciliation_json_schema)
from theseus.stimulus_log import StimulusLog, new_id
from theseus.tools.recall import RECALL_TOOL_NAME
from theseus.wisdom_layer import WisdomLayer, WisdomRecord

# RRF constant: 60 is the standard choice; it flattens score-scale differences
# between layers so rank order, not raw scores, drives fusion.
_RRF_K = 60

# Chars of overlap between consecutive chunks of one oversized event, so a
# sentence straddling a chunk boundary isn't lost from both sides' context.
# The resulting repeated text is why extraction runs a dedup pass afterward.
_CHUNK_OVERLAP_CHARS = 200


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
        reconciliation_context_k: int = 6,
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
        if type(reconciliation_context_k) is not int or reconciliation_context_k < 1:
            raise ValueError("reconciliation_context_k must be a positive integer")
        self._max_input_chars = max_input_chars
        self._max_output_tokens = max_output_tokens
        self._reconciliation_context_k = reconciliation_context_k
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

        context_text = "\n".join(e.to_json() for e in context_only)
        # Real overhead: the prompt shape plus whatever context this episode actually
        # carries, not a fixed guess — a large recalled-context block eats real budget.
        overhead = len(build_extraction_prompt("", context_text))
        solo_budget = self._max_input_chars - overhead
        if solo_budget <= 0:
            raise ValueError("episode has too many events for the extraction budget")

        units, oversized_events = _pack_units(evidence, solo_budget)

        tokens_in = tokens_out = calls = 0
        reported_usage = []
        parsed_units = []
        unit_eligible_ids = []
        for unit_text, unit_meta in units:
            unit_prompt = build_extraction_prompt(unit_text, context_text)
            parsed = None
            for provider in self._model_providers:
                try:
                    if not getattr(provider, "is_available", lambda: True)():
                        continue
                    calls += 1
                    tokens_in += estimate_tokens(unit_prompt)
                    raw = provider.chat(unit_prompt, json_schema=extraction_json_schema(),
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
                        "tokens_out_estimated": tokens_out, **unit_meta,
                    }))
            if parsed is None:
                # Nothing durable has been written yet (pending.json is built only after
                # every unit succeeds), so a failed chunk leaves the whole episode
                # pending, exactly like a single-request failure always has.
                raise RuntimeError("no model provider produced a valid extraction; episode remains pending")
            parsed_units.append(parsed)
            unit_eligible_ids.append(set(unit_meta["event_ids"]))

        # Multiple requests over one episode is a v0 side effect of chunking, not a
        # reason to change what the episode record looks like: one merged summary
        # covering every request, one dead-letter/accept decision per raw candidate
        # (using only the support ids its own request actually saw evidence for — a
        # chunk of one oversized event cannot back a claim with some other event's
        # id just because that other event happens to fall inside the same episode),
        # and only then a dedup pass over what was accepted. Validating before
        # deduping matters: the same claim can be genuinely out of scope in one
        # request (e.g. the shared pack, which never saw the oversized event) and
        # legitimately supported in another (that event's own chunks) — deduping
        # first could let the invalid occurrence claim the slot and silently drop
        # the valid one.
        summary = " ".join(p["summary"].strip() for p in parsed_units if p["summary"].strip())
        records = {name: [] for name in ("knowledge", "memory", "wisdom")}
        dead = []
        routed = {}
        supersessions = 0
        before_embeddings = self._embedding_calls
        before_embedding_tokens = self._embedding_tokens
        ts = max(e.ts for e in evidence)
        accepted = []
        for i, parsed in enumerate(parsed_units):
            request_eligible_events = {eid: eligible_events[eid] for eid in unit_eligible_ids[i]}
            for candidate in parsed["assertions"]:
                rid = new_id()
                reason = _validate_assertion(candidate, request_eligible_events, context_only_ids,
                                             lenient=self._lenient_fact_routing)
                if reason:
                    dead.append({"episode_id": episode.episode_id, "assertion_id": rid,
                                 "candidate": candidate, "reason": reason})
                else:
                    accepted.append((rid, candidate))
        assertions, duplicates = _dedupe_assertions(accepted)

        fact_indices = [i for i, (_, c) in enumerate(assertions) if _route_write(c) == "knowledge"]
        reconciliation_calls, decisions, reconciliation_usage = self._reconcile_facts(
            episode.episode_id, assertions, fact_indices)
        reported_usage.extend(reconciliation_usage)
        decision_counts: dict[str, int] = {}
        local_supersessions: dict[str, str] = {}  # id this episode already superseded -> its replacement

        for i, (rid, candidate) in enumerate(assertions):
            layer = _route_write(candidate)
            metadata = {
                "support_event_ids": tuple(candidate["support_event_ids"]),
                "attribution": candidate["attribution"],
                "reported_by": candidate.get("reported_by"),
                "action_status": candidate["action_status"],
            }
            if layer == "knowledge":
                subject = candidate["subject"].strip()
                predicate = candidate["predicate"].strip()
                value = candidate["value"].strip()
                decision, target_id = decisions.get(i, (None, None))
                if decision is None:
                    # No reconciliation opinion for this candidate (no existing
                    # knowledge for its subject at all, or the call failed/skipped
                    # it) — fall back to the pre-reconciliation default: replace an
                    # exact subject+predicate match, otherwise write it as new.
                    same_key = self.knowledge.current(subject, predicate)
                    decision, target_id = ("replace", same_key[0].id) if same_key else ("new", None)
                elif decision == "reinforce":
                    # A guard, not a trust exercise: if the "reinforced" record's
                    # value actually differs, this is a real change reconciliation
                    # mislabeled — treat it as a replacement so it isn't lost.
                    target_record = self.knowledge.get(target_id)
                    if target_record is not None and target_record.value.strip().casefold() != value.casefold():
                        decision = "replace"
                if target_id is not None:
                    target_id = local_supersessions.get(target_id, target_id)
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
                supersedes = target_id if decision in ("reinforce", "replace") else None
                contradicts = (target_id,) if decision == "contradiction" and target_id else None
                record = KnowledgeRecord(rid, ts, subject, predicate, value,
                                         source_episode_id=episode.episode_id,
                                         supersedes=supersedes, reconciliation=decision,
                                         contradicts=contradicts, **metadata)
                if supersedes is not None:
                    local_supersessions[supersedes] = rid
                supersessions += int(decision == "replace")
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
        vector = self._embed(summary) or []
        record = MemoryRecord(new_id(), ts, "\n".join(e.to_json() for e in evidence),
                              summary, vector, source_episode_id=episode.episode_id,
                              embedding_model=self._last_embedding_model,
                              support_event_ids=tuple(e.id for e in evidence))
        records["memory"].append(record.to_json())
        routed["memory"] = routed.get("memory", 0) + 1
        elapsed = time.monotonic() - started
        extracted = len(dead) + len(assertions)
        trace = {"episode_id": episode.episode_id, "candidates_extracted": extracted,
                 "routed": routed, "supersessions": supersessions, "schema_failures": len(dead),
                 "tokens_in": tokens_in, "tokens_out": tokens_out, "wall_time_s": elapsed,
                 "token_counts_estimated": True, "chat_calls": calls + reconciliation_calls,
                 "embedding_calls": self._embedding_calls - before_embeddings}
        trace["oversized_events"] = len(oversized_events)
        trace["extraction_requests"] = len(units)
        trace["duplicate_assertions_dropped"] = duplicates
        trace["reported_chat_usage"] = reported_usage
        trace["reported_embedding_tokens"] = self._embedding_tokens - before_embedding_tokens
        trace["reconciliation_calls"] = reconciliation_calls
        trace["reconciliation_decisions"] = decision_counts
        atomic_json(self.memory_dir / "pending.json", {
            "episode_id": episode.episode_id, "start_id": episode.start_id, "end_id": episode.end_id,
            "records": records, "dead_letters": dead, "trace": trace,
        })
        self._recover_pending()
        return ConsolidationResult(episode.episode_id, extracted, routed,
                                   supersessions, len(dead), tokens_in, tokens_out,
                                   time.monotonic() - started)

    def _reconcile_facts(
        self, episode_id: str, assertions: list[tuple[str, Any]], fact_indices: list[int],
    ) -> tuple[int, dict[int, tuple[str, str | None]], list[dict]]:
        """Decide how each fact-kind candidate relates to existing current
        knowledge for its subject — reinforcing it, replacing it, coexisting
        beside it, contradicting it, or filed as historical relative to it —
        instead of the blunt "same subject+predicate key, newest timestamp
        wins" rule that can't tell an alternate-worded correction from an
        unrelated fact, or a stale report from a current one.

        Bounded and cheap by construction: a subject with no existing current
        knowledge has nothing to reconcile against (result is always "new"),
        so it costs no call at all; every subject that does gets at most
        `_reconciliation_context_k` of its current records as context. One
        request covers every fact candidate in the episode that has any
        context to reconcile against.

        Never blocks the episode: a failed, unavailable, or malformed
        reconciliation response leaves `decisions` empty for the candidates it
        should have covered, and the caller's own fallback (replace an exact
        subject+predicate match, otherwise write new) takes over — the same
        outcome consolidation always had before this step existed. Recorded
        to `reconciliation_failures.jsonl` for visibility, same as extraction.
        """
        context: dict[int, dict[str, KnowledgeRecord]] = {}
        targets: list[int] = []
        for i in fact_indices:
            candidate = assertions[i][1]
            existing = self.knowledge.current(subject=candidate["subject"])[: self._reconciliation_context_k]
            if existing:
                context[i] = {record.id: record for record in existing}
                targets.append(i)

        decisions: dict[int, tuple[str, str | None]] = {}
        reported_usage: list[dict] = []
        if not targets:
            return 0, decisions, reported_usage

        candidate_payload = [assertions[i][1] for i in targets]
        existing_union = {rid: record for i in targets for rid, record in context[i].items()}
        prompt = build_reconciliation_prompt(candidate_payload, list(existing_union.values()))
        calls = 0
        parsed = None
        for provider in self._model_providers:
            try:
                if not getattr(provider, "is_available", lambda: True)():
                    continue
                calls += 1
                raw = provider.chat(prompt, json_schema=reconciliation_json_schema(),
                                    max_tokens=self._max_output_tokens)
                usage = getattr(provider, "last_chat_usage", None)
                if isinstance(usage, dict):
                    reported_usage.append({"provider": _provider_identity(provider), **usage})
                candidate_response = parse_json_response(raw)
                if (not isinstance(candidate_response, dict)
                    or not isinstance(candidate_response.get("decisions"), list)):
                    raise ValueError("reconciliation requires a decisions list")
                parsed = candidate_response
                break
            except Exception as exc:
                append_record(self.memory_dir / "reconciliation_failures.jsonl", json.dumps({
                    "episode_id": episode_id, "provider": _provider_identity(provider), "error": str(exc),
                }))
        if parsed is None:
            return calls, decisions, reported_usage

        for row in parsed["decisions"]:
            if not isinstance(row, dict):
                continue
            position = row.get("candidate_index")
            if type(position) is not int or not (0 <= position < len(targets)):
                continue
            decision = row.get("decision")
            if decision not in RECONCILIATION_DECISIONS:
                continue
            i = targets[position]
            target_id = row.get("target_id")
            if decision in ("new", "coexist", "historical"):
                # target_id is informational at most for these — "historical" never
                # supersedes or contradicts anything, it just never becomes current.
                target_id = None
            elif not isinstance(target_id, str) or target_id not in context[i]:
                continue  # a decision needing a real target that doesn't resolve is dropped, falls back
            decisions[i] = (decision, target_id)
        return calls, decisions, reported_usage

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


# -- extraction packing -----------------------------------------------------------

def _water_fill(sizes: list[int], budget: int) -> list[int]:
    """Per-item share of `budget`, redistributing what small items don't need
    onto the items still competing for room — instead of handing every item
    the same fixed slice regardless of whether it needed it.

    Classic water-filling: take the smallest remaining item; if it fits inside
    an equal split of what is left, it keeps its own (smaller) size and drops
    out, growing the share for what remains. The moment the smallest
    remaining item no longer fits an equal split, nothing larger will either
    (the list is sorted ascending), so everything still in play gets that
    same equal share and allocation stops.
    """
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    allocation = [0] * len(sizes)
    remaining_budget = max(0, budget)
    remaining = list(order)
    while remaining:
        share = remaining_budget // len(remaining)
        smallest = remaining[0]
        if sizes[smallest] <= share:
            allocation[smallest] = sizes[smallest]
            remaining_budget -= sizes[smallest]
            remaining.pop(0)
        else:
            for i in remaining:
                allocation[i] = share
            break
    return allocation


def _chunk_event(event, budget: int, overlap: int = _CHUNK_OVERLAP_CHARS) -> list:
    """Split one event's content into ordered, bounded pieces that together
    cover it completely, each still carrying the event's own id.

    This is what keeps decisive evidence near the end of a long event (a
    success/failure marker, a final total) reachable by extraction: nothing
    is cut away the way a prefix truncation would — the tail simply lands in
    a later chunk instead of an earlier one.
    """
    content_str = json.dumps(event.content, ensure_ascii=False)
    skeleton = len(replace(event, content={}).to_json())
    size = max(0, budget - skeleton - 64)
    while size > 0:
        probe = replace(event, content={
            "chunk_of": event.id, "chunk_index": 0, "chunk_count": 1,
            "excerpt": content_str[:size],
        }).to_json()
        if len(probe) <= budget:
            break
        size //= 2
    if size <= 0:
        raise ValueError("episode has too many events for the extraction budget")
    pieces = []
    step = max(1, size - overlap)
    start = 0
    while True:
        pieces.append(content_str[start:start + size])
        if start + size >= len(content_str):
            break
        start += step
    total = len(pieces)
    return [
        replace(event, content={
            "chunk_of": event.id, "chunk_index": i, "chunk_count": total, "excerpt": piece,
        })
        for i, piece in enumerate(pieces)
    ]


def _pack_units(evidence: list, solo_budget: int) -> tuple[list[tuple[str, dict]], list]:
    """Complete events that fit a shared request, packed into one; anything
    left over gets its own bounded request(s).

    Two passes: first try every event at full size — if the episode simply
    fits, nothing is split at all. Otherwise, water-fill the budget: events
    small enough to be satisfied stay whole and share one request; the rest
    (individually too large for what redistribution leaves them) are pulled
    out entirely and given a dedicated request each, at the full per-request
    budget rather than a cramped leftover share — chunked further only if
    even that isn't enough.
    """
    full_lines = [e.to_json() for e in evidence]
    sizes = [len(line) for line in full_lines]
    separators = max(0, len(evidence) - 1)
    pack_budget = max(0, solo_budget - separators)

    if sum(sizes) <= pack_budget:
        packed_events, oversized_events = evidence, []
    else:
        allocation = _water_fill(sizes, pack_budget)
        packed_events = [e for e, sz, alloc in zip(evidence, sizes, allocation) if sz <= alloc]
        oversized_events = [e for e, sz, alloc in zip(evidence, sizes, allocation) if sz > alloc]

    units: list[tuple[str, dict]] = []
    if packed_events:
        units.append((
            "\n".join(e.to_json() for e in packed_events),
            {"unit": "pack", "event_ids": [e.id for e in packed_events]},
        ))
    for event in oversized_events:
        chunks = [event] if len(event.to_json()) <= solo_budget else _chunk_event(event, solo_budget)
        for chunk in chunks:
            units.append((chunk.to_json(), {"unit": "chunk", "event_ids": [event.id]}))
    if not units:
        raise ValueError("episode has too many events for the extraction budget")
    return units, oversized_events


def _assertion_key(candidate: Any) -> tuple | None:
    """Identity used to dedupe already-*validated* assertions across extraction
    requests. Facts key on their triple (case/whitespace-insensitive) so the
    same fact restated with a differently worded `statement` still merges;
    everything else keys on kind + exact statement text. `support_event_ids`
    (order-insensitive) is always part of the key: two candidates with the
    same content but different supporting evidence are different claims, not
    a chunking repeat, and must never be collapsed into one."""
    if not isinstance(candidate, dict):
        return None
    support = candidate.get("support_event_ids")
    support_key = tuple(sorted(support)) if isinstance(support, list) else None
    kind = candidate.get("kind")
    if kind == "fact" and _has_triple(candidate):
        return (support_key, "fact", candidate["subject"].strip().casefold(),
                candidate["predicate"].strip().casefold(), candidate["value"].strip().casefold())
    statement = candidate.get("statement")
    if isinstance(statement, str) and statement.strip():
        return (support_key, kind, statement.strip().casefold())
    return None


def _dedupe_assertions(candidates: list[tuple[str, Any]]) -> tuple[list[tuple[str, Any]], int]:
    """Drop exact repeats among already-validated `(assertion_id, candidate)`
    pairs — the predictable side effect of the same evidence reaching more
    than one extraction request (overlapping chunk context, or one oversized
    event's chunks each restating the same fact). First occurrence wins."""
    seen: set = set()
    kept = []
    dropped = 0
    for rid, candidate in candidates:
        key = _assertion_key(candidate)
        if key is not None:
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
        kept.append((rid, candidate))
    return kept, dropped


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
