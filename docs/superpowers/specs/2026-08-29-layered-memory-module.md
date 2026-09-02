# BRIEF: Layered Memory Module

**Project:** Theseus Agent Construction Kit
**Status:** Draft for review
**Supersedes:** A-MEM as sole memory system (A-MEM remains as an evaluation baseline)

---

## Problem

Theseus currently uses A-MEM as its only memory system. It works, but it applies one set of persistence semantics to every kind of knowledge the agent accumulates. A fact that has been superseded, an episode that happened once, and a behavioural heuristic that generalises across episodes are all stored and aged identically.

"The Missing Knowledge Layer in Cognitive Architectures for AI Agents" (arXiv 2604.11364) argues that the semantic/episodic distinction in CoALA is named but never operationalised, and re-keys the layers on **persistence semantics** rather than content type:

| Layer | Persistence semantics |
|---|---|
| Knowledge | Supersession. Facts are replaced, not decayed. |
| Memory | Bi-temporal, append-only. Decay applied at query time, never at storage time. |
| Wisdom | Evidence-gated revision. Changes only on corroboration. |
| Intelligence | Ephemeral. Session-scoped, no persistence. |

We want the smallest system that covers all four, built so that a single layer can be rewritten later without touching the others, and measured by a standing evaluation harness from day one.

## Goals

1. Cover all four layers with genuinely distinct persistence semantics.
2. Present a single module boundary to the CognitiveCore. All routing internal.
3. Optimise for observability: plaintext stores, full retrieval and consolidation tracing.
4. Optimise for evaluability: a standing harness so per-layer iteration produces measurable deltas.
5. Be backbone-robust. Append-only stores degrade gracefully on weak local models; graph and reflective architectures do not.

## Non-goals (v1)

- Wisdom promotion, revision, or gating logic. Wisdom v0 is deliberately a stub.
- Any vector database. Brute-force cosine in numpy is sufficient at expected scale.
- Autonomous memory-system evolution. Traces are collected; analysis and extension are out of scope.
- Replacing or modifying the StimulusLog or Segmenter.
- Cross-layer reconciliation of duplicate assertions. Identity is preserved so this is possible later; it is not implemented now.
- Shared or multi-agent memory.

---

## Architecture

### Module boundary

A single `MemoryModule` satisfies the memory port on the OODA membrane. Routing, fusion, and layer selection are internal implementation details.

Public surface, approximately:

```
recall(query, budget) -> RecallResult
consolidate(episode) -> ConsolidationResult
```

**Leak test:** if a layer name can appear in any public signature, argument, or return key that the CognitiveCore must interpret, the boundary has leaked. The core may read provenance for logging; it must never need to understand it in order to act.

`RecallResult` carries:
- fused entries, ranked
- per-entry provenance: originating layer, record id, pre-fusion rank and score
- **misses**: sub-queries that returned nothing, as first-class data
- cost metrics: per-layer and total latency

Misses are load-bearing. Only the module knows a lookup came back empty, and the StimulusLog recall-event design requires failed attempts to be reported so Orient can write them as flagged events.

### Division of labour with Orient

- **Orient** decides *what is worth asking* given goals, current stimulus, and task. This is cognition and stays in the core.
- **MemoryModule** decides *where a question gets answered and how results fuse*. This is memory and lives behind the port.

The `InterrogativeContextAssembler` splits along this seam: question formulation in Orient, question routing inside the module.

### Stores

Three persisted JSONL files. Intelligence is a read strategy, not a file.

#### 1. Knowledge — `knowledge.jsonl`

Append-only log with a read-time projection. Supersession, no decay.

Record fields: `assertion_id`, `subject`, `predicate`, `object` (or freeform statement), `supersedes` (assertion_id or null), `valid_from`, `source_episode_id`, `confidence`.

- Projection built at load: an in-memory index where the latest non-superseded record wins. Rebuild is a fold over the log.
- Retrieval: entity/predicate lookup against the projection. No embeddings on the primary path; optional cosine fallback for unmatched queries.
- **Supersession is always an explicit, logged operation.** The superseding record names the id it replaces. The extractor never silently overwrites.
- The superseded record stays in the log. It leaves *retrieval*, not *history*.

Add log snapshotting only if projection rebuild exceeds the load-time budget. Do not build it speculatively.

#### 2. Memory — `memory.jsonl`

Append-only, bi-temporal, never mutated.

Record fields: `episode_id`, `summary`, `occurred_at` (event time), `recorded_at` (ingest time), `participants`, `embedding`, `stimulus_range`, `assertion_ids`.

- Retrieval: brute-force cosine over the embedding array in numpy, with recency weighting applied **at query time**. Decay is a ranking function, not a storage mutation.
- Records are never edited. A correction is a new record; the old one remains and simply ranks lower or is superseded at the Knowledge layer.

#### 3. Wisdom — `wisdom.jsonl`

Present, retrievable, and deliberately embarrassing.

Record fields: `heuristic_id`, `statement`, `evidence_refs[]`, `evidence_count`, `first_seen`, `last_seen`, `embedding`.

- v0 retrieval: cosine similarity filtered by `evidence_count >= threshold`.
- v0 has **no** promotion logic, no revision, no evidence gating beyond an integer counter.
- It exists so the other layers have somewhere to route, and so there is a baseline worth beating when it is built properly.

#### 4. Intelligence — no file

Served as a read strategy over the tail of the StimulusLog, which already includes recall events. Session-scoped and ephemeral by construction. Creating a fourth file would duplicate a store we already have.

### Shared assertion identity

Extraction assigns an `assertion_id` **before** routing. An assertion that is simultaneously verifiable and behavioural lands in both Knowledge and Wisdom, by design, sharing that id.

v1 does not reconcile these. It only guarantees they are traceable. Retrofitting identity later is significantly worse than assigning it now.

### Read routing

Internal to the module, two-stage:

1. **Deterministic first.** Cheap classification on query shape produces a candidate layer set. Default behaviour on ambiguity is to fan out to all layers.
2. **Fusion.** Reciprocal rank fusion across whichever layers were queried.

An LLM classification path may exist as an escalation, cached by query shape, and is **off by default in v1**.

Rationale: in the source paper, typed routing beat a flat store by roughly 13 points with an oracle classifier and lost by roughly 13 points with a keyword heuristic. Fan-out plus fusion fails soft; a router that picks one store fails hard. Latency also matters — putting a model call in front of every Orient is the failure mode that makes some published systems unusable interactively.

---

## Offline consolidation

Runs out of band, triggered by a Segmenter episode boundary. Never on the interactive path. Mutation-time LLM calls are affordable here; recall-time ones are not.

### Pipeline

1. Read the episode's stimulus range from the StimulusLog.
2. **Filter recall-flagged events out of the evidence set.** They may be read for context; they may never be a provenance source. Without this, retrieval frequency silently becomes evidence weight and the memory becomes a record of the agent's own retrieval habits.
3. Extract candidate assertions (LLM).
4. Assign `assertion_id` to each candidate.
5. Route each candidate to one or more layers.
6. For Knowledge candidates, run an explicit supersession check against the projection. Any supersession is logged with the id it replaces.
7. Append to the target stores.

### Write-routing rules

Start deterministic; escalate to the LLM only for unresolved candidates.

| Signal | Layer |
|---|---|
| Verifiable, atemporal, entity–attribute shaped | Knowledge |
| Event-bound, has an `occurred_at`, participant-scoped | Memory |
| Behavioural or directive, generalises across episodes | Wisdom |
| Both factual and directive | Knowledge **and** Wisdom, shared `assertion_id` |

### Guards

- **Idempotency.** Re-consolidating an episode must not duplicate. Key on `(episode_id, assertion_id)`.
- **Failure isolation.** One episode's extraction failure must not block the queue. Dead-letter and continue.
- **Schema validation.** Local backbones produce malformed structured output at non-trivial rates, and the failure is silent — the agent converses normally while its writes corrupt. Validate every extracted record against schema, dead-letter on failure, and **count it as a first-class metric**.
- **Cost budget.** Log tokens and wall-clock time per episode.

### Explicitly out of scope for v1

Cross-episode reflection, Wisdom evidence accumulation beyond an integer counter, memory merging or rewriting, and any automatic promotion between layers.

---

## Instrumentation

This is not a nice-to-have. It is the phase-0 data that the eventual memory-evolution loop trains on, and neither of us currently knows the real query distribution.

**Per recall**, one JSONL trace record: original query, formulated sub-queries, layers queried, per-layer hit counts, pre- and post-fusion ranks, returned set, misses, per-layer latency, total latency, and whether the caller re-queried within N turns.

**Per consolidation**, one JSONL trace record: `episode_id`, candidates extracted, per-layer routing counts, supersessions performed, schema failures, tokens consumed, wall-clock time.

Re-query and downstream-correction signals are the cheapest available proxies for retrieval success. Both are already derivable from the StimulusLog and Segmenter.

---

## Evaluation harness

- **Golden query set built from real Theseus traces**, not LoCoMo. A community audit found LoCoMo's answer key materially wrong and its judge accepting a majority of intentionally incorrect answers. Target roughly 50 queries to start; grow from traces.
- **Report cost alongside accuracy, always.** Retrieval accuracy, retrieval latency p50/p95, consolidation cost per episode, schema failure rate, store sizes. The field's inability to distinguish a 15-hour system from a 60ms one comes directly from not doing this.
- **LLM judges rank, they do not threshold.** Judge orderings are stable across rubrics; absolute scores move with prompt strictness. Use the judge to compare Wisdom v1 against Wisdom v2. Never set a promotion threshold from a judge score.
- **Two baselines.** Current A-MEM, and a flat "append everything to one file, retrieve by cosine" control. If the layered system does not beat the flat control, that is a legitimate finding and should be reported as one.

---

## Acceptance criteria

- [ ] All four layers reachable through one module interface, with no layer name in any public signature the core must interpret.
- [ ] Knowledge supersession removes the superseded value from retrieval while its record remains in the log.
- [ ] Memory records are never mutated after write.
- [ ] Wisdom v0 exists, is retrievable, and contains no promotion logic.
- [ ] Intelligence is served from the StimulusLog tail; no fourth store file exists.
- [ ] `recall` returns misses explicitly, and Orient writes both hits and misses to the StimulusLog as recall-flagged events.
- [ ] Consolidation is idempotent per `(episode_id, assertion_id)`.
- [ ] Recall-flagged stimuli are excluded from the consolidation evidence set.
- [ ] Schema-invalid extractions are dead-lettered and counted, not dropped silently.
- [ ] Trace records emitted for every recall and every consolidation.
- [ ] Eval harness runs against the golden set and emits accuracy, latency, and cost in a single report.
- [ ] Layered system benchmarked against both A-MEM and the flat-file control.

---

## Risks

| Risk | Mitigation |
|---|---|
| Write-routing classifier quality | Deterministic rules first, LLM escalation only; read-side fan-out means a routing error is recoverable rather than fatal |
| Local backbone format errors during consolidation | Schema validation, dead-letter queue, failure rate tracked as a metric |
| Inscribe-time LLM regime weakens deletion precision | Supersession is an explicit logged operation, never inferred by the extractor |
| Knowledge projection rebuild cost as the log grows | Acceptable at expected scale; add snapshotting only if load-time budget is exceeded |
| Scope creep into Wisdom | Stubbed by explicit non-goal and acceptance criterion |
| Module becomes a god object | Layers are separate collaborators behind a common store interface; the module is a thin router and fuser |

## Open questions

- Embedding model and dimension. Should match whatever the Segmenter's drift detection already uses if practical.
- Does Memory generate its own summary, or reuse the Segmenter's episode summary?
- Retention policy. Does anything ever get truncated, or do the logs grow without bound?
- Does `budget` arrive as a token count or an entry count?

## Suggested issue sequencing

1. Store primitives: record schemas, JSONL append, Knowledge projection.
2. Module boundary: `recall`, deterministic routing, RRF fusion, miss reporting.
3. Instrumentation: recall and consolidation trace records.
4. Consolidation pipeline: extraction, write routing, supersession, guards.
5. Eval harness: golden set, metrics report, A-MEM and flat-file baselines.

Issues 1–3 are the ones that unblock everything else. Issue 5 should land before any tuning work begins.
