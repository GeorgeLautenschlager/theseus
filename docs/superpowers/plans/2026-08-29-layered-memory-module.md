# Plan: Layered Memory Module

Governing spec: `docs/superpowers/specs/2026-08-29-layered-memory-module.md` (BRIEF-layered-memory-module).
Resolved open questions (from George, 2026-08-29): budget is **tokens** (naive `len(text)//4` per entry);
**no Segmenter** — `consolidate(episode)` is caller-triggered, `Episode = id + stimulus range`;
embeddings via local Ollama `nomic-embed-text` (768-d), chat via existing `ModelProvider`s.

## Architecture

```
MemoryModule                      public surface (leak-tested)
├── recall(query, budget_tokens) -> RecallResult
├── consolidate(episode)          -> ConsolidationResult
├── routing: deterministic fan-out to all layers + reciprocal rank fusion
└── instrumentation: one JSONL trace per recall / per consolidation

Layers (separate collaborators behind a common store interface):
  knowledge_layer.KnowledgeLayer   append-only JSONL + load-time projection;
                                   supersession = new record with `supersedes` id; no decay
  memory_layer.MemoryLayer         append-only bi-temporal JSONL episode records;
                                   numpy cosine, recency weighting at query time only
  wisdom_layer.WisdomLayer         append-only JSONL stub; retrievable; evidence_count filter;
                                   NO promotion/revision logic
  intelligence_layer.IntelligenceLayer  read strategy over StimulusLog tail; no file

Shared primitives:
  layer_store.py   append_record / load_lines — the write discipline (append, flush, fsync,
                   torn-final-line tolerance) shared by all three file-backed layers
```

## Files

| File | Contents |
|---|---|
| `src/theseus/layer_store.py` | `append_record(path, line)`, `load_lines(path)` (torn-line tolerant), `ensure_dir(path)` |
| `src/theseus/knowledge_layer.py` | `KnowledgeRecord` (id, ts, subject, predicate, value, source_episode_id, supersedes), `KnowledgeLayer` (append with supersession, projection, lookup) |
| `src/theseus/memory_layer.py` | `MemoryRecord` (id, ts, content, summary, embedding, source_episode_id), `MemoryLayer` (append, query = cosine × recency weight, records never mutated) |
| `src/theseus/wisdom_layer.py` | `WisdomRecord` (id, ts, statement, embedding, evidence_count, source_episode_id), `WisdomLayer` (append, query with min_evidence filter) |
| `src/theseus/intelligence_layer.py` | `IntelligenceLayer` — reads StimulusLog tail N, returns ranked entries; no file |
| `src/theseus/memory_module.py` | `MemoryModule`, `Episode`, `RecallEntry`, `Provenance`, `RecallResult`, `ConsolidationResult`; routing + RRF + budget fill + misses; consolidation pipeline (extract → validate → route → write, ledger idempotency, dead-letter) |
| `src/theseus/memory_prompts.py` | add `extraction_json_schema()` + `build_extraction_prompt(evidence_text, context_text)` (append to existing module) |
| `tests/test_layered_memory.py` | acceptance-criterion tests (matrix below) |
| `eval/layered_memory_eval.py` | issue-5 harness: golden set builder from `agents/a_mem.jsonl`, baselines (AgenticMemory, flat-file cosine control), single report |

Module dir layout (`memory_dir` passed to `MemoryModule`):
`knowledge.jsonl`, `memory.jsonl`, `wisdom.jsonl`, `consolidation_ledger.jsonl`,
`dead_letter.jsonl`, `traces/recall.jsonl`, `traces/consolidation.jsonl`.

## Key semantics

- **Recall**: embed query once → fan out: knowledge = token-overlap lookup on subject/predicate;
  memory = cosine × recency weight (half-life days, default 30); wisdom = cosine, min_evidence=0;
  intelligence = tail N events by recency. RRF `score(d) = Σ 1/(k + rank_i)` (k=60), fill fused
  order under token budget (greedy skip-and-continue). Misses: per-layer "no matches" strings in
  `RecallResult.misses` — data, never exceptions.
- **Consolidation**: read stimulus range; recall-flagged events (`type == "tool_result"` and
  `content.tool == RECALL_TOOL_NAME`) are context-only, never evidence/provenance. LLM extracts
  candidate assertions (schema-validated → dead-letter on failure). Each assertion gets an id
  before routing. Deterministic write routing: fact w/ subject+predicate+value → knowledge
  (supersession on same subject+predicate); principle → wisdom; else/unresolved → memory. One
  episode record per consolidation into the memory layer. Ledger makes re-consolidating an
  episode a no-op.
- **Leak test**: no layer name in `recall`/`consolidate` signatures or `RecallResult` top-level
  fields; `Provenance.layer` is opaque data (core may log it, never branch on it).

## Test matrix (acceptance criteria → named tests)

| # | Criterion | Test |
|---|---|---|
| 1 | separate persistence semantics; no decay/mutation of own records | `test_layers_persist_independently_without_mutation` |
| 2/3 | recall returns misses explicitly, as data | `test_recall_reports_misses_as_data` |
| 4 | idempotent per (episode_id, assertion_id) | `test_consolidate_is_idempotent_per_episode` |
| 5 | recall-flagged stimuli excluded from evidence | `test_recall_flagged_stimuli_excluded_from_evidence` |
| 6 | supersession explicit + logged | `test_knowledge_supersession_is_explicit_and_logged` |
| 7 | wisdom retrievable w/ evidence_count filter | `test_wisdom_retrieval_filters_by_evidence_count` |
| 8 | token budget fills window | `test_recall_respects_token_budget` |
| 9 | intelligence from log tail, no fourth file | `test_intelligence_reads_log_tail_without_a_file` |
| 10 | no Segmenter / episode detection | `test_no_episode_detection_in_public_surface` |
| 11 | traces per recall + per consolidation | `test_traces_emitted_per_recall_and_consolidation` |
| 12 | schema-invalid dead-lettered AND counted | `test_schema_invalid_extractions_dead_lettered_and_counted` |
| — | leak test | `test_no_layer_names_leak_into_public_surface` |

Plus issue-5: `eval/layered_memory_eval.py` → single report (accuracy, latency p50/p95, cost/episode, store sizes) vs AgenticMemory + flat control.

## Issue order (per spec)

1. Store primitives + Knowledge projection (+ memory/wisdom stores — same slice, same discipline)
2. Module boundary: recall + RRF + misses + budget
3. Instrumentation: trace writer + per-recall/per-consolidation records
4. Consolidation pipeline: extraction, routing, supersession, ledger, dead-letter
5. Eval harness + golden set + report

Each slice: targeted tests green + `poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py` green before moving on.
