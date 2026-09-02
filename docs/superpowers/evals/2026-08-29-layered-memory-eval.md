# Layered Memory Eval — 2026-08-29

Single report per the brief: `MemoryModule` (layered) vs existing `AgenticMemory`
(A-MEM) vs flat append-everything-cosine control, over identical episodes and one
golden set. Machine-readable results:
[`2026-08-29-layered-memory-eval.json`](./2026-08-29-layered-memory-eval.json).

## Setup

- **Episodes (15):** 5 real chunks from in-repo logs (`auto_alty/stimulus_log.jsonl`,
  `stimulus_log.jsonl`; 10 events each, file order) + 10 synthetic episodes
  (4 durable facts + noise, deterministic). Real traces were too thin for 50 real
  queries, so the golden set is topped up with clearly-labeled synthetic queries.
- **Golden set (50):** 10 real queries (answers verbatim in the named log) +
  40 synthetic (one per fact).
- **Models:** local Ollama — `gemma4:e4b` (extraction, note construction, judge),
  `nomic-embed-text` (768-dim embeddings). Judge used for **ranking only**
  (shuffled candidates + "no relevant information" sentinel, one call per query
  ranking all three systems' lists); never thresholds.
- **Recall budget:** 2048 tokens, top-3 candidates scored.

## Headline numbers

| Metric | layered | agentic (A-MEM) | flat control |
|---|---|---|---|
| Hit rate (answer verbatim in top-3) | **8%** | **18%** | **80%** |
| — real queries (n=10) | 2/10 | **9/10** | 7/10 |
| — synthetic queries (n=40) | 2/40 | 0/40 | **33/40** |
| Judge: answer found in ranking | 8% | 18% | 68% |
| Judge: best answer rank (mean, 1 = top) | **1.25** | 1.33 | 1.71 |
| Sentinel last / omitted (%) | 74 | 86 | 80 |
| Recall latency p50 / p95 (ms) | 64 / 446 | 43 / 52 | **34 / 58** |
| Consolidation wall time (15 episodes) | 5067 s (~5.6 min/ep) | not instrumented¹ | 26 s (embed-only) |
| Consolidation tokens in/out | 12,944 / 2,704 (~863/180 per ep) | — | 0 chat / embed only |
| Store size on disk | 557 KB² | 67 KB | **32 KB** |
| Records kept | 23 (0 knowledge, 8 memory assertions, 7 wisdom, +15 episode records) | **3 notes** | 15 episode blobs |

¹ `AgenticMemory` is the pre-existing module (untouched per goal constraints) and
emits no consolidation traces; its cost was not measured rather than estimated.
² Layered store includes 132 KB of trace records and 304 KB `memory.jsonl`
(episode records carry their full evidence by design).

## Findings

1. **Layered does not beat the flat control — reported as a finding, per the
   brief.** Flat append-everything-cosine hits 80% of golden queries; layered
   hits 8%. The gap is at **consolidation, not retrieval**: when any system
   actually has the answer in its top-3, it ranks it ~1.25–1.71 (layered's own
   retriever is the *best* ranker of the three). Flat wins because raw episode
   text keeps every fact verbatim and embeds it directly.

2. **The bottleneck is extraction quality on `gemma4:e4b`, in two distinct
   failure modes:**
   - **Schema failures: 84.3%** (43 of 51 candidate assertions dead-lettered).
     The model labels claims as `kind: "fact"` but almost never fills the
     subject/predicate/value triple — so **zero records reached the Knowledge
     layer**. The pipeline did exactly what it was designed to do (validate,
     dead-letter, count); the extractor simply cannot produce valid triples.
   - **Fact collapsing.** Where it does emit assertions, it merges 4 distinct
     facts into one vague event summary ("During this session, I discussed
     several details with Mara, including…"), which embeds as a blur and fails
     verbatim substring recall.

3. **A-MEM's failure mode is different: narrative retention.** It kept only
   **3 notes for 15 episodes** and scored 0/40 on synthetic — its first-person
   agent-voice note construction paraphrases away third-party facts (Mara's
     preferences) that a later query asks about verbatim. On real queries it was
     the best system (9/10) because those notes describe the agent's own actions,
     which survive in the agent's voice.

4. **Costs are dominated by the CPU LLM, not the architecture.** gemma4:e4b
   generates ~1 token/s on this machine: layered consolidation is ~5.6 min/episode
   for ~180 output tokens; the judge phase ran ~5 min/query. Any system design
   that pays per-episode chat costs inherits this; flat's embed-only consolidation
   (26 s total) shows the floor.

## What this means (tuning candidates — not applied, per issue order)

The layered module's *retrieval* design is validated (best-in-class ranking when
the material exists); its *consolidation* does not survive a weak backbone:

1. **Prompt:** force atomic extraction — "one assertion per claim; never merge
   multiple exchanges into one assertion; if you cannot state subject/predicate/
   value, emit kind `event`". The 84% schema failure rate is mostly the model
   misusing `fact`; a fallback rule would route those to Memory with the
   statement text instead of dead-lettering.
2. **Model:** extraction is the quality gate; a stronger local model (or a
   cheaper cloud one) should be A/B'd on the same golden set before any further
   architectural work.
3. **Knowledge ingestion is currently dead** on this backbone — until triples
   validate, the Knowledge layer adds projection machinery with no content.

## Reproduce

```bash
# full run (fresh workdir; ~6 h on this CPU)
poetry run python -m theseus.memory_eval --workdir /tmp/mem-eval
# per-query detail: /tmp/mem-eval/query_details.jsonl
```

Not part of the offline suite (needs live Ollama), like `tests/test_fact_retention.py`.
