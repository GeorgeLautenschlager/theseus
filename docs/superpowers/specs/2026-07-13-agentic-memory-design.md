# Agentic Memory (A-MEM-inspired) — design

**Date:** 2026-07-13
**Status:** implemented

## Problem

Theseus agents had no long-term memory: `ContextAssembler` dumped the entire
stimulus log into every Decide/Act prompt. Context grew without bound, and the
agent had no distilled, retrievable recall of past experience.

## Approach

An A-MEM-inspired memory module (Xu et al., *A-MEM: Agentic Memory for LLM
Agents*, arXiv 2502.12110), implementing the subset of the paper that carries
most of its measured benefit:

1. **Note construction** — batches of stimulus events are distilled by the LLM
   into atomic `MemoryNote`s: raw content plus generated context, keywords,
   and tags, plus an embedding.
2. **Agentic linking** — a new note's nearest neighbors (cosine similarity)
   are offered to the LLM, which chooses which are genuinely related; chosen
   ids become the note's `links`.
3. **Retrieval** — embedding similarity top-k, expanded one hop along links,
   rendered into a `<memories>` section of the Decide/Act prompts.

**Memory evolution** (the paper's third mechanism, where new notes trigger LLM
rewrites of old notes' context/tags) is deliberately deferred: it is the most
expensive part of the pipeline (several LLM calls per write) and the
least-validated in follow-up literature. Deferring it lets notes be immutable
and the store append-only. When it lands, it will likely be modeled as
note-superseding records rather than in-place rewrites, preserving the
append-only discipline.

## Memory as a disposable projection

Per `stimulus_log.py`'s charter, everything downstream of the log is a
disposable projection. Notes honor this: each records the inclusive
`source_span` of stimulus-event ids it was formed from. The formation
high-water mark is *derived* (`MemoryStore.last_consolidated_id()` = max span
end), never stored separately — so deleting the notes file and replaying
cycles rebuilds memory from the log alone.

## Components

- `memory_note.py` — `MemoryNote`, frozen dataclass, JSONL-serializable,
  mirroring `StimulusEvent`.
- `memory_store.py` — `MemoryStore`, append-only JSONL with `StimulusLog`'s
  write discipline (append/flush/fsync, torn-tail tolerance). In-process
  pure-Python cosine `top_k`; no vector-store or numpy dependency.
- `memory_prompts.py` — pure prompt builders + JSON schemas for the two LLM
  steps (note construction, link decision), offline-testable like
  `cognitive_prompts.py`.
- `agentic_memory.py` — `AgenticMemory.form(events)` and `.retrieve(query)`.
  All failures are caught and reported: memory must never crash the loop.
- `ModelProvider.embed(text)` — OpenAI-compatible `/v1/embeddings`, gated on a
  new optional `embedding_model` constructor param on each provider.

## Integration

- `ContextAssembler` now returns `AssembledContext(recent_events, memories)`:
  the last `window_size` (default 50) events verbatim, plus notes retrieved
  against that window. The full-log dump is gone.
- `CognitiveCore` accepts `memory: AgenticMemory | None = None`. Orient
  assembles context (with retrieval), runs Decide/Act, then consolidates all
  events past the high-water mark into a new note — post-cycle, so memory
  writes never delay the agent's outward response.
- Formation timing is synchronous and post-cycle; no threads.

## Non-goals (this iteration)

- Memory evolution (see above).
- Aldric / `ChatCognitiveCore` migration — `tests/test_fact_retention.py`
  remains a known-failing live-LLM eval until Aldric moves to
  `CognitiveCore`.
- Background/async formation, forgetting/decay, deduplication.

## Wiring example

```python
provider = LmStudioProvider(model="...", embedding_model="text-embedding-nomic-embed-text-v1.5")
memory = AgenticMemory(model_providers=[provider], store=MemoryStore("memory.jsonl"))
core = CognitiveCore(
    constitution=..., model_providers=[provider],
    effectors=..., stimulus_log=StimulusLog("stimulus_log.jsonl"),
    memory=memory,
)
```
