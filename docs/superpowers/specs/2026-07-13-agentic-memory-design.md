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

## The Memory protocol

`AgenticMemory` is the first memory module in the construction kit, not the
last. The core therefore sees memory only through the `Memory` protocol
(`memory.py`, mirroring `Effector`):

- `form()` — the core's signal, sent at loop termination. What to consolidate
  (and whether anything is new enough to bother) is the module's own business;
  the module tracks its own high-water mark and reads the log itself. Must
  never raise into the loop.
- `retrieve(query) -> str` — a rendered, prompt-ready block of memories, `""`
  when nothing is relevant.

`retrieve` returning a string is what keeps the interface leak-proof: neither
`CognitiveCore` nor `ContextAssembler` imports note types or stores.

## Loop discipline

`CognitiveCore` steps never pass data to each other — any step may kick back
to Orient, so everything a loop accumulates lives in `loop_memory`
(`recent_events`, `memories`, `decision`). Every terminal path of Act funnels
through `loop_termination()`, which is skipped whenever another cycle is
triggered. Loop termination currently hard-codes the `memory.form()` signal
and resets `loop_memory`; a later iteration should accumulate callbacks and
iterate over them.

## Components

- `memory_note.py` — `MemoryNote`, frozen dataclass, JSONL-serializable,
  mirroring `StimulusEvent`.
- `memory_store.py` — `MemoryStore`, append-only JSONL with `StimulusLog`'s
  write discipline (append/flush/fsync, torn-tail tolerance). In-process
  pure-Python cosine `top_k`; no vector-store or numpy dependency.
- `memory_prompts.py` — pure prompt builders + JSON schemas for the two LLM
  steps (note construction, link decision), offline-testable like
  `cognitive_prompts.py`.
- `memory.py` — the `Memory` protocol (see above).
- `agentic_memory.py` — `AgenticMemory`, implementing the protocol. All
  failures are caught and reported: memory must never crash the loop.
- `ModelProvider.embed(text)` — OpenAI-compatible `/v1/embeddings` using the
  instance's one model. A provider class is a *place we get models*; an
  embedding provider is just another instance, e.g.
  `OllamaProvider(model="nomic-embed-text")`. `AgenticMemory` takes separate
  `model_providers` (chat) and `embedding_providers` priority lists.

## Integration

- `ContextAssembler` now returns `AssembledContext(recent_events, memories)`:
  the last `window_size` (default 50) events verbatim, plus whatever the
  memory module's `retrieve` returns against that window. The full-log dump
  is gone.
- `CognitiveCore` accepts `memory: Memory | None = None`. Orient assembles
  context (with retrieval) into `loop_memory`, runs Decide/Act, and
  `loop_termination()` signals `memory.form()` — post-cycle, so memory writes
  never delay the agent's outward response.
- Formation timing is synchronous and post-cycle; no threads.

## Non-goals (this iteration)

- Memory evolution (see above).
- Aldric / `ChatCognitiveCore` migration — `tests/test_fact_retention.py`
  remains a known-failing live-LLM eval until Aldric moves to
  `CognitiveCore`.
- Background/async formation, forgetting/decay, deduplication.

## Wiring example

```python
provider = OllamaProvider(model="gemma4:e4b")
embedder = OllamaProvider(model="nomic-embed-text")
stimulus_log = StimulusLog("stimulus_log.jsonl")
memory = AgenticMemory(
    model_providers=[provider],
    embedding_providers=[embedder],
    store=MemoryStore("memory.jsonl"),
    stimulus_log=stimulus_log,
)
core = CognitiveCore(
    constitution=..., model_providers=[provider],
    effectors=..., stimulus_log=stimulus_log,
    memory=memory,
)
```
