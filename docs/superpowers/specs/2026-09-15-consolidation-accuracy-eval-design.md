# Multi-event consolidation accuracy evaluation

Design for issue #63 — *Evaluate consolidation accuracy across multi-event
episodes*. First increment of the Episode Consolidation: Accuracy Roadmap
(2026-09-15). Provides the accuracy baseline the remaining consolidation issues
(#64–#68) build on.

## Problem

The existing offline experiment (`memory_experiment_eval.py`) consolidates
**single-event** episodes — every scenario is one stimulus event whose
`Episode` spans `event.id, event.id`. That establishes retrieval mechanics but
not extraction accuracy across the situations the agent actually meets: actions
followed by their outcome, corrections phrased differently from the original
claim, coexisting preferences, historical reports that must not read as current,
decisive evidence buried at the end of an oversized event, and repeated or
contradictory principles. It also reports one blended retrieval score rather
than separating what was correctly retained from what was wrongly asserted.

## Goal

Add a **reusable multi-event accuracy evaluation** with labeled supporting
events and expected knowledge transitions, keeping deterministic pipeline checks
strictly separate from live extraction evaluation. The primary deterministic
signal is **knowledge transitions** — after consolidating a scenario's
episode(s), the durable knowledge projection reflects the expected current value
and the superseded values drop out.

## Non-goals

- No changes to `MemoryModule`, the layers, or extraction/packing behavior. This
  issue *measures*; #64–#68 change behavior. In particular the decisive-tail
  budget hazard (#65) is detected and reported here, not fixed.
- No automatic historical reconsolidation.
- `memory_experiment_eval.py` and its tests are left untouched — this is a new
  parallel module, not a rewrite.

## Approach

A new module `src/theseus/memory_accuracy_eval.py` and a new offline test file
`tests/test_memory_accuracy_eval.py`. The module owns a labeled scenario
dataset, a deterministic offline run, and an opt-in live run that shares the same
scenarios. Two eval entry points coexist deliberately: the older module keeps
its raw-lexical retrieval control; this one is organized around knowledge
transitions and separated retention/update/unsupported metrics.

## Components

### Scenario dataset — the labeled core

One frozen `Scenario` per acceptance-criteria category. Each describes a
multi-event episode (or a short sequence of episodes):

```python
@dataclass(frozen=True)
class EpisodeSpec:
    event_indices: tuple[int, ...]      # which scenario events form this episode (contiguous)
    reference_summary: str
    reference_assertions: tuple[dict, ...]   # what a correct extractor should yield for the whole episode

@dataclass(frozen=True)
class Claim:
    subject: str
    predicate: str
    value: str

@dataclass(frozen=True)
class Transition:
    subject: str
    predicate: str
    current_value: str
    superseded_values: tuple[str, ...]

@dataclass(frozen=True)
class Query:
    question: str
    expected: str | None                # None == the store must answer "unknown"
    forbidden: tuple[str, ...]

@dataclass(frozen=True)
class Scenario:
    name: str
    category: str
    events: tuple[tuple[str, str, dict, str], ...]   # (actor, type, content, role) role ∈ {"evidence","recall_context"}
    episodes: tuple[EpisodeSpec, ...]
    transitions: tuple[Transition, ...]
    supported: tuple[Claim, ...]         # must be retained / recallable
    unsupported: tuple[Claim, ...]       # must NOT appear (plan-as-done, historical-as-current, fabrication)
    queries: tuple[Query, ...]
```

Key property: **one reference extraction per multi-event episode**, not per
event — the episode is the consolidation unit, and the label is what a correct
extractor would produce having read every supporting event together.

Scenario categories (each is multi-event):

1. **Plan → outcome**, both branches: a plan/attempt event followed by a failure
   event, and a separate scenario where a plan is followed by a success event.
   The transition must reflect the *outcome*, and the *plan* must not be recorded
   as a completed action (an `unsupported` claim).
2. **Alternate-wording correction**: an initial fact, then a correction that uses
   different phrasing than the original. Transition supersedes the old value.
3. **Coexisting preferences**: two non-contradictory preferences established in
   separate events; both must be retained (neither supersedes the other).
4. **Historical report**: a past-tense report of a prior state; it must be
   retrievable as history but must not become the current fact
   (`unsupported` = the historical value presented as current).
5. **Decisive evidence at end of oversized event**: one event whose content
   exceeds the per-event extraction budget, with the decisive fact at the tail.
6. **Recall repetition**: the same fact queried multiple times; recall must stay
   consistent and the repeated recall context must not corrupt the answer.
7. **Split action/result pair**: an action stated in one event, its result in a
   later event within the same episode; the extracted knowledge reflects the
   combined outcome.
8. **Repeated + contradictory principles**: a principle asserted twice
   (idempotent) and later contradicted (the contradiction is surfaced, the stale
   principle does not read as current guidance).

### Deterministic run — `run_offline(workdir) -> dict`

Reference extractor only; no LLM, no live endpoint.

1. Guard: refuse a non-empty `workdir` (trials must not reuse memories), matching
   the existing eval's contract.
2. Build a `StimulusLog` and `MemoryModule` with a `ReferenceExtractor` whose
   response is set per episode from that episode's `reference_summary` /
   `reference_assertions` before each `consolidate(...)` call.
3. For each scenario, append its events (recording their stimulus ids), then
   consolidate each `EpisodeSpec` as one `Episode` spanning the first and last
   event id in its slice.
4. **Measure, separately:**
   - **correct_updates** — for every `Transition`, `knowledge.current(subject,
     predicate)` equals `current_value` and no `superseded_value` is still
     current.
   - **supported_retained** — every `supported` claim is recallable (present in a
     `recall(...)` result and/or the knowledge projection).
   - **unsupported_present** — count of `unsupported` claims that wrongly appear
     in recall or knowledge (target 0).
5. **Restart**: re-instantiate the `MemoryModule` from the same directory and
   re-run the recall checks — recall must survive a cold start.
6. **Context departure**: append a run of routine stimulus events so the source
   events fall out of the live/intelligence tail, then re-run recall — answers
   must come from durable stores, not the live window.
7. Retain per-scenario evidence in the report (the recalled entries and the
   current knowledge records that back each judgement) for manual review.

### Oversized-event handling

The oversized scenario places the decisive fact at the end of a single large
event. Current packing truncates head-first (`content[:size]`), so the tail is
dropped from the extraction budget. The eval **detects and reports** this as
`decisive_tail_in_budget: false` (computed from the same packing math the module
uses) rather than asserting success, and confirms the full event text remains
searchable in the Memory record. The test asserts the eval *correctly flags* the
loss — a red baseline that #65 will turn green.

### Live run — `run_live(workdir, *, extractor, embedder=None, answerer=None) -> dict`

Same scenarios, real providers, explicitly selected via CLI — never part of
offline CI. The real extractor produces the assertions; the eval measures the
same three metric families but **does not** treat a substring match or a valid
citation ID as proof of semantic support. Raw model answers and citation IDs are
retained verbatim for manual review, and the limitations note says so plainly.

### Reporting

`run_offline` and `run_live` both write `report.json` to `workdir` and return the
dict. Top level:

- `mode`: `"reference-extraction-offline"` | `"live-extraction"`.
- `provider` / `model` / `embedding` (identity strings; `None` offline).
- `metrics`: three distinct counters — `supported_retained`,
  `unsupported_present`, `correct_updates` — with per-category and totals, never
  collapsed into a single score.
- `restart_recall` and `context_departure_recall`: pass/fail per checked query.
- `costs`: extraction chat calls, embedding calls at consolidation and recall,
  estimated tokens in/out, and any provider-reported usage (live). Usage may omit
  failed requests and provider retries — stated as a limitation.
- `limitations`: reference extractions are labels, not model output; exact-text
  checks are retrieval diagnostics, not semantic correctness; substring/citation
  matches are not semantic proof; the decisive tail is dropped under the current
  head-first packing.
- `scenarios`: per-scenario evidence and per-metric outcomes.

### CLI (`main`)

Mirrors the existing eval's argument style: `--workdir` (required), optional
`--provider`/`--model`, `--embedding-provider`/`--embedding-model`, and
`--answers`. Provider and model must be given together; live embeddings/answers
require a live extraction provider. With no provider, runs `run_offline`.

## Tests (`tests/test_memory_accuracy_eval.py`, offline only)

- `run_offline` on an empty `tmp_path`: `correct_updates` equals the expected
  count, `unsupported_present == 0`, `supported_retained` covers every supported
  claim.
- Restart and context-departure recall both pass for the checked queries.
- The oversized scenario reports `decisive_tail_in_budget: false` and the full
  event text is still searchable.
- The empty-workdir guard raises on a re-run.
- No live endpoints are contacted (no provider configured).

## Files

- `src/theseus/memory_accuracy_eval.py` — new module (dataset, `run_offline`,
  `run_live`, `main`).
- `tests/test_memory_accuracy_eval.py` — new offline test file.
- `docs/memory-reliability.md` — short pointer to the new eval and how to run it
  live.

## Relation to the roadmap

- #64 (link assertions to supporting events): scenarios already label supporting
  events; this eval can later assert provenance links.
- #65 (preserve decisive evidence within budgets): the oversized scenario's
  `decisive_tail_in_budget: false` is the baseline to flip.
- #66 (reconcile with existing knowledge): the correction / contradictory-
  principle scenarios exercise supersession the reconciliation work must respect.
- #67 (interaction boundaries): multi-event episodes make boundary correctness
  measurable.
- #68 (provisional principles): the repeated/contradictory-principle scenario is
  the fixture for that work.
