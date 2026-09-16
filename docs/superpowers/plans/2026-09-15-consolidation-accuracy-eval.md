# Multi-event Consolidation Accuracy Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: this plan is executed with
> `steward:steward-local-sdd` — planning/review stay on the frontier model, each
> task's code generation is dispatched to the local Pi agent ("Blueberry") via
> `pi -p`. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Handoff note (local-sdd):** each task gives the **full test code** (the
> contract Pi must satisfy) and the **full scenario data** (the labeled reference
> — it *is* the deliverable), but describes the *logic* of `run_offline`,
> `run_live`, and the metric functions as precise specifications rather than
> transcribed source. Pi writes the logic to make the tests pass. Do not paste
> finished function bodies into the plan. See memory
> `steward-plans-must-not-embed-source`.

**Goal:** Add a reusable multi-event consolidation accuracy evaluation
(`memory_accuracy_eval.py`) with labeled supporting events and expected knowledge
transitions, keeping deterministic offline checks separate from opt-in live
extraction runs.

**Architecture:** A new module holds a frozen `Scenario` dataset (one per
acceptance-criteria category, each a multi-event episode), a deterministic
`run_offline` path driven by a scripted `ReferenceExtractor`, an opt-in
`run_live` path sharing the same scenarios, and a `main` CLI mirroring the
existing `memory_experiment_eval.py`. The primary deterministic signal is
knowledge transitions; supported-claim retention, unsupported claims, and correct
updates are reported as three independent metrics. `memory_experiment_eval.py`
is left untouched.

**Tech Stack:** Python 3.12, Poetry, pytest. Uses `theseus.memory_module`
(`MemoryModule`, `Episode`), `theseus.knowledge_layer`, `theseus.stimulus_log`,
`theseus.tools.recall.RECALL_TOOL_NAME`. All offline tests run without a live
endpoint.

**Spec:** `docs/superpowers/specs/2026-09-15-consolidation-accuracy-eval-design.md`

**Run tests with:** `env -u VIRTUAL_ENV poetry run pytest <path> -v`
(the `env -u VIRTUAL_ENV` guard is required in this workspace — see memory
`virtualenv-shadows-poetry`; the whole suite otherwise shadows Poetry's venv).

---

## File Structure

- **Create** `src/theseus/memory_accuracy_eval.py` — the whole module: data
  model, `SCENARIOS`, `validate_scenarios`, `ReferenceExtractor`,
  `drive_scenarios`, metric functions, `run_offline`, `run_live`, `main`.
- **Create** `tests/test_memory_accuracy_eval.py` — offline tests only.
- **Modify** `docs/memory-reliability.md` — add a short pointer to the new eval.

Everything lives in one module because the dataset, driver, metrics, and runs are
tightly coupled around the `Scenario` shape and are always changed together.

---

## Task 1: Scenario data model, dataset, and well-formedness validator

**Files:**
- Create: `src/theseus/memory_accuracy_eval.py`
- Test: `tests/test_memory_accuracy_eval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_accuracy_eval.py`:

```python
from __future__ import annotations

import json

import pytest

from theseus.memory_accuracy_eval import (
    SCENARIOS,
    REQUIRED_CATEGORIES,
    EpisodeSpec,
    Scenario,
    Transition,
    validate_scenarios,
)


def test_dataset_is_well_formed_and_covers_every_category():
    validate_scenarios(SCENARIOS)  # raises on any structural problem
    categories = {s.category for s in SCENARIOS}
    assert REQUIRED_CATEGORIES <= categories
    # Every scenario is multi-event OR multi-episode — never a single lone event.
    for s in SCENARIOS:
        assert len(s.events) >= 2 or len(s.episodes) >= 2, s.name
    # Episode indices partition the scenario's events.
    for s in SCENARIOS:
        covered = [i for ep in s.episodes for i in ep.event_indices]
        assert sorted(covered) == list(range(len(s.events))), s.name


def test_validator_rejects_a_transition_without_a_supporting_fact():
    broken = Scenario(
        name="broken", category="plan_then_failure",
        events=(("human", "chat_message", {"message": "x"}, "evidence"),
                ("human", "chat_message", {"message": "y"}, "evidence")),
        episodes=(EpisodeSpec((0, 1), "summary", ()),),  # no assertions back the transition
        transitions=(Transition("Atlas", "payment status", "failed"),),
    )
    with pytest.raises(ValueError, match="transition"):
        validate_scenarios((broken,))


def test_validator_rejects_noncontiguous_or_out_of_range_episode():
    broken = Scenario(
        name="broken2", category="plan_then_failure",
        events=(("human", "chat_message", {"message": "x"}, "evidence"),),
        episodes=(EpisodeSpec((0, 5), "summary", ()),),  # index 5 out of range
    )
    with pytest.raises(ValueError):
        validate_scenarios((broken2 := broken,))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'theseus.memory_accuracy_eval'`.

- [ ] **Step 3: Write the module's data model, helpers, dataset, and validator**

Create `src/theseus/memory_accuracy_eval.py`. Start with the module docstring
and `from __future__ import annotations`, then implement exactly these pieces.

**Data model (frozen dataclasses — give these verbatim, they are the contract):**

```python
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    event_indices: tuple[int, ...]        # which scenario events form this episode (contiguous, sorted)
    summary: str                          # reference episode summary
    assertions: tuple[dict, ...] = ()     # what a correct extractor should yield for the whole episode


@dataclass(frozen=True, slots=True)
class Transition:
    subject: str
    predicate: str
    current_value: str
    superseded_values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Query:
    question: str
    expected: str | None                  # None == the store should have no answer
    forbidden: tuple[str, ...] = ()        # used only in live-answer diagnostics


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    category: str
    events: tuple[tuple[str, str, dict, str], ...]   # (actor, type, content, role); role in {"evidence","recall_context"}
    episodes: tuple[EpisodeSpec, ...]
    transitions: tuple[Transition, ...] = ()
    supported: tuple[str, ...] = ()        # fragments that MUST be retrievable via recall or current knowledge
    unsupported: tuple[str, ...] = ()      # fragments that MUST NOT be a current knowledge value
    queries: tuple[Query, ...] = ()
```

**Helpers (module-level, to keep the dataset readable):**

- `def _ev(actor, message, role="evidence"): return (actor, "chat_message", {"message": message}, role)`
- `def _fact(subject, predicate, value, statement=None): return {"kind": "fact", "subject": subject, "predicate": predicate, "value": value, "statement": statement or f"{subject} {predicate}: {value}"}`
- `def _principle(statement): return {"kind": "principle", "statement": statement}`
- `def _note(statement): return {"kind": "event", "statement": statement}`

**`REQUIRED_CATEGORIES`** — a frozenset of exactly:
`{"plan_then_failure", "plan_then_success", "alternate_wording_correction", "coexisting_preferences", "historical_report", "recall_repetition", "split_action_result", "principles", "oversized_tail"}`

**`SCENARIOS`** — a tuple of these nine `Scenario` values (give the data
verbatim; it is the labeled deliverable):

1. `plan_then_failure` / name `"atlas-payment"`:
   - events: `_ev("Alpha", "I will attempt the Atlas payment now.")`, `_ev("tool", "Atlas payment attempt failed; no funds were received.")`
   - episodes: one `EpisodeSpec((0, 1), "I attempted the Atlas payment; it failed and no funds arrived.", (_fact("Atlas", "payment status", "failed", "Atlas payment attempt failed; no funds received."),))`
   - transitions: `(Transition("Atlas", "payment status", "failed"),)`
   - supported: `("failed",)`  ·  unsupported: `("completed", "received")`
   - queries: `(Query("What is Atlas's payment status?", "failed", ("completed", "received", "paid")), Query("What is Atlas's bank account number?", None))`

2. `plan_then_success` / name `"project-backup"`:
   - events: `_ev("Beta", "I plan to back up the project database.")`, `_ev("tool", "Project database backup completed successfully.")`
   - episodes: `EpisodeSpec((0, 1), "Beta planned the backup and it completed successfully.", (_fact("Project", "backup status", "completed"),))`
   - transitions: `(Transition("Project", "backup status", "completed"),)`
   - supported: `("completed",)`  ·  unsupported: `("planned", "not yet")`
   - queries: `(Query("What is the project backup status?", "completed", ("planned", "not yet")),)`

3. `alternate_wording_correction` / name `"atlas-deadline"`:
   - events: `_ev("human", "The Atlas prototype is due Friday.")`, `_ev("human", "Scratch that — we pushed Atlas delivery to Monday.")`
   - episodes: `EpisodeSpec((0,), "Atlas prototype is due Friday.", (_fact("Atlas", "prototype deadline", "Friday"),))`, `EpisodeSpec((1,), "Atlas delivery was pushed to Monday.", (_fact("Atlas", "prototype deadline", "Monday"),))`
   - transitions: `(Transition("Atlas", "prototype deadline", "Monday", ("Friday",)),)`
   - supported: `("Monday",)`  ·  unsupported: `("Friday",)`
   - queries: `(Query("What is Atlas's prototype deadline?", "Monday", ("Friday",)),)`

4. `coexisting_preferences` / name `"user-prefs"`:
   - events: `_ev("human", "I prefer morning meetings.")`, `_ev("human", "I prefer concise written summaries.")`
   - episodes: `EpisodeSpec((0,), "The user prefers morning meetings.", (_fact("George", "meeting time preference", "morning"),))`, `EpisodeSpec((1,), "The user prefers concise written summaries.", (_fact("George", "summary format preference", "concise written"),))`
   - transitions: `(Transition("George", "meeting time preference", "morning"), Transition("George", "summary format preference", "concise written"))`
   - supported: `("morning", "concise")`  ·  unsupported: `()`
   - queries: `(Query("What meeting time does the user prefer?", "morning"), Query("What summary format does the user prefer?", "concise"))`

5. `historical_report` / name `"atlas-phase"`:
   - events: `_ev("Alpha", "Last quarter Atlas was in the pilot phase.")`, `_ev("human", "Atlas is now in the production phase.")`
   - episodes: `EpisodeSpec((0,), "Alpha reported that last quarter Atlas was in the pilot phase.", (_note("Last quarter Atlas was in the pilot phase."),))`, `EpisodeSpec((1,), "Atlas is now in the production phase.", (_fact("Atlas", "phase", "production"),))`
   - transitions: `(Transition("Atlas", "phase", "production"),)`
   - supported: `("production", "pilot")`  ·  unsupported: `("pilot",)`  *(historical value must not become a current knowledge fact, but stays retrievable in memory)*
   - queries: `(Query("What phase is Atlas in now?", "production", ("pilot",)),)`

6. `recall_repetition` / name `"raven-code"`:
   - events: `_ev("human", "The Atlas access code is RAVEN-42.")`, `_ev("agent", "(recalled) The Atlas access code was once QUAIL-7.", role="recall_context")`
   - episodes: `EpisodeSpec((0, 1), "The Atlas access code is RAVEN-42.", (_fact("Atlas", "access code", "RAVEN-42"),))`
   - transitions: `(Transition("Atlas", "access code", "RAVEN-42"),)`
   - supported: `("RAVEN-42",)`  ·  unsupported: `("QUAIL-7",)`  *(recall-context value must not become a fact)*
   - queries: three identical `Query("What is the Atlas access code?", "RAVEN-42")` entries.

7. `split_action_result` / name `"boreal-invoice"`:
   - events: `_ev("Alpha", "Sending the Boreal invoice now.")`, `_ev("tool", "Boreal invoice sent; confirmation BOR-77.")`
   - episodes: `EpisodeSpec((0, 1), "Alpha sent the Boreal invoice; confirmation BOR-77.", (_fact("Boreal", "invoice status", "sent", "Boreal invoice sent; confirmation BOR-77."),))`
   - transitions: `(Transition("Boreal", "invoice status", "sent"),)`
   - supported: `("sent", "BOR-77")`  ·  unsupported: `("draft",)`
   - queries: `(Query("What is Boreal's invoice status?", "sent", ("draft", "not sent")),)`

8. `principles` / name `"confirm-policy"`:
   - events: `_ev("human", "Always confirm risky actions before executing.")`, `_ev("human", "Reminder: confirm risky actions before executing.")`, `_ev("human", "For low-risk actions, skip the confirmation step.")`
   - episodes: `EpisodeSpec((0,), "The user set a rule to always confirm risky actions.", (_principle("Always confirm risky actions before executing."),))`, `EpisodeSpec((1,), "The user repeated the rule to confirm risky actions.", (_principle("Always confirm risky actions before executing."),))`, `EpisodeSpec((2,), "The user added that low-risk actions can skip confirmation.", (_principle("For low-risk actions, skip the confirmation step."),))`
   - transitions: `()`  *(principles route to Wisdom, not Knowledge)*
   - supported: `("confirm risky actions", "low-risk")`  ·  unsupported: `()`
   - queries: `(Query("What is the policy for risky actions?", "confirm"), Query("What is the policy for low-risk actions?", "low-risk"))`

9. `oversized_tail` / name `"atlas-token"`:
   - Define a module-level constant `_OVERSIZED_MESSAGE = ("padding. " * 8000) + "DECISIVE: The Atlas security token is ZULU-9."` (~64 KB, comfortably over the ~46 KB per-event budget; the decisive fact is at the very end).
   - events: `("tool", "tool_result", {"message": _OVERSIZED_MESSAGE}, "evidence")`
   - episodes: `EpisodeSpec((0,), "Atlas security token noted.", (_fact("Atlas", "security token", "ZULU-9"),))`  *(the reference label is what a correct extractor would produce IF it saw the tail)*
   - transitions: `(Transition("Atlas", "security token", "ZULU-9"),)`
   - supported: `("ZULU-9",)`  ·  unsupported: `()`
   - queries: `(Query("What is the Atlas security token?", "ZULU-9"),)`
   - This is the one scenario allowed to be a single event (its point is one oversized event); the well-formedness test's `len(events) >= 2 or len(episodes) >= 2` must still hold, so **give it a second trailing evidence event** to satisfy the multi-event rule and keep the decisive event oversized: add `_ev("human", "Store the Atlas security token safely.")` as event index 1, and make the episode span `(0, 1)`. Keep `_OVERSIZED_MESSAGE` on event 0.

**`validate_scenarios(scenarios)`** — raise `ValueError` (with a message naming the
offending scenario) when any of these hold, otherwise return `None`:
- an `event_indices` tuple is empty, out of range for its scenario's events, not
  sorted, or not contiguous;
- the union of all `event_indices` across a scenario's episodes is not exactly
  `range(len(events))` (every event belongs to exactly one episode);
- a `Transition(subject, predicate, current_value)` has no `_fact` assertion
  anywhere in that scenario whose subject/predicate match (case-insensitive) and
  whose value equals `current_value` — the message must contain the word
  `"transition"`;
- an assertion has `kind == "fact"` but is missing a non-empty
  subject/predicate/value, or a non-fact assertion is missing a non-empty
  `statement`;
- an event's role is not in `{"evidence", "recall_context"}`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/theseus/memory_accuracy_eval.py tests/test_memory_accuracy_eval.py
git commit -m "feat(eval): scenario data model + dataset + validator for consolidation accuracy (#63)"
```

---

## Task 2: ReferenceExtractor and the consolidation driver

**Files:**
- Modify: `src/theseus/memory_accuracy_eval.py`
- Test: `tests/test_memory_accuracy_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_memory_accuracy_eval.py`:

```python
from datetime import datetime, timezone

from theseus.memory_accuracy_eval import ReferenceExtractor, build_memory, drive_scenarios


def _scenario(name):
    return next(s for s in SCENARIOS if s.name == name)


def test_driver_applies_correction_across_episodes(tmp_path):
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    drive_scenarios(memory, log, (_scenario("atlas-deadline"),),
                    reference=True, extractor=extractor,
                    start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    current = memory.knowledge.current("Atlas", "prototype deadline")
    assert current and current[0].value == "Monday"          # correction applied
    assert all(r.value != "Friday" for r in memory.knowledge.current())  # old value superseded


def test_driver_never_promotes_recall_context_to_a_fact(tmp_path):
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    trace = drive_scenarios(memory, log, (_scenario("raven-code"),),
                            reference=True, extractor=extractor,
                            start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    values = [r.value for r in memory.knowledge.current()]
    assert "RAVEN-42" in values
    assert "QUAIL-7" not in values                            # recall context is not evidence
    # The driver records the prompt the extractor saw for each episode.
    assert trace["raven-code"]["episode_prompts"]
    prompt = next(iter(trace["raven-code"]["episode_prompts"].values()))
    assert "QUAIL-7" in prompt                                # present as context_only, not as evidence
```

- [ ] **Step 2: Run to verify it fails**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -k driver -v`
Expected: FAIL with `ImportError` (build_memory / drive_scenarios / ReferenceExtractor not defined).

- [ ] **Step 3: Implement `ReferenceExtractor`, `build_memory`, `drive_scenarios`**

Add to the module:

**`ReferenceExtractor`** — a scripted stand-in for a chat provider:
- class attribute `model = "ReferenceExtractor"`;
- `__init__` sets `self.response = ""` and `self.last_prompt = None`;
- `chat(self, prompt, **kwargs)` records `self.last_prompt = prompt` and returns
  `self.response`.

**`build_memory(workdir, *, extractor, embedder)` -> `(MemoryModule, StimulusLog)`:**
- `log = StimulusLog(Path(workdir) / "stimulus.jsonl")`;
- `memory = MemoryModule(Path(workdir) / "memory", log, model_providers=[extractor], embedding_providers=[embedder] if embedder else [])`;
- return `(memory, log)`.

**`drive_scenarios(memory, log, scenarios, *, reference, extractor, start)` -> `dict`:**
- Import `Episode` from `theseus.memory_module`, `RECALL_TOOL_NAME` from
  `theseus.tools.recall`, `timedelta` from `datetime`.
- Maintain a monotonically increasing `day` counter (one per appended event) so
  every event gets a distinct timestamp `start + timedelta(days=day)`.
- For each scenario, append its events in order, collecting their event ids:
  - role `"recall_context"` → append as
    `log.append(actor, "tool_result", {**content, "tool": RECALL_TOOL_NAME}, ts=...)`;
  - otherwise → `log.append(actor, ev_type, content, ts=...)` using the tuple's
    `ev_type`.
- Then for each `EpisodeSpec` (index `i`), build
  `episode_id = f"{scenario.name}-ep{i}"`,
  `start_id = ids[event_indices[0]]`, `end_id = ids[event_indices[-1]]`.
  - When `reference` is True, set
    `extractor.response = json.dumps({"summary": ep.summary, "assertions": [dict(a) for a in ep.assertions]})`
    before consolidating. When False (live), leave the extractor alone.
  - Call `memory.consolidate(Episode(episode_id, start_id, end_id))`.
  - After the call, record `episode_prompts[episode_id] = getattr(extractor, "last_prompt", None)`.
- Return `{scenario.name: {"event_ids": ids, "episode_prompts": {...}} for each scenario}`.

- [ ] **Step 4: Run to verify it passes**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -k driver -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/theseus/memory_accuracy_eval.py tests/test_memory_accuracy_eval.py
git commit -m "feat(eval): reference extractor + multi-event consolidation driver (#63)"
```

---

## Task 3: The three separated metric functions

**Files:**
- Modify: `src/theseus/memory_accuracy_eval.py`
- Test: `tests/test_memory_accuracy_eval.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from theseus.memory_accuracy_eval import (
    measure_correct_updates,
    measure_supported_retained,
    measure_unsupported_present,
)


def _drive_all(tmp_path):
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    drive_scenarios(memory, log, SCENARIOS, reference=True, extractor=extractor,
                    start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    return memory, log


def test_metrics_all_pass_on_the_reference_run(tmp_path):
    memory, _ = _drive_all(tmp_path)
    updates = measure_correct_updates(memory, SCENARIOS)
    retained = measure_supported_retained(memory, SCENARIOS, budget_tokens=2000)
    unsupported = measure_unsupported_present(memory, SCENARIOS)
    assert updates["total"] > 0 and updates["passed"] == updates["total"]
    assert retained["total"] > 0 and retained["passed"] == retained["total"]
    assert unsupported["total"] > 0 and unsupported["violations"] == 0


def test_unsupported_metric_flags_a_leaked_current_fact(tmp_path):
    from theseus.knowledge_layer import KnowledgeRecord
    memory, _ = _drive_all(tmp_path)
    # Force the historical value to become a current fact — the metric must catch it.
    memory.knowledge.add(KnowledgeRecord(
        "leak", datetime(2030, 1, 1, tzinfo=timezone.utc), "Atlas", "phase", "pilot"))
    unsupported = measure_unsupported_present(memory, SCENARIOS)
    assert unsupported["violations"] >= 1
    assert any("pilot" in v["fragment"] for v in unsupported["failures"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -k metric -v`
Expected: FAIL with ImportError for the three `measure_*` functions.

- [ ] **Step 3: Implement the metric functions**

Each returns a plain dict and never raises for a missing record (misses are
data). Comparisons are case-insensitive substring unless stated otherwise.

**`measure_correct_updates(memory, scenarios)` -> dict:**
- For every `Transition` in every scenario:
  - `current = memory.knowledge.current(t.subject, t.predicate)`;
  - a transition **passes** iff `current` is non-empty and
    `t.current_value.casefold()` equals (not just substring) the casefolded value
    of `current[0].value`, **and** none of `t.superseded_values` equals any
    current value for that subject/predicate.
  - Record failures as `{"scenario", "subject", "predicate", "expected", "found"}`.
- Return `{"passed": int, "total": int, "failures": [...]}`.

**`measure_supported_retained(memory, scenarios, *, budget_tokens=2000)` -> dict:**
- For each scenario build the retrieval union: call `memory.recall(q.question, budget_tokens)`
  for every query and collect all `entry.text`; also add every current knowledge
  value `r.value` from `memory.knowledge.current()`.
- Each `fragment` in `scenario.supported` **passes** iff its casefold appears in
  the casefolded union text. Record failures as
  `{"scenario", "fragment"}`.
- Return `{"passed": int, "total": int, "failures": [...]}`.

**`measure_unsupported_present(memory, scenarios)` -> dict:**
- `current_values = [r.value for r in memory.knowledge.current()]`.
- Each `fragment` in `scenario.unsupported` is a **violation** iff its casefold
  appears in any casefolded current value. Record violations as
  `{"scenario", "fragment"}`.
- Return `{"violations": int, "total": int, "failures": [...]}`.

- [ ] **Step 4: Run to verify it passes**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -k metric -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/theseus/memory_accuracy_eval.py tests/test_memory_accuracy_eval.py
git commit -m "feat(eval): separated retention / unsupported / correct-update metrics (#63)"
```

---

## Task 4: `run_offline` — orchestration, restart, context departure, oversized, report

**Files:**
- Modify: `src/theseus/memory_accuracy_eval.py`
- Test: `tests/test_memory_accuracy_eval.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from theseus.memory_accuracy_eval import run_offline


def test_run_offline_reports_separated_metrics_and_survives_restart(tmp_path):
    report = run_offline(tmp_path)
    assert report["mode"] == "reference-extraction-offline"
    m = report["metrics"]
    assert m["correct_updates"]["passed"] == m["correct_updates"]["total"] > 0
    assert m["supported_retained"]["passed"] == m["supported_retained"]["total"] > 0
    assert m["unsupported_present"]["violations"] == 0
    # Recall survives a cold restart and after source events leave live context.
    assert report["restart_recall"]["passed"] == report["restart_recall"]["total"] > 0
    assert (report["context_departure_recall"]["passed"]
            == report["context_departure_recall"]["total"] > 0)
    # No live calls happened offline.
    assert report["provider"] is None
    assert report["costs"]["answer_chat_calls"] == 0
    assert report["costs"]["embedding_calls_at_consolidation"] == 0
    # report.json is written and round-trips.
    assert json.loads((tmp_path / "report.json").read_text())["mode"] == report["mode"]


def test_run_offline_detects_the_dropped_decisive_tail(tmp_path):
    report = run_offline(tmp_path)
    oversized = report["oversized"]
    assert oversized["decisive_tail_in_budget"] is False   # head-first truncation drops the tail (#65 baseline)
    assert oversized["full_event_searchable"] is True      # full evidence still stored/searchable


def test_run_offline_refuses_a_nonempty_workdir(tmp_path):
    run_offline(tmp_path)
    with pytest.raises(ValueError, match="empty workdir"):
        run_offline(tmp_path)
```

- [ ] **Step 2: Run to verify it fails**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -k run_offline -v`
Expected: FAIL with ImportError for `run_offline`.

- [ ] **Step 3: Implement `run_offline(workdir)`**

Signature: `def run_offline(workdir, *, budget_tokens=2000) -> dict:`

Algorithm:
1. `workdir = Path(workdir)`. If it exists and is non-empty, raise
   `ValueError("use an empty workdir so trials cannot reuse earlier memories")`
   (message must contain "empty workdir"). Then `mkdir(parents=True, exist_ok=True)`.
2. `validate_scenarios(SCENARIOS)`.
3. `extractor = ReferenceExtractor()`; `memory, log = build_memory(workdir, extractor=extractor, embedder=None)`.
4. `traces = drive_scenarios(memory, log, SCENARIOS, reference=True, extractor=extractor, start=datetime(2026, 1, 1, tzinfo=timezone.utc))`.
5. Compute the three metrics on `memory`.
6. **Restart recall:** re-open the store —
   `memory2, _ = build_memory(...)` **must not** be used (it would guard on the
   non-empty dir); instead re-instantiate directly:
   `restarted = MemoryModule(workdir / "memory", log, model_providers=[extractor], embedding_providers=[])`.
   Compute `measure_supported_retained(restarted, SCENARIOS, budget_tokens=budget_tokens)`
   → `restart_recall`.
7. **Context departure:** append ~30 routine events to `log`
   (`log.append("agent", "decision", {"text": f"Routine housekeeping {i}"})`),
   re-instantiate the module again (`departed = MemoryModule(...)` as above), and
   compute `measure_supported_retained(departed, SCENARIOS, budget_tokens=budget_tokens)`
   → `context_departure_recall`. This proves recall comes from durable stores,
   not the live tail.
8. **Oversized detection:** for the `oversized_tail` scenario,
   - `prompt = next(iter(traces["atlas-token"]["episode_prompts"].values()))`;
   - `decisive_tail_in_budget = bool(prompt) and "ZULU-9" in prompt`;
   - `full_event_searchable = any("ZULU-9" in r.content for r in memory.memory.read_all())`;
   - store both under `report["oversized"]`.
9. **Costs:** read the consolidation traces exactly as `memory_experiment_eval.py`
   does — `traces_jsonl = [json.loads(line) for line in load_lines(workdir / "memory" / "traces" / "consolidation.jsonl")]` — and build
   `costs = {"extraction_chat_calls": sum(t["chat_calls"] for t in traces_jsonl), "answer_chat_calls": 0, "embedding_calls_at_consolidation": sum(t["embedding_calls"] for t in traces_jsonl), "embedding_calls_at_recall": memory._embedding_calls, "tokens_in_estimated": sum(t["tokens_in"] for t in traces_jsonl), "tokens_out_estimated": sum(t["tokens_out"] for t in traces_jsonl), "reported_chat_usage": [u for t in traces_jsonl for u in t["reported_chat_usage"]]}`.
10. Build the report dict:
    ```
    {
      "mode": "reference-extraction-offline",
      "provider": None, "model": "ReferenceExtractor", "embedding": None,
      "scenarios_count": len(SCENARIOS),
      "categories": sorted({s.category for s in SCENARIOS}),
      "metrics": {"correct_updates": ..., "supported_retained": ..., "unsupported_present": ...},
      "restart_recall": ..., "context_departure_recall": ...,
      "oversized": {"decisive_tail_in_budget": ..., "full_event_searchable": ...},
      "costs": costs,
      "limitations": (
        "Reference extractions are labels, not model output. Exact-text checks are "
        "retrieval diagnostics, not semantic answer correctness. The oversized-tail "
        "scenario shows the decisive fact is dropped from the extraction budget under "
        "the current head-first packing (see #65); the reference label includes it only "
        "because it is scripted. Reported usage may omit failed requests and provider "
        "retries. Live semantic accuracy requires run_live with an explicit model."),
      "scenarios": [per-scenario evidence, see below],
    }
    ```
11. **Per-scenario evidence** (`report["scenarios"]`): a list of dicts, one per
    scenario, each `{"name", "category", "current_knowledge": [f"{r.subject} {r.predicate}: {r.value}" for the scenario's transition subjects], "recall": {q.question: [entry.text for entry in memory.recall(q.question, budget_tokens).entries[:3]]}}` — enough to review why each judgement held.
12. Write `report.json` (`json.dumps(report, indent=2)`), return `report`.

Imports needed at top: `json`, `time` (optional), `from pathlib import Path`,
`from datetime import datetime, timezone, timedelta`, `from theseus.layer_store import load_lines`,
`from theseus.memory_module import MemoryModule, Episode`, `from theseus.stimulus_log import StimulusLog`.

- [ ] **Step 4: Run to verify it passes**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -k run_offline -v`
Expected: 3 passed.

- [ ] **Step 5: Run the whole new test file**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/theseus/memory_accuracy_eval.py tests/test_memory_accuracy_eval.py
git commit -m "feat(eval): run_offline with restart, context-departure, and oversized-tail detection (#63)"
```

---

## Task 5: `run_live` and the CLI

**Files:**
- Modify: `src/theseus/memory_accuracy_eval.py`
- Test: `tests/test_memory_accuracy_eval.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from theseus.memory_accuracy_eval import run_live, main


class _ScriptedProvider:
    """A fake live provider — no network. Returns a valid extraction per episode
    and a JSON answer that cites an id the harness never offered."""
    model = "scripted-live"
    last_chat_usage = {"total_tokens": 7}

    def chat(self, prompt, **kwargs):
        if "assertions" in prompt or "consolidation" in prompt:
            return json.dumps({"summary": "ok", "assertions": [
                {"kind": "fact", "subject": "Atlas", "predicate": "phase",
                 "value": "production", "statement": "Atlas phase: production."}]})
        return json.dumps({"answer": "production", "evidence_ids": ["invented-id"]})


def test_run_live_stamps_mode_and_never_treats_citations_as_proof(tmp_path):
    provider = _ScriptedProvider()
    report = run_live(tmp_path, extractor=provider, answerer=provider)
    assert report["mode"] == "live-extraction"
    assert report["provider"] == "scripted-live" or report["model"] == "scripted-live"
    assert report["costs"]["answer_chat_calls"] > 0
    assert report["costs"]["reported_answer_usage"]              # usage recorded when present
    # Every answer that cited an unknown id is flagged, never counted as support.
    flagged = [a for a in report["answers"] if not a["valid_citations"]]
    assert flagged
    assert "not" in report["limitations"].lower() and "proof" in report["limitations"].lower()


def test_main_offline_writes_a_report(tmp_path, capsys):
    main(["--workdir", str(tmp_path / "run")])
    assert json.loads((tmp_path / "run" / "report.json").read_text())["mode"] == "reference-extraction-offline"
```

- [ ] **Step 2: Run to verify it fails**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -k "live or main" -v`
Expected: FAIL with ImportError for `run_live` / `main`.

- [ ] **Step 3: Implement `run_live` and `main`**

**`run_live(workdir, *, extractor, embedder=None, answerer=None, budget_tokens=2000)` -> dict:**
- Same empty-workdir guard and `validate_scenarios` as `run_offline`.
- `build_memory(workdir, extractor=extractor, embedder=embedder)`.
- `drive_scenarios(..., reference=False, extractor=extractor, ...)` — the real
  model produces assertions; the driver does not set `extractor.response`.
- Compute the same three metrics + restart + context-departure exactly as
  `run_offline` (factor the shared body into a private helper
  `_evaluate(memory, log, workdir, extractor, budget_tokens)` returning the
  metrics/restart/departure/oversized/costs blocks, and call it from both runs —
  DRY). In `run_live` add to costs:
  `embedding_calls_at_consolidation` from traces (already there) and, if
  `embedder` is provided, `getattr(embedder, "model", None)` as `report["embedding"]`.
- **Answers (optional):** when `answerer` is given, for each scenario query call
  `answerer.chat(<answer prompt>, max_tokens=512)` where the answer prompt asks
  the model to answer only from the recalled records and return JSON
  `{"answer", "evidence_ids"}` — reuse the exact prompt wording from
  `memory_experiment_eval.py`. Parse with
  `theseus.json_utils.parse_json_response`. For each, record
  `{"scenario", "question", "answer", "valid_citations", "expected_present", "forbidden_present"}`
  where `valid_citations` is True only if every cited id is among the ids of the
  records actually shown to the model, and `expected_present` / `forbidden_present`
  are substring diagnostics **recorded, never asserted as semantic proof**.
  Collect provider usage via `getattr(answerer, "last_chat_usage", None)` into
  `costs["reported_answer_usage"]`; set `costs["answer_chat_calls"]` to the number
  of answer calls. When `answerer` is None, `answers = []` and
  `answer_chat_calls = 0`.
- `report["mode"] = "live-extraction"`,
  `report["provider"] = getattr(extractor, "model", type(extractor).__name__)`,
  `report["model"] = getattr(extractor, "model", None)`.
- `report["limitations"]` must include the sentence:
  `"Substring matches and valid citation IDs are recorded for review but are not treated as proof of semantic support."`
- Write `report.json`, return `report`.

**`main(argv=None)`:** argparse mirroring `memory_experiment_eval.py`:
- `--workdir` (required, `Path`), `--provider` / `--model`,
  `--embedding-provider` / `--embedding-model`, `--answers` (store_true).
- Validate: provider and model must be given together (same for embedding pair);
  `--answers`/embedding require a provider — reuse the existing eval's
  `parser.error(...)` messages.
- With no provider → `run_offline(args.workdir)`. With a provider →
  build providers from `PROVIDER_REGISTRY` (import from `theseus.model_providers`)
  and call `run_live(...)` passing `answerer=extractor if args.answers else None`.
- Print `json.dumps({k: v for k, v in report.items() if k not in ("results", "scenarios")}, indent=2)`.
- Guard the module entrypoint with `if __name__ == "__main__": main()`.

- [ ] **Step 4: Run to verify it passes**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -k "live or main" -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/theseus/memory_accuracy_eval.py tests/test_memory_accuracy_eval.py
git commit -m "feat(eval): opt-in live run + CLI, citations never counted as semantic proof (#63)"
```

---

## Task 6: Docs pointer and full-suite verification

**Files:**
- Modify: `docs/memory-reliability.md`
- Verify: whole offline suite

- [ ] **Step 1: Add a docs pointer**

Append a short subsection to `docs/memory-reliability.md` titled
`### Multi-event consolidation accuracy eval` that states:
- what the module is (`src/theseus/memory_accuracy_eval.py`) and that it is a
  second, parallel eval focused on knowledge transitions (the older
  `memory_experiment_eval.py` keeps the raw-lexical retrieval control);
- how to run it offline: `env -u VIRTUAL_ENV poetry run python -m theseus.memory_accuracy_eval --workdir /tmp/acc-run`;
- how to run it live (explicitly selected, not CI):
  `... --provider <p> --model <m> [--embedding-provider <p> --embedding-model <m>] [--answers]`;
- that the three metrics (correct updates, supported retention, unsupported
  claims) are reported separately, and that substring/citation matches are
  diagnostics, not proof of semantic support.

- [ ] **Step 2: Run the new test file**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py -v`
Expected: all pass (Tasks 1–5 tests green).

- [ ] **Step 3: Run the memory + eval regression subset**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/test_memory_accuracy_eval.py tests/test_memory_experiment_eval.py tests/test_memory_agent_lifecycle.py tests/test_memory_reliability.py -q`
Expected: all pass — the new module must not disturb the existing evals.

- [ ] **Step 4: Run the full offline suite**

Run: `env -u VIRTUAL_ENV poetry run pytest tests/ -q --ignore=tests/test_fact_retention.py`
Expected: pass (this ignores the one test that needs a live endpoint; see memory
`make-test-hits-live-endpoints`). If other tests hang on a live endpoint,
narrow with additional `--ignore` and note which were skipped.

- [ ] **Step 5: Commit**

```bash
git add docs/memory-reliability.md
git commit -m "docs: point to the multi-event consolidation accuracy eval (#63)"
```

---

## Self-Review

**Spec coverage:**
- Scenario coverage of all eight acceptance categories → Task 1 dataset +
  `REQUIRED_CATEGORIES` coverage test. (plan→failure/success are two scenarios
  for one bullet; corrections, coexisting prefs, historical, oversized-tail,
  recall repetition, split action/result, repeated+contradictory principles all
  present.)
- Report supported retention / unsupported claims / correct updates separately,
  retain per-scenario evidence → Task 3 metrics + Task 4 `report["scenarios"]`.
- Recall after restart and after source events leave live context → Task 4
  `restart_recall` + `context_departure_recall`.
- Distinguish reference-extraction from live runs; don't treat substring or
  citation IDs as proof → Task 4 `mode` stamp + Task 5 `run_live` answers block
  and limitations sentence + the `_ScriptedProvider` citation test.
- Record provider/model, request counts, usage, limitations; live explicitly
  selected → Task 4/5 `provider`/`model`/`costs`/`limitations`; live only via
  `run_live`/`--provider`, never in offline tests.
- Decisive evidence at end of oversized events → Task 4 oversized detection.

**Placeholder scan:** none — every metric, key, and scenario value is specified;
logic is specified as algorithm + signatures per the local-sdd handoff note.

**Type consistency:** `EpisodeSpec.event_indices`, `Scenario.events` (4-tuple
with role), `Transition(subject, predicate, current_value, superseded_values)`,
`Query(question, expected, forbidden)`, and the metric return dict keys
(`passed`/`total`/`failures`, `violations`/`total`/`failures`) are used
identically across Tasks 1–5. `build_memory`/`drive_scenarios`/`run_offline`/
`run_live`/`main` signatures match their call sites in the tests.
