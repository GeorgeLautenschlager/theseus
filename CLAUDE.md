# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Theseus is a **modular construction kit for cognitive language agents**. The intended workflow: clone the repo, add a new agent file alongside `src/agents/simple_agent.py`, declare whichever modules suit that agent's purpose, and run it. `TheseusAgent` does the composition wiring; an agent file just selects parts.

## Commands

There is no packaging file (no `requirements.txt`/`pyproject.toml`). Runtime deps: `openai`, `textual`. Dev: `pytest`. Python 3.12.

```bash
# Run the full test suite (from repo root)
python -m pytest tests/ -q

# Run a single test
python -m pytest tests/test_theseus_agent.py::TestRun::test_run_delegates_to_transport -v

# Run the reference agent (launches a Textual TUI; needs the local LLM
# server reachable at the URL configured in the provider, e.g. LlamaProvider)
python src/agents/simple_agent.py
```

Tests are Mock-based and target `TheseusAgent` directly — no real modules or network needed.

## Architecture

The big picture lives across `src/modules/theseus_agent.py` (the core) and the per-protocol module directories. Read the core first.

### TheseusAgent is the composition container

`src/modules/theseus_agent.py` owns everything structural: the OODA loop, slot resolution, run-loop entry, the Protocols, and the transfer objects (`Observation`, `Orientation`, `Action`). **It is the thing that handles composition wiring** — agent files do not wire modules together by hand.

`TheseusAgent.__init__` resolves each slot in priority order: **explicit kwarg → subclass class attribute → validation**. Direct injection (`TheseusAgent(observer=..., ...)`) still works, which is why the core's own tests use it.

### A concrete agent is a thin subclass

An agent subclasses `TheseusAgent` and declares its modules as class attributes. `src/agents/simple_agent.py` is the **reference implementation** — the smallest possible example, not the composition root:

```python
class SimpleAgent(TheseusAgent):
    observer = NaiveObserver()
    orienter = NaiveOrienter()
    decider  = LLMDecider(LlamaProvider())
    actor    = NaiveActor()
    memory   = [NaiveContextAssembler()]
    ui       = TextualUI()
```

To build a new agent, add a sibling file in `src/agents/` that does the same with different modules. Don't push wiring or run-loop logic back into the agent file.

### The OODA loop

`process(user_input)` runs Observe → Orient → Decide → Act each cycle until an `Action` has `emit=True` (or `max_cycles` is hit → `MaxCyclesExceeded`). Each phase is a structural `Protocol` — `Observer`, `Orienter`, `Decider`, `Actor` — and every phase receives the shared `memory: list[MemoryModule]`, so any phase can read/write memory. Output is pushed via `ui.render(...)` on emit; the actor stays unaware of presentation.

### The transport owns the input loop

`TheseusAgent.run()` delegates to the transport: `self.ui.start(self)` (and raises `RuntimeError` if no transport is set — headless/library use calls `process()` directly and never `run()`). The `UI` protocol therefore has **two** methods: `start(agent)` (owns the input loop) and `render(content)` (output). `TextualUI.start()` builds and runs the Textual app; `AgentApp` is an internal detail of `user_interfaces/textual_ui.py` and is constructed nowhere else. A future network/server transport is just another module implementing the same `UI` protocol — no change to `TheseusAgent`.

### Module organization

`src/modules/` has **one directory per pluggable protocol category**, each holding one concrete class per file (file named after the class in snake_case):

```
observers/  orienters/  deciders/  actors/      # the four OODA stages
memory/  model_providers/  user_interfaces/     # MemoryModule, providers, transports
```

- **No `__init__.py`** anywhere — these are implicit namespace packages. Keep it that way.
- **Dependency direction is one-way:** concrete modules import transfer objects/protocols *from* `theseus_agent.py`; the core imports **no** concrete module. Don't introduce a core→module import.

## Conventions

- Every module starts with `from __future__ import annotations`. Annotations are never evaluated at runtime, so forward references (e.g. `UI.start(self, agent: TheseusAgent)`) and annotation-only imports are fine.
- **Single-instance assumption:** modules declared as subclass class attributes are instantiated once at class-definition time and shared across instances of that subclass. Fine for these single-instance agents. If an agent ever needs multiple live instances, give it a subclass `__init__` that calls `super().__init__(...)` with freshly constructed modules.
- New work goes through the spec → plan → implement flow under `docs/superpowers/` (see Reference docs).

## Deliberate decisions — do NOT "fix" these

These are intentional and were considered; changing them is scope creep unless explicitly asked:

- The `UI` protocol keeps its name even though it now owns the input loop (not renamed to `Transport`).
- `LLMDecider` type-hints the concrete `LlamaProvider`; there is no `ModelProvider` protocol yet.
- `src/modules/context_assemblers/` is **reserved and intentionally empty** — a concept distinct from the Orienter. Leave it.
- `NaiveContextAssembler` lives under `memory/` (it implements `MemoryModule`) despite its name.
- The root-level `agent.py` is the original pre-framework prototype and is dead relative to this architecture — it is **not** the entry point (that's `src/agents/simple_agent.py`). Slated for eventual removal.

## Reference docs

- `docs/notes_on_organization.md` — the construction-kit vision.
- `docs/superpowers/specs/2026-06-16-theseus-composition-and-run-loop-design.md` — the composition + run-loop design (this architecture).
- `docs/superpowers/specs/2026-06-15-theseus-agent-design.md` — the original `TheseusAgent` core design.
- `docs/superpowers/plans/` — implementation plans.
