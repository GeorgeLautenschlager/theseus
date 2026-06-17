# TheseusAgent Composition & Run-Loop Design

**Date:** 2026-06-16
**Status:** Approved
**Builds on:** [2026-06-15-theseus-agent-design.md](2026-06-15-theseus-agent-design.md)

## Overview

This change makes `TheseusAgent` the owner of composition wiring and the run loop, so that a concrete agent is expressed as a thin subclass that only declares its modules. It also relocates the OODA-stage implementations that currently live inline in `simple_agent.py` into per-protocol module directories, and moves ownership of the input loop from the Textual `AgentApp` into the transport (UI) module.

The goal is the "modular construction kit" described in [notes_on_organization.md](../../notes_on_organization.md): clone the repo, drop a new file next to `simple_agent.py`, declare the modules appropriate for that agent's purpose, and run it.

`simple_agent.py` is a **reference implementation** — the smallest possible example of standing up a `TheseusAgent`. It is not the composition root; `TheseusAgent` is.

## Motivation

Today `simple_agent.py` does the work that should belong to the framework:

- It defines four OODA-stage classes inline (`NaiveObserver`, `NaiveOrienter`, `LLMDecider`, `NaiveActor`).
- It constructs every module and injects them into `TheseusAgent`.
- It constructs the Textual `AgentApp` and runs it, so the agent file knows about the UI framework.

A new agent should not have to repeat any of that. After this change it subclasses `TheseusAgent`, declares modules, and calls `.run()`.

## Architecture

### 1. Agents are subclasses

A concrete agent subclasses `TheseusAgent` and declares its modules as class attributes. The base class owns slot resolution, construction validation, the OODA loop, and `.run()`.

Reference implementation (`src/agents/simple_agent.py`):

```python
class SimpleAgent(TheseusAgent):
    observer = NaiveObserver()
    orienter = NaiveOrienter()
    decider  = LLMDecider(LlamaProvider())
    actor    = NaiveActor()
    memory   = [NaiveContextAssembler()]
    ui       = TextualUI()


if __name__ == "__main__":
    SimpleAgent().run()
```

The reference file's only remaining cost is a stack of explicit imports — accepted deliberately. A construction kit favors explicit module selection over hidden magic.

**Single-instance assumption:** modules declared as class attributes are instantiated at class-definition time and shared across instances of that subclass. These agents are instantiated once per process, so this is acceptable. If multiple live instances of one agent class are ever needed, switch that agent to a subclass `__init__` that calls `super().__init__(...)` with freshly constructed modules. The base constructor already supports this (see slot resolution).

### 2. Slot resolution in `TheseusAgent.__init__`

The constructor resolves each slot in priority order: **explicit keyword argument → subclass class attribute → default/validation**. This keeps the existing direct-injection interface (and all 16 current tests) working unchanged, because an explicitly passed module always wins.

Base class gains class-level slot attributes:

```python
class TheseusAgent:
    observer: Observer | None = None
    orienter: Orienter | None = None
    decider: Decider | None = None
    actor: Actor | None = None
    memory: list[MemoryModule] | None = None
    ui: UI | None = None
    max_cycles: int = 10

    def __init__(self, observer=None, orienter=None, decider=None, actor=None,
                 memory=None, ui=None, max_cycles=None):
        observer = observer if observer is not None else type(self).observer
        orienter = orienter if orienter is not None else type(self).orienter
        decider  = decider  if decider  is not None else type(self).decider
        actor    = actor    if actor    is not None else type(self).actor

        if observer is None:
            raise ValueError("observer is required")
        if orienter is None:
            raise ValueError("orienter is required")
        if decider is None:
            raise ValueError("decider is required")
        if actor is None:
            raise ValueError("actor is required")

        max_cycles = max_cycles if max_cycles is not None else type(self).max_cycles
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

        memory = memory if memory is not None else type(self).memory

        self.observer = observer
        self.orienter = orienter
        self.decider = decider
        self.actor = actor
        self.memory = memory if memory is not None else []
        self.ui = ui if ui is not None else type(self).ui
        self.max_cycles = max_cycles
```

Notes:
- `type(self).<slot>` reads the class attribute, so a subclass's declared module is picked up while a bare `TheseusAgent` resolves to `None` and fails validation as before.
- The four OODA params remain positional-or-keyword, preserving positional call sites.
- No concrete module is imported by the core. `theseus_agent.py` has no dependency on any `modules/*` implementation; subclasses supply them.

### 3. Run loop owned by the transport

`TheseusAgent` gains a `run()` method that hands control to the injected transport:

```python
def run(self) -> None:
    if self.ui is None:
        raise RuntimeError(
            "no transport configured; set `ui` or call process() directly"
        )
    self.ui.start(self)
```

The `UI` protocol grows a `start` method. It now owns input/lifecycle as well as output rendering:

```python
class UI(Protocol):
    def start(self, agent: "TheseusAgent") -> None: ...   # owns the input loop
    def render(self, content: str) -> None: ...           # called by process() on emit
```

Division of responsibility:
- **Input / lifecycle:** the transport's `start(agent)` owns the loop and calls `agent.process(text)` for each input.
- **Output:** `process()` continues to call `self.ui.render(result)` on emit. Output remains a push from the agent; the transport does not separately display the return value.

A future network transport (`ServerUI`) implements the same two methods (`start` listens on a host/port; `render` writes to the connected client) and drops in as a module with no change to `TheseusAgent`. Only the local Textual transport is built in this change.

The protocol keeps the name `UI`. It now owns the loop, so `Transport` would be more precise, but renaming is deferred — out of scope here.

### 4. Transport absorbs `AgentApp`

`TextualUI` gains `start()`, which constructs and runs the Textual app. `AgentApp` becomes an internal detail of the `user_interfaces` module — nothing outside it constructs `AgentApp` anymore.

```python
class TextualUI:
    def __init__(self) -> None:
        self._log = None

    def start(self, agent) -> None:
        AgentApp(agent, self).run()

    def set_log(self, log) -> None:
        self._log = log

    def render(self, content) -> None:
        if self._log is not None:
            self._log.write(f"[bold]agent:[/bold] {content}")
```

`AgentApp` is unchanged in behavior: on mount it registers the `RichLog` with the `TextualUI` via `set_log`; on input submit it echoes the user line, calls `agent.process(text)` (which renders the reply via `set_log`'d `RichLog`), and clears the input.

### 5. Module relocation

The four OODA-stage classes move out of `simple_agent.py` into per-protocol directories, mirroring the existing `memory/`, `model_providers/`, and `user_interfaces/` convention (one directory per protocol; file named after the concrete class; no `__init__.py`, consistent with current implicit-namespace-package usage).

```
src/modules/
├── theseus_agent.py              # core: protocols, dataclasses, TheseusAgent (loop + run)
├── observers/
│   └── naive_observer.py         # NaiveObserver
├── orienters/
│   └── naive_orienter.py         # NaiveOrienter
├── deciders/
│   └── llm_decider.py            # LLMDecider
├── actors/
│   └── naive_actor.py            # NaiveActor
├── memory/
│   └── naive_context_assembler.py
├── model_providers/
│   ├── llama_provider.py
│   └── lm_studio_provider.py
├── user_interfaces/
│   └── textual_ui.py             # TextualUI + (internal) AgentApp
└── context_assemblers/           # reserved: a distinct future concept, left empty
```

The relocated class bodies are unchanged from their current form in `simple_agent.py`; only their location changes. Each imports its transfer objects and `MemoryModule` from `theseus_agent`.

`context_assemblers/` is intentionally left empty. It is reserved for a concept distinct from the Orienter and is out of scope for this change.

## Out of Scope

- The network/IP:PORT transport — designed for as a future drop-in, not built here.
- Renaming the `UI` protocol to `Transport`.
- Defining a `ModelProvider` protocol. `LLMDecider` currently type-hints `LlamaProvider`; loosening that to a structural provider type is a reasonable future change but not required here.
- Renaming `NaiveContextAssembler` or relocating it out of `memory/`.

## Error Handling

- **`run()` with no transport** raises `RuntimeError` with a clear message. `ui` remains optional by design (headless/library use calls `process()` directly and never `run()`), so this is a call-time check, not construction validation.
- **Construction validation** is unchanged: missing `observer`/`orienter`/`decider`/`actor` (after slot resolution) raises `ValueError`; `max_cycles < 1` raises `ValueError`.
- **The OODA loop** (`process`) is untouched, including `MaxCyclesExceeded` and the `ui.render` on emit.

## Testing

The existing 16 tests in `tests/test_theseus_agent.py` must continue to pass unchanged — slot resolution is designed to be backward compatible with direct injection.

New tests:

1. **`run()` delegates to transport** — `ui = Mock()`; build an agent with `ui=ui`; `agent.run()`; assert `ui.start.assert_called_once_with(agent)`.
2. **`run()` without transport raises** — agent with `ui=None`; `agent.run()` raises `RuntimeError`.
3. **Subclass declares modules via class attributes** — define a throwaway `TheseusAgent` subclass whose class attributes are `Mock`s; instantiate with no args; assert each resolved slot is the declared mock and `memory`/`max_cycles` resolve correctly.
4. **Explicit kwarg overrides class attribute** — instantiate the same subclass passing one slot explicitly; assert the explicit value wins over the class attribute.

`TextualUI.start()` launches a real TUI and remains untested, consistent with the current absence of UI tests.

## Migration Summary

| File | Change |
|---|---|
| `src/modules/theseus_agent.py` | Add class-level slots + slot resolution in `__init__`; add `run()`; add `start` to `UI` protocol. `process()` unchanged. |
| `src/modules/user_interfaces/textual_ui.py` | Add `TextualUI.start()`; `AgentApp` becomes internal (no external constructors). |
| `src/modules/observers/naive_observer.py` | New — `NaiveObserver` moved from `simple_agent.py`. |
| `src/modules/orienters/naive_orienter.py` | New — `NaiveOrienter` moved from `simple_agent.py`. |
| `src/modules/deciders/llm_decider.py` | New — `LLMDecider` moved from `simple_agent.py`. |
| `src/modules/actors/naive_actor.py` | New — `NaiveActor` moved from `simple_agent.py`. |
| `src/agents/simple_agent.py` | Becomes `class SimpleAgent(TheseusAgent)` with class-attribute module declarations + `SimpleAgent().run()`. |
| `tests/test_theseus_agent.py` | Add the four new tests above. |
