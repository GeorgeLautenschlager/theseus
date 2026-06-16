# TheseusAgent Design

**Date:** 2026-06-15
**Status:** Approved

## Overview

`TheseusAgent` is the composable container for cognitive language agents in this repo. It owns the core OODA loop (Observe → Orient → Decide → Act) and enforces the architectural decisions that apply to every agent. All behavioral variation is expressed by swapping injected modules.

Public interface: `process(user_input: str) -> str`.

## Architecture

`TheseusAgent` lives in `src/modules/theseus_agent.py`. It holds one instance of each OODA phase module plus a list of memory modules and an optional UI component.

### Constructor Slots

| Slot | Type | Required | Notes |
|---|---|---|---|
| `observer` | `Observer` | Yes | Collects input and retrieves from memory |
| `orienter` | `Orienter` | Yes | Assembles context for the Decider |
| `decider` | `Decider` | Yes | Chooses next action; owns loop termination |
| `actor` | `Actor` | Yes | Executes the action; writes to memory |
| `memory` | `list[MemoryModule]` | No | Defaults to `[]` |
| `ui` | `UI` | No | Defaults to `None` |
| `max_cycles` | `int` | No | Defaults to `10` |

### Core Loop

```python
def process(self, user_input: str) -> str:
    for cycle in range(self.max_cycles):
        observation  = self.observer.observe(user_input, self.memory)
        orientation  = self.orienter.orient(observation, self.memory)
        action       = self.decider.decide(orientation, self.memory)
        result       = self.actor.act(action, self.memory)
        if action.emit:
            if self.ui:
                self.ui.render(result)
            return result
    raise MaxCyclesExceeded(self.max_cycles, action)
```

Each phase receives `memory` directly — all four phases can read from and write to any memory module. The loop continues until `action.emit` is `True` or `max_cycles` is hit.

## Protocols and Data Types

### Transfer Objects

```python
@dataclass
class Observation:
    user_input: str
    memory_context: str   # assembled from memory modules
    cycle: int            # which OODA cycle we're on

@dataclass
class Orientation:
    observation: Observation
    context: str          # fully assembled prompt/context for the Decider

@dataclass
class Action:
    emit: bool
    response: str | None = None      # set when emit=True
    tool_name: str | None = None     # set when calling a tool
    tool_args: dict | None = None
    thought: str | None = None       # internal reasoning / reflection
```

`Action.thought` allows reflection systems to pass internal reasoning to Actor for memory writes without surfacing it to the user.

### Protocols

```python
class Observer(Protocol):
    def observe(self, user_input: str, memory: list[MemoryModule]) -> Observation: ...

class Orienter(Protocol):
    def orient(self, observation: Observation, memory: list[MemoryModule]) -> Orientation: ...

class Decider(Protocol):
    def decide(self, orientation: Orientation, memory: list[MemoryModule]) -> Action: ...

class Actor(Protocol):
    def act(self, action: Action, memory: list[MemoryModule]) -> str: ...

class MemoryModule(Protocol):
    def retrieve(self, query: str) -> str: ...
    def store(self, content: str) -> None: ...

class UI(Protocol):
    def render(self, content: str) -> None: ...
```

All protocols use structural typing (duck typing). `@runtime_checkable` is optional and left to implementors.

## Error Handling

**`MaxCyclesExceeded`** — raised when the loop exhausts `max_cycles` without `action.emit=True`. Carries `cycles: int` and `last_action: Action` for diagnostics.

**Construction-time validation** — `observer`, `orienter`, `decider`, and `actor` are validated at `__init__`. Missing required slots raise `ValueError` immediately rather than failing mid-loop.

**Module exceptions** propagate unchanged. `TheseusAgent` does not catch or wrap errors from phase modules.

## UI Component

The UI is injected as a slot like any other module. It is called by `TheseusAgent` after `action.emit=True` — Actor returns a plain string and remains unaware of the presentation layer. This keeps Actor testable and the UI swappable (TUI, CLI, web, etc.) without touching agent logic.

## Testing

Tests target `TheseusAgent` directly using `unittest.mock.Mock` stubs for all four protocols — no real module implementations needed.

Key cases:
1. **Single-cycle emit** — Decider returns `Action(emit=True)` on cycle 1.
2. **Multi-cycle emit** — Decider loops N times before emitting; assert each phase was called N times.
3. **Max cycles exceeded** — Decider never emits; assert `MaxCyclesExceeded` is raised with correct metadata.
4. **Data flow** — assert each phase receives the return value of the previous phase.
5. **Construction validation** — missing required slots raise `ValueError`.
6. **UI called on emit** — assert `ui.render` is called with Actor's return value when `ui` is set.
7. **UI skipped when None** — no error when `ui=None`.
