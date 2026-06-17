# TheseusAgent Composition & Run-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:local-subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `TheseusAgent` own composition wiring and the run loop so a concrete agent is a thin subclass that declares its modules; relocate the inline OODA stages into per-protocol module directories; give the transport ownership of the input loop.

**Architecture:** `TheseusAgent.__init__` resolves each slot as *explicit kwarg → subclass class attribute → validation*, preserving the existing direct-injection interface. A new `run()` delegates to the transport's `start(agent)`. The `UI` protocol gains `start`; `TextualUI` absorbs the Textual app loop. The four naive OODA classes move out of `simple_agent.py`, which becomes a `SimpleAgent(TheseusAgent)` subclass.

**Tech Stack:** Python 3.12, pytest, `unittest.mock`, Textual, OpenAI client.

**Spec:** [docs/superpowers/specs/2026-06-16-theseus-composition-and-run-loop-design.md](../specs/2026-06-16-theseus-composition-and-run-loop-design.md)

---

## File Structure

| File | Responsibility |
|---|---|
| `src/modules/theseus_agent.py` | Core: protocols, transfer objects, `TheseusAgent` (slot resolution, OODA loop, `run()`). Modified. |
| `src/modules/user_interfaces/textual_ui.py` | `TextualUI` transport (owns input loop via `start()`, renders output) + internal `AgentApp`. Modified. |
| `src/modules/observers/naive_observer.py` | `NaiveObserver`. Created (moved). |
| `src/modules/orienters/naive_orienter.py` | `NaiveOrienter`. Created (moved). |
| `src/modules/deciders/llm_decider.py` | `LLMDecider`. Created (moved). |
| `src/modules/actors/naive_actor.py` | `NaiveActor`. Created (moved). |
| `src/agents/simple_agent.py` | Reference agent: `SimpleAgent(TheseusAgent)` declaring modules. Modified. |
| `tests/test_theseus_agent.py` | Existing 16 tests + new slot-resolution and `run()` tests. Modified. |

Note: existing module directories use implicit namespace packages (no `__init__.py`); the new directories follow the same convention. All test commands run from the repo root `/home/aldric/theseus`.

---

### Task 1: Slot resolution in `TheseusAgent.__init__`

**Files:**
- Modify: `src/modules/theseus_agent.py:60-88` (the `TheseusAgent` class header and `__init__`)
- Test: `tests/test_theseus_agent.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_theseus_agent.py`:

```python
class TestSubclassDeclaration:
    def test_subclass_class_attributes_resolve_into_slots(self):
        class Declared(TheseusAgent):
            observer = Mock()
            orienter = Mock()
            decider = Mock()
            actor = Mock()
            memory = [Mock()]
            ui = Mock()
            max_cycles = 7

        agent = Declared()
        assert agent.observer is Declared.observer
        assert agent.orienter is Declared.orienter
        assert agent.decider is Declared.decider
        assert agent.actor is Declared.actor
        assert agent.memory is Declared.memory
        assert agent.ui is Declared.ui
        assert agent.max_cycles == 7

    def test_explicit_kwarg_overrides_class_attribute(self):
        class Declared(TheseusAgent):
            observer = Mock()
            orienter = Mock()
            decider = Mock()
            actor = Mock()

        override = Mock()
        agent = Declared(observer=override)
        assert agent.observer is override
        assert agent.orienter is Declared.orienter
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_theseus_agent.py::TestSubclassDeclaration -v`
Expected: FAIL — `Declared()` raises `TypeError: __init__() missing 4 required positional arguments: 'observer', 'orienter', 'decider', and 'actor'`.

- [ ] **Step 3: Implement slot resolution**

In `src/modules/theseus_agent.py`, replace the `TheseusAgent` class header and `__init__` (current lines 60-88) with:

```python
class TheseusAgent:
    observer: Observer | None = None
    orienter: Orienter | None = None
    decider: Decider | None = None
    actor: Actor | None = None
    memory: list[MemoryModule] | None = None
    ui: UI | None = None
    max_cycles: int = 10

    def __init__(
        self,
        observer: Observer | None = None,
        orienter: Orienter | None = None,
        decider: Decider | None = None,
        actor: Actor | None = None,
        memory: list[MemoryModule] | None = None,
        ui: UI | None = None,
        max_cycles: int | None = None,
    ):
        observer = observer if observer is not None else type(self).observer
        orienter = orienter if orienter is not None else type(self).orienter
        decider = decider if decider is not None else type(self).decider
        actor = actor if actor is not None else type(self).actor

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
        self.memory: list[MemoryModule] = memory if memory is not None else []
        self.ui = ui if ui is not None else type(self).ui
        self.max_cycles = max_cycles
```

The `process()` method below `__init__` is unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_theseus_agent.py -q`
Expected: PASS — 18 passed (16 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/modules/theseus_agent.py tests/test_theseus_agent.py
git commit -m "feat: resolve TheseusAgent slots from subclass class attributes"
```

---

### Task 2: `run()` method and `UI.start` protocol

**Files:**
- Modify: `src/modules/theseus_agent.py` (the `UI` protocol; add `run()` to `TheseusAgent`)
- Test: `tests/test_theseus_agent.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_theseus_agent.py`:

```python
class TestRun:
    def test_run_delegates_to_transport(self):
        observer, orienter, decider, actor = _mocks()
        ui = Mock()
        agent = TheseusAgent(observer, orienter, decider, actor, ui=ui)
        agent.run()
        ui.start.assert_called_once_with(agent)

    def test_run_without_transport_raises(self):
        observer, orienter, decider, actor = _mocks()
        agent = TheseusAgent(observer, orienter, decider, actor, ui=None)
        with pytest.raises(RuntimeError, match="transport"):
            agent.run()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_theseus_agent.py::TestRun -v`
Expected: FAIL — `AttributeError: 'TheseusAgent' object has no attribute 'run'`.

- [ ] **Step 3: Implement `run()` and extend the `UI` protocol**

In `src/modules/theseus_agent.py`, update the `UI` protocol (currently lines 40-41) to:

```python
class UI(Protocol):
    def start(self, agent: TheseusAgent) -> None: ...
    def render(self, content: str) -> None: ...
```

(`from __future__ import annotations` is already at the top of the file, so the forward reference to `TheseusAgent` is not evaluated at runtime.)

Then add a `run()` method to `TheseusAgent`, immediately after `process()`:

```python
    def run(self) -> None:
        if self.ui is None:
            raise RuntimeError(
                "no transport configured; set `ui` or call process() directly"
            )
        self.ui.start(self)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_theseus_agent.py -q`
Expected: PASS — 20 passed.

- [ ] **Step 5: Commit**

```bash
git add src/modules/theseus_agent.py tests/test_theseus_agent.py
git commit -m "feat: add TheseusAgent.run() delegating to transport.start()"
```

---

### Task 3: `TextualUI.start()` owns the loop

**Files:**
- Modify: `src/modules/user_interfaces/textual_ui.py` (add `start()` to `TextualUI`)

No unit test: `start()` launches a real Textual app. It is verified by a smoke import here and the manual run at the end of the plan.

- [ ] **Step 1: Add `start()` to `TextualUI`**

In `src/modules/user_interfaces/textual_ui.py`, add a `start` method to `TextualUI` so the class reads:

```python
class TextualUI:
    def __init__(self) -> None:
        self._log: RichLog | None = None

    def start(self, agent) -> None:
        AgentApp(agent, self).run()

    def set_log(self, log: RichLog) -> None:
        self._log = log

    def render(self, content: str) -> None:
        if self._log is not None:
            self._log.write(f"[bold]agent:[/bold] {content}")
```

`AgentApp` is defined below `TextualUI` in the same module; it is a module global by the time `start()` is called, so the forward reference resolves at runtime. Leave the `AgentApp` class unchanged — it is now constructed only by `TextualUI.start()`.

- [ ] **Step 2: Smoke-verify the import and method**

Run: `python -c "from src.modules.user_interfaces.textual_ui import TextualUI; assert callable(TextualUI.start); print('ok')"`
Expected: prints `ok` with no import error.

- [ ] **Step 3: Commit**

```bash
git add src/modules/user_interfaces/textual_ui.py
git commit -m "feat: TextualUI.start() owns the input loop"
```

---

### Task 4: Relocate OODA stages and convert `simple_agent` to a subclass

**Files:**
- Create: `src/modules/observers/naive_observer.py`
- Create: `src/modules/orienters/naive_orienter.py`
- Create: `src/modules/deciders/llm_decider.py`
- Create: `src/modules/actors/naive_actor.py`
- Modify: `src/agents/simple_agent.py` (replace inline classes + wiring with a subclass)

These changes are grouped so there is no intermediate state with classes defined twice.

- [ ] **Step 1: Create `src/modules/observers/naive_observer.py`**

```python
from __future__ import annotations

from src.modules.theseus_agent import MemoryModule, Observation


class NaiveObserver:
    def observe(self, user_input: str, memory: list[MemoryModule], cycle: int) -> Observation:
        parts = [m.retrieve(user_input) for m in memory]
        context = "\n".join(p for p in parts if p)
        for m in memory:
            m.store(f"User: {user_input}")
        return Observation(user_input=user_input, memory_context=context, cycle=cycle)
```

- [ ] **Step 2: Create `src/modules/orienters/naive_orienter.py`**

```python
from __future__ import annotations

from src.modules.theseus_agent import MemoryModule, Observation, Orientation


class NaiveOrienter:
    def orient(self, observation: Observation, memory: list[MemoryModule]) -> Orientation:
        prefix = f"{observation.memory_context}\n" if observation.memory_context else ""
        context = f"{prefix}User: {observation.user_input}\nAgent:"
        return Orientation(observation=observation, context=context)
```

- [ ] **Step 3: Create `src/modules/deciders/llm_decider.py`**

```python
from __future__ import annotations

from src.modules.model_providers.llama_provider import LlamaProvider
from src.modules.theseus_agent import Action, MemoryModule, Orientation


class LLMDecider:
    def __init__(self, provider: LlamaProvider) -> None:
        self.provider = provider

    def decide(self, orientation: Orientation, memory: list[MemoryModule]) -> Action:
        response = self.provider.chat(orientation.context)
        return Action(emit=True, response=response)
```

- [ ] **Step 4: Create `src/modules/actors/naive_actor.py`**

```python
from __future__ import annotations

from src.modules.theseus_agent import Action, MemoryModule


class NaiveActor:
    def act(self, action: Action, memory: list[MemoryModule]) -> str:
        for m in memory:
            m.store(f"Agent: {action.response}")
        return action.response
```

- [ ] **Step 5: Rewrite `src/agents/simple_agent.py`**

Replace the entire file with:

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.modules.actors.naive_actor import NaiveActor
from src.modules.deciders.llm_decider import LLMDecider
from src.modules.memory.naive_context_assembler import NaiveContextAssembler
from src.modules.model_providers.llama_provider import LlamaProvider
from src.modules.observers.naive_observer import NaiveObserver
from src.modules.orienters.naive_orienter import NaiveOrienter
from src.modules.theseus_agent import TheseusAgent
from src.modules.user_interfaces.textual_ui import TextualUI


class SimpleAgent(TheseusAgent):
    observer = NaiveObserver()
    orienter = NaiveOrienter()
    decider = LLMDecider(LlamaProvider())
    actor = NaiveActor()
    memory = [NaiveContextAssembler()]
    ui = TextualUI()


if __name__ == "__main__":
    SimpleAgent().run()
```

- [ ] **Step 6: Smoke-verify the reference agent constructs**

Run: `python -c "from src.agents.simple_agent import SimpleAgent; from src.modules.theseus_agent import TheseusAgent; a = SimpleAgent(); assert isinstance(a, TheseusAgent); assert a.observer is SimpleAgent.observer; assert a.ui is SimpleAgent.ui; print('ok')"`
Expected: prints `ok`. (Importing instantiates the class-attribute modules, which creates `context.db` in the working directory — expected.)

- [ ] **Step 7: Run the full test suite**

Run: `python -m pytest tests/ -q`
Expected: PASS — 20 passed.

- [ ] **Step 8: Commit**

```bash
git add src/modules/observers/naive_observer.py src/modules/orienters/naive_orienter.py src/modules/deciders/llm_decider.py src/modules/actors/naive_actor.py src/agents/simple_agent.py
git commit -m "refactor: relocate OODA stages into module dirs; SimpleAgent subclass"
```

---

## Manual Verification (after all tasks)

The Textual transport and the live model call cannot be unit-tested. Confirm end-to-end:

Run: `python src/agents/simple_agent.py`
Expected: the TUI launches; typing a line echoes `you: ...`, and after the model responds, `agent: ...` is rendered. Requires the Llama server reachable at the URL configured in `LlamaProvider` (`http://100.126.84.49:8080/v1`). If the server is unreachable, the TUI still launches and input still echoes — only the model reply will error.

---

## Self-Review

**Spec coverage:**
- Subclass agents (spec §1) → Task 4 Step 5.
- Slot resolution kwarg→class-attr→validate (spec §2) → Task 1.
- `run()` + transport-owned loop, `RuntimeError` when no transport (spec §3) → Task 2.
- `UI.start` protocol (spec §3) → Task 2 Step 3.
- `TextualUI.start()` absorbs `AgentApp` (spec §4) → Task 3.
- Module relocation into per-protocol dirs (spec §5) → Task 4 Steps 1-4.
- `context_assemblers/` left untouched (spec §5) → no task touches it (correct).
- Backward-compatible with 16 existing tests → verified in Task 1 Step 4 and Task 4 Step 7.
- New tests (run delegates, run raises, subclass declaration, kwarg override) → Tasks 1-2.

**Placeholder scan:** none — every code step contains complete code; every command has expected output.

**Type consistency:** `start(self, agent)`, `render(self, content)`, `run(self)`, and the slot names (`observer`/`orienter`/`decider`/`actor`/`memory`/`ui`/`max_cycles`) are used identically across the protocol, the core, the transport, and `SimpleAgent`.
