# TheseusAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:local-subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `TheseusAgent` — the OODA-loop container that wires together Observer, Orienter, Decider, Actor, memory modules, and a UI component into a decision-controlled agent.

**Architecture:** Strict Protocol-based slots for all four OODA phases. TheseusAgent owns the loop (`process(user_input) -> str`) and calls phases in order, passing `memory` to each. The Decider controls termination via `Action.emit`.

**Tech Stack:** Python 3.12, `dataclasses`, `typing.Protocol`, `pytest`, `unittest.mock`

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `src/modules/theseus_agent.py` | Modify | All protocols, transfer objects, `MaxCyclesExceeded`, `TheseusAgent` |
| `tests/test_theseus_agent.py` | Create | All tests for `TheseusAgent` |
| `conftest.py` | Create | Add project root to `sys.path` so tests can import `src.*` |

---

## Task 1: Type Definitions

Write all dataclasses, the exception class, and all Protocols into `src/modules/theseus_agent.py`. No behavior here — no tests needed.

**Files:**
- Modify: `src/modules/theseus_agent.py`

- [ ] **Step 1: Replace file contents with type definitions**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Observation:
    user_input: str
    memory_context: str
    cycle: int


@dataclass
class Orientation:
    observation: Observation
    context: str


@dataclass
class Action:
    emit: bool
    response: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    thought: str | None = None


class MaxCyclesExceeded(Exception):
    def __init__(self, cycles: int, last_action: Action):
        self.cycles = cycles
        self.last_action = last_action
        super().__init__(f"Agent exceeded {cycles} OODA cycles without emitting")


class MemoryModule(Protocol):
    def retrieve(self, query: str) -> str: ...
    def store(self, content: str) -> None: ...


class UI(Protocol):
    def render(self, content: str) -> None: ...


class Observer(Protocol):
    def observe(self, user_input: str, memory: list[MemoryModule], cycle: int) -> Observation: ...


class Orienter(Protocol):
    def orient(self, observation: Observation, memory: list[MemoryModule]) -> Orientation: ...


class Decider(Protocol):
    def decide(self, orientation: Orientation, memory: list[MemoryModule]) -> Action: ...


class Actor(Protocol):
    def act(self, action: Action, memory: list[MemoryModule]) -> str: ...
```

- [ ] **Step 2: Verify the file parses cleanly**

```bash
cd /home/aldric/theseus && python -c "from src.modules.theseus_agent import Observation, Orientation, Action, MaxCyclesExceeded, Observer, Orienter, Decider, Actor, MemoryModule, UI; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/modules/theseus_agent.py
git commit -m "feat: add OODA transfer objects, protocols, and MaxCyclesExceeded"
```

---

## Task 2: TDD — TheseusAgent Construction

Write construction tests, then implement `TheseusAgent.__init__`.

**Files:**
- Create: `conftest.py`
- Create: `tests/test_theseus_agent.py`
- Modify: `src/modules/theseus_agent.py`

- [ ] **Step 1: Create `conftest.py` at repo root**

```python
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
```

- [ ] **Step 2: Write construction tests**

Create `tests/test_theseus_agent.py`:

```python
import pytest
from unittest.mock import Mock
from src.modules.theseus_agent import (
    TheseusAgent, Observation, Orientation, Action, MaxCyclesExceeded,
)


def _mocks():
    return Mock(), Mock(), Mock(), Mock()


class TestInit:
    def test_missing_observer_raises(self):
        _, orienter, decider, actor = _mocks()
        with pytest.raises(ValueError, match="observer"):
            TheseusAgent(None, orienter, decider, actor)

    def test_missing_orienter_raises(self):
        observer, _, decider, actor = _mocks()
        with pytest.raises(ValueError, match="orienter"):
            TheseusAgent(observer, None, decider, actor)

    def test_missing_decider_raises(self):
        observer, orienter, _, actor = _mocks()
        with pytest.raises(ValueError, match="decider"):
            TheseusAgent(observer, orienter, None, actor)

    def test_missing_actor_raises(self):
        observer, orienter, decider, _ = _mocks()
        with pytest.raises(ValueError, match="actor"):
            TheseusAgent(observer, orienter, decider, None)

    def test_max_cycles_zero_raises(self):
        observer, orienter, decider, actor = _mocks()
        with pytest.raises(ValueError, match="max_cycles"):
            TheseusAgent(observer, orienter, decider, actor, max_cycles=0)

    def test_max_cycles_negative_raises(self):
        observer, orienter, decider, actor = _mocks()
        with pytest.raises(ValueError, match="max_cycles"):
            TheseusAgent(observer, orienter, decider, actor, max_cycles=-1)

    def test_defaults(self):
        observer, orienter, decider, actor = _mocks()
        agent = TheseusAgent(observer, orienter, decider, actor)
        assert agent.memory == []
        assert agent.ui is None
        assert agent.max_cycles == 10

    def test_all_args_stored(self):
        observer, orienter, decider, actor = _mocks()
        memory = [Mock()]
        ui = Mock()
        agent = TheseusAgent(observer, orienter, decider, actor, memory=memory, ui=ui, max_cycles=5)
        assert agent.observer is observer
        assert agent.orienter is orienter
        assert agent.decider is decider
        assert agent.actor is actor
        assert agent.memory is memory
        assert agent.ui is ui
        assert agent.max_cycles == 5
```

- [ ] **Step 3: Run tests — expect failures**

```bash
cd /home/aldric/theseus && python -m pytest tests/test_theseus_agent.py::TestInit -v
```

Expected: all tests FAIL with `ImportError` or `TypeError` (TheseusAgent not implemented yet)

- [ ] **Step 4: Implement `TheseusAgent.__init__`**

Append to `src/modules/theseus_agent.py`:

```python


class TheseusAgent:
    def __init__(
        self,
        observer: Observer,
        orienter: Orienter,
        decider: Decider,
        actor: Actor,
        memory: list[MemoryModule] | None = None,
        ui: UI | None = None,
        max_cycles: int = 10,
    ):
        if observer is None:
            raise ValueError("observer is required")
        if orienter is None:
            raise ValueError("orienter is required")
        if decider is None:
            raise ValueError("decider is required")
        if actor is None:
            raise ValueError("actor is required")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

        self.observer = observer
        self.orienter = orienter
        self.decider = decider
        self.actor = actor
        self.memory: list[MemoryModule] = memory if memory is not None else []
        self.ui = ui
        self.max_cycles = max_cycles
```

- [ ] **Step 5: Run tests — expect all pass**

```bash
cd /home/aldric/theseus && python -m pytest tests/test_theseus_agent.py::TestInit -v
```

Expected: 8 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add conftest.py tests/test_theseus_agent.py src/modules/theseus_agent.py
git commit -m "feat: implement TheseusAgent.__init__ with construction validation"
```

---

## Task 3: TDD — TheseusAgent.process()

Write tests for the OODA loop, then implement `process()`.

**Files:**
- Modify: `tests/test_theseus_agent.py`
- Modify: `src/modules/theseus_agent.py`

- [ ] **Step 1: Append process() tests to `tests/test_theseus_agent.py`**

```python


class TestProcess:
    def _obs(self, cycle=0):
        return Observation(user_input="hi", memory_context="", cycle=cycle)

    def _ori(self, obs=None):
        obs = obs or self._obs()
        return Orientation(observation=obs, context="ctx")

    def test_single_cycle_returns_actor_result(self):
        obs = self._obs(0)
        ori = self._ori(obs)
        action = Action(emit=True)

        observer = Mock()
        observer.observe = Mock(return_value=obs)
        orienter = Mock()
        orienter.orient = Mock(return_value=ori)
        decider = Mock()
        decider.decide = Mock(return_value=action)
        actor = Mock()
        actor.act = Mock(return_value="hello")

        agent = TheseusAgent(observer, orienter, decider, actor)
        assert agent.process("hi") == "hello"

    def test_single_cycle_each_phase_called_once_with_correct_args(self):
        obs = self._obs(0)
        ori = self._ori(obs)
        action = Action(emit=True)

        observer = Mock()
        observer.observe = Mock(return_value=obs)
        orienter = Mock()
        orienter.orient = Mock(return_value=ori)
        decider = Mock()
        decider.decide = Mock(return_value=action)
        actor = Mock()
        actor.act = Mock(return_value="hello")

        agent = TheseusAgent(observer, orienter, decider, actor)
        agent.process("hi")

        observer.observe.assert_called_once_with("hi", [], 0)
        orienter.orient.assert_called_once_with(obs, [])
        decider.decide.assert_called_once_with(ori, [])
        actor.act.assert_called_once_with(action, [])

    def test_multi_cycle_each_phase_called_n_times(self):
        obs = self._obs(0)
        ori = self._ori(obs)

        call_count = 0

        def decide_side_effect(orientation, memory):
            nonlocal call_count
            call_count += 1
            return Action(emit=(call_count == 3))

        observer = Mock()
        observer.observe = Mock(return_value=obs)
        orienter = Mock()
        orienter.orient = Mock(return_value=ori)
        decider = Mock()
        decider.decide = Mock(side_effect=decide_side_effect)
        actor = Mock()
        actor.act = Mock(return_value="done")

        agent = TheseusAgent(observer, orienter, decider, actor)
        result = agent.process("hi")

        assert result == "done"
        assert observer.observe.call_count == 3
        assert orienter.orient.call_count == 3
        assert decider.decide.call_count == 3
        assert actor.act.call_count == 3

    def test_cycle_counter_increments_each_pass(self):
        cycle_args = []

        def observe_side_effect(user_input, memory, cycle):
            cycle_args.append(cycle)
            return Observation(user_input=user_input, memory_context="", cycle=cycle)

        call_count = 0

        def decide_side_effect(orientation, memory):
            nonlocal call_count
            call_count += 1
            return Action(emit=(call_count == 2))

        observer = Mock()
        observer.observe = Mock(side_effect=observe_side_effect)
        orienter = Mock()
        orienter.orient = Mock(
            side_effect=lambda obs, mem: Orientation(observation=obs, context="")
        )
        decider = Mock()
        decider.decide = Mock(side_effect=decide_side_effect)
        actor = Mock()
        actor.act = Mock(return_value="done")

        agent = TheseusAgent(observer, orienter, decider, actor)
        agent.process("hi")

        assert cycle_args == [0, 1]

    def test_max_cycles_exceeded_raises_with_metadata(self):
        obs = self._obs(0)
        ori = self._ori(obs)
        last_action = Action(emit=False)

        observer = Mock()
        observer.observe = Mock(return_value=obs)
        orienter = Mock()
        orienter.orient = Mock(return_value=ori)
        decider = Mock()
        decider.decide = Mock(return_value=last_action)
        actor = Mock()
        actor.act = Mock(return_value="unused")

        agent = TheseusAgent(observer, orienter, decider, actor, max_cycles=3)

        with pytest.raises(MaxCyclesExceeded) as exc_info:
            agent.process("hi")

        assert exc_info.value.cycles == 3
        assert exc_info.value.last_action is last_action

    def test_memory_passed_to_all_phases(self):
        memory = [Mock()]
        obs = self._obs(0)
        ori = self._ori(obs)
        action = Action(emit=True)

        observer = Mock()
        observer.observe = Mock(return_value=obs)
        orienter = Mock()
        orienter.orient = Mock(return_value=ori)
        decider = Mock()
        decider.decide = Mock(return_value=action)
        actor = Mock()
        actor.act = Mock(return_value="done")

        agent = TheseusAgent(observer, orienter, decider, actor, memory=memory)
        agent.process("hi")

        observer.observe.assert_called_once_with("hi", memory, 0)
        orienter.orient.assert_called_once_with(obs, memory)
        decider.decide.assert_called_once_with(ori, memory)
        actor.act.assert_called_once_with(action, memory)

    def test_ui_render_called_on_emit(self):
        obs = self._obs(0)
        ori = self._ori(obs)
        action = Action(emit=True)

        observer = Mock()
        observer.observe = Mock(return_value=obs)
        orienter = Mock()
        orienter.orient = Mock(return_value=ori)
        decider = Mock()
        decider.decide = Mock(return_value=action)
        actor = Mock()
        actor.act = Mock(return_value="rendered output")
        ui = Mock()

        agent = TheseusAgent(observer, orienter, decider, actor, ui=ui)
        agent.process("hi")

        ui.render.assert_called_once_with("rendered output")

    def test_ui_not_called_when_none(self):
        obs = self._obs(0)
        ori = self._ori(obs)
        action = Action(emit=True)

        observer = Mock()
        observer.observe = Mock(return_value=obs)
        orienter = Mock()
        orienter.orient = Mock(return_value=ori)
        decider = Mock()
        decider.decide = Mock(return_value=action)
        actor = Mock()
        actor.act = Mock(return_value="output")

        agent = TheseusAgent(observer, orienter, decider, actor, ui=None)
        result = agent.process("hi")

        assert result == "output"
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /home/aldric/theseus && python -m pytest tests/test_theseus_agent.py::TestProcess -v
```

Expected: all tests FAIL with `AttributeError: 'TheseusAgent' object has no attribute 'process'`

- [ ] **Step 3: Implement `TheseusAgent.process()`**

Append to the `TheseusAgent` class in `src/modules/theseus_agent.py` (inside the class body, after `__init__`):

```python
    def process(self, user_input: str) -> str:
        last_action: Action | None = None
        for cycle in range(self.max_cycles):
            observation = self.observer.observe(user_input, self.memory, cycle)
            orientation = self.orienter.orient(observation, self.memory)
            last_action = self.decider.decide(orientation, self.memory)
            result = self.actor.act(last_action, self.memory)
            if last_action.emit:
                if self.ui:
                    self.ui.render(result)
                return result
        raise MaxCyclesExceeded(self.max_cycles, last_action)
```

- [ ] **Step 4: Run all tests — expect all pass**

```bash
cd /home/aldric/theseus && python -m pytest tests/test_theseus_agent.py -v
```

Expected: 16 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/test_theseus_agent.py src/modules/theseus_agent.py
git commit -m "feat: implement TheseusAgent.process() OODA loop"
```
