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
