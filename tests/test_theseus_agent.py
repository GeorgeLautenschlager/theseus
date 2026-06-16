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
