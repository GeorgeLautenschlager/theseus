from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from theseus.agentic_memory import AgenticMemory
from theseus.cognitive_core import CognitiveCore
from theseus.cognitive_prompts import WAIT_ACTION
from theseus.memory_store import MemoryStore
from theseus.stimulus_log import StimulusLog


class StubEffector:
    def __init__(self, name="respond_in_web_chat"):
        self.name = name
        self.description = "Send a chat message to George."
        self.act_instruction = "Compose your chat message now."
        self.executed_with = None

    def execute(self, payload):
        self.executed_with = payload


def make_provider(responses):
    """A stub ModelProvider whose .chat() returns each of `responses` in turn."""
    provider = MagicMock()
    provider.is_available.return_value = True
    provider.chat.side_effect = list(responses)
    return provider


def make_core(tmp_path, provider, effectors=None, memory=None):
    stimulus_log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    return CognitiveCore(
        constitution="You are Tam.",
        model_providers=[provider],
        effectors=effectors or {},
        stimulus_log=stimulus_log,
        memory=memory,
    )


class TestDecide:
    def test_passes_json_schema_to_provider(self, tmp_path):
        effector = StubEffector()
        decision = {"rationale": "Nothing to do.", "action": WAIT_ACTION}
        provider = make_provider([json.dumps(decision)])
        core = make_core(tmp_path, provider, effectors={effector.name: effector})

        core.decide(context="")

        _, kwargs = provider.chat.call_args_list[0]
        assert "json_schema" in kwargs
        assert kwargs["json_schema"]["properties"]["action"]["enum"] == [
            effector.name,
            WAIT_ACTION,
        ]

    def test_logs_decision_event(self, tmp_path):
        effector = StubEffector()
        decision = {"rationale": "Nothing to do.", "action": WAIT_ACTION}
        provider = make_provider([json.dumps(decision)])
        core = make_core(tmp_path, provider, effectors={effector.name: effector})

        core.decide(context="")

        events = core.stimulus_log.read_all()
        decision_events = [e for e in events if e.type == "decision"]
        assert len(decision_events) == 1
        assert decision_events[0].content["action"] == WAIT_ACTION
        assert decision_events[0].content["rationale"] == "Nothing to do."

    def test_parses_fenced_json_response(self, tmp_path):
        effector = StubEffector()
        decision = {"rationale": "Nothing to do.", "action": WAIT_ACTION}
        fenced = f"```json\n{json.dumps(decision)}\n```"
        provider = make_provider([fenced])
        core = make_core(tmp_path, provider, effectors={effector.name: effector})

        core.decide(context="")

        events = core.stimulus_log.read_all()
        assert events[0].content["action"] == WAIT_ACTION


class TestAct:
    def test_wait_decision_makes_no_effector_call_and_only_one_model_call(self, tmp_path):
        effector = StubEffector()
        decision = {"rationale": "Nothing to do.", "action": WAIT_ACTION}
        provider = make_provider([json.dumps(decision)])
        core = make_core(tmp_path, provider, effectors={effector.name: effector})

        core.decide(context="")

        assert provider.chat.call_count == 1
        assert effector.executed_with is None

    def test_effector_decision_executes_effector_with_act_phase_output(self, tmp_path):
        effector = StubEffector()
        decision = {"rationale": "George said hello.", "action": effector.name}
        provider = make_provider([json.dumps(decision), "Hello, George!"])
        core = make_core(tmp_path, provider, effectors={effector.name: effector})

        core.decide(context="")

        assert provider.chat.call_count == 2
        assert effector.executed_with == "Hello, George!"

    def test_effector_decision_logs_action_event(self, tmp_path):
        effector = StubEffector()
        decision = {"rationale": "George said hello.", "action": effector.name}
        provider = make_provider([json.dumps(decision), "Hello, George!"])
        core = make_core(tmp_path, provider, effectors={effector.name: effector})

        core.decide(context="")

        events = core.stimulus_log.read_all()
        action_events = [e for e in events if e.type == "action"]
        assert len(action_events) == 1
        assert action_events[0].content["action"] == effector.name
        assert action_events[0].content["output"] == "Hello, George!"

    def test_act_phase_prompt_includes_rationale_and_act_instruction(self, tmp_path):
        effector = StubEffector()
        decision = {"rationale": "George said hello.", "action": effector.name}
        provider = make_provider([json.dumps(decision), "Hello, George!"])
        core = make_core(tmp_path, provider, effectors={effector.name: effector})

        core.decide(context="")

        act_call_kwargs = provider.chat.call_args_list[1].kwargs
        assert "George said hello." in act_call_kwargs["prompt"]
        assert effector.act_instruction in act_call_kwargs["prompt"]

    def test_unknown_action_neither_raises_nor_calls_effector(self, tmp_path, capsys):
        effector = StubEffector()
        decision = {"rationale": "Confused.", "action": "not_a_real_action"}
        provider = make_provider([json.dumps(decision)])
        core = make_core(tmp_path, provider, effectors={effector.name: effector})

        core.decide(context="")

        assert effector.executed_with is None
        assert provider.chat.call_count == 1

    def test_stimulus_log_actor_uses_core_name(self, tmp_path):
        effector = StubEffector()
        decision = {"rationale": "George said hello.", "action": effector.name}
        provider = make_provider([json.dumps(decision), "Hello, George!"])
        stimulus_log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        core = CognitiveCore(
            constitution="You are Tam.",
            model_providers=[provider],
            effectors={effector.name: effector},
            stimulus_log=stimulus_log,
            name="Aldric",
        )

        core.decide(context="")

        events = core.stimulus_log.read_all()
        assert all(e.actor == "Aldric" for e in events)


WAIT_DECISION = json.dumps({"rationale": "Nothing to do.", "action": WAIT_ACTION})
ENRICHMENT = json.dumps(
    {"context": "George introduced himself.", "keywords": ["George"], "tags": ["identity"]}
)


def make_memory(tmp_path, provider):
    return AgenticMemory(
        model_providers=[provider],
        store=MemoryStore(tmp_path / "memory.jsonl"),
    )


class TestMemoryIntegration:
    def test_orient_forms_a_note_post_cycle(self, tmp_path):
        provider = make_provider([WAIT_DECISION, ENRICHMENT])
        provider.embed.return_value = [1.0, 0.0]
        memory = make_memory(tmp_path, provider)
        core = make_core(tmp_path, provider, memory=memory)
        stimulus = core.stimulus_log.append(
            actor="george", type="exchange", content={"message": "Hello, my name is George."}
        )

        core.orient()

        notes = memory.store.read_all()
        assert len(notes) == 1
        assert notes[0].context == "George introduced himself."
        # The note spans everything in the cycle: the stimulus and the decision it produced.
        decision_event = core.stimulus_log.read_all()[-1]
        assert notes[0].source_span == (stimulus.id, decision_event.id)

    def test_formation_resumes_from_high_water_mark(self, tmp_path):
        provider = make_provider(
            [
                WAIT_DECISION, ENRICHMENT,                       # cycle 1: decide, construct
                WAIT_DECISION, ENRICHMENT, json.dumps({"links": []}),  # cycle 2: + link decision
            ]
        )
        provider.embed.return_value = [1.0, 0.0]
        memory = make_memory(tmp_path, provider)
        core = make_core(tmp_path, provider, memory=memory)

        core.stimulus_log.append(actor="george", type="exchange", content={"message": "First."})
        core.orient()
        second = core.stimulus_log.append(
            actor="george", type="exchange", content={"message": "Second."}
        )
        core.orient()

        notes = memory.store.read_all()
        assert len(notes) == 2
        # The second note starts exactly where the first left off — no re-consolidation.
        assert notes[1].source_span[0] == second.id
        assert "First." not in notes[1].content

    def test_decide_prompt_contains_retrieved_memories(self, tmp_path):
        provider = make_provider(
            [ENRICHMENT, WAIT_DECISION, ENRICHMENT, json.dumps({"links": []})]
        )
        provider.embed.return_value = [1.0, 0.0]
        memory = make_memory(tmp_path, provider)
        core = make_core(tmp_path, provider, memory=memory)
        # Seed one note through the real pipeline, then start a fresh cycle.
        core.stimulus_log.append(actor="george", type="exchange", content={"message": "I am George."})
        memory.form(core.stimulus_log.read_all())
        core.stimulus_log.append(actor="george", type="exchange", content={"message": "Who am I?"})

        core.orient()

        decide_prompt = provider.chat.call_args_list[1].kwargs["prompt"]
        assert "<memories>" in decide_prompt
        assert "George introduced himself." in decide_prompt

    def test_without_memory_behaves_as_before(self, tmp_path):
        provider = make_provider([WAIT_DECISION])
        core = make_core(tmp_path, provider)
        core.stimulus_log.append(actor="george", type="exchange", content={"message": "Hello."})

        core.orient()

        assert provider.chat.call_count == 1
        assert provider.embed.call_count == 0
        assert "<memories>" not in provider.chat.call_args_list[0].kwargs["prompt"]
