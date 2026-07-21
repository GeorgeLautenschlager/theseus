from __future__ import annotations

import json
from unittest.mock import MagicMock

from theseus.agentic_memory import AgenticMemory
from theseus.cognitive_core import CognitiveCore
from theseus.memory_store import MemoryStore
from theseus.stimulus_log import StimulusLog
from theseus.tools.tool import AssistantTurn, ToolCall, ToolResult


class StubTool:
    def __init__(self, name="terminal_chat"):
        self.name = name
        self.description = "Send a chat message to George."
        self.parameters = {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }
        self.executed_with = None

    def execute(self, **kwargs):
        self.executed_with = kwargs
        return ToolResult("delivered", details={**kwargs})


def turn(*tool_calls, text=None):
    """An AssistantTurn with zero or more (name, arguments) tool calls."""
    calls = tuple(
        ToolCall(id=f"call_{i}", name=name, arguments=args)
        for i, (name, args) in enumerate(tool_calls)
    )
    return AssistantTurn(text=text, tool_calls=calls)


def make_provider(turns):
    """A stub ModelProvider whose .complete_with_tools() returns each turn in order."""
    provider = MagicMock()
    provider.is_available.return_value = True
    provider.complete_with_tools.side_effect = list(turns)
    return provider


def make_core(tmp_path, provider, tools=None, memory=None):
    stimulus_log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    return CognitiveCore(
        constitution="You are Tam.",
        model_providers=[provider],
        tools=tools or {},
        stimulus_log=stimulus_log,
        memory=memory,
    )


def run_decide(core, recent_events="", memories=""):
    """Enter the loop at Decide: steps read from loop_memory, never parameters."""
    core.loop_memory["recent_events"] = recent_events
    core.loop_memory["memories"] = memories
    core.decide()


class TestDecide:
    def test_offers_the_tools_to_the_provider(self, tmp_path):
        tool = StubTool()
        provider = make_provider([turn()])
        core = make_core(tmp_path, provider, tools={tool.name: tool})

        run_decide(core)

        args, _ = provider.complete_with_tools.call_args
        offered_tools = args[1]
        assert tool in offered_tools

    def test_system_and_user_prompts_are_sent_as_messages(self, tmp_path):
        tool = StubTool()
        provider = make_provider([turn()])
        core = make_core(tmp_path, provider, tools={tool.name: tool})

        run_decide(core, recent_events="George said hi.")

        messages = provider.complete_with_tools.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert "You are Tam." in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "George said hi." in messages[1]["content"]

    def test_logs_decision_event_with_tool_calls(self, tmp_path):
        tool = StubTool()
        provider = make_provider([turn((tool.name, {"message": "Hi!"}), text="greeting")])
        core = make_core(tmp_path, provider, tools={tool.name: tool})

        run_decide(core)

        events = core.stimulus_log.read_all()
        decision_events = [e for e in events if e.type == "decision"]
        assert len(decision_events) == 1
        assert decision_events[0].content["text"] == "greeting"
        assert decision_events[0].content["tool_calls"] == [
            {"name": tool.name, "arguments": {"message": "Hi!"}}
        ]


class TestAct:
    def test_tool_call_executes_tool_with_its_arguments(self, tmp_path):
        tool = StubTool()
        provider = make_provider([turn((tool.name, {"message": "Hello, George!"}))])
        core = make_core(tmp_path, provider, tools={tool.name: tool})

        run_decide(core)

        assert tool.executed_with == {"message": "Hello, George!"}

    def test_no_tool_call_is_the_wait_path(self, tmp_path):
        tool = StubTool()
        provider = make_provider([turn(text="nothing to do")])
        core = make_core(tmp_path, provider, tools={tool.name: tool})

        run_decide(core)

        assert tool.executed_with is None
        events = core.stimulus_log.read_all()
        assert not [e for e in events if e.type == "action"]

    def test_action_event_records_the_tool_result(self, tmp_path):
        tool = StubTool()
        provider = make_provider([turn((tool.name, {"message": "Hello, George!"}))])
        core = make_core(tmp_path, provider, tools={tool.name: tool})

        run_decide(core)

        events = core.stimulus_log.read_all()
        action_events = [e for e in events if e.type == "action"]
        assert len(action_events) == 1
        assert action_events[0].content["action"] == tool.name
        assert action_events[0].content["arguments"] == {"message": "Hello, George!"}
        assert action_events[0].content["output"] == "delivered"
        assert action_events[0].content["is_error"] is False

    def test_unknown_tool_neither_raises_nor_executes(self, tmp_path, capsys):
        tool = StubTool()
        provider = make_provider([turn(("not_a_real_tool", {"message": "x"}))])
        core = make_core(tmp_path, provider, tools={tool.name: tool})

        run_decide(core)

        assert tool.executed_with is None
        events = core.stimulus_log.read_all()
        assert not [e for e in events if e.type == "action"]

    def test_only_one_model_call_per_cycle(self, tmp_path):
        tool = StubTool()
        provider = make_provider([turn((tool.name, {"message": "Hi!"}))])
        core = make_core(tmp_path, provider, tools={tool.name: tool})

        run_decide(core)

        # Native tool-calling collapses Decide+Act into a single model call.
        assert provider.complete_with_tools.call_count == 1

    def test_stimulus_log_actor_uses_core_name(self, tmp_path):
        tool = StubTool()
        provider = make_provider([turn((tool.name, {"message": "Hi!"}))])
        stimulus_log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        core = CognitiveCore(
            constitution="You are Tam.",
            model_providers=[provider],
            tools={tool.name: tool},
            stimulus_log=stimulus_log,
            name="Aldric",
        )

        run_decide(core)

        events = core.stimulus_log.read_all()
        assert all(e.actor == "Aldric" for e in events)


class TestLoopTermination:
    def test_every_terminal_path_signals_memory_formation_once(self, tmp_path):
        tool = StubTool()
        terminal_turns = [
            turn(text="nothing to do"),                       # wait
            turn(("not_a_real_tool", {"message": "x"})),      # unknown tool
            turn((tool.name, {"message": "Hello!"})),         # real tool call
        ]
        for t in terminal_turns:
            provider = make_provider([t])
            memory = MagicMock()
            memory.retrieve.return_value = ""
            core = make_core(tmp_path, provider, tools={tool.name: tool}, memory=memory)

            run_decide(core)

            assert memory.form.call_count == 1, f"path: {t}"

    def test_termination_resets_loop_memory(self, tmp_path):
        provider = make_provider([turn(text="nothing to do")])
        core = make_core(tmp_path, provider)

        run_decide(core, recent_events="something", memories="a memory")

        assert core.loop_memory == {}


WAIT_TURN = turn(text="nothing to do")
ENRICHMENT = json.dumps(
    {"context": "George introduced himself.", "keywords": ["George"], "tags": ["identity"]}
)


def make_embedder(embedding=None):
    embedder = MagicMock()
    embedder.is_available.return_value = True
    embedder.embed.return_value = embedding or [1.0, 0.0]
    return embedder


def make_memory(tmp_path, provider, stimulus_log, embedder=None):
    return AgenticMemory(
        model_providers=[provider],
        embedding_providers=[embedder or make_embedder()],
        store=MemoryStore(tmp_path / "memory.jsonl"),
        stimulus_log=stimulus_log,
    )


def make_core_with_memory(tmp_path, provider, embedder=None):
    stimulus_log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    memory = make_memory(tmp_path, provider, stimulus_log, embedder=embedder)
    core = CognitiveCore(
        constitution="You are Tam.",
        model_providers=[provider],
        tools={},
        stimulus_log=stimulus_log,
        memory=memory,
    )
    return core, memory


class TestMemoryIntegration:
    def test_orient_forms_a_note_post_cycle(self, tmp_path):
        # Decide is a wait turn; memory formation (form()) uses provider.chat.
        provider = make_provider([WAIT_TURN])
        provider.chat.side_effect = [ENRICHMENT]
        core, memory = make_core_with_memory(tmp_path, provider)
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

    def test_decide_messages_contain_retrieved_memories(self, tmp_path):
        provider = make_provider([WAIT_TURN, WAIT_TURN])
        provider.chat.side_effect = [ENRICHMENT, json.dumps({"links": []}), ENRICHMENT]
        core, memory = make_core_with_memory(tmp_path, provider)
        # Seed one note through the real pipeline, then start a fresh cycle.
        core.stimulus_log.append(actor="george", type="exchange", content={"message": "I am George."})
        memory.form()
        core.stimulus_log.append(actor="george", type="exchange", content={"message": "Who am I?"})

        core.orient()

        messages = provider.complete_with_tools.call_args.args[0]
        decide_user_prompt = messages[1]["content"]
        assert "<memories>" in decide_user_prompt
        assert "George introduced himself." in decide_user_prompt

    def test_without_memory_behaves_as_before(self, tmp_path):
        provider = make_provider([WAIT_TURN])
        core = make_core(tmp_path, provider)
        core.stimulus_log.append(actor="george", type="exchange", content={"message": "Hello."})

        core.orient()

        assert provider.complete_with_tools.call_count == 1
        messages = provider.complete_with_tools.call_args.args[0]
        assert "<memories>" not in messages[1]["content"]
