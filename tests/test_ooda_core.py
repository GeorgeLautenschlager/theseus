from __future__ import annotations

import json
from unittest.mock import MagicMock

from theseus.agentic_memory import AgenticMemory
from theseus.context_assembler import ContextAssembler
from theseus.ooda_core import OODACore
from theseus.memory_store import MemoryStore
from theseus.stimulus_log import StimulusLog
from theseus.tools.recall import RecallTool
from theseus.tools.tool import AssistantTurn, ToolCall, ToolResult


class StubTool:
    def __init__(self, name="terminal_chat", ends_turn=True):
        self.name = name
        self.description = "Send a chat message to George."
        self.parameters = {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }
        self.ends_turn = ends_turn
        self.executed_with = None

    def execute(self, **kwargs):
        self.executed_with = kwargs
        return ToolResult(f"{self.name}-output", details={**kwargs})


def instrumental(name="ls"):
    """A non-terminal tool: running it should trigger another pass through the loop."""
    return StubTool(name=name, ends_turn=False)


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


def make_core(tmp_path, provider, tools=None, memory=None, max_loops=10, name="Tam"):
    stimulus_log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    return OODACore(
        name=name,
        constitution="You are Tam.",
        persona="Direct and curious.",
        context_assembler=ContextAssembler(stimulus_log=stimulus_log),
        model_providers=[provider],
        tools=tools or {},
        stimulus_log=stimulus_log,
        memory=memory,
        max_loops=max_loops,
    )


def run_decide(core, recent_events=""):
    """Enter the loop at Decide: steps read from loop_memory, never parameters.

    Seeds exactly what Orient would have left behind, `window_chars` included — Decide
    reports that back to the assembler to calibrate the next window.
    """
    core.loop_memory["recent_events"] = recent_events
    core.loop_memory["window_chars"] = len(recent_events)
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

    def test_reports_prompt_cost_back_to_the_assembler(self, tmp_path):
        # The window is sized closed-loop: no OpenAI-compatible endpoint will tell us the
        # context window it loaded, but every response reports what the prompt cost, so
        # Decide hands that measurement back for the next window to be sized against.
        tool = StubTool()
        provider = make_provider([AssistantTurn(text="ok", prompt_tokens=250)])
        core = make_core(tmp_path, provider, tools={tool.name: tool})

        run_decide(core, recent_events="George said hi.")

        messages = provider.complete_with_tools.call_args.args[0]
        sent_chars = sum(len(m["content"]) for m in messages)
        assembler = core.context_assembler
        assert assembler.chars_per_token == sent_chars / 250
        # Everything that wasn't the window is overhead: constitution, persona, schemas.
        assert assembler.overhead_tokens == (
            sent_chars - len("George said hi.")
        ) / assembler.chars_per_token

    def test_a_provider_without_usage_leaves_the_estimate_alone(self, tmp_path):
        # ClaudeProvider shells out to the CLI and reports no usage.
        tool = StubTool()
        provider = make_provider([AssistantTurn(text="ok", prompt_tokens=None)])
        core = make_core(tmp_path, provider, tools={tool.name: tool})
        seeded = core.context_assembler.chars_per_token

        run_decide(core, recent_events="George said hi.")

        assert core.context_assembler.chars_per_token == seeded


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
        assert not [e for e in events if e.type == "tool_result"]

    def test_tool_result_event_records_the_outcome(self, tmp_path):
        tool = StubTool()
        provider = make_provider([turn((tool.name, {"message": "Hello, George!"}))])
        core = make_core(tmp_path, provider, tools={tool.name: tool})

        run_decide(core)

        events = core.stimulus_log.read_all()
        results = [e for e in events if e.type == "tool_result"]
        assert len(results) == 1
        assert results[0].content["tool"] == tool.name
        assert results[0].content["arguments"] == {"message": "Hello, George!"}
        assert results[0].content["output"] == "terminal_chat-output"
        assert results[0].content["is_error"] is False

    def test_terminal_tool_ends_the_loop(self, tmp_path):
        tool = StubTool()  # ends_turn=True
        provider = make_provider([turn((tool.name, {"message": "Hi!"}))])
        core = make_core(tmp_path, provider, tools={tool.name: tool})

        run_decide(core)

        # A terminal tool ends the turn: exactly one model call, loop_memory reset.
        assert provider.complete_with_tools.call_count == 1
        assert core.loop_memory == {}

    def test_instrumental_tool_triggers_another_pass(self, tmp_path):
        ls = instrumental("ls")
        chat = StubTool("terminal_chat")  # terminal
        provider = make_provider(
            [
                turn(("ls", {"path": "."})),
                turn(("terminal_chat", {"message": "Here are the files."})),
            ]
        )
        core = make_core(tmp_path, provider, tools={"ls": ls, "terminal_chat": chat})

        run_decide(core)

        # The ls result triggered a second Decide, which then replied and ended the turn.
        assert provider.complete_with_tools.call_count == 2
        assert ls.executed_with == {"path": "."}
        assert chat.executed_with == {"message": "Here are the files."}
        # The second Decide actually saw the ls output in its assembled context.
        second_messages = provider.complete_with_tools.call_args_list[1].args[0]
        assert "ls-output" in second_messages[1]["content"]
        assert core.loop_memory == {}

    def test_max_loops_caps_a_runaway_loop(self, tmp_path):
        ls = instrumental("ls")
        provider = make_provider([turn(("ls", {"path": "."}))] * 5)
        core = make_core(tmp_path, provider, tools={"ls": ls}, max_loops=3)

        run_decide(core)

        # Never terminal, so only the cap stops it.
        assert provider.complete_with_tools.call_count == 3
        assert core.loop_memory == {}

    def test_unknown_tool_logs_error_result(self, tmp_path):
        tool = StubTool()
        provider = make_provider([turn(("not_a_real_tool", {"message": "x"}))])
        core = make_core(tmp_path, provider, tools={tool.name: tool}, max_loops=1)

        run_decide(core)

        assert tool.executed_with is None
        events = core.stimulus_log.read_all()
        results = [e for e in events if e.type == "tool_result"]
        assert len(results) == 1
        assert results[0].content["tool"] == "not_a_real_tool"
        assert results[0].content["is_error"] is True

    def test_stimulus_log_actor_uses_core_name(self, tmp_path):
        tool = StubTool()
        provider = make_provider([turn((tool.name, {"message": "Hi!"}))])
        core = make_core(tmp_path, provider, tools={tool.name: tool}, name="Aldric")

        run_decide(core)

        events = core.stimulus_log.read_all()
        assert all(e.actor == "Aldric" for e in events)


class TestLoopTermination:
    def test_terminal_paths_signal_memory_formation_once(self, tmp_path):
        tool = StubTool()  # terminal
        terminal_turns = [
            turn(text="nothing to do"),                  # wait
            turn((tool.name, {"message": "Hello!"})),    # terminal tool
        ]
        for t in terminal_turns:
            provider = make_provider([t])
            memory = MagicMock()
            memory.retrieve.return_value = ""
            core = make_core(tmp_path, provider, tools={tool.name: tool}, memory=memory)

            run_decide(core)

            assert memory.form.call_count == 1, f"path: {t}"

    def test_max_loops_termination_forms_memory_once(self, tmp_path):
        ls = instrumental("ls")
        memory = MagicMock()
        memory.retrieve.return_value = ""
        provider = make_provider([turn(("ls", {"path": "."}))] * 3)
        core = make_core(tmp_path, provider, tools={"ls": ls}, memory=memory, max_loops=2)

        run_decide(core)

        # Memory forms once for the whole multi-pass turn, not per pass.
        assert memory.form.call_count == 1

    def test_termination_resets_loop_memory(self, tmp_path):
        provider = make_provider([turn(text="nothing to do")])
        core = make_core(tmp_path, provider)

        run_decide(core, recent_events="something")

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


def make_core_with_memory(tmp_path, provider, embedder=None, tools=None):
    stimulus_log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    memory = make_memory(tmp_path, provider, stimulus_log, embedder=embedder)
    core = OODACore(
        name="Tam",
        constitution="You are Tam.",
        persona="Direct and curious.",
        context_assembler=ContextAssembler(stimulus_log=stimulus_log),
        model_providers=[provider],
        tools=tools or {},
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

    def test_calling_recall_puts_the_recollection_in_the_next_decide_prompt(self, tmp_path):
        """The whole point of recall-as-a-tool: what the agent remembers re-enters the
        loop through the stimulus log, on the same path as every other tool result, and
        the non-terminal pass gives it a chance to act on what came back."""
        provider = make_provider([turn(("recall", {"query": "who is George?"})), WAIT_TURN])
        provider.chat.side_effect = [ENRICHMENT, json.dumps({"links": []}), ENRICHMENT]
        core, memory = make_core_with_memory(tmp_path, provider)
        core.tools = {"recall": RecallTool(memory)}
        # Seed one note through the real pipeline, then start a fresh cycle.
        core.stimulus_log.append(actor="george", type="exchange", content={"message": "I am George."})
        memory.form()
        core.stimulus_log.append(actor="george", type="exchange", content={"message": "Who am I?"})

        core.orient()

        # Two Decide calls: the one that chose to recall, and the one that saw the result.
        assert provider.complete_with_tools.call_count == 2
        second_prompt = provider.complete_with_tools.call_args.args[0][1]["content"]
        assert "<stimulus_log>" in second_prompt
        assert "George introduced himself." in second_prompt
        assert "<memories>" not in second_prompt

    def test_recall_output_is_not_consolidated_into_a_new_note(self, tmp_path):
        # Guards the compounding echo end to end: the note formed at loop termination
        # must not contain the memories recalled during that same loop.
        provider = make_provider([turn(("recall", {"query": "who is George?"})), WAIT_TURN])
        provider.chat.side_effect = [ENRICHMENT, json.dumps({"links": []}), ENRICHMENT]
        core, memory = make_core_with_memory(tmp_path, provider)
        core.tools = {"recall": RecallTool(memory)}
        core.stimulus_log.append(actor="george", type="exchange", content={"message": "I am George."})
        memory.form()
        core.stimulus_log.append(actor="george", type="exchange", content={"message": "Who am I?"})

        core.orient()

        formation_prompt = provider.chat.call_args_list[-1].kwargs["prompt"]
        assert "who is George?" in formation_prompt  # the act of recalling is kept
        assert "George introduced himself." not in formation_prompt  # its payload is not

    def test_without_memory_behaves_as_before(self, tmp_path):
        provider = make_provider([WAIT_TURN])
        core = make_core(tmp_path, provider)
        core.stimulus_log.append(actor="george", type="exchange", content={"message": "Hello."})

        core.orient()

        assert provider.complete_with_tools.call_count == 1
        messages = provider.complete_with_tools.call_args.args[0]
        assert "<memories>" not in messages[1]["content"]
