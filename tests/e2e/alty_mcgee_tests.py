"""Live end-to-end tests for Alty McGee.

These drive the *real* agent against *real* local models — Ollama `gemma4:e4b` for
reasoning and `nomic-embed-text` for memory embeddings. Nothing is stubbed: a test
is a genuine cognitive loop (Orient → Decide → Act) plus A-MEM formation and
retrieval. That makes them slow (minutes for the full file) and non-deterministic,
so every assertion is structural or semantic — never an exact string match.

The scenario replays the conversation George actually had with Alty (captured in a
stimulus log) and asserts the behaviour that conversation exercised: Alty answers
through the `terminal_chat` tool, each turn is recorded on the stimulus log, and the
A-MEM layer forms notes and can retrieve them again.

Every test gets a fully isolated stimulus log and memory store under `tmp_path`, so
running this suite never touches Alty's real `a_mem.jsonl`.

This file is deliberately named so pytest's default discovery ignores it — `make
test` will not pick it up. Run it explicitly:

    poetry run pytest tests/e2e/alty_mcgee_tests.py -v

If Ollama isn't reachable the whole module skips rather than failing.
"""

from __future__ import annotations

import pytest

from theseus.agents.alty_mcgee import AltyMcGee
from theseus.memory_store import MemoryStore
from theseus.model_providers.ollama_provider import OllamaProvider
from theseus.stimulus_log import StimulusLog

pytestmark = pytest.mark.skipif(
    not OllamaProvider(model="gemma4:e4b").is_available(),
    reason="live e2e: needs a reachable Ollama with gemma4:e4b and nomic-embed-text",
)

# The three turns George actually sent, verbatim from the captured conversation.
GREETING = (
    "Hello I'm George. You're name is Alty McGee, the resident test agent for this "
    "agent construction kit. Can you help me with some testing?"
)
E2E_PLAN = (
    "I'm thinking we ought to set up some e2e tests, so this conversation we're "
    "having is going to become set up for those e2e tests."
)
MEMORY_PLAN = (
    "The first thing we need to test is your memory. I've given you and A-mem style "
    "memory layer that will assemble notes from the log of our conversation."
)


def make_alty(tmp_path, log_name="stimulus_log.jsonl", store=None) -> AltyMcGee:
    """An Alty whose stimulus log and memory store live under tmp_path."""
    return AltyMcGee(
        stimulus_log=StimulusLog(path=tmp_path / log_name),
        memory_store=store if store is not None else MemoryStore(tmp_path / "a_mem.jsonl"),
    )


@pytest.fixture
def alty(tmp_path) -> AltyMcGee:
    return make_alty(tmp_path)


def say(alty: AltyMcGee, message: str) -> str | None:
    """Feed a user message the way TerminalChatObserver does, run one full cognitive loop,
    and return what Alty said back (None if he chose not to reply)."""
    alty.stimulus_log.append(
        actor="user", type="chat_message", content={"message": message}
    )
    alty.terminal_chat.response = None  # so a stale reply can't satisfy an assertion
    alty.core.orient()
    return alty.terminal_chat.response


def tools_called(alty: AltyMcGee) -> list[str]:
    """Every tool Alty decided to call, in order, across the whole log."""
    return [
        call["name"]
        for event in alty.stimulus_log.read_all()
        if event.type == "decision"
        for call in event.content.get("tool_calls", [])
    ]


class TestConversation:
    def test_alty_answers_a_greeting_through_terminal_chat(self, alty):
        reply = say(alty, GREETING)

        assert reply, "Alty should answer a direct greeting"
        assert "terminal_chat" in tools_called(alty)

    def test_a_turn_is_recorded_as_chat_message_decision_and_tool_result(self, alty):
        say(alty, GREETING)

        events = alty.stimulus_log.read_all()
        types = [e.type for e in events]
        assert types[0] == "chat_message"
        assert "decision" in types
        assert "tool_result" in types

        # The reply was actually delivered, not merely decided on.
        delivered = [
            e for e in events
            if e.type == "tool_result" and e.content["tool"] == "terminal_chat"
        ]
        assert delivered, "the terminal_chat call should have produced a tool_result"
        assert delivered[-1].content["is_error"] is False

    def test_alty_holds_up_across_the_whole_recorded_conversation(self, alty):
        replies = [say(alty, message) for message in (GREETING, E2E_PLAN, MEMORY_PLAN)]

        assert all(replies), "Alty should answer every turn of the conversation"
        assert tools_called(alty).count("terminal_chat") >= 3


class TestMemory:
    def test_a_note_is_formed_from_a_turn(self, alty):
        say(alty, GREETING)

        notes = alty.memory.store.read_all()
        assert len(notes) == 1
        note = notes[0]
        assert note.context.strip(), "the note should carry a distilled context"
        assert note.embedding, "the note should be embedded so it can be retrieved"

    def test_one_note_forms_per_turn(self, alty):
        for message in (GREETING, E2E_PLAN, MEMORY_PLAN):
            say(alty, message)

        # Memory forms once per cognitive loop, i.e. once per user turn.
        assert len(alty.memory.store.read_all()) == 3

    def test_earlier_turns_come_back_as_retrieved_memories(self, alty):
        say(alty, GREETING)
        say(alty, E2E_PLAN)

        assembled = alty.core.mono_memory.assemble_context()

        assert assembled.memories, "retrieval should surface notes from earlier turns"

    def test_alty_recalls_georges_name_from_memory_alone(self, tmp_path):
        # First session: George introduces himself; a note is formed on disk.
        first = make_alty(tmp_path, log_name="session_one.jsonl")
        say(first, GREETING)
        assert first.memory.store.read_all(), "the first session should leave a note"

        # Second session: same memory store, a brand-new conversation log. The name
        # is no longer in the recent-events window, so it can only come back through
        # A-MEM retrieval.
        second = make_alty(
            tmp_path,
            log_name="session_two.jsonl",
            store=MemoryStore(tmp_path / "a_mem.jsonl"),
        )

        # The retrieval layer itself must surface the name. This is the real memory
        # assertion — it exercises embed → top-k → render directly, so no agent
        # behaviour can make it pass by accident.
        assert "george" in second.memory.retrieve("What is my name?").lower()

        reply = say(second, "What is my name?")

        assert reply, "Alty should answer the question"
        assert "george" in reply.lower()

        # ...and he must have answered *from memory*, not by reading the logs off
        # disk. all_tools() is rooted at the process cwd (the repo root under pytest),
        # where the real stimulus_log.jsonl — which contains "Hello I'm George" —
        # lives. Grepping it would satisfy the assertion above while proving nothing
        # about memory, so a filesystem answer has to fail this test.
        used = set(tools_called(second))
        assert used <= {"terminal_chat"}, (
            f"expected a memory-based answer, but Alty used tools: {sorted(used)}"
        )

    
