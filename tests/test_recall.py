from __future__ import annotations

from datetime import datetime, timezone

import pytest

from theseus.memory_note import MemoryNote
from theseus.memory_store import MemoryStore
from theseus.tools.recall import RecallTool


class StubMemory:
    """A Memory-protocol stand-in. Real AgenticMemory needs live providers; the tool
    only ever sees `retrieve`, so the protocol boundary is the honest seam here."""

    def __init__(self, recollection: str = "", raises: Exception | None = None):
        self.recollection = recollection
        self.raises = raises
        self.queries: list[str] = []

    def form(self) -> None:  # pragma: no cover - unused by the tool
        ...

    def retrieve(self, query: str) -> str:
        self.queries.append(query)
        if self.raises is not None:
            raise self.raises
        return self.recollection


def test_recall_hands_back_what_the_memory_module_returned():
    memory = StubMemory("[n1] George drinks tea, not coffee.")
    result = RecallTool(memory).execute(query="what does George drink?")

    assert not result.is_error
    assert "George drinks tea" in result.content
    assert memory.queries == ["what does George drink?"]


def test_recall_reports_an_empty_result_rather_than_an_empty_string():
    # An empty tool result reads as a malfunction to the model; "nothing came to mind"
    # is a fact it can act on.
    result = RecallTool(StubMemory("")).execute(query="George's cat")

    assert not result.is_error
    assert "george's cat" in result.content.lower()
    assert result.details == {"query": "George's cat", "found": False}


def test_recall_marks_a_hit_in_details():
    result = RecallTool(StubMemory("[n1] something")).execute(query="q")

    assert result.details == {"query": "q", "found": True}


def test_memory_failure_is_an_error_result_not_a_raise():
    # Recall must never take the cognitive loop down; the model sees the failure and
    # can carry on without it.
    tool = RecallTool(StubMemory(raises=RuntimeError("embedding endpoint down")))

    result = tool.execute(query="anything")

    assert result.is_error
    assert "embedding endpoint down" in result.content


def test_recall_is_non_terminal_so_the_agent_can_act_on_what_it_remembered():
    assert getattr(RecallTool(StubMemory()), "ends_turn", False) is False


def test_recall_declares_a_query_parameter():
    tool = RecallTool(StubMemory())
    assert tool.name == "recall"
    assert tool.parameters["required"] == ["query"]
    assert tool.parameters["properties"]["query"]["type"] == "string"


def test_recall_against_a_real_memory_store(tmp_path):
    """End-to-end through the real retrieval path: a seeded note comes back by
    embedding similarity, rendered into the tool result."""
    from theseus.agentic_memory import AgenticMemory
    from theseus.stimulus_log import StimulusLog

    class StubEmbedder:
        def is_available(self) -> bool:
            return True

        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0]

    store = MemoryStore(tmp_path / "memory.jsonl")
    store.add(
        MemoryNote(
            id="n1",
            ts=datetime.now(timezone.utc),
            content="raw events",
            context="George prefers tea.",
            embedding=[1.0, 0.0],
        )
    )
    memory = AgenticMemory(
        model_providers=[],
        embedding_providers=[StubEmbedder()],
        store=store,
        stimulus_log=StimulusLog(tmp_path / "stimulus_log.jsonl"),
    )

    result = RecallTool(memory).execute(query="what does George drink?")

    assert not result.is_error
    assert "George prefers tea." in result.content
    assert "[n1]" in result.content
