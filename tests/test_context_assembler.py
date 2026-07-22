from __future__ import annotations

from unittest.mock import MagicMock

from theseus.context_assembler import ContextAssembler
from theseus.stimulus_log import StimulusLog


def fill_log(tmp_path, n) -> StimulusLog:
    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    for i in range(n):
        log.append(actor="george", type="exchange", content={"message": f"msg {i}"})
    return log


class TestRecentEventsWindow:
    def test_includes_all_events_when_under_window(self, tmp_path):
        log = fill_log(tmp_path, 3)
        assembled = ContextAssembler(stimulus_log=log, window_size=50).assemble_context()

        assert assembled.recent_events.count("\n") == 2
        assert "msg 0" in assembled.recent_events
        assert "msg 2" in assembled.recent_events

    def test_truncates_to_most_recent_window(self, tmp_path):
        log = fill_log(tmp_path, 5)
        assembled = ContextAssembler(stimulus_log=log, window_size=2).assemble_context()

        assert "msg 2" not in assembled.recent_events
        assert "msg 3" in assembled.recent_events
        assert "msg 4" in assembled.recent_events


class TestMemories:
    def test_empty_without_memory_system(self, tmp_path):
        log = fill_log(tmp_path, 1)
        assembled = ContextAssembler(stimulus_log=log).assemble_context()

        assert assembled.memories == ""

    def test_uses_module_rendered_memories(self, tmp_path):
        log = fill_log(tmp_path, 1)
        memory = MagicMock()
        memory.retrieve.return_value = "George prefers tea."

        assembled = ContextAssembler(stimulus_log=log, memory=memory).assemble_context()

        assert assembled.memories == "George prefers tea."
        memory.retrieve.assert_called_once_with(assembled.recent_events)

    def test_skips_retrieval_on_empty_log(self, tmp_path):
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        memory = MagicMock()

        assembled = ContextAssembler(stimulus_log=log, memory=memory).assemble_context()

        assert assembled.memories == ""
        memory.retrieve.assert_not_called()

    def test_retrieval_query_bounded_while_prompt_window_stays_full(self, tmp_path):
        # The prompt window can be huge (fine for the LLM), but the retrieval query is
        # embedded, so it must stay within a configurable budget.
        log = fill_log(tmp_path, 10)
        memory = MagicMock()
        memory.retrieve.return_value = ""

        assembled = ContextAssembler(
            stimulus_log=log, memory=memory, retrieval_query_chars=50
        ).assemble_context()

        # Every event is still in the prompt window...
        assert "msg 0" in assembled.recent_events
        assert "msg 9" in assembled.recent_events
        # ...but retrieval only saw the most-recent tail, capped to the budget.
        query = memory.retrieve.call_args.args[0]
        assert len(query) <= 50
        assert query == assembled.recent_events[-50:]
