from __future__ import annotations

import inspect

from theseus.mono_memory import MonoMemory
from theseus.stimulus_log import StimulusLog


def fill_log(tmp_path, n) -> StimulusLog:
    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    for i in range(n):
        log.append(actor="george", type="exchange", content={"message": f"msg {i}"})
    return log


class TestRecentEventsWindow:
    def test_includes_all_events_when_under_window(self, tmp_path):
        log = fill_log(tmp_path, 3)
        assembled = MonoMemory(stimulus_log=log, window_size=50).assemble_context()

        assert assembled.recent_events.count("\n") == 2
        assert "msg 0" in assembled.recent_events
        assert "msg 2" in assembled.recent_events

    def test_truncates_to_most_recent_window(self, tmp_path):
        log = fill_log(tmp_path, 5)
        assembled = MonoMemory(stimulus_log=log, window_size=2).assemble_context()

        assert "msg 2" not in assembled.recent_events
        assert "msg 3" in assembled.recent_events
        assert "msg 4" in assembled.recent_events


class TestNoInvoluntaryRetrieval:
    """The assembler no longer reaches into memory behind the agent's back. Recall is a
    tool the agent calls; its result reaches the next prompt through the stimulus log
    like any other tool_result, so there is nothing for the assembler to do."""

    def test_assembler_takes_only_the_log_and_a_window(self):
        # `constitution` used to sit here unread — the same dead-parameter trap that hid
        # persona from the system prompt. The assembler assembles the window; nothing else.
        params = set(inspect.signature(MonoMemory.__init__).parameters) - {"self"}
        assert params == {"stimulus_log", "window_size"}

    def test_assembled_context_carries_only_the_event_window(self, tmp_path):
        log = fill_log(tmp_path, 1)

        assembled = MonoMemory(stimulus_log=log).assemble_context()

        assert not hasattr(assembled, "memories")
        assert "msg 0" in assembled.recent_events

    def test_empty_log_assembles_empty_context(self, tmp_path):
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")

        assert MonoMemory(stimulus_log=log).assemble_context().recent_events == ""
