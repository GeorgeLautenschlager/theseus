from __future__ import annotations

from unittest.mock import MagicMock

from theseus.stimulus_log import StimulusLog
from theseus.tools import ReadTool, ToolRunner, WriteTool
from theseus.tools.tool import AssistantTurn, ToolCall, ToolResult


def make_provider(turns):
    """A stub provider whose complete_with_tools returns each AssistantTurn in order."""
    provider = MagicMock()
    provider.complete_with_tools.side_effect = list(turns)
    return provider


def test_returns_text_when_no_tool_calls():
    provider = make_provider([AssistantTurn(text="just an answer")])
    runner = ToolRunner(provider, tools={})
    assert runner.run("hi") == "just an answer"


def test_dispatches_tool_then_returns_final_text(tmp_path):
    (tmp_path / "a.txt").write_text("file body")
    provider = make_provider(
        [
            AssistantTurn(tool_calls=(ToolCall(id="c1", name="read", arguments={"path": "a.txt"}),)),
            AssistantTurn(text="done"),
        ]
    )
    log = StimulusLog(path=tmp_path / "log.jsonl")
    runner = ToolRunner(provider, tools={"read": ReadTool(tmp_path)}, stimulus_log=log)

    assert runner.run("read a.txt") == "done"

    # The read result was fed back to the provider on the second call.
    second_call_messages = provider.complete_with_tools.call_args_list[1].args[0]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "file body" in tool_msg["content"]

    # Both a tool_call and a tool_result event were logged.
    events = [e.type for e in log.read_all()]
    assert "tool_call" in events and "tool_result" in events


def test_unknown_tool_reports_error_without_raising(tmp_path):
    provider = make_provider(
        [
            AssistantTurn(tool_calls=(ToolCall(id="c1", name="nope", arguments={}),)),
            AssistantTurn(text="recovered"),
        ]
    )
    runner = ToolRunner(provider, tools={})
    assert runner.run("x") == "recovered"
    messages = provider.complete_with_tools.call_args_list[1].args[0]
    tool_msg = [m for m in messages if m["role"] == "tool"][0]
    assert "Unknown tool" in tool_msg["content"]


def test_tool_exception_is_caught(tmp_path):
    boom = MagicMock()
    boom.name = "boom"
    boom.execute.side_effect = RuntimeError("kaboom")
    provider = make_provider(
        [
            AssistantTurn(tool_calls=(ToolCall(id="c1", name="boom", arguments={}),)),
            AssistantTurn(text="ok"),
        ]
    )
    runner = ToolRunner(provider, tools={"boom": boom})
    assert runner.run("x") == "ok"
    messages = provider.complete_with_tools.call_args_list[1].args[0]
    tool_msg = [m for m in messages if m["role"] == "tool"][0]
    assert "kaboom" in tool_msg["content"]


def test_stops_at_max_iterations():
    # Always asks for a tool, never finishes.
    provider = MagicMock()
    provider.complete_with_tools.return_value = AssistantTurn(
        tool_calls=(ToolCall(id="c", name="read", arguments={"path": "x"}),)
    )
    tool = MagicMock()
    tool.name = "read"
    tool.execute.return_value = ToolResult("stuff")
    runner = ToolRunner(provider, tools={"read": tool}, max_iterations=3)
    assert "max iterations" in runner.run("loop")
    assert provider.complete_with_tools.call_count == 3
