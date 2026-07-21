from __future__ import annotations

from theseus.tools.terminal_chat import TerminalChat
from theseus.tools.tool import ToolResult, to_openai_tool


def test_terminal_chat_has_tool_shape():
    tool = TerminalChat()
    assert tool.name == "terminal_chat"
    assert isinstance(tool.description, str) and tool.description
    assert tool.parameters["type"] == "object"
    assert "message" in tool.parameters["properties"]
    assert tool.parameters["required"] == ["message"]


def test_terminal_chat_execute_delivers_message_and_returns_result():
    tool = TerminalChat()

    result = tool.execute(message="hello George")

    assert isinstance(result, ToolResult)
    assert not result.is_error
    assert tool.response == "hello George"
    assert result.details["message"] == "hello George"


def test_terminal_chat_serializes_to_openai_tool():
    entry = to_openai_tool(TerminalChat())

    assert entry["type"] == "function"
    assert entry["function"]["name"] == "terminal_chat"
    assert entry["function"]["parameters"]["required"] == ["message"]
