from __future__ import annotations

from unittest.mock import MagicMock

from theseus.tools.tool import ToolResult, to_openai_tool
from theseus.tools.web_chat import WebChat


def test_web_chat_has_tool_shape():
    tool = WebChat(web_observer=MagicMock())
    assert tool.name == "respond_in_web_chat"
    assert isinstance(tool.description, str) and tool.description
    assert tool.parameters["type"] == "object"
    assert "message" in tool.parameters["properties"]
    assert tool.parameters["required"] == ["message"]


def test_web_chat_execute_delivers_message_and_returns_result():
    web_observer = MagicMock()
    tool = WebChat(web_observer=web_observer)

    result = tool.execute(message="hello George")

    assert isinstance(result, ToolResult)
    assert not result.is_error
    assert tool.response == "hello George"
    assert result.details["message"] == "hello George"
    # The message was actually pushed out to connected browsers.
    assert web_observer.publish_assistant_chunk.called


def test_web_chat_serializes_to_openai_tool():
    entry = to_openai_tool(WebChat(web_observer=MagicMock()))

    assert entry["type"] == "function"
    assert entry["function"]["name"] == "respond_in_web_chat"
    assert entry["function"]["parameters"]["required"] == ["message"]
