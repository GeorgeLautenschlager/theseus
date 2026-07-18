from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from theseus.model_providers.lm_studio_provider import LmStudioProvider
from theseus.tools import ReadTool, to_openai_tool


def test_to_openai_tool_wraps_json_schema():
    entry = to_openai_tool(ReadTool())
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "read"
    # The canonical JSON Schema is passed through untouched.
    assert entry["function"]["parameters"] == ReadTool.parameters


def _fake_completion(*, content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_complete_with_tools_serializes_and_requests_non_streamed():
    provider = LmStudioProvider()
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _fake_completion(content="hi")

    turn = provider.complete_with_tools(
        [{"role": "user", "content": "hello"}], tools=[ReadTool()]
    )

    kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert kwargs["tools"][0]["function"]["name"] == "read"
    assert kwargs["tool_choice"] == "auto"
    assert "stream" not in kwargs or kwargs["stream"] is False  # default is non-streamed
    assert turn.text == "hi" and turn.tool_calls == ()


def test_complete_with_tools_parses_tool_calls():
    provider = LmStudioProvider()
    provider._client = MagicMock()
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="read", arguments=json.dumps({"path": "a.txt"})),
    )
    provider._client.chat.completions.create.return_value = _fake_completion(
        content=None, tool_calls=[tool_call]
    )

    turn = provider.complete_with_tools([{"role": "user", "content": "read a.txt"}], tools=[ReadTool()])

    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.id == "call_1" and call.name == "read"
    assert call.arguments == {"path": "a.txt"}  # JSON string decoded to a dict


def test_complete_with_tools_tolerates_bad_arguments_json():
    provider = LmStudioProvider()
    provider._client = MagicMock()
    tool_call = SimpleNamespace(
        id="call_2", function=SimpleNamespace(name="ls", arguments="{not json")
    )
    provider._client.chat.completions.create.return_value = _fake_completion(tool_calls=[tool_call])

    turn = provider.complete_with_tools([{"role": "user", "content": "x"}], tools=[ReadTool()])
    assert turn.tool_calls[0].arguments == {}
