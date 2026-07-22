from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from theseus.model_providers.claude_provider import ClaudeProvider
from theseus.tools.tool import AssistantTurn, ToolResult


@dataclass(frozen=True)
class FakeTool:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult("")


def make_tools():
    return [
        FakeTool(
            "ls",
            "List a directory.",
            {"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        FakeTool(
            "terminal_chat",
            "Reply in chat.",
            {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
        ),
    ]


def make_provider(raw: str) -> ClaudeProvider:
    """A ClaudeProvider whose CLI call (chat) is stubbed to return `raw`."""
    provider = ClaudeProvider(model="claude-sonnet-4-6")
    provider.chat = MagicMock(return_value=raw)
    return provider


MESSAGES = [
    {"role": "system", "content": "You are Tam."},
    {"role": "user", "content": "George asked: what files are here?"},
]


def test_tool_call_decision_maps_to_one_toolcall():
    raw = json.dumps({"rationale": "Listing the dir.", "action": "ls", "arguments": {"path": "."}})
    provider = make_provider(raw)

    turn = provider.complete_with_tools(MESSAGES, make_tools())

    assert isinstance(turn, AssistantTurn)
    assert turn.text == "Listing the dir."
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "ls"
    assert call.arguments == {"path": "."}
    assert call.id  # a generated id stands in for the wire id


def test_wait_decision_yields_no_tool_calls():
    provider = make_provider(json.dumps({"rationale": "Nothing to do.", "action": "wait"}))

    turn = provider.complete_with_tools(MESSAGES, make_tools())

    assert turn.text == "Nothing to do."
    assert turn.tool_calls == ()


def test_unknown_action_is_treated_as_wait():
    provider = make_provider(json.dumps({"action": "not_a_tool", "arguments": {"x": 1}}))

    turn = provider.complete_with_tools(MESSAGES, make_tools())

    assert turn.tool_calls == ()


def test_schema_enum_lists_tool_names_plus_wait():
    provider = make_provider(json.dumps({"action": "wait"}))

    provider.complete_with_tools(MESSAGES, make_tools())

    _, kwargs = provider.chat.call_args
    enum = kwargs["json_schema"]["properties"]["action"]["enum"]
    assert enum == ["ls", "terminal_chat", "wait"]


def test_system_prompt_includes_tool_parameter_schemas():
    provider = make_provider(json.dumps({"action": "wait"}))

    provider.complete_with_tools(MESSAGES, make_tools())

    _, kwargs = provider.chat.call_args
    system_prompt = kwargs["system_prompt"]
    assert "You are Tam." in system_prompt          # original system content preserved
    assert '"message"' in system_prompt             # terminal_chat's parameter schema present
    assert "ls:" in system_prompt                   # ls listed by name


def test_multi_turn_history_renders_into_prompt():
    provider = make_provider(json.dumps({"action": "wait"}))
    messages = [
        {"role": "system", "content": "You are Tam."},
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c0", "type": "function"}]},
        {"role": "tool", "tool_call_id": "c0", "content": "a.txt\nb.txt"},
    ]

    provider.complete_with_tools(messages, make_tools())

    prompt = provider.chat.call_args.args[0]
    assert "list files" in prompt
    assert "a.txt" in prompt


def test_no_tools_returns_plain_completion():
    provider = make_provider("Hello, George!")

    turn = provider.complete_with_tools(MESSAGES, [])

    assert turn.text == "Hello, George!"
    assert turn.tool_calls == ()
    _, kwargs = provider.chat.call_args
    assert "json_schema" not in kwargs  # plain completion — no schema constraint


def test_chat_feeds_prompt_via_stdin_not_argv(monkeypatch):
    # A large prompt as a CLI argument overflows the OS single-argument limit
    # (Linux MAX_ARG_STRLEN, 128 KiB) → OSError [Errno 7]. The prompt must go on stdin.
    from theseus.model_providers import claude_provider

    recorded = {}

    class _Result:
        stdout = "the reply\n"

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return _Result()

    monkeypatch.setattr(claude_provider.subprocess, "run", fake_run)

    provider = ClaudeProvider(model="claude-sonnet-4-6")
    big_prompt = "A" * 200_000
    out = provider.chat(big_prompt, system_prompt="you are tam")

    assert out == "the reply"
    # Prompt is fed on stdin, never as an argv...
    assert recorded["kwargs"]["input"] == big_prompt
    assert big_prompt not in recorded["cmd"]
    # ...while the (bounded) system prompt stays a CLI argument.
    assert "-p" in recorded["cmd"]
    assert "you are tam" in recorded["cmd"]
