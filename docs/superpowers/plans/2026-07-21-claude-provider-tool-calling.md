# Claude Provider Tool-Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `ClaudeProvider.complete_with_tools` a working implementation so Claude can drive the cognitive core's Decide step, using the `claude` CLI (subscription auth) rather than the Anthropic API.

**Architecture:** `complete_with_tools` constrains the CLI's output with `--json-schema` to a single `{rationale, action, arguments}` decision, appends each tool's parameter schema to the system prompt so the model can fill `arguments`, and maps the parsed decision to the `AssistantTurn`/`ToolCall` the core already consumes. `action == "wait"` (or an unknown action) yields empty `tool_calls` — the native-tools "wait". Reuses the existing `chat()` CLI plumbing and `parse_json_response()`.

**Tech Stack:** Python 3.12, Poetry, pytest, `unittest.mock`. No new dependencies (no `anthropic` SDK, no API key).

Spec: `docs/superpowers/specs/2026-07-21-claude-provider-tool-calling-design.md`.

---

## File Structure

- **Modify:** `src/theseus/model_providers/claude_provider.py` — replace the stubbed `complete_with_tools` (currently raises `NotImplementedError`) with the CLI implementation plus four private helpers. `is_available()`, `chat()`, and `embed()` are unchanged.
- **Create:** `tests/test_claude_provider.py` — offline unit tests that mock `ClaudeProvider.chat` (so no `claude` CLI is invoked).

---

### Task 1: Implement `complete_with_tools` over the `claude` CLI

**Files:**
- Modify: `src/theseus/model_providers/claude_provider.py`
- Test: `tests/test_claude_provider.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_claude_provider.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/test_claude_provider.py -q`
Expected: FAIL — the current `complete_with_tools(self, *args, **kwargs)` raises `NotImplementedError`.

- [ ] **Step 3: Implement `complete_with_tools` and its helpers**

Replace the entire contents of `src/theseus/model_providers/claude_provider.py` with:

```python
from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from typing import Any

from theseus.json_utils import parse_json_response
from theseus.tools.tool import AssistantTurn, Tool, ToolCall

from .model_provider import ModelProvider

WAIT_ACTION = "wait"


class ClaudeProvider(ModelProvider):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 8196,
        temperature: float = 0.7,
        json_schema: dict | None = None,
    ) -> str:
        # --safe-mode and --tools "" keep this call a plain completion: without
        # them the CLI loads this project's CLAUDE.md/skills and the model
        # responds as an interactive Claude Code session (e.g. offering to
        # invoke skills) instead of just answering the prompt.
        cmd = ["claude", "-p", prompt, "--model", self.model, "--safe-mode", "--tools", ""]
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]
        if json_schema is not None:
            cmd += ["--json-schema", json.dumps(json_schema)]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        max_tokens: int = 8196,
        temperature: float = 0.7,
    ) -> AssistantTurn:
        """One tool-calling turn over the `claude` CLI.

        The CLI has no native tool-calling loop, so we constrain its output with
        `--json-schema` to a single {rationale, action, arguments} decision and map
        that to the `AssistantTurn` the cognitive core expects. `action == "wait"`
        (or an unknown action) means take no action this cycle — an empty tool-call
        tuple, which the core reads as the native-tools "wait".
        """
        system_prompt, prompt = self._split_messages(messages)
        tools = list(tools or [])

        if not tools:
            text = self.chat(
                prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return AssistantTurn(text=text, tool_calls=())

        raw = self.chat(
            prompt,
            system_prompt=self._augment_system_prompt(system_prompt, tools),
            max_tokens=max_tokens,
            temperature=temperature,
            json_schema=self._tool_call_schema(tools),
        )
        return self._to_assistant_turn(parse_json_response(raw), tools)

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("The Claude CLI does not expose an embeddings endpoint.")

    # -- CLI tool-calling helpers ---------------------------------------------

    @staticmethod
    def _split_messages(messages: list[dict[str, Any]]) -> tuple[str | None, str]:
        """Split OpenAI-style `messages` into (system_prompt, user_prompt) for the CLI.

        System-role content is concatenated into the system prompt. A lone user turn
        (the cognitive core's case) becomes the prompt verbatim; a longer history is
        rendered best-effort as labelled turns.
        """
        system_parts = [m.get("content") or "" for m in messages if m.get("role") == "system"]
        turns = [m for m in messages if m.get("role") != "system"]
        system_prompt = "\n\n".join(p for p in system_parts if p) or None
        if len(turns) == 1 and turns[0].get("role") == "user":
            prompt = turns[0].get("content") or ""
        else:
            prompt = "\n\n".join(
                f"{(m.get('role') or '').capitalize()}: {m.get('content') or ''}" for m in turns
            )
        return system_prompt, prompt

    @staticmethod
    def _augment_system_prompt(system_prompt: str | None, tools: list[Tool]) -> str:
        """Append each tool's argument schema so the CLI can populate `arguments`."""
        schemas = "\n".join(f"- {tool.name}: {json.dumps(tool.parameters)}" for tool in tools)
        block = (
            "# Choosing an action\n"
            "Set `action` to one tool name and fill `arguments` to match that tool's JSON "
            f'Schema below, or set `action` to "{WAIT_ACTION}" to do nothing this cycle. '
            "Tool argument schemas:\n"
            f"{schemas}"
        )
        return f"{system_prompt}\n\n{block}" if system_prompt else block

    @staticmethod
    def _tool_call_schema(tools: list[Tool]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "action": {"type": "string", "enum": [t.name for t in tools] + [WAIT_ACTION]},
                "arguments": {"type": "object"},
            },
            "required": ["action"],
        }

    @staticmethod
    def _to_assistant_turn(decision: dict[str, Any], tools: list[Tool]) -> AssistantTurn:
        rationale = decision.get("rationale")
        action = decision.get("action")
        if action not in {tool.name for tool in tools}:  # "wait", missing, or unknown
            return AssistantTurn(text=rationale, tool_calls=())
        arguments = decision.get("arguments") or {}
        call = ToolCall(id=uuid.uuid4().hex, name=action, arguments=arguments)
        return AssistantTurn(text=rationale, tool_calls=(call,))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/test_claude_provider.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Run the full offline suite**

Run: `poetry run pytest tests/ -q --ignore=tests/test_fact_retention.py`
Expected: PASS — no regressions (previous baseline was 117 passed; now 124 with the 7 new tests).

- [ ] **Step 6: Commit**

```bash
git add src/theseus/model_providers/claude_provider.py tests/test_claude_provider.py
git commit -m "feat: ClaudeProvider.complete_with_tools via the claude CLI

Constrain the CLI output with --json-schema to a single {rationale, action,
arguments} decision and map it to AssistantTurn/ToolCall. action == wait yields
empty tool_calls. No anthropic dependency, no API key.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2 (optional): Live smoke test against the real `claude` CLI

Only if a `claude` CLI is on PATH and the operator wants an end-to-end check. This makes a real (subscription-billed) call; it is not part of the automated suite.

**Files:**
- Create (throwaway, not committed): a scratch script, or run inline via `poetry run python -c`.

- [ ] **Step 1: Run one real decision**

Run:

```bash
poetry run python -c "
from theseus.model_providers.claude_provider import ClaudeProvider
from theseus.tools.terminal_chat import TerminalChat
p = ClaudeProvider(model='claude-sonnet-4-6')
assert p.is_available(), 'claude CLI not on PATH'
messages = [
    {'role': 'system', 'content': 'You are Tam. Decide your next action.'},
    {'role': 'user', 'content': 'George says: please reply and say hello back.'},
]
turn = p.complete_with_tools(messages, [TerminalChat()])
print('text:', turn.text)
print('tool_calls:', [(c.name, c.arguments) for c in turn.tool_calls])
"
```

Expected: a `terminal_chat` tool call with a `message` argument (or, acceptably, a `wait` with empty `tool_calls` if the model judges no reply is needed). Confirms the schema round-trips through the real CLI.

---

## Notes for the implementer

- **Do not** add `anthropic` to dependencies or touch `pyproject.toml` — this path deliberately stays on the `claude` CLI.
- **Do not** change `cognitive_core.py`, `cognitive_prompts.py`, the other providers, or Tam. Wiring Claude into Tam's core is a separate step.
- The tests stub `ClaudeProvider.chat` (the subprocess seam), so the suite never shells out to `claude`.
