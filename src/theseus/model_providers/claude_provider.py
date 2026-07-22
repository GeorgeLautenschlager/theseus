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
