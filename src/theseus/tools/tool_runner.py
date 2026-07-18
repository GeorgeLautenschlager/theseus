from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from theseus.tools.tool import Tool, ToolResult

if TYPE_CHECKING:
    from theseus.model_providers.model_provider import ModelProvider
    from theseus.stimulus_log import StimulusLog


class ToolRunner:
    """Drives an OpenAI-style tool-calling loop over a set of `Tool`s.

    Given a provider that can return tool calls (`complete_with_tools`), it repeatedly:
    ask the model → dispatch any requested tools → feed results back — until the model
    replies with plain text or the iteration cap is hit. Each call and result is recorded
    to the `StimulusLog` (when provided) as `tool_call` / `tool_result` events.
    """

    def __init__(
        self,
        provider: "ModelProvider",
        tools: dict[str, Tool],
        stimulus_log: "StimulusLog | None" = None,
        actor: str = "tam",
        max_iterations: int = 10,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.stimulus_log = stimulus_log
        self.actor = actor
        self.max_iterations = max_iterations

    def run(self, prompt: str, system_prompt: str | None = None) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for _ in range(self.max_iterations):
            turn = self.provider.complete_with_tools(messages, list(self.tools.values()))
            if not turn.tool_calls:
                return turn.text or ""

            messages.append(self._assistant_message(turn))
            for call in turn.tool_calls:
                self._log("tool_call", {"id": call.id, "name": call.name, "arguments": call.arguments})
                result = self._dispatch(call.name, call.arguments)
                self._log(
                    "tool_result",
                    {"id": call.id, "name": call.name, "content": result.content, "is_error": result.is_error},
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result.content}
                )

        return "[tool loop hit max iterations without a final answer]"

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(f"Unknown tool: {name}", is_error=True)
        try:
            return tool.execute(**arguments)
        except TypeError as exc:
            return ToolResult(f"Bad arguments for {name}: {exc}", is_error=True)
        except Exception as exc:  # a tool bug shouldn't kill the whole loop
            return ToolResult(f"{name} raised {type(exc).__name__}: {exc}", is_error=True)

    @staticmethod
    def _assistant_message(turn) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": turn.text or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in turn.tool_calls
            ],
        }

    def _log(self, event_type: str, content: dict[str, Any]) -> None:
        if self.stimulus_log is not None:
            self.stimulus_log.append(actor=self.actor, type=event_type, content=content)
