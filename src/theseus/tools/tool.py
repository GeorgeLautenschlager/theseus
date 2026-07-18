from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolResult:
    """The structured outcome of a tool invocation.

    `content` is the text handed back to the model as the tool result. `is_error`
    flags a failure (the model still sees `content`, which should explain what went
    wrong). `details` carries structured extras for logging/UI and is never required.
    """

    content: str
    is_error: bool = False
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool, normalized from any provider's wire format.

    Providers differ in how they represent a tool call (Anthropic returns an object,
    OpenAI a JSON string that must be parsed), but every OpenAI-compatible endpoint we
    target does that parsing server-side. By the time a call reaches here, `arguments`
    is always a plain dict.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantTurn:
    """One assistant response: free text, tool calls, or both."""

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class Tool(Protocol):
    """A structured capability the model can invoke with typed arguments.

    Distinct from `Effector`: effectors are the agent's freeform "how I act" surface
    (two-phase text), while tools are invoked with a validated argument object and
    return a structured `ToolResult` the loop feeds back to the model.

    `parameters` is a plain JSON Schema object describing the arguments, e.g.::

        {"type": "object",
         "properties": {"path": {"type": "string"}},
         "required": ["path"]}

    Because that is exactly what an OpenAI-compatible `tools` field wants, serializing a
    tool for the wire is a near pass-through (see `to_openai_tool`).
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the already-parsed argument object."""
        ...


def to_openai_tool(tool: Tool) -> dict[str, Any]:
    """Serialize a `Tool` into the OpenAI-compatible `tools` entry.

    This is the entire "schema translation" step: the canonical `parameters` is already
    JSON Schema, so we only wrap it in the `{"type": "function", ...}` envelope that
    LM Studio / Ollama / llama.cpp expect.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
