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
    """One assistant response: free text, tool calls, or both.

    `prompt_tokens` is the backend's own count of what the prompt cost, taken from the
    response's `usage`. It is the only reliable signal we get about context consumption —
    the OpenAI-compatible `/v1/models` endpoint does not report a context window, and the
    vendor-native endpoints that do mostly report the model's *trained* maximum rather
    than the window the server actually loaded. So instead of predicting the budget we
    measure the spend: `ContextAssembler.observe` feeds this back to calibrate the next window.
    `None` when the provider reports no usage (the Claude CLI, or a server that omits it).
    """

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int | None = None


class Tool(Protocol):
    """A structured capability the model can invoke with typed arguments.

    A tool is invoked with a validated argument object and returns a structured
    `ToolResult` that the loop feeds back to the model.

    `parameters` is a plain JSON Schema object describing the arguments, e.g.::

        {"type": "object",
         "properties": {"path": {"type": "string"}},
         "required": ["path"]}

    Because that is exactly what an OpenAI-compatible `tools` field wants, serializing a
    tool for the wire is a near pass-through (see `to_openai_tool`).

    `ends_turn` marks a *terminal* tool (a chat reply — `TerminalChat`, `WebChat`) whose
    call completes the cognitive turn. Tools that omit it are non-terminal: running one
    triggers another Orient→Decide→Act pass so the agent can act on the result. The core
    reads it as `getattr(tool, "ends_turn", False)`, so only terminal tools declare it.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    ends_turn: bool = False

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
