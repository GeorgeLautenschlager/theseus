"""TerminalChat — delivers agent chat responses to the terminal chat UI.

Pairs with `ChatObserver` (see `chat_observer.py`). Plays the same role as
`WebChat.respond` — the agent's "mouth" — but instead of Server-Sent
Events, it just prints the reply to stdout. Exposed to the cognitive core as a
`Tool` the model can invoke during Decide.
"""

from __future__ import annotations

from typing import Any

from theseus.tools.tool import ToolResult


class TerminalChat:
    name = "terminal_chat"
    ends_turn = True  # a chat reply completes the cognitive turn
    description = (
        "Send a chat message through the terminal chat UI. The message is delivered "
        "verbatim as your reply."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The chat message to send, delivered verbatim.",
            },
        },
        "required": ["message"],
    }

    def __init__(self):
        self.response = None

    def execute(self, message) -> ToolResult:
        """Satisfies the Tool protocol; delegates to respond()."""
        self.respond(message)
        return ToolResult(
            "Message delivered to the terminal chat UI.",
            details={"message": message},
        )

    def respond(self, response: str) -> None:
        """Send the response to the chat UI."""
        self.response = response
        print(f"Agent: {response}")

    def respond_callback(self, response: str) -> None:
        """Callback to be invoked by the cognitive core."""
        self.respond(response)
