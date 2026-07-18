"""ChatEffector — delivers agent chat responses to the terminal chat UI.

Pairs with `ChatObserver` (see `chat_observer.py`). Plays the same role as
`WebChatUIEffector.respond` — the `act` phase's mouth — but instead of
Server-Sent Events, it just prints the reply to stdout.
"""

from __future__ import annotations


class ChatEffector:
    name = "respond_in_chat"
    description = (
        "Send a chat message to George through the terminal chat UI. Choose this when the recent "
        "history contains a message or situation that calls for a reply from you."
    )
    act_instruction = (
        "Compose your chat message to George now. Your entire output is delivered verbatim as "
        "your message in the terminal chat UI — output only the message itself, with no JSON, no "
        "preamble, and no notes about these instructions."
    )

    def __init__(self):
        self.response = None

    def execute(self, payload: str) -> None:
        """Satisfies the Effector protocol; delegates to respond()."""
        self.respond(payload)

    def respond(self, response: str) -> None:
        """Send the response to the chat UI."""
        self.response = response
        print(f"Agent: {response}")

    def respond_callback(self, response: str) -> None:
        """Callback to be invoked by the cognitive core."""
        self.respond(response)
