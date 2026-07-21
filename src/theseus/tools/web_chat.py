"""WebChat — delivers agent chat responses to the Theseus Chat web UI.

Pairs with `WebChatUIObserver` (see `web_chat_ui_observer.py`). Plays the same
role as `TerminalChat.respond` — the agent's "mouth" — but instead of `print`,
it hands the reply to whichever browser tabs are connected, over Server-Sent
Events. Exposed to the cognitive core as a `Tool` the model can invoke during
Decide.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, TYPE_CHECKING

from theseus.tools.tool import ToolResult

if TYPE_CHECKING:
    from theseus.web_chat_ui_observer import WebChatUIObserver

_STEP_SECONDS = 0.05


class WebChat:
    name = "respond_in_web_chat"
    ends_turn = True  # a chat reply completes the cognitive turn
    description = (
        "Send a chat message to George through the web chat UI. The message is delivered "
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

    def __init__(self, web_observer: "WebChatUIObserver"):
        self.response = None
        self.web_observer = web_observer

    def execute(self, message: str) -> ToolResult:
        """Satisfies the Tool protocol; delegates to respond()."""
        self.respond(message)
        return ToolResult(
            "Message delivered to the web chat UI.",
            details={"message": message},
        )

    def respond(self, response: str) -> None:
        """Send the response to the chat UI, revealed in word-sized chunks.

        `ModelProvider.chat()` isn't a streaming API, so the full reply is
        already in hand by the time this runs — the chunking below is a
        presentation choice (matching the design's "streams in
        incrementally" behavior), not real token-level LLM streaming. If a
        provider ever exposes token streaming, drive
        `publish_assistant_chunk` from that generator instead of this loop.
        """
        self.response = response
        message_id = uuid.uuid4().hex
        tokens = re.split(r"(\s+)", response)
        partial = ""
        for i, token in enumerate(tokens):
            partial += token
            is_last = i == len(tokens) - 1
            self.web_observer.publish_assistant_chunk(message_id, partial, done=is_last)
            if not is_last:
                time.sleep(_STEP_SECONDS)

    def respond_callback(self, response: str) -> None:
        """Callback to be invoked by the cognitive core."""
        self.respond(response)
