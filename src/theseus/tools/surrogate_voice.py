"""SaySurrogate / NotifySurrogate — the host's "mouth" for commanding a surrogate.

These tools append host-origin `command.<verb>` events to the host `StimulusLog`
(see `theseus.commands`); `command_feed.CommandFeed` then streams them to the
surrogate. They mirror `WebChat`: constructed with their collaborators, exposed to
the cognitive core as `Tool`s. This is the foundation for issue #94's surrogate
surface — no TTL argument here; the surrogate applies its own default.
"""

from __future__ import annotations

from typing import Any

from theseus.commands import command_content, command_type
from theseus.stimulus_log import StimulusLog
from theseus.tools.tool import ToolResult


class SaySurrogate:
    name = "say_to_surrogate"
    ends_turn = True  # a reply completes the cognitive turn, like WebChat
    description = (
        "Send a spoken message to the surrogate. Use this to say something to the "
        "person on the other side; it is delivered as your voice."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "What to say to the surrogate."},
        },
        "required": ["text"],
    }

    def __init__(self, target: str, stimulus_log: StimulusLog):
        self.target = target
        self.stimulus_log = stimulus_log

    def execute(self, text: str) -> ToolResult:
        self.stimulus_log.append(
            "host",
            command_type("say"),
            command_content(target=self.target, payload={"text": text}),
        )
        return ToolResult(
            f"Command sent to {self.target}.",
            details={"target": self.target, "text": text},
        )


class NotifySurrogate:
    name = "notify_user"
    ends_turn = False  # a proactive attention-grab; the agent may keep acting
    description = (
        "Show a notification on the surrogate. Use this to grab attention without "
        "waiting for a reply, e.g. an alert or a completed background task."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Notification title."},
            "body": {"type": "string", "description": "Notification body."},
        },
        "required": ["title", "body"],
    }

    def __init__(self, target: str, stimulus_log: StimulusLog):
        self.target = target
        self.stimulus_log = stimulus_log

    def execute(self, title: str, body: str) -> ToolResult:
        self.stimulus_log.append(
            "host",
            command_type("notify"),
            command_content(target=self.target, payload={"title": title, "body": body}),
        )
        return ToolResult(
            f"Notification sent to {self.target}.",
            details={"target": self.target, "title": title, "body": body},
        )
