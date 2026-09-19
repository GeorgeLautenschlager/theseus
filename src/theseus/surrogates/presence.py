"""The presence seams between a surrogate and its OS-facing surfaces (issue #95).

A surrogate "shows up" through two OS-facing surfaces: the on-screen chat it speaks
into and the native notifications it raises. Neither belongs to the runtime — a
Windows surrogate uses a command-rendered chat window and toast notifications, the
web observer (#98) implements `ChatSurface` against the browser, and headless/dev
runs just print. These protocols are the seam that makes those swappable, so the
runtime is testable on Linux without any Windows or web-framework dependency.

Pure interfaces only: no OS imports, no I/O beyond `ConsoleNotifier`'s print.
"""

from __future__ import annotations

import sys
from typing import Any, Protocol, runtime_checkable

from theseus.command_reports import Executed, Failed
from theseus.commands import command_type
from theseus.stimulus_log import StimulusEvent


@runtime_checkable
class ChatSurface(Protocol):
    """Where the surrogate's spoken messages appear, and whether anyone is watching.

    `is_focused` lets presence logic behave differently when the user is looking —
    e.g. a notification can be skipped if the chat window already has focus.
    """

    def publish_agent_message(self, text: str) -> None:
        """Show one agent message on the surface."""
        ...

    def is_focused(self) -> bool:
        """Whether the surface currently has the user's attention."""
        ...


@runtime_checkable
class Notifier(Protocol):
    """How the surrogate raises the user's attention outside the chat surface."""

    def notify(self, title: str, body: str) -> None:
        """Show one native notification."""
        ...


class ConsoleNotifier:
    """Prints notifications to stdout — the headless/dev fallback for `Notifier`."""

    def notify(self, title: str, body: str) -> None:
        print(f"[notify] {title}: {body}", file=sys.stdout)


class WindowsPresence:
    """Renders host commands into the OS-facing surfaces (issue #96).

    The `CommandExecutor`'s renderer: turns a `command.say` into a chat message
    (plus a toast when nobody is watching the chat) and a `command.notify` into a
    native notification. Every command yields exactly one `Outcome` — a malformed
    command is a `Failed` report, never an exception, so the report path home
    stays honest even when the host sends garbage.
    """

    def __init__(self, chat: ChatSurface, notifier: Notifier) -> None:
        self.chat = chat
        self.notifier = notifier

    def __call__(self, event: StimulusEvent) -> Executed | Failed:
        payload = self._payload(event)
        if payload is None:
            return Failed(reason="missing or malformed payload")
        if event.type == command_type("say"):
            text = payload.get("text")
            if not isinstance(text, str) or not text:
                return Failed(reason="command.say payload missing 'text'")
            self.chat.publish_agent_message(text)
            if not self.chat.is_focused():
                # ponytail: fixed toast title; a per-message title needs a host contract change
                self.notifier.notify("New message", text)
            return Executed()
        if event.type == command_type("notify"):
            title = payload.get("title")
            body = payload.get("body")
            if not isinstance(title, str) or not isinstance(body, str):
                return Failed(reason="command.notify payload missing 'title'/'body'")
            self.notifier.notify(title, body)
            return Executed()
        return Failed(reason=f"unknown command verb: {event.type!r}")

    @staticmethod
    def _payload(event: StimulusEvent) -> dict[str, Any] | None:
        content = event.content
        if not isinstance(content, dict):
            return None
        payload = content.get("payload")
        return payload if isinstance(payload, dict) else None
