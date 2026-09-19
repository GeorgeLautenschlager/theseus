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
from typing import Protocol, runtime_checkable


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
