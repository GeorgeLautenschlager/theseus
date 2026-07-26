"""Plain-text previews of agent replies, for desktop notifications.

The chat UI renders replies as markdown, but a `Notification` body is plain
text — so the preview is built from the raw reply text before rendering,
rather than by stripping tags back out of the rendered HTML.
"""

from __future__ import annotations

import re

_LINE_MARKER = re.compile(r"^\s*(?:#{1,6}\s*|[-*]\s+)")
_WHITESPACE = re.compile(r"\s+")
_FALLBACK = "New message"


def notification_preview(text: str, limit: int = 140) -> str:
    """Condense `text` into a single line of at most ~`limit` characters.

    Markdown structure that reads as noise out of context (heading markers,
    list bullets, backticks) is dropped; the result is truncated on a word
    boundary and marked with an ellipsis.
    """
    lines = (_LINE_MARKER.sub("", line) for line in text.splitlines())
    condensed = _WHITESPACE.sub(" ", " ".join(lines)).replace("`", "").strip()
    if not condensed:
        return _FALLBACK
    if len(condensed) <= limit:
        return condensed
    head = condensed[:limit]
    boundary = head.rfind(" ")
    if boundary > 0:
        head = head[:boundary]
    return head.rstrip() + "…"
