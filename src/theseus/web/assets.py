"""Shared web assets and SSE frame formatting (issue #116, item 1).

Public seam for the packaged chat templates/static directories and the SSE
`event: message` frame shape, used by both `WebChatUIObserver` and
`SurrogateWebUI` (replacing cross-module private reuse).
"""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def format_sse_event(html_fragment: str) -> str:
    lines = html_fragment.splitlines() or [""]
    payload = "\n".join(f"data: {line}" for line in lines)
    return f"event: message\n{payload}\n\n"