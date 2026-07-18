from __future__ import annotations

from pathlib import Path
from typing import Any

from theseus.tools.path_utils import resolve_path
from theseus.tools.tool import ToolResult
from theseus.tools.truncate import truncate

# Hard caps on what a single read hands back to the model.
DEFAULT_LINE_LIMIT = 2000
MAX_BYTES = 256 * 1024


class ReadTool:
    name = "read"
    description = (
        "Read the contents of a text file. Optionally start at a 1-based line `offset` "
        "and cap the number of lines with `limit`."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative or absolute."},
            "offset": {"type": "integer", "description": "1-based line to start from."},
            "limit": {"type": "integer", "description": "Maximum number of lines to return."},
        },
        "required": ["path"],
    }

    def __init__(self, cwd: str | Path | None = None) -> None:
        self.cwd = Path(cwd) if cwd else Path.cwd()

    def execute(
        self, path: str, offset: int | None = None, limit: int | None = None
    ) -> ToolResult:
        target = resolve_path(path, self.cwd)
        if not target.exists():
            return ToolResult(f"File not found: {target}", is_error=True)
        if target.is_dir():
            return ToolResult(f"Path is a directory, not a file: {target}", is_error=True)
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(f"Could not read {target}: {exc}", is_error=True)

        lines = content.splitlines(keepends=True)
        start = (offset - 1) if offset else 0
        if start < 0 or (lines and start >= len(lines)):
            return ToolResult(
                f"offset {offset} is out of range (file has {len(lines)} lines).",
                is_error=True,
            )
        window_limit = limit if limit is not None else DEFAULT_LINE_LIMIT
        selected = lines[start : start + window_limit]
        result = truncate("".join(selected), max_lines=window_limit, max_bytes=MAX_BYTES)

        note = ""
        shown_through = start + result.text.count("\n")
        if result.truncated or (limit is None and len(lines) > shown_through):
            note = (
                f"\n\n[truncated — file has {len(lines)} lines; "
                f"re-read with a higher `offset`/`limit` to see more]"
            )
        return ToolResult(result.text + note, details={"lines": len(lines)})
