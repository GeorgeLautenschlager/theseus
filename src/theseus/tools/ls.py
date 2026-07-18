from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from theseus.tools.path_utils import resolve_path
from theseus.tools.tool import ToolResult
from theseus.tools.truncate import truncate

DEFAULT_LIMIT = 500
MAX_BYTES = 64 * 1024


class LsTool:
    name = "ls"
    description = (
        "List the entries of a directory, sorted alphabetically. Directories are shown "
        "with a trailing '/'. Dotfiles are included."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (defaults to cwd)."},
            "limit": {"type": "integer", "description": "Maximum entries to return."},
        },
        "required": [],
    }

    def __init__(self, cwd: str | Path | None = None) -> None:
        self.cwd = Path(cwd) if cwd else Path.cwd()

    def execute(self, path: str | None = None, limit: int | None = None) -> ToolResult:
        target = resolve_path(path, self.cwd) if path else Path(self.cwd)
        if not target.exists():
            return ToolResult(f"Directory not found: {target}", is_error=True)
        if not target.is_dir():
            return ToolResult(f"Not a directory: {target}", is_error=True)
        try:
            entries = sorted(os.listdir(target), key=str.lower)
        except OSError as exc:
            return ToolResult(f"Could not list {target}: {exc}", is_error=True)

        cap = limit if limit is not None else DEFAULT_LIMIT
        rendered = []
        for name in entries[:cap]:
            suffix = "/" if (target / name).is_dir() else ""
            rendered.append(f"{name}{suffix}")
        result = truncate("\n".join(rendered), max_lines=cap, max_bytes=MAX_BYTES)

        note = ""
        if len(entries) > cap or result.truncated:
            note = f"\n[truncated — {len(entries)} entries total]"
        body = result.text or "(empty directory)"
        return ToolResult(body + note, details={"count": len(entries)})
