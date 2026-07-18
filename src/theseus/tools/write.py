from __future__ import annotations

from pathlib import Path
from typing import Any

from theseus.tools.path_utils import resolve_path
from theseus.tools.tool import ToolResult


class WriteTool:
    name = "write"
    description = (
        "Write content to a file. Creates the file (and any missing parent directories) "
        "if it does not exist, and overwrites it if it does."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative or absolute."},
            "content": {"type": "string", "description": "The full contents to write."},
        },
        "required": ["path", "content"],
    }

    def __init__(self, cwd: str | Path | None = None) -> None:
        self.cwd = Path(cwd) if cwd else Path.cwd()

    def execute(self, path: str, content: str) -> ToolResult:
        target = resolve_path(path, self.cwd)
        if target.is_dir():
            return ToolResult(f"Path is a directory, not a file: {target}", is_error=True)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(f"Could not write {target}: {exc}", is_error=True)
        byte_count = len(content.encode("utf-8"))
        return ToolResult(
            f"Wrote {byte_count} bytes to {target}",
            details={"path": str(target), "bytes": byte_count},
        )
