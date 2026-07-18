from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from theseus.tools.path_utils import resolve_path
from theseus.tools.tool import ToolResult
from theseus.tools.truncate import truncate

DEFAULT_LIMIT = 1000
MAX_BYTES = 64 * 1024

# Pure-Python fallback can't parse .gitignore; skip the usual noise directories instead.
_PRUNE_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache"}


class FindTool:
    name = "find"
    description = (
        "Search for files by glob pattern (e.g. '*.py', '**/*.json', 'src/**/*_test.py'). "
        "Returns matching paths relative to the search directory. Uses `fd` when available "
        "(honoring .gitignore); otherwise walks the tree, skipping common noise directories."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern to match file names."},
            "path": {"type": "string", "description": "Directory to search (defaults to cwd)."},
            "limit": {"type": "integer", "description": "Maximum results to return."},
        },
        "required": ["pattern"],
    }

    def __init__(self, cwd: str | Path | None = None) -> None:
        self.cwd = Path(cwd) if cwd else Path.cwd()

    def execute(
        self, pattern: str, path: str | None = None, limit: int | None = None
    ) -> ToolResult:
        root = resolve_path(path, self.cwd) if path else Path(self.cwd)
        if not root.is_dir():
            return ToolResult(f"Not a directory: {root}", is_error=True)
        cap = limit if limit is not None else DEFAULT_LIMIT

        fd = shutil.which("fd") or shutil.which("fdfind")
        try:
            matches = (
                self._find_with_fd(fd, pattern, root, cap)
                if fd
                else self._find_with_glob(pattern, root, cap)
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ToolResult(f"find failed: {exc}", is_error=True)

        if not matches:
            return ToolResult("(no matches)", details={"count": 0})
        result = truncate("\n".join(matches[:cap]), max_lines=cap, max_bytes=MAX_BYTES)
        note = f"\n[truncated — {len(matches)}+ matches]" if (len(matches) > cap or result.truncated) else ""
        return ToolResult(result.text + note, details={"count": len(matches)})

    @staticmethod
    def _find_with_fd(fd: str, pattern: str, root: Path, cap: int) -> list[str]:
        proc = subprocess.run(
            [fd, "--glob", pattern, "--type", "f", "--max-results", str(cap), ".", str(root)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln]
        return [str(Path(ln).relative_to(root)) if Path(ln).is_absolute() else ln for ln in lines]

    @staticmethod
    def _find_with_glob(pattern: str, root: Path, cap: int) -> list[str]:
        # Recursive-by-default when the pattern is a bare name (mirrors fd); honor an
        # explicit path/'**' pattern as written.
        globber = root.glob if ("/" in pattern or "**" in pattern) else root.rglob
        out: list[str] = []
        for p in globber(pattern):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(part in _PRUNE_DIRS for part in rel.parts):
                continue
            out.append(str(rel))
            if len(out) >= cap + 1:  # one extra so callers can detect truncation
                break
        return sorted(out)
