from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from theseus.tools.path_utils import resolve_path
from theseus.tools.tool import ToolResult
from theseus.tools.truncate import truncate

DEFAULT_LIMIT = 100
MAX_LINE_LENGTH = 200
MAX_BYTES = 64 * 1024
_PRUNE_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache"}


class GrepTool:
    name = "grep"
    description = (
        "Search file contents for a pattern, returning matching lines with file paths and "
        "line numbers. Uses ripgrep when available (honoring .gitignore); otherwise scans "
        "with Python's regex engine."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex (or literal, if `literal`) to search for."},
            "path": {"type": "string", "description": "File or directory to search (defaults to cwd)."},
            "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'."},
            "ignoreCase": {"type": "boolean", "description": "Case-insensitive search."},
            "literal": {"type": "boolean", "description": "Treat `pattern` as a literal string."},
            "context": {"type": "integer", "description": "Lines of context before/after each match."},
            "limit": {"type": "integer", "description": "Maximum matches to return."},
        },
        "required": ["pattern"],
    }

    def __init__(self, cwd: str | Path | None = None) -> None:
        self.cwd = Path(cwd) if cwd else Path.cwd()

    def execute(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignoreCase: bool = False,
        literal: bool = False,
        context: int = 0,
        limit: int | None = None,
    ) -> ToolResult:
        root = resolve_path(path, self.cwd) if path else Path(self.cwd)
        if not root.exists():
            return ToolResult(f"Path not found: {root}", is_error=True)
        cap = limit if limit is not None else DEFAULT_LIMIT

        rg = shutil.which("rg")
        try:
            if rg:
                lines = self._grep_with_rg(rg, pattern, root, glob, ignoreCase, literal, context, cap)
            else:
                lines = self._grep_with_re(pattern, root, glob, ignoreCase, literal, cap)
        except re.error as exc:
            return ToolResult(f"Invalid regex: {exc}", is_error=True)
        except (OSError, subprocess.SubprocessError) as exc:
            return ToolResult(f"grep failed: {exc}", is_error=True)

        if not lines:
            return ToolResult("(no matches)", details={"count": 0})
        result = truncate("\n".join(lines[:cap]), max_lines=cap, max_bytes=MAX_BYTES)
        note = f"\n[truncated — {len(lines)}+ matches]" if (len(lines) > cap or result.truncated) else ""
        return ToolResult(result.text + note, details={"count": len(lines)})

    def _grep_with_rg(
        self, rg, pattern, root, glob, ignore_case, literal, context, cap
    ) -> list[str]:
        args = [rg, "--json", "--max-count", str(cap)]
        if ignore_case:
            args.append("--ignore-case")
        if literal:
            args.append("--fixed-strings")
        if context:
            args += ["--context", str(context)]
        if glob:
            args += ["--glob", glob]
        args += [pattern, str(root)]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
        out: list[str] = []
        for raw in proc.stdout.splitlines():
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("type") not in ("match", "context"):
                continue
            data = event["data"]
            rel = self._relative(data["path"]["text"], root)
            line_no = data["line_number"]
            text = data["lines"]["text"].rstrip("\n")[:MAX_LINE_LENGTH]
            sep = ":" if event["type"] == "match" else "-"
            out.append(f"{rel}{sep}{line_no}{sep}{text}")
            if len([o for o in out if ":" in o]) >= cap:
                break
        return out

    def _grep_with_re(self, pattern, root, glob, ignore_case, literal, cap) -> list[str]:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(re.escape(pattern) if literal else pattern, flags)
        files = [root] if root.is_file() else self._walk(root, glob)
        out: list[str] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    rel = self._relative(str(f), root)
                    out.append(f"{rel}:{i}:{line[:MAX_LINE_LENGTH]}")
                    if len(out) >= cap:
                        return out
        return out

    @staticmethod
    def _walk(root: Path, glob: str | None):
        for p in root.rglob(glob or "*"):
            if not p.is_file():
                continue
            if any(part in _PRUNE_DIRS for part in p.relative_to(root).parts):
                continue
            yield p

    @staticmethod
    def _relative(path_text: str, root: Path) -> str:
        p = Path(path_text)
        try:
            base = root if root.is_dir() else root.parent
            return str(p.relative_to(base))
        except ValueError:
            return path_text
