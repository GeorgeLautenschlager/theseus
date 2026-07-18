from __future__ import annotations

from pathlib import Path
from typing import Any

from theseus.tools.path_utils import resolve_path
from theseus.tools.tool import ToolResult

_BOM = "﻿"


class EditTool:
    name = "edit"
    description = (
        "Edit a file by exact text replacement. Each `oldText` must match exactly once in "
        "the file; all edits are applied against the original content, so overlapping edits "
        "must be merged into a single one."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative or absolute."},
            "edits": {
                "type": "array",
                "description": "One or more exact-match replacements.",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {"type": "string", "description": "Text to find (must be unique)."},
                        "newText": {"type": "string", "description": "Replacement text."},
                    },
                    "required": ["oldText", "newText"],
                },
            },
        },
        "required": ["path", "edits"],
    }

    def __init__(self, cwd: str | Path | None = None) -> None:
        self.cwd = Path(cwd) if cwd else Path.cwd()

    def execute(self, path: str, edits: list[dict[str, str]]) -> ToolResult:
        if not edits:
            return ToolResult("No edits provided.", is_error=True)
        target = resolve_path(path, self.cwd)
        if not target.exists() or target.is_dir():
            return ToolResult(f"File not found: {target}", is_error=True)
        try:
            # newline="" disables universal-newline translation so we can detect and
            # faithfully restore the file's original CRLF/LF style.
            with target.open("r", encoding="utf-8", newline="") as f:
                raw = f.read()
        except OSError as exc:
            return ToolResult(f"Could not read {target}: {exc}", is_error=True)

        # Normalize line endings + BOM for matching, remember them to restore on write.
        had_bom = raw.startswith(_BOM)
        body = raw[len(_BOM):] if had_bom else raw
        uses_crlf = "\r\n" in body
        normalized = body.replace("\r\n", "\n")

        # Resolve every edit to a non-overlapping span in the ORIGINAL content.
        spans: list[tuple[int, int, str]] = []
        for i, edit in enumerate(edits):
            old = edit.get("oldText", "").replace("\r\n", "\n")
            new = edit.get("newText", "").replace("\r\n", "\n")
            if old == "":
                return ToolResult(f"Edit {i}: `oldText` is empty.", is_error=True)
            count = normalized.count(old)
            if count == 0:
                return ToolResult(f"Edit {i}: `oldText` not found in file.", is_error=True)
            if count > 1:
                return ToolResult(
                    f"Edit {i}: `oldText` is not unique ({count} matches); add surrounding "
                    "context to disambiguate.",
                    is_error=True,
                )
            start = normalized.index(old)
            spans.append((start, start + len(old), new))

        spans.sort(key=lambda s: s[0])
        for (_, prev_end, _), (next_start, _, _) in zip(spans, spans[1:]):
            if next_start < prev_end:
                return ToolResult(
                    "Edits overlap; merge them into a single edit.", is_error=True
                )

        out: list[str] = []
        cursor = 0
        for start, end, new in spans:
            out.append(normalized[cursor:start])
            out.append(new)
            cursor = end
        out.append(normalized[cursor:])
        result = "".join(out)

        if uses_crlf:
            result = result.replace("\n", "\r\n")
        if had_bom:
            result = _BOM + result
        try:
            # newline="" leaves our explicit line endings exactly as assembled.
            with target.open("w", encoding="utf-8", newline="") as f:
                f.write(result)
        except OSError as exc:
            return ToolResult(f"Could not write {target}: {exc}", is_error=True)
        return ToolResult(
            f"Applied {len(edits)} edit(s) to {target}",
            details={"path": str(target), "edits": len(edits)},
        )
