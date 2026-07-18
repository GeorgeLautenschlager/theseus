from __future__ import annotations

from pathlib import Path

from theseus.tools.bash import BashTool
from theseus.tools.edit import EditTool
from theseus.tools.find import FindTool
from theseus.tools.grep import GrepTool
from theseus.tools.ls import LsTool
from theseus.tools.read import ReadTool
from theseus.tools.tool import Tool
from theseus.tools.write import WriteTool


def _keyed(tools: list[Tool]) -> dict[str, Tool]:
    return {tool.name: tool for tool in tools}


def all_tools(cwd: str | Path | None = None) -> dict[str, Tool]:
    """Every tool, keyed by name."""
    return _keyed(
        [
            ReadTool(cwd),
            WriteTool(cwd),
            EditTool(cwd),
            LsTool(cwd),
            FindTool(cwd),
            GrepTool(cwd),
            BashTool(cwd),
        ]
    )


def read_only_tools(cwd: str | Path | None = None) -> dict[str, Tool]:
    """Inspection-only tools — no filesystem mutation, no shell."""
    return _keyed([ReadTool(cwd), LsTool(cwd), FindTool(cwd), GrepTool(cwd)])


def coding_tools(cwd: str | Path | None = None) -> dict[str, Tool]:
    """The mutation-capable subset: read, write, edit, bash."""
    return _keyed([ReadTool(cwd), WriteTool(cwd), EditTool(cwd), BashTool(cwd)])
