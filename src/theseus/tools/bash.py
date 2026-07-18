from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Any

from theseus.tools.tool import ToolResult
from theseus.tools.truncate import truncate

DEFAULT_TIMEOUT = 120  # seconds; used when the model doesn't specify one
MAX_LINES = 5000
MAX_BYTES = 1024 * 1024


class BashTool:
    name = "bash"
    description = (
        "Run a bash command in the working directory and return its combined stdout/stderr. "
        "Output is truncated to the last 5000 lines or 1MB. A non-zero exit or a timeout is "
        "reported as an error, with the exit code in the result details."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute."},
            "timeout": {"type": "integer", "description": "Max seconds to run (default 120)."},
        },
        "required": ["command"],
    }

    def __init__(self, cwd: str | Path | None = None) -> None:
        self.cwd = Path(cwd) if cwd else Path.cwd()

    def execute(self, command: str, timeout: int | None = None) -> ToolResult:
        seconds = timeout if timeout and timeout > 0 else DEFAULT_TIMEOUT
        # start_new_session puts the child in its own process group so a timeout can kill
        # the whole tree (child + any grandchildren), not just the top-level shell.
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = proc.communicate(timeout=seconds)
            exit_code = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            output, _ = proc.communicate()
            exit_code = None
            timed_out = True

        clamped = truncate(output or "", max_lines=MAX_LINES, max_bytes=MAX_BYTES, keep="tail")
        body = clamped.text
        if clamped.truncated:
            body = "[output truncated to the last 5000 lines / 1MB]\n" + body

        if timed_out:
            return ToolResult(
                f"Command timed out after {seconds}s.\n{body}",
                is_error=True,
                details={"exit_code": None, "timed_out": True},
            )
        if exit_code != 0:
            return ToolResult(
                f"Command exited with code {exit_code}.\n{body}",
                is_error=True,
                details={"exit_code": exit_code},
            )
        return ToolResult(body or "(no output)", details={"exit_code": 0})

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
