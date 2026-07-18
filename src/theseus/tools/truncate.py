from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Truncated:
    text: str
    truncated: bool


def _cap_bytes(text: str, max_bytes: int, keep: str) -> tuple[str, bool]:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text, False
    clipped = data[:max_bytes] if keep == "head" else data[-max_bytes:]
    # errors="ignore" drops a partial multibyte char left at the cut boundary.
    return clipped.decode("utf-8", errors="ignore"), True


def truncate(text: str, max_lines: int, max_bytes: int, keep: str = "head") -> Truncated:
    """Clamp `text` to at most `max_lines` lines and `max_bytes` bytes.

    `keep="head"` retains the beginning (used by `read`, `ls`, `find`); `keep="tail"`
    retains the end (used by `bash`, where the last lines of output matter most).
    """
    lines = text.splitlines(keepends=True)
    truncated = False
    if len(lines) > max_lines:
        truncated = True
        lines = lines[:max_lines] if keep == "head" else lines[-max_lines:]
    text = "".join(lines)
    text, byte_truncated = _cap_bytes(text, max_bytes, keep)
    return Truncated(text=text, truncated=truncated or byte_truncated)
