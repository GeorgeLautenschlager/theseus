from __future__ import annotations

from pathlib import Path


def resolve_path(path: str, cwd: str | Path) -> Path:
    """Resolve `path` against `cwd` (Pi's `resolveToCwd`).

    Relative paths are taken relative to the tool's working directory; `~` is expanded.
    Absolute paths are honored as-is. The result is normalized but not required to exist
    (so `write` can target a new file).
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()
