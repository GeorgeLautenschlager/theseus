"""Shared JSONL persistence discipline for the layered memory stores.

One write contract, shared by every file-backed layer (knowledge, memory, wisdom)
and by the module's ledger and dead-letter files: one record per line, append,
flush, fsync. Reads tolerate a torn trailing line (crash mid-write); corruption
of an *interior* line is a real error and raises — same contract as StimulusLog
and MemoryStore, so a crash in any of these stores recovers the same way.

Layers are separate collaborators behind this common interface; nothing here
knows what a record means. `LayerHit` is the one shared retrieval shape: a
ranked (id, text, score) triple, so the module can fuse layers without knowing
their record types.
"""

from __future__ import annotations

import json
import os
import tempfile
import re
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LayerHit:
    """One ranked retrieval hit from a layer. `text` is what an agent would read;
    `score` is layer-internal (higher is better); fusion only ever sees ranks."""

    id: str
    text: str
    score: float


def terms(text: str) -> set[str]:
    stop = {"the", "what", "when", "where", "which", "who", "how", "does", "did",
            "has", "have", "was", "were", "are", "for", "and", "with", "about", "that", "this"}
    return {t for t in re.findall(r"\w+", text.casefold()) if len(t) > 2 and t not in stop}


def lexical_score(query: str, text: str) -> float:
    wanted = terms(query)
    return len(wanted & terms(text)) / len(wanted) if wanted else 0.0


def valid_vector(vector, dimension: int | None = None) -> bool:
    return (isinstance(vector, (list, tuple)) and bool(vector)
            and (dimension is None or len(vector) == dimension)
            and all(isinstance(v, (float, int)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in vector)
            and any(v != 0 for v in vector))


@contextmanager
def _file_lock(stream):
    if os.name == "nt":
        import msvcrt
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def ensure_store(path: str | os.PathLike[str]) -> Path:
    """Create the store's parent dir and an empty file if absent; return a Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)
    return p


def append_record(path: str | os.PathLike[str], line: str) -> None:
    """Append one serialized record, durable before return. Creates parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    json.loads(line)
    with open(p, "a+b") as f, _file_lock(f):
        # Repair only the unterminated suffix; complete older records stay intact.
        f.seek(0, os.SEEK_END)
        end = f.tell()
        start = end
        tail = b""
        while start:
            size = min(start, 4096)
            start -= size
            f.seek(start)
            tail = f.read(size) + tail
            boundary = tail.rfind(b"\n")
            if boundary >= 0:
                start += boundary + 1
                tail = tail[boundary + 1:]
                break
        if tail:
            try:
                json.loads(tail)
            except (ValueError, UnicodeDecodeError):
                f.truncate(start)
            else:
                f.seek(0, os.SEEK_END)
                f.write(b"\n")
        f.seek(0, os.SEEK_END)
        committed = f.tell()
        try:
            f.write((line + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            f.truncate(committed)
            f.flush()
            raise
    fsync_directory(p.parent)


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return  # Windows does not expose directory fsync through os.open.
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: dict) -> None:
    """Publish a complete preparation record before any layer is changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as f:
            name = Path(f.name)
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        fsync_directory(path.parent)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


@contextmanager
def store_lock(directory: Path):
    """Serialize module transactions across threads/processes owning one home."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".memory.lock").open("a+b") as f, _file_lock(f):
        yield


def load_lines(path: str | os.PathLike[str]) -> list[str]:
    """All complete lines. A torn final line (crash mid-write) is dropped, never
    raised; a corrupt interior line raises ValueError."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "rb") as f:
        lines = f.readlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.rstrip(b"\n")
        if not stripped:
            continue
        try:
            stripped = stripped.decode("utf-8")
            json.loads(stripped)  # interior corruption check; layers parse for real
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_last = i == len(lines) - 1
            if is_last and not line.endswith(b"\n"):
                break  # torn final write — recover by dropping it
            raise ValueError(f"corrupt interior record at line {i}: {exc}") from exc
        out.append(stripped)
    return out
