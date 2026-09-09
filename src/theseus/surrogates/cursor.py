"""The surrogate's durable memory of how far its host has acknowledged its stream.

The replicator ships the local log from this cursor forward, so the cursor is the
surrogate's position in the protocol — and unlike `HighWaterMarks`, it cannot be
derived from the log. The log records what the surrogate *did*; only the host knows
what it *accepted*, and that answer arrives over the wire as a `2xx`. So this is a
sidecar beside the log, written after each ack, never before or during: behind is
recoverable (a re-sent batch is deduped to another `2xx`), ahead is not (skipped
events are gone). A process crash between the ack and the write costs exactly one
re-sent batch. A *power* loss can cost one more: `os.replace` is atomic, but the rename
is not durable until the parent directory is fsynced, which this does not do. Both
failures land on the recoverable side, which is why the gap is affordable.

Recovery fails only in the safe direction. A missing file means "nothing acked yet" —
deliberately `None`, not `0`. An unreadable, malformed, or foreign-origin file also
loads as `None`: the surrogate re-sends from the start and the host dedupes it, which
is redundant work; refusing to start would take a surrogate offline over state whose
only failure mode is that same redundant work.

This cursor has a second use on the downstream command channel (issue #34): there it
is how far the surrogate has **executed** the host's commands, not how far the host
has acked the surrogate's stream — the roles flip, but the mechanism is identical
(durable, monotonic, never backwards, `None` until first advanced). The
`SseCommandChannel` sends `acked_seq` as `Last-Event-ID` on every reconnect, and the
caller advances after executing, not on receipt — at-least-once, because nothing
dedupes a spoken sentence. A reader who has only seen the replicator will think the
file is misplaced; it is the same position, read in the other direction.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading

logger = logging.getLogger(__name__)


class AckedCursor:
    """The highest seq this surrogate's host has acknowledged, persisted beside the log."""

    def __init__(self, path: str | os.PathLike[str], origin: str) -> None:
        self._path = os.fspath(path)
        self._origin = origin
        # `advance` is a read-modify-write and the replicator may be driven from more
        # than one thread; unlocked, two threads can interleave their check-and-set and
        # move the cursor *backwards* — the direction that re-sends batches forever.
        # Same reason `HighWaterMarks` locks its marks.
        self._lock = threading.Lock()
        self._acked_seq: int | None = self._load()

    @property
    def acked_seq(self) -> int | None:
        """Highest acked seq, or None if nothing has ever been acknowledged."""
        with self._lock:
            return self._acked_seq

    def advance(self, seq: int) -> None:
        """Record an ack, durably. Never moves backwards."""
        if seq < 1:
            # Matching `StimulusLog`'s rule: seqs start at 1, and a cursor below that
            # would sit under every real seq.
            raise ValueError(f"seqs start at 1; got {seq!r}")
        with self._lock:
            current = self._acked_seq
            if current is not None and seq <= current:
                return  # behind or equal to what's already durable: nothing to record
            payload = json.dumps({"origin": self._origin, "acked_seq": seq}) + "\n"
            self._write_replaced(payload)
            # Only after the replace landed: a failed write must not leave memory
            # ahead of disk, which is the direction that skips real events.
            self._acked_seq = seq

    def _load(self) -> int | None:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None  # fresh start: nothing acked yet, and no log about it
        except (OSError, ValueError) as exc:
            logger.error(
                "acked cursor %s is unreadable or malformed (%s); treating it as "
                "nothing-acked — the host will dedupe the re-send",
                self._path, exc,
            )
            return None
        origin = data.get("origin") if isinstance(data, dict) else None
        seq = data.get("acked_seq") if isinstance(data, dict) else None
        if origin != self._origin or not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
            logger.error(
                "acked cursor %s does not hold a usable position for origin %r (found "
                "%r); treating it as nothing-acked — the host will dedupe the re-send",
                self._path, self._origin, data,
            )
            return None
        return seq

    def _write_replaced(self, payload: str) -> None:
        # Atomic replace on top of `StimulusLog`'s durable-write idiom (flush + fsync):
        # a crash mid-write leaves the *previous* file whole, never a truncated cursor.
        # The temp file lives in the same directory so os.replace is a rename(2) on one
        # filesystem; writing anywhere else could leave two files and no replace.
        directory = os.path.dirname(self._path) or "."
        fd, tmp = tempfile.mkstemp(prefix=".acked-cursor-", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path)
        except BaseException:
            # The replace is what makes it durable; if anything before it failed, the
            # previous cursor file is untouched and the temp file is ours to clean up.
            try:
                os.unlink(tmp)
            except OSError:
                pass  # best effort: a stray temp file is harmless, a leaked one isn't
            raise
