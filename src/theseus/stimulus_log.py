"""StimulusLog — the immutable, append-only event log.

This is the bedrock. Everything downstream (segments, episodes) is a disposable
projection over this; if a transform improves, we replay the log and rebuild.
So the log's one job is to be the dumbest, most bulletproof link in the chain:
append, fsync, survive crashes, and bake in *no* substrate assumptions.

A record is a typed StimulusEvent — NOT a "turn", NOT a prompt/response pair.
A conversational exchange is just one `type` whose payload lives in `content`.
An egocentric capture or a game observation is another. All three agents can
therefore share one log.

Appends are also announced to in-process listeners (`subscribe`) — how a running
loop learns that something landed without polling the file. The notification
carries no policy: it is a bare "this happened", fired after the write is durable,
so the log still assumes nothing about who is listening or what they do about it.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

# --- Sortable IDs (minimal ULID) ------------------------------------------------
# 48-bit ms timestamp + 80-bit randomness, Crockford base32. Lexically sortable
# by creation time, no coordination needed. Good enough for a v1 WAL at human
# rates; strict intra-millisecond monotonicity is not guaranteed (file order is
# the tiebreaker, and the log is append-only so file order is stable).
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _b32(n: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


def new_id(ms: int | None = None) -> str:
    ms = int(time.time() * 1000) if ms is None else ms
    rand = int.from_bytes(os.urandom(10), "big")
    return _b32(ms, 10) + _b32(rand, 16)  # 26 chars


# The origin a log stamps on its own events when none is configured. `origin` answers
# *where* an event entered the system (`kitchen-surrogate`, `android-01`, `webchat`);
# `actor` answers *who* produced it. They stay separate: the same mind reaches the agent
# through several channels, and collapsing them makes the agent either believe in two
# users or unable to decide which mouth to answer from.
DEFAULT_ORIGIN = "local"


# --- Event ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StimulusEvent:
    """One thing that happened, plus the envelope that lets it be replicated.

    `content` is the surrogate protocol's `payload` under its original name — the alias
    is documented rather than renamed, because renaming it churns every module for no gain.
    Likewise `ts` is the protocol's `event_ts`.

    The two timestamps are both retained, and they answer different questions:
    `appended_ts` is authoritative for *log order*, `ts` for *meaning*. The gap between
    them is observable clock skew — a surrogate drifting 900ms shows up in the trace
    instead of silently scrambling the agent's sense of before-and-after.

    `appended_ts` is `None` only as a constructor sentinel meaning "derive it from `ts`". On a
    constructed instance it is always a `datetime`, so downstream ordering code needs no
    defensive `or event.ts`.
    """

    id: str
    ts: datetime         # event_ts: when it happened, by the producer's clock
    actor: str           # who/what produced it ("george", "tam", "env", "sensor")
    type: str            # "exchange" | "capture" | "observation" | ...
    content: dict[str, Any]  # type-specific payload; e.g. {"prompt":..,"response":..}
    origin: str = DEFAULT_ORIGIN  # where it entered the system; assigned by the producer
    seq: int | None = None        # monotonic per origin. Not contiguous — gaps are legal.
    appended_ts: datetime | None = None  # when it landed on this log

    def __post_init__(self) -> None:
        # An event that no log has appended yet still needs an arrival timestamp, so
        # ordering code never has to special-case None.
        #
        # Deriving it only when it is None also means `dataclasses.replace(event, ts=...)`
        # leaves `appended_ts` where it was — correcting an event's own clock must not move
        # the moment it arrived.
        if self.appended_ts is None:
            object.__setattr__(self, "appended_ts", self.ts)

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "ts": self.ts.astimezone(timezone.utc).isoformat(),
                "actor": self.actor,
                "type": self.type,
                "content": self.content,
                "origin": self.origin,
                "seq": self.seq,
                "appended_ts": self.appended_ts.astimezone(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(
        cls, line: str, *, default_origin: str = DEFAULT_ORIGIN
    ) -> "StimulusEvent":
        """Parse one log line. Lines written before the envelope existed are still valid:
        `origin` falls back to `default_origin` (the reading log's own origin), `seq` to
        None, and `appended_ts` to `ts`. Old lines stay readable in place — there is no
        migration.

        An `origin` that is present but empty is treated as absent and coerced to
        `default_origin` too: an empty origin name is a misconfiguration, and filing those
        events under the reading log's own origin is the least surprising thing to do with
        them. It does mean `origin=""` is the one value that does not survive a round-trip.
        """
        d = json.loads(line)
        ts = datetime.fromisoformat(d["ts"])
        appended_ts = d.get("appended_ts")
        return cls(
            id=d["id"],
            ts=ts,
            actor=d["actor"],
            type=d["type"],
            content=d["content"],
            origin=d.get("origin") or default_origin,
            seq=d.get("seq"),
            appended_ts=datetime.fromisoformat(appended_ts) if appended_ts else ts,
        )


# --- Log ------------------------------------------------------------------------
class StimulusLog:
    """Append-only JSONL. One event per line. fsync per append.

    Reads tolerate a torn trailing line (crash mid-write): the partial final
    line is dropped on read, never raised. Corruption of an *interior* line is
    a real error and is raised, because that should never happen to an
    append-only file and silently skipping it would hide data loss.

    Listeners registered with `subscribe` are called with each appended event, on
    the appending thread, once the write is durable.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._listeners: list[Callable[[StimulusEvent], None]] = []
        self._listener_lock = threading.Lock()

    def subscribe(
        self, listener: Callable[[StimulusEvent], None]
    ) -> Callable[[], None]:
        """Call `listener` with every event appended from here on; returns a callable
        that unsubscribes it again.

        Notification is synchronous, on whichever thread did the append, and happens
        only after the record is fsynced — a listener can never see an event that a
        crash would un-append. It runs inside `append`, so listeners must be cheap and
        must not block: the canonical one sets a `threading.Event` (see `Autocore.wake`)
        and returns. Anything heavier belongs on its own thread.

        This is an in-process signal, not a file watch. A writer in another process
        appends to the same file without anyone here hearing about it; readers that
        must survive that keep polling `read_all`.
        """
        with self._listener_lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._listener_lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def append(
        self,
        actor: str,
        type: str,
        content: dict[str, Any],
        ts: datetime | None = None,
    ) -> StimulusEvent:
        ts = ts or datetime.now(timezone.utc)
        event = StimulusEvent(id=new_id(int(ts.timestamp() * 1000)),
                              ts=ts, actor=actor, type=type, content=content)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._notify(event)
        return event

    def _notify(self, event: StimulusEvent) -> None:
        """Fan the event out to listeners, snapshotting the list so a listener may
        subscribe or unsubscribe from inside its own callback."""
        with self._listener_lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                # The log is the bedrock: a buggy listener must never turn a durable
                # append into a raise, nor stop the other listeners hearing about it.
                traceback.print_exc()

    def read_all(self) -> list[StimulusEvent]:
        events: list[StimulusEvent] = []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            try:
                events.append(StimulusEvent.from_json(stripped))
            except (json.JSONDecodeError, KeyError) as exc:
                is_last = i == len(lines) - 1
                if is_last and not line.endswith("\n"):
                    break  # torn final write — recover by dropping it
                raise ValueError(f"corrupt interior record at line {i}: {exc}") from exc
        return events

    def read_range(self, start_id: str, end_id: str) -> list[StimulusEvent]:
        """Inclusive [start_id, end_id]. IDs are lexically sortable, so a span
        is just a string range — robust to re-encoding, unlike line offsets."""
        return [e for e in self.read_all() if start_id <= e.id <= end_id]

    def __iter__(self) -> Iterator[StimulusEvent]:
        return iter(self.read_all())
