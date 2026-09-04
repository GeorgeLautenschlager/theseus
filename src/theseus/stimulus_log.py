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
from dataclasses import dataclass
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


def _id_run(ms: int, count: int) -> list[str]:
    """`count` ids for one instant, strictly increasing.

    `new_id` gives each id an independent random suffix, which is fine at human rates
    where the millisecond prefix does the ordering. A replicated batch is not that: fifty
    events land in one write, share a millisecond, and are then distinguished only by
    chance — while `older_batch` in the debug pagination bisects a list of ids and needs it
    sorted ascending.

    So a batch draws one random base and walks it. The base is drawn below `2**80 - count`
    so the walk cannot wrap, which would put the run out of order at exactly the moment it
    matters.

    This orders a batch against *itself*, and nothing more. Two separate writes landing in
    one millisecond — a local `append` beside an `append_many`, or two appends — still draw
    independent bases and can come out in either order, because the two paths share no
    state. Measured on a memory-backed filesystem, that is a coin flip whenever it happens,
    and `older_batch` in the debug pagination bisects file-order ids, so it silently returns
    an empty page when they disagree. Closing that needs the log to remember the last id it
    minted and walk up from it; until then, id order is arrival order within a write and a
    near-certainty between writes, not a guarantee.
    """
    base = int.from_bytes(os.urandom(10), "big") % ((1 << 80) - count)
    return [_b32(ms, 10) + _b32(base + i, 16) for i in range(count)]


# The origin a log stamps on its own events when none is configured. `origin` answers
# *where* an event entered the system (`kitchen-surrogate`, `android-01`, `webchat`);
# `actor` answers *who* produced it. They stay separate: the same mind reaches the agent
# through several channels, and collapsing them makes the agent either believe in two
# users or unable to decide which mouth to answer from.
DEFAULT_ORIGIN = "local"


def _aware(value: datetime) -> datetime:
    """A parsed timestamp, guaranteed comparable.

    A line this log wrote always carries an offset — `to_json` normalises to UTC. A line
    from somewhere else need not, and a naive one mixed with an aware one raises
    `TypeError` inside any sort, which since #28 means every context assembly and so the
    whole cognitive loop. Attaching the host's zone is the same reading `to_json` already
    gives a naive datetime on the way out, applied here so nothing downstream can meet a
    mixed pair.
    """
    return value.astimezone() if value.tzinfo is None else value


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
    seq: int | None = None        # monotonic per origin, not contiguous — gaps are legal.
                                  # None only on a line written before the envelope existed.
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
        ts = _aware(datetime.fromisoformat(d["ts"]))
        appended_ts = d.get("appended_ts")
        return cls(
            id=d["id"],
            ts=ts,
            actor=d["actor"],
            type=d["type"],
            content=d["content"],
            origin=d.get("origin") or default_origin,
            seq=d.get("seq"),
            appended_ts=_aware(datetime.fromisoformat(appended_ts)) if appended_ts else ts,
        )


# --- Log ------------------------------------------------------------------------
class StimulusLog:
    """Append-only JSONL. One event per line. fsync per append, or per batch — see
    `append_many`.

    Reads tolerate a torn trailing line (crash mid-write): the partial final
    line is dropped on read, never raised. Corruption of an *interior* line is
    a real error and is raised, because that should never happen to an
    append-only file and silently skipping it would hide data loss. A batch appended
    through `append_many` is one write, so the same holds for it: a crash can only tear
    its final line.

    Listeners registered with `subscribe` are called with each appended event, on
    the appending thread, once the write is durable.

    A log has an `origin` — the name of the place its own events enter the system. It
    allocates a monotonic `seq` per origin, recovered from the file on first append, and
    accepts replicated events that carry the origin and seq their producer assigned.

    That allocator is per-instance and in-memory, so **exactly one writer per (file,
    origin)** is a hard requirement. Two `StimulusLog` objects appending to one file under
    the same origin — in one process or two — each recover the counter once and then drift
    apart permanently, issuing the same seq twice. Concurrent *readers* are fine, and so is
    a second writer under a genuinely different origin.
    """

    def __init__(
        self, path: str | os.PathLike[str], origin: str = DEFAULT_ORIGIN
    ) -> None:
        if not origin:
            # Configuration, so it fails before anything is created: an origin is the key
            # everything downstream dedupes and routes on.
            raise ValueError("origin must be a non-empty name")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.origin = origin
        self._listeners: list[Callable[[StimulusEvent], None]] = []
        self._listener_lock = threading.Lock()
        self._append_lock = threading.Lock()
        self._next_seq: int | None = None  # recovered from the log on first local append

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

        Notification order is not file order. Listeners are notified after the append lock
        is released, so a listener that appends re-entrantly can have its own event
        announced to later listeners before the one that triggered it. That is the price of
        letting a listener append at all; a listener that needs true order should read the
        file rather than trust the sequence of callbacks.
        """
        with self._listener_lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._listener_lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _recover_next_seq(self) -> int:
        """The seq this log should issue next for its own origin, read back off the file.

        The log is the only durable state, so the counter is derived from it rather than
        kept in a sidecar: a sidecar that disagreed with the file after a crash would
        either skip real events or issue the same seq twice. Seqs start at 1, so 0 is
        always below every real seq and is a safe comparison floor for a reader tracking
        what it has accepted. ("Nothing seen yet" is its own answer, distinct from 0 —
        see `HighWaterMarks.high_water`.)
        """
        highest = 0
        for event in self.read_all():
            if event.origin == self.origin and event.seq is not None:
                highest = max(highest, event.seq)
        return highest + 1

    def _check_replicated(self, origin: str | None, seq: int | None) -> None:
        """The contract for an event this log did not produce: `origin` and `seq` are
        supplied together, the origin is somebody else's, and the seq is a real one.

        Shared by `append` and `append_many` so the two cannot drift — an ingress that
        could get a batch past a check a single append would have caught is exactly the
        hole this protocol's dedupe depends on not existing.
        """
        if not origin:
            raise ValueError("origin must be a non-empty name")
        if origin == self.origin:
            raise ValueError(
                f"{origin!r} is this log's own origin and it allocates those seqs "
                f"itself. A replicated append must arrive under its producer's own "
                f"origin name — two producers sharing one name break the per-origin "
                f"monotonicity that duplicate suppression depends on."
            )
        if seq is None:
            raise ValueError(
                f"a replicated append (origin {origin!r}) must carry the seq its "
                f"producer assigned"
            )
        if isinstance(seq, bool) or not isinstance(seq, int):
            raise ValueError(f"seq must be an integer (got {seq!r})")
        if seq < 1:
            raise ValueError(
                f"seq must be 1 or greater (got {seq!r}); starting at 1 keeps 0 below "
                f"every real seq, as a safe comparison floor for a reader tracking what "
                f"it has accepted"
            )

    def _write_durably(self, payload: str) -> None:
        """Append `payload` and fsync, leaving the file untouched if anything fails.

        Without the rollback, a write that dies part-way — ENOSPC is the realistic one —
        commits a prefix and leaves the file ending mid-line. The next successful append
        then concatenates onto that stump, turning it into an *interior* corrupt record,
        which `read_all` raises on by design and forever. Since `HighWaterMarks` derives
        itself by reading the whole log, that is an agent that cannot boot again.

        A crash, as opposed to an exception, needs no help: it can only ever tear the
        final line, which `read_all` already drops.

        Caller must hold `_append_lock`, so the truncate cannot race another writer.
        """
        committed = self.path.stat().st_size
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            os.truncate(self.path, committed)
            raise

    def _next_local_seq(self) -> int:
        """The next seq for this log's own origin. Caller must hold `_append_lock`."""
        if self._next_seq is None:
            self._next_seq = self._recover_next_seq()
        seq = self._next_seq
        self._next_seq = seq + 1
        return seq

    def append(
        self,
        actor: str,
        type: str,
        content: dict[str, Any],
        ts: datetime | None = None,
        *,
        origin: str | None = None,
        seq: int | None = None,
    ) -> StimulusEvent:
        """Append one event and notify listeners.

        `ts` is the event's own clock — when it happened. `appended_ts` is always minted
        here, and the id is minted from it, so id order stays arrival order however far a
        producer's clock has drifted.

        `origin` and `seq` are supplied together or not at all. Omit both for a local
        append and this log allocates the next seq for its own origin. Supply both for a
        replicated append, carrying the origin and seq the producer already assigned. A
        log never allocates a seq on another producer's behalf, and never accepts one for
        its own origin — either would put two numbering authorities on one origin name and
        break the per-origin monotonicity that duplicate suppression depends on.

        A replicated event is re-identified here: the id is minted from this log's
        `appended_ts`, so the same event has a different id on the producer and on this log.
        Identity across nodes is `(origin, seq)`, never `id`.
        """
        origin = self.origin if origin is None else origin
        # A local append is the one shape that carries no seq, because the allocator below
        # supplies it. Everything else is somebody else's event and goes through the
        # replicated contract.
        if not (origin == self.origin and seq is None):
            self._check_replicated(origin, seq)

        with self._append_lock:
            if seq is None:
                seq = self._next_local_seq()

            # Minted under the lock, not before it: a thread that stamped `ts` and then
            # blocked on another thread's fsync would otherwise land after an event with a
            # later clock, inverting arrival against chronology in a plain single-producer
            # log. A caller-supplied `ts` is the producer's own and is left alone.
            ts = ts or datetime.now(timezone.utc)
            appended_ts = datetime.now(timezone.utc)
            event = StimulusEvent(
                id=new_id(int(appended_ts.timestamp() * 1000)),
                ts=ts,
                actor=actor,
                type=type,
                content=content,
                origin=origin,
                seq=seq,
                appended_ts=appended_ts,
            )
            self._write_durably(event.to_json() + "\n")

        # Outside the lock: a listener is free to append, and holding the lock across a
        # callback would deadlock it.
        self._notify(event)
        return event

    def append_many(self, events: Iterable[StimulusEvent]) -> list[StimulusEvent]:
        """Append a replicated batch, all of it or none of it.

        The spec forbids a partial state the surrogate would have to reason about, so the
        whole batch goes down under one `open`/`write`/`fsync`. That also makes a crash
        mid-batch harmless in a way a per-event loop is not: the only line that can tear is
        the last one, which `read_all` already recovers from, while an interior tear — the
        one case it raises on — becomes unreachable.

        Takes events as parsed off the wire and returns the ones actually written. `id` and
        `appended_ts` are re-minted here, exactly as in `append`: identity across nodes is
        `(origin, seq)`, never `id`. `ts` is the producer's own and is left alone.

        A batch may mix this log's own events with replicated ones, and the ingress needs
        it to: a host-minted gap marker explaining a hole in a surrogate's stream carries
        this log's origin, while the events it explains carry the surrogate's. Splitting
        those across two writes would mean a crash between them loses the explanation while
        the events it described are already committed. Local events are numbered here from
        this log's own counter; replicated ones keep the seq their producer assigned.

        The log does not police batch shape beyond that. Whether the seqs ascend, repeat,
        or come from one producer is the wire protocol's business and is enforced at the
        door by `replication_batch.parse_batch` — restating it here would be a second,
        divergent copy of a rule in the layer least able to explain a rejection.
        """
        events = list(events)
        if not events:
            return []

        for event in events:
            # The same two shapes `append` takes: a local event this log numbers itself,
            # or a replicated one carrying its producer's origin and seq. A batch may mix
            # them, and the ingress needs it to — a host-minted gap marker explaining a
            # hole in a surrogate's stream carries this log's origin while the events it
            # explains carry the surrogate's, and the two have to land in one write or a
            # crash between them loses the explanation.
            if not (event.origin == self.origin and event.seq is None):
                self._check_replicated(event.origin, event.seq)

        with self._append_lock:
            appended_ts = datetime.now(timezone.utc)
            event_ids = _id_run(int(appended_ts.timestamp() * 1000), len(events))
            minted = [
                StimulusEvent(
                    id=event_id,
                    ts=event.ts,
                    actor=event.actor,
                    type=event.type,
                    content=event.content,
                    origin=event.origin,
                    seq=self._next_local_seq() if event.seq is None else event.seq,
                    appended_ts=appended_ts,
                )
                for event_id, event in zip(event_ids, events)
            ]
            self._write_durably("".join(event.to_json() + "\n" for event in minted))

        # Outside the lock, and only once the whole batch is durable: a listener must never
        # see the first event of a batch while the last could still be lost.
        for event in minted:
            self._notify(event)
        return minted

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
                events.append(
                    StimulusEvent.from_json(stripped, default_origin=self.origin)
                )
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
