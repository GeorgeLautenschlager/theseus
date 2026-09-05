"""Per-origin high-water marks — the host's memory of what it has already accepted.

Duplicate suppression is the host's whole defence against double-appending a retried batch,
and it hinges on one number per origin: the highest `seq` committed so far. That number has
to survive a restart, because the surrogate's retry does not care that the host bounced.

The marks are **derived from the log**, not persisted beside it. The log is the bedrock; a
sidecar checkpoint can disagree with it after a crash, and disagreeing here means either
silently dropping real events or double-appending them — precisely the two failures this
state exists to prevent. The cost is one boot-time pass over the log, which the process
already makes elsewhere. If that pass ever becomes too slow, the fallback is a sidecar that
is only ever a *hint*: always reconciled forward against the log, never trusted past it.

Tracking marks is all this does. Deciding what to do with a batch that sits below, straddles
or jumps past a mark is the ingress's job.
"""

from __future__ import annotations

import threading

from theseus.stimulus_log import StimulusEvent, StimulusLog


def _committed_seq(event: StimulusEvent) -> int | None:
    """The seq to fold into a mark, or None for a line that carries none.

    Recovery reads whatever is on disk, and disk is not the same trust boundary as
    `StimulusLog.append`: a foreign writer — another implementation, a hand-edited file —
    can put a string or a negative number where a seq belongs. Folding one of those into a
    mark yields a mark that compares wrongly or not at all, and an under-counted mark is
    the direction that re-appends a batch the host already has. So recovery fails loudly
    here rather than booting with a mark that lies, which is what `read_all` already does
    for a corrupt interior line.
    """
    if event.seq is None:
        # Written before the envelope existed. History, not a delivery: no mark.
        return None
    if isinstance(event.seq, bool) or not isinstance(event.seq, int):
        raise ValueError(
            f"event {event.id} carries a non-integer seq {event.seq!r}; "
            f"a mark derived from it would compare wrongly"
        )
    if event.seq < 1:
        raise ValueError(
            f"event {event.id} carries seq {event.seq!r}; seqs start at 1, and a mark "
            f"below that would sit under every real seq"
        )
    return event.seq


class HighWaterMarks:
    """Highest committed `seq` per origin, recovered from the log at construction.

    Deliberately separate from `StimulusLog`'s own seq allocator, though both begin by
    scanning the same file. The allocator answers "what should I issue next, for myself";
    these marks answer "what have I already accepted, from everyone" — and the two diverge
    the moment an ingress advances a mark on commit without appending anything of its own.
    Folding them together would couple the writer's counter to the reader's dedupe state.

    An instance is a **snapshot taken at construction**, advanced only by callers. It never
    re-reads the log, so two preconditions hold: there is exactly one `HighWaterMarks` per
    log, and every path that commits a replicated event calls `advance`. Break either — a
    second instance, another process appending to the same file, or an ingress that appends
    and forgets to advance — and these marks go stale silently, which arms exactly the
    duplicate they exist to suppress. A restart repairs it, because the log is the truth and
    recovery reads it back.
    """

    def __init__(self, log: StimulusLog) -> None:
        self._marks: dict[str, int] = {}
        # `advance` is a read-modify-write, and the ingress that will call it is a FastAPI
        # endpoint: a surrogate whose HTTP client times out and retries produces two
        # concurrent requests for one origin while the host is still committing the first.
        # Unlocked, two threads can interleave their check-and-set and move a mark
        # *backwards* — the direction that re-appends a retried batch into the agent's
        # permanent memory. `StimulusLog` locks its own counter for the same reason.
        self._lock = threading.Lock()

        # Held by the ingress across its whole read-mark → plan → write → advance sequence,
        # which the two locks above cannot cover: they make each individual read and write
        # indivisible, and what the dedupe rules need indivisible is the *span between* them.
        #
        # It lives here rather than on the ingress because the invariant belongs to this
        # object. One `HighWaterMarks` per log is this class's stated precondition, so a lock
        # here is one lock per log; a lock on the ingress is one per ingress, and two
        # ingresses sharing one marks object — the arrangement this docstring recommends —
        # would then hold different locks and double-append a concurrently retried batch.
        # Measured: two ingresses, one marks object, one batch delivered to both at once, and
        # the log came back holding seqs [1, 2, 3, 1, 2, 3].
        self.commit_lock = threading.Lock()

        for event in log.read_all():
            seq = _committed_seq(event)
            if seq is not None:
                self.advance(event.origin, seq)

    def high_water(self, origin: str) -> int | None:
        """Highest seq committed for `origin`, or `None` if nothing has ever arrived from it.

        `None` is distinct from `0`: seqs start at 1, so `0` would claim a seq had been
        seen, while an origin whose first event is still in flight has no mark at all.

        Read under the lock so a caller cannot observe a torn transition — a mark that is
        neither its old value nor its new one.
        """
        with self._lock:
            return self._marks.get(origin)

    def advance(self, origin: str, seq: int) -> None:
        """Record that `seq` from `origin` is committed.

        Never moves a mark backwards, including under concurrent callers. A batch
        straddling the mark commits only the events above it and a duplicate commits
        nothing, so the mark is a maximum rather than a last-write.

        `seq` is not validated here, because both paths that reach it have already checked
        it: a live commit came through `StimulusLog.append`, which rejects a seq below 1,
        and recovery came through `_committed_seq`, which rejects anything a foreign writer
        may have left on disk. Validating an untrusted *batch* — shape, size, origin — is
        still the ingress's job, at the door.
        """
        with self._lock:
            current = self._marks.get(origin)
            if current is None or seq > current:
                self._marks[origin] = seq
