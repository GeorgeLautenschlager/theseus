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

from theseus.stimulus_log import StimulusLog


class HighWaterMarks:
    """Highest committed `seq` per origin, recovered from the log at construction.

    Deliberately separate from `StimulusLog`'s own seq allocator, though both begin by
    scanning the same file. The allocator answers "what should I issue next, for myself";
    these marks answer "what have I already accepted, from everyone" — and the two diverge
    the moment an ingress advances a mark on commit without appending anything of its own.
    Folding them together would couple the writer's counter to the reader's dedupe state.
    """

    def __init__(self, log: StimulusLog) -> None:
        self._marks: dict[str, int] = {}
        for event in log.read_all():
            # Lines written before the envelope existed carry no seq. They are history,
            # not deliveries, and must not invent a mark.
            if event.seq is not None:
                self.advance(event.origin, event.seq)

    def high_water(self, origin: str) -> int | None:
        """Highest seq committed for `origin`, or `None` if nothing has ever arrived from it.

        `None` is distinct from `0`: seqs start at 1, so `0` would claim a seq had been
        seen, while an origin whose first event is still in flight has no mark at all.
        """
        return self._marks.get(origin)

    def advance(self, origin: str, seq: int) -> None:
        """Record that `seq` from `origin` is committed.

        Never moves a mark backwards. A batch straddling the mark commits only the events
        above it and a duplicate commits nothing, so the mark is a maximum rather than a
        last-write.

        `seq` is not validated here: a mark advances *on commit*, so the only path that can
        produce one has already been through `StimulusLog.append`, which rejects a seq below
        1. Validating untrusted input is the ingress's job, at the door.
        """
        current = self._marks.get(origin)
        if current is None or seq > current:
            self._marks[origin] = seq
