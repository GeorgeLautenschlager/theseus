"""What a batch actually adds to the log, given what the host already has.

`replication_batch` decides whether a batch is acceptable. This module decides what an
acceptable batch *adds*. Nothing here rejects: a duplicate is discarded and answered `2xx`,
a straddle commits its tail and is answered `2xx`, and a jump past the mark is committed
*with a marker explaining the hole* and answered `2xx`. The `4xx` class belongs at the door,
and by the time a batch reaches this module it is well-formed — which is never something a
surrogate should be told to throw away.

Pure: no log, no marks object, no clock, no I/O. Everything the decision needs is an
argument and the decision itself is data, which is what lets the ingress's three cases be
tested without a server, a file, or a sleep.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from theseus.replication_events import GAP, inferred_gap
from theseus.stimulus_log import StimulusEvent

# The host's name for its own diagnostic events. `origin` says which machine wrote it;
# `actor` says who on that machine did, and a gap the host inferred was written by no one
# the agent was talking to.
HOST_ACTOR = "host"

# `StimulusLog.append_many` re-mints `id` for everything it writes, so this is never the id
# of anything on the tape. It is here because `StimulusEvent` requires one, and a value that
# says so is better than a plausible-looking ULID that would be a lie if it ever survived.
PLACEHOLDER_ID = "01HOSTMINTEDGAPPLACEHOLDER"


@dataclass(frozen=True)
class BatchPlan:
    """What the ingress should do with one parsed batch.

    `to_append` is already in the order it must be written: the host's own gap marker, when
    it minted one, ahead of the events whose arrival revealed the hole. It goes to
    `append_many` as a single call, because a crash between two writes would commit the hole
    and lose the explanation.

    An **empty `to_append` means the batch was entirely a duplicate** — every event at or
    below the mark. The ingress commits nothing, advances nothing, and answers `2xx`: a
    surrogate retrying after a lost ack is asking to stop worrying, not to be told it was
    wrong.

    `new_high_water` is the mark to advance to *after* the write is durable, and is `None`
    exactly when there is nothing to write. Advancing before the commit would leave the mark
    claiming events the log does not have — the direction that silently drops a retry.

    `inferred_holes` are the inclusive ranges the host minted markers for — empty when there
    was no hole, or when the batch's own declared markers already accounted for all of it. It
    is the plan's answer to "how much did the host have to guess", which is what an operator
    wants to count.

    There can be more than one. `parse_batch` accepts a batch whose seqs ascend without being
    contiguous — a surrogate that evicted events under storage pressure has real holes in its
    buffer — so a single batch can reveal a hole before its first event *and* between any two
    of them.
    """

    to_append: tuple[StimulusEvent, ...]
    new_high_water: int | None
    inferred_holes: tuple[tuple[int, int], ...] = ()


def plan_batch(
    events: Sequence[StimulusEvent],
    *,
    high_water: int | None,
    host_origin: str,
    now: datetime,
    previous_ts: datetime | None = None,
) -> BatchPlan:
    """Apply the dedupe rules to one well-formed batch.

    `high_water` is the highest seq already committed for this batch's origin, or `None` if
    nothing has ever arrived from it. `now` is the host's clock, passed in rather than read
    so a test can pin it. `previous_ts` is the `ts` of the last event committed from this
    origin, if the caller knows it — it is only ever the lower bound of an inferred gap's
    span, and `None` is honest when nothing remembers it.

    Raises `ValueError` on a batch this function's arithmetic cannot describe: empty, spanning
    two origins, claiming the host's own origin, or not ascending by integer seq. Those are
    caller bugs, not wire conditions — `replication_batch` answers the wire versions with a
    `4xx` long before this — so they surface as the exception a programmer gets, not as a
    silently wrong hole.
    """
    if not events:
        raise ValueError("a batch must carry at least one event")

    origins = {event.origin for event in events}
    if len(origins) != 1:
        raise ValueError(f"a batch must come from one origin, got {sorted(origins)}")
    origin = events[0].origin
    if origin == host_origin:
        raise ValueError(
            f"{origin!r} is the host's own origin; a replicated batch arrives under its "
            f"producer's name, or two producers number one seq stream"
        )
    _check_ascending(events)

    # An origin with no mark is a mark of 0: seqs start at 1, so nothing real sits at or
    # below it, and the first-contact hole `[1, first_seq - 1]` falls out of the same
    # arithmetic as every other jump rather than needing a case of its own.
    mark = 0 if high_water is None else high_water
    tail = tuple(event for event in events if event.seq > mark)
    if not tail:
        return BatchPlan(to_append=(), new_high_water=None)

    # Every hole the commit will step over, then whatever the batch explains for itself
    # subtracted from it. The host mints for the residue only: what nobody accounted for.
    unexplained = _unexplained(_holes(mark, tail), _declared_ranges(tail, origin))

    markers = tuple(
        _inferred_marker(
            origin=origin,
            hole=hole,
            host_origin=host_origin,
            now=now,
            lower_bound=previous_ts,
            upper_bound=tail[0].ts,
        )
        for hole in unexplained
    )

    return BatchPlan(
        to_append=markers + tail,
        new_high_water=tail[-1].seq,
        inferred_holes=unexplained,
    )


def _holes(mark: int, tail: Sequence[StimulusEvent]) -> tuple[tuple[int, int], ...]:
    """Every seq range this commit steps over without delivering, inclusive.

    Two kinds, and they were not always treated alike. The *leading* hole sits between the
    mark and the batch's first event. An *internal* hole sits between two events of the same
    batch — which exists because `parse_batch` deliberately accepts ascending-but-not-
    contiguous batches, so that a surrogate whose buffer has real holes in it is not told to
    throw away the events it still has.

    Before internal holes were counted, delivering `[5, 9]` after a mark of 4 advanced the
    mark to 9 and recorded nothing: seqs 6-8 became permanently undeliverable *and*
    unaccounted for, while the very same hole split across two batches (`[5]` then `[9]`) was
    recorded properly. The tape should not depend on how a surrogate happened to chunk its
    backlog.
    """
    holes = []
    if tail[0].seq > mark + 1:
        holes.append((mark + 1, tail[0].seq - 1))
    for earlier, later in zip(tail, tail[1:]):
        if later.seq > earlier.seq + 1:
            holes.append((earlier.seq + 1, later.seq - 1))
    return tuple(holes)


def _unexplained(
    holes: Sequence[tuple[int, int]], declared: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """The holes left over once the batch's own declared markers are subtracted.

    Subtraction, rather than asking whether any single marker covers a hole. Two markers can
    jointly account for one hole and often will: a surrogate that evicts 5-7 under storage
    pressure and then loses 8-9 to a dead link declares two ranges for what the host sees as
    one gap. Requiring a single covering marker mints an `inferred` marker beside them saying
    the surrogate may be dead — erasing the one distinction the gap vocabulary carries, and
    inflating the count an operator reads as "how often did the host have to guess".

    A partial explanation still leaves a residue, and the residue still earns a marker: 5-7
    explained of a 5-9 hole leaves 8-9 genuinely unaccounted for, and a readable hole beats a
    silence.
    """
    remaining: list[tuple[int, int]] = []
    covers = sorted(declared)
    for start, end in holes:
        cursor = start
        for cover_start, cover_end in covers:
            if cover_end < cursor:
                continue
            if cover_start > end:
                break
            if cover_start > cursor:
                remaining.append((cursor, min(cover_start - 1, end)))
            cursor = max(cursor, cover_end + 1)
            if cursor > end:
                break
        if cursor <= end:
            remaining.append((cursor, end))
    return tuple(remaining)


def _check_ascending(events: Sequence[StimulusEvent]) -> None:
    """The precondition the tail slice and the hole arithmetic both rest on.

    `replication_batch` checks this too, and that is not the duplication its own fix round
    argued against: there the wire's rule was being restated in a layer that did not depend
    on it, whereas here it is this function's own correctness. A caller that skips the parser
    — the #31 surrogate side, a replay tool, a test — gets a `ValueError` naming the pair
    rather than a silently wrong hole.
    """
    for event in events:
        if isinstance(event.seq, bool) or not isinstance(event.seq, int):
            raise ValueError(
                f"event {event.id!r} carries a non-integer seq {event.seq!r}; the tail "
                f"slice and the hole arithmetic are only correct on integers"
            )
    for earlier, later in zip(events, events[1:]):
        if later.seq <= earlier.seq:
            raise ValueError(
                f"a batch must ascend by seq; {later.seq} follows {earlier.seq}"
            )


def _declared_ranges(
    tail: Sequence[StimulusEvent], origin: str
) -> tuple[tuple[int, int], ...]:
    """The ranges this batch explains about its own stream, for subtracting from the holes.

    A surrogate's marker rides in the batch that opens the hole, because `seq` is assigned at
    its local write time: it evicts 5-9, then writes the marker as seq 10.

    Only the `tail` is scanned, never the whole batch. A marker below the mark is one the host
    already has, and is discarded here along with every other duplicate — counting it would
    let an already-committed marker silence a hole nothing in this batch explains.

    The content is read defensively even though `parse_batch` now re-applies
    `replication_events`' rules at the door: this function is reachable from a caller that
    never went through that door, and a marker it cannot read is a marker that explains
    nothing.
    """
    ranges = []
    for event in tail:
        if event.type != GAP or not isinstance(event.content, dict):
            continue
        if event.content.get("origin") != origin:
            continue
        from_seq = _as_seq(event.content.get("from_seq"))
        to_seq = _as_seq(event.content.get("to_seq"))
        if from_seq is not None and to_seq is not None and to_seq >= from_seq:
            ranges.append((from_seq, to_seq))
    return tuple(ranges)


def _as_seq(value: Any) -> int | None:
    """A wire value usable as a seq bound, or `None`.

    `bool` is an `int` in Python, so `"from_seq": true` would otherwise compare as 1 and let
    a malformed marker silence a real hole.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _inferred_marker(
    *,
    origin: str,
    hole: tuple[int, int],
    host_origin: str,
    now: datetime,
    lower_bound: datetime | None,
    upper_bound: datetime,
) -> StimulusEvent:
    """The host's own account of a hole nobody declared.

    It carries the **host's** origin and no seq, because the host wrote it; the surrogate's
    origin names whose stream has the hole and lives in the content. That is what makes this
    one of the two origins `StimulusLog.append_many` permits in a single write — and it has
    to be in that write, or a crash between them commits the hole and loses the explanation.

    The span is the host's honest bound, not a claim about when the missing events happened.
    Its upper end is the first event that did arrive. Its lower end is whatever the caller
    could supply, and when there is none both ends collapse onto the upper bound: **a
    zero-width span on an inferred gap reads as "noticed here, no lower bound known"**, which
    is the truth on first contact rather than a fabricated interval.
    """
    from_seq, to_seq = hole
    end = _utc(upper_bound)
    # A surrogate's clock can sit ahead of the host's mark. Taking the minimum keeps the span
    # from inverting, which `inferred_gap` would reject — and rejecting here would lose the
    # marker over a clock skew the marker itself exists to make visible.
    start = end if lower_bound is None else min(_utc(lower_bound), end)
    return StimulusEvent(
        id=PLACEHOLDER_ID,
        ts=now,
        actor=HOST_ACTOR,
        type=GAP,
        content=inferred_gap(
            origin=origin,
            from_seq=from_seq,
            to_seq=to_seq,
            span_start=start,
            span_end=end,
        ),
        origin=host_origin,
    )


def _utc(value: datetime) -> datetime:
    """Comparable, by the same rule the rest of the package uses: a naive datetime is
    host-local. Two timestamps from different nodes must never meet as a mixed pair."""
    return value.astimezone(timezone.utc)
