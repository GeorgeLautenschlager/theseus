"""The host's replication ingress: where a surrogate's events actually land.

`replication_batch` decides whether a batch is acceptable, `replication_dedupe` decides
what it adds, and `StimulusLog.append_many` commits it. This module is the only piece that
touches all three, plus the clock, the marks and the network — everything the other two
were kept pure of.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from theseus.high_water import HighWaterMarks
from theseus.replication_batch import (
    DEFAULT_MAX_BATCH_BYTES,
    DEFAULT_MAX_BATCH_EVENTS,
    BatchRejected,
    parse_batch,
)
from theseus.replication_dedupe import plan_batch
from theseus.stimulus_log import StimulusLog


class CoalescingTrigger:
    """One cognitive turn per burst of arrivals, not one per batch.

    A surrogate returning from an outage chunks its backlog into limit-sized batches and
    drains them "in `seq` order with no inter-batch delay" — the spec's words — so fifty
    batches can land as fifty HTTP requests inside a second. Calling the core once per
    request is wrong in two different ways depending on which core is behind it. In front
    of an `Autocore` it is merely redundant: `wake` sets a flag and returns. In front of an
    `OODACore` the callback is `orient_and_wait`, which blocks for a whole cognitive cycle
    — so a fifty-batch backlog would queue fifty cycles, each one reading a context window
    that the next batch has already invalidated.

    So arrivals are coalesced. `request()` never blocks and never runs cognition; one
    worker thread turns any number of requests into one callback per burst. The guarantee
    that matters is not "exactly one call" but this: **every `request()` is followed by a
    callback that starts after it**. That is what makes it safe for the endpoint to answer
    `2xx` before the agent has looked at anything — the surrogate is being told the events
    are durable, which they are, not that they have been thought about.

    This deliberately does not live in the core. It needs a thread with a lifecycle, and
    the two cores it must sit in front of take incompatible callbacks; a coalescer built
    into either would be the wrong shape for the other.
    """

    def __init__(
        self, callback: Callable[[], None], *, name: str = "replication-trigger"
    ) -> None:
        self._callback = callback
        self._name = name
        self._pending = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin serving requests. Idempotent; starting twice is a no-op, not two threads.

        `is_alive` rather than a bare `is not None`, so that a `stop` whose join timed out
        cannot be followed by a `start` that adds a second worker beside the one still
        running. Two workers is the failure this idempotence exists to prevent, and a
        timed-out stop is another way to reach it.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        # A `stop` sets `_pending` to wake the worker. Left set, the next `start` would fire
        # a callback nobody asked for.
        self._pending.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop serving, and wait for any callback already running to finish.

        A request still pending when this is called is **dropped**, not drained. The
        alternative — run one more cognitive cycle on the way out — is worse: shutdown is
        exactly when the log, the model endpoint and the process itself are least likely to
        still be there. Nothing is lost that matters, because the events are already on the
        tape; only the nudge to think about them is skipped, and the next start reads the
        same tape.
        """
        thread = self._thread
        if thread is None:
            return
        self._stopping.set()
        self._pending.set()  # wake the worker so it can notice it is stopping
        thread.join(timeout)
        # Only forgotten once it is actually gone. A worker still running past the timeout —
        # a cognitive cycle can outlast it — stays on record, so a later `start` sees it and
        # declines to add a second beside it.
        if not thread.is_alive():
            self._thread = None

    def request(self) -> None:
        """Ask for a callback. Never blocks, never runs the callback on this thread.

        Safe from a request handler, a listener, or anywhere else: it sets a flag. The
        endpoint calls this while holding nothing, having already answered the surrogate.
        """
        self._pending.set()

    def _run(self) -> None:
        while True:
            self._pending.wait()
            if self._stopping.is_set():
                return
            # Cleared *before* the callback, never after. A request that arrives while the
            # callback is running must survive it: clearing afterwards would wipe that flag
            # and lose the wakeup, leaving a batch on the tape that nothing ever looks at
            # until the next one happens to arrive. Clearing first means the worst case is
            # one redundant callback, which is the direction to err in.
            self._pending.clear()
            try:
                self._callback()
            except Exception:
                # A core that raised — a model endpoint down, a bad tool result — must not
                # take the ingress's thread down with it. The events are committed either
                # way, and the next arrival gets a fresh attempt.
                traceback.print_exc()


@dataclass(frozen=True)
class Ingested:
    """What one batch did, in the terms the surrogate and an operator both need.

    `status` is what to answer: `2xx` for anything committed *or* already committed,
    `4xx` for a batch that will never be acceptable. There is no `5xx` here — a `5xx` is
    what an unhandled exception becomes, and the point of the two pure modules below this
    one is that a well-formed batch cannot produce one.

    `appended` counts every event written, including a gap marker the host minted itself,
    which is why `inferred_holes` is reported beside it rather than folded into it.
    """

    status: int
    origin: str | None
    appended: int
    high_water: int | None
    inferred_holes: tuple[tuple[int, int], ...] = ()
    reason: str | None = None

    @property
    def duplicate(self) -> bool:
        """Committed nothing because the host already had all of it. Still a `2xx`: a
        surrogate retrying after a lost ack is asking to stop worrying, not to be told it
        was wrong."""
        return self.status < 400 and self.appended == 0

    def payload(self) -> dict[str, Any]:
        """The response body. `high_water` is the useful field: it tells a surrogate where
        the host actually is, which is what a cursor-holder needs to resynchronise after
        any confusion about what got through."""
        body: dict[str, Any] = {
            "origin": self.origin,
            "appended": self.appended,
            "high_water": self.high_water,
        }
        if self.inferred_holes:
            body["inferred_gaps"] = [
                {"from_seq": start, "to_seq": end} for start, end in self.inferred_holes
            ]
        if self.reason is not None:
            body["reason"] = self.reason
        return body


class ReplicationIngress:
    """Where a surrogate's events land: parse, dedupe, commit, nudge.

    The three steps before the nudge each belong to something else — `parse_batch` owns
    acceptability and the `4xx` class, `plan_batch` owns what a batch adds, `append_many`
    owns the all-or-nothing write. This class owns the order they happen in, the lock that
    makes the sequence atomic, and the decision to answer the surrogate before the agent
    has thought about anything.

    **Why there is a lock.** The spec allows one batch in flight per surrogate, but that is
    the surrogate's rule and it does not survive its own HTTP client: a request that times
    out client-side and is retried puts two concurrent requests for one origin in front of
    this class while the first is still committing. Reading the mark, planning against it,
    writing, and advancing it must therefore be one atomic sequence. Split them and both
    requests read the same mark, both plan a full commit, and the batch lands twice — in
    the agent's permanent memory, which is the exact failure duplicate suppression exists
    to prevent. `HighWaterMarks` locks its own read and its own write, but that is not
    enough: what needs to be indivisible is the span between them.

    One lock, not one per origin. It costs nothing real — `append_many` already serialises
    every writer on the log's own lock and one fsync at a time is the actual ceiling — and
    a lock per origin would be more state to get wrong for concurrency the host cannot use.
    It is taken from the `HighWaterMarks` rather than created here, so that it is one lock per
    *log* rather than one per ingress; see that class for why the difference bites.

    **A listener must not call `ingest`.** `StimulusLog` notifies listeners outside its own
    lock, and explicitly allows one to append — but this class holds `_commit_lock` across
    `append_many`, so a listener runs underneath it. `threading.Lock` is not reentrant, so a
    listener that ingests deadlocks that thread permanently and no batch from any origin ever
    commits again. A relay — a host forwarding what it receives — is the plausible way to
    write this by accident.

    **Why the mark advances after the write.** A mark claiming events the log does not have
    silently discards the retry that would have delivered them. If the process dies between
    the write and the advance, the mark is merely behind, and a retry re-delivers a batch
    the host already has — which the dedupe rules answer with a `2xx` and no second write.
    Behind is recoverable; ahead is not.
    """

    def __init__(
        self,
        log: StimulusLog,
        marks: HighWaterMarks,
        *,
        on_arrival: Callable[[], None] | None = None,
        max_events: int = DEFAULT_MAX_BATCH_EVENTS,
        max_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    ) -> None:
        self._log = log
        self._marks = marks
        self._max_events = max_events
        self._max_bytes = max_bytes
        # From the marks object, not a fresh one: see `HighWaterMarks.commit_lock`. A lock
        # created here would be one per ingress, and two ingresses over one log would each
        # hold their own and double-append a concurrently retried batch.
        self._commit_lock = marks.commit_lock
        # The last committed `ts` per origin, in memory only. It is the lower bound of an
        # inferred gap's span — and it dies with the process on purpose: after a restart
        # the honest answer is `None`, which mints a zero-width span saying "no lower bound
        # known" rather than one that pretends to remember. Deriving it from the log at boot
        # would touch `HighWaterMarks`; that is a follow-up, not this round.
        self._last_ts: dict[str, datetime] = {}
        self._trigger = (
            CoalescingTrigger(on_arrival) if on_arrival is not None else None
        )

    def start(self) -> None:
        """Start the trigger's worker, if there is one. Safe to call without."""
        if self._trigger is not None:
            self._trigger.start()

    def stop(self) -> None:
        if self._trigger is not None:
            self._trigger.stop()

    def ingest(self, body: str | bytes) -> Ingested:
        """Apply one batch. Blocking — an fsync and a lock — so never call this from a
        coroutine on an event loop; `add_routes` hands it to a worker thread.

        Exceptions are not caught here beyond `BatchRejected`. `plan_batch`'s `ValueError`s
        are unreachable from this path — `parse_batch` has already rejected an empty batch,
        two origins, a claim on the host's own name, and seqs that do not ascend — so one
        escaping would be a genuine bug in this host, and a `5xx` is the right answer to
        that: it tells the surrogate to keep the events and try again, which is exactly what
        it should do while somebody fixes the host.
        """
        try:
            events = parse_batch(
                body,
                host_origin=self._log.origin,
                max_events=self._max_events,
                max_bytes=self._max_bytes,
            )
        except BatchRejected as rejected:
            return Ingested(
                status=rejected.status,
                origin=None,
                appended=0,
                high_water=None,
                reason=rejected.reason,
            )

        origin = events[0].origin
        with self._commit_lock:
            plan = plan_batch(
                events,
                high_water=self._marks.high_water(origin),
                host_origin=self._log.origin,
                now=datetime.now(timezone.utc),
                previous_ts=self._last_ts.get(origin),
            )
            if plan.to_append:
                self._log.append_many(plan.to_append)
                self._marks.advance(origin, plan.new_high_water)
                # Markers are prepended to the write, so its last event is a replicated one —
                # the origin's clock, not the host's mint time. A duplicate appends nothing
                # and moves nothing: a retry must not rewrite where the next span starts.
                self._last_ts[origin] = plan.to_append[-1].ts
            high_water = self._marks.high_water(origin)

        # Outside the lock, and only for a batch that actually added something. A duplicate
        # is the surrogate re-asking a question already answered; waking the agent for it
        # would turn a lost ack into a cognitive turn about nothing.
        if plan.to_append and self._trigger is not None:
            self._trigger.request()

        return Ingested(
            status=200,
            origin=origin,
            appended=len(plan.to_append),
            high_water=high_water,
            inferred_holes=plan.inferred_holes,
        )

    def add_routes(self, app: FastAPI, *, path: str = "/replicate") -> None:
        """Mount the endpoint on an existing app, so a host already serving a chat UI does
        not need a second port for this."""

        @app.post(path)
        async def replicate(request: Request):
            # Refused on the declared length, before a byte is buffered. `parse_batch` also
            # checks the size, but only once `request.body()` has read the whole thing into
            # memory: a 256 MB POST against a 1 KB limit was measured answering a correct 413
            # after growing the process by half a gigabyte. On a Pi-class host that is the
            # agent dying rather than a batch being refused.
            #
            # This trusts the header, so it closes the honest-client case and not the
            # adversarial one — a chunked or lying client still reaches the check below.
            # Authentication is a Phase-1 non-goal, so an unbounded stream is still a way to
            # hurt this endpoint; the header check is the cheap half of the answer.
            declared = request.headers.get("content-length")
            if declared is not None and declared.isdigit():
                if int(declared) > self._max_bytes:
                    rejected = Ingested(
                        status=413,
                        origin=None,
                        appended=0,
                        high_water=None,
                        reason=(
                            f"batch declares {declared} bytes, over the "
                            f"{self._max_bytes} byte limit"
                        ),
                    )
                    return JSONResponse(rejected.payload(), status_code=413)

            body = await request.body()
            # `ingest` blocks on a lock and an fsync. Run on the event loop it would stall
            # every other request in the process — the chat UI included, if they share an
            # app — for the duration of a disk write.
            result = await run_in_threadpool(self.ingest, body)
            return JSONResponse(result.payload(), status_code=result.status)

    def build_app(self) -> FastAPI:
        """A standalone app, for a host running the ingress on its own port."""
        app = FastAPI()
        self.add_routes(app)
        return app
