"""CommandFeed — the host side of the command channel (issue #34, Task 3).

The host needs to act with no triggering stimulus, and hours of silence is the normal
case — so the downstream channel is a real subscription, not "reply to the surrogate's
next POST". This feed reads the host's own log, filters the commands it issued for one
target above a cursor, and serves them as SSE. It is the streaming half of the
protocol's `CommandChannel` interface; the push-as-doorbell transport later differs only
in how bytes move, never in what the protocol means — which is why the replay half
(`pending`) is pulled out of the generator and testable without a connection.

Semantics mirror upstream: commands carry a monotonic `seq`, the surrogate holds a
cursor (`Last-Event-ID` on SSE reconnect), and reconnection resumes from that cursor
rather than dropping what it missed. Absent or unparseable cursor means from the
beginning: the cursor arrives off the network, and replaying is the direction this
protocol already tolerates.

The SSE shape follows `web_chat_ui_observer._sse_stream` — per-listener `Queue`,
`request.is_disconnected()` to end the generator, a heartbeat on idleness,
`run_in_threadpool` for the blocking get — without importing it: that observer is a
chat-UI concern.
"""

from __future__ import annotations

import time
from queue import Empty, Full, Queue

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from theseus.commands import command_target, is_command
from theseus.stimulus_log import StimulusEvent, StimulusLog


class CommandFeed:
    """Serves the host's own commands per surrogate as SSE, above a reconnect cursor.

    One `CommandFeed` per host log; `add_routes` mounts `GET /commands/{target}` on an
    app the host already serves (chat UI, replication ingress — no third port), and
    `build_app` stands alone for a host that runs the feed on its own port.
    """

    def __init__(
        self,
        log: StimulusLog,
        *,
        heartbeat_seconds: float = 20.0,
        poll_seconds: float = 1.0,
    ) -> None:
        self._log = log
        self._heartbeat_seconds = heartbeat_seconds
        self._poll_seconds = poll_seconds

    # -- replay ---------------------------------------------------------

    def pending(self, target: str, *, after: int | None) -> list[StimulusEvent]:
        """The commands this host issued for `target` above `after`, in `seq` order.

        `after=None` means from the beginning. Foreign-origin events are never served:
        a replicated command entered the log from somewhere else and carries the
        producer's `seq` — its numbering is the producer's cursor domain, not this
        log's, so comparing it against a surrogate's cursor would be a category error.
        """
        events = [
            event
            for event in self._log.read_all()
            if self._matches(event, target)
            and event.seq is not None
            and (after is None or event.seq > after)
        ]
        events.sort(key=lambda event: event.seq)
        return events

    # -- SSE ------------------------------------------------------------

    def add_routes(self, app: FastAPI, *, path: str = "/commands") -> None:
        """Mount `GET {path}/{target}` on an existing app, so a host already serving a
        chat UI and a replication ingress does not need a third port.

        A composer running this under uvicorn **must** pass
        `timeout_graceful_shutdown` to `uvicorn.run`, the way
        `web_chat_ui_observer.serve` does: the stream is infinite and only ends when
        the client disconnects, so Ctrl+C otherwise hangs forever with a surrogate
        connected.
        """

        @app.get(f"{path}/{{target}}")
        async def commands(request: Request, target: str):
            return StreamingResponse(
                self._stream(request, target), media_type="text/event-stream"
            )

    def build_app(self) -> FastAPI:
        """A standalone app, for a host running the feed on its own port."""
        app = FastAPI()
        self.add_routes(app)
        return app

    async def _stream(self, request: Request, target: str):
        """Replay above the cursor, then live, with a heartbeat when idle.

        The live half treats the subscription as a **doorbell**, never as the delivery:
        the log's own docstring warns that notification order is not file order (the
        notify runs outside the append lock), so a queue-fed stream with a running-max
        discard silently drops a command whose notification arrives after a later one —
        permanently, because the caller's cursor then skips it on every reconnect. The
        truth is the file, which *is* seq-ordered; the listener only rings, and every
        ring triggers a fresh `pending(target, after=highest)` fetch. No discard is
        needed: `pending` is defined as strictly above the cursor, so an event covered
        by the previous fetch is never served twice.

        The ordering here is still the sharpest trap in the task: **subscribe first,
        replay second**. Reading first and subscribing second leaves a window where an
        appended command lands in neither the read nor the subscription — nothing rings
        for it, ever. Subscribe-first puts anything appended in between in *both*
        halves, and the cursor rule serves it exactly once.

        The cost traded into: every ring re-reads the log above the cursor. That is the
        price of ordering correctness, it happens on a worker thread (`run_in_threadpool`
        — a big tape must never freeze the endpoints mounted beside this one), and it
        only happens when something was actually appended.
        """
        after = _parse_last_event_id(request.headers.get("last-event-id"))
        yield "retry: 2000\n\n"

        # maxsize=1 and best-effort put: a ring dropped because one is already pending
        # is harmless — the fetch after it reads the whole log above `highest`. The
        # ring runs on the appending thread, the critical path of every host append,
        # so it must be near-free and must never grow.
        bell: Queue = Queue(maxsize=1)
        unsubscribe = self._log.subscribe(lambda _event: _ring(bell))
        try:
            highest = after
            for event in self.pending(target, after=after):
                highest = _advance(highest, event.seq)
                yield _format_sse(event)
            # The clock measures time since **bytes were sent to the client**, not
            # since the doorbell rang: a busy host — the chat UI writes to this very
            # log — rings the bell forever with non-matching events, and a ring-path
            # clock resets on each of them, so the stream sends no bytes at all on
            # exactly the host whose proxy declares the silent connection dead.
            idle_since = time.monotonic()
            while True:
                if await request.is_disconnected():
                    break
                sent = False
                try:
                    await run_in_threadpool(bell.get, timeout=self._poll_seconds)
                except Empty:
                    pass
                else:
                    for event in await run_in_threadpool(
                        self.pending, target, after=highest
                    ):
                        highest = _advance(highest, event.seq)
                        yield _format_sse(event)
                        sent = True
                # Reachable on every path, not just the doorbell-timeout one: after
                # `heartbeat_seconds` of client silence, send bytes so proxies and NAT
                # tables never declare an otherwise-silent connection dead.
                if not sent and (
                    time.monotonic() - idle_since >= self._heartbeat_seconds
                ):
                    yield ": heartbeat\n\n"
                    idle_since = time.monotonic()
        finally:
            # A listener left behind on every dropped connection grows with reconnects —
            # the normal case on a flaky link — so unsubscribing runs on every exit.
            unsubscribe()

    def _matches(self, event: StimulusEvent, target: str) -> bool:
        return (
            is_command(event)
            and command_target(event) == target
            and event.origin == self._log.origin
        )


def _ring(bell: Queue) -> None:
    """Best-effort doorbell ring: drop when already rung. A dropped signal is harmless —
    the reader's next fetch sees everything the log holds above its cursor."""
    try:
        bell.put_nowait(None)
    except Full:
        pass


def _advance(highest: int | None, seq: int | None) -> int | None:
    """The replay boundary after serving `seq`. A `seq`-less event cannot be compared
    against a cursor, so it leaves the boundary where it was."""
    if seq is None:
        return highest
    return seq if highest is None else max(highest, seq)


def _parse_last_event_id(raw: str | None) -> int | None:
    """Absent or unparseable is `None`: the cursor arrives off the network, and
    refusing to serve is worse than replaying."""
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def _format_sse(event: StimulusEvent) -> str:
    """One SSE frame: `id:` carries the cursor (`seq`), `event:` the log type, `data:`
    the event's JSON — split across multiple `data:` lines, because a raw newline
    inside one `data:` line ends the event early. Same shape as
    `web_chat_ui_observer._format_sse_event`, plus the `id:` field that replay needs."""
    lines = event.to_json().splitlines() or [""]
    payload = "\n".join(f"data: {line}" for line in lines)
    id_line = "" if event.seq is None else f"id: {event.seq}\n"
    # Strip CR/LF from the type: the feed serves the log, not `command_type`'s output,
    # and the log does not validate types — a crafted type with a newline could forge
    # extra `event:`/`data:` lines, i.e. commands the surrogate never saw issued.
    # `command_type` refusing whitespace is not sufficient here; this framing function
    # is the last place that can guarantee the wire is well-formed.
    event_type = event.type.replace("\r", "").replace("\n", "")
    return f"{id_line}event: {event_type}\n{payload}\n\n"
