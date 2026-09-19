"""The surrogate runtime's replication half: own the local log, ship upstream.

This is the drain half — own-origin events ship to the host by a `Replicator`, driven
by a `CoalescingTrigger` — plus the command-execution half: host commands arrive over
an injected `CommandChannel`, render via a `CommandExecutor`, and land one
`command_report.*` per command in the same log, so the drain ships the reports
upstream with no new drain code. Each half holds its own `AckedCursor`: the upstream
cursor is the replication/ack position, the command cursor the execution position.
Concrete collaborators are injected; choosing them belongs to the entry point (#105).
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from theseus.replication_ingress import CoalescingTrigger
from theseus.stimulus_log import StimulusLog
from theseus.surrogates.command_channel import CommandChannel
from theseus.surrogates.command_executor import CommandExecutor, Renderer
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.replicator import Replicator
from theseus.surrogates.transport import StimulusTransport


class SurrogateRuntime:
    """Composes the surrogate edge of replication: local appends ring a drain.

    Holds a `Replicator` over the injected log/transport/cursor and a
    `CoalescingTrigger` whose single worker runs `Replicator.drain()` — so drains never
    overlap, and a burst of appends (a user typing, sensory events arriving) becomes one
    drain. The drain reads all own-origin events above the cursor, so anything appended
    after `start()` — chat messages now, command reports and sensory events later —
    ships for free.

    `submit_user_message` is called synchronously from the web UI's `POST /chat`
    handler, so it does a local append plus a non-blocking trigger ring and nothing
    more; the network cost is paid by the drain worker.

    When `command_channel`, `renderer`, and `command_cursor` are all provided, a second
    thread runs `CommandExecutor.run(channel)` — commands render locally and one report
    event per command lands in the same log, so the drain ships them upstream like any
    own-origin event. The three are all-or-none: a partial set is a wiring bug.
    """

    def __init__(
        self,
        log: StimulusLog,
        transport: StimulusTransport,
        upstream_cursor: AckedCursor,
        command_channel: CommandChannel | None = None,
        renderer: Renderer | None = None,
        command_cursor: AckedCursor | None = None,
        *,
        user_actor: str = "user",
        flush_interval_seconds: float = 30.0,
    ) -> None:
        self._log = log
        self._user_actor = user_actor
        self._flush_interval_seconds = flush_interval_seconds
        self._replicator = Replicator(log, transport, upstream_cursor)
        self._trigger = CoalescingTrigger(self._drain_once, name="surrogate-drain")
        self._unsubscribe: Callable[[], None] | None = None
        self._flush_stop = threading.Event()
        self._flush_thread: threading.Thread | None = None
        command_parts = (command_channel, renderer, command_cursor)
        if any(p is not None for p in command_parts) and not all(
            p is not None for p in command_parts
        ):
            raise ValueError(
                "command_channel, renderer, and command_cursor are all-or-none; "
                "pass all three or none"
            )
        if command_channel is not None:
            self._executor: CommandExecutor | None = CommandExecutor(
                log, renderer, command_cursor
            )
            self._command_channel: CommandChannel | None = command_channel
        else:
            self._executor = None
            self._command_channel = None
        self._command_thread: threading.Thread | None = None

    def submit_user_message(self, text: str) -> None:
        """Append a user chat message locally and ring the drain; never blocks on I/O."""
        self._log.append(
            actor=self._user_actor, type="chat_message", content={"message": text}
        )
        # Explicit ring covers the pre-start / manual-drain path; once start() has
        # subscribed, the append above also rings via the listener. Both are kept —
        # the rings coalesce, so the redundancy is free and neither path is left silent.
        self._trigger.request()

    def start(self) -> None:
        """Wire the append listener, the trigger worker, and the periodic flush.

        Idempotent: every step is either inherently idempotent (subscribe, trigger
        start) or guarded, so calling `start` twice does not double the drain.
        """
        if self._command_channel is not None and (
            self._command_thread is None or not self._command_thread.is_alive()
        ):
            self._command_thread = threading.Thread(
                target=self._executor.run,
                args=(self._command_channel,),
                name="surrogate-command",
                daemon=True,
            )
            self._command_thread.start()
        if self._unsubscribe is None:
            self._unsubscribe = self._log.subscribe(
                lambda _event: self._trigger.request()
            )
        self._trigger.start()
        if self._flush_thread is None or not self._flush_thread.is_alive():
            self._flush_stop.clear()
            self._flush_thread = threading.Thread(
                target=self._flush_loop, name="surrogate-drain-flush", daemon=True
            )
            self._flush_thread.start()

    def stop(self) -> None:
        """Close the command channel and join the command thread, then tear down the
        drain (unsubscribe → flush → trigger).

        Command-first ordering: closing the channel ends `stream()`, so the executor
        finishes its last report and exits while the drain listener is still
        subscribed — that final `command_report.*` still rings the trigger and ships
        upstream before the drain stops.
        """
        if self._command_channel is not None:
            self._command_channel.close()
        if self._command_thread is not None:
            self._command_thread.join(timeout=10.0)
            self._command_thread = None
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._flush_stop.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=self._flush_interval_seconds + 1.0)
            self._flush_thread = None
        self._trigger.stop()

    def _drain_once(self) -> None:
        """The trigger's callback: one drain. Transport failures are `DrainResult`s;
        `CoalescingTrigger` swallows and logs anything else."""
        self._replicator.drain()

    def _flush_loop(self) -> None:
        """Ring the trigger every `flush_interval_seconds` so a drain that failed
        while the host was unreachable gets retried without a new append."""
        interval = self._flush_interval_seconds
        # ponytail: sliced sleep so stop() returns promptly rather than waiting out
        # the whole interval — same shape as SseCommandChannel's sliced backoff.
        while not self._flush_stop.wait(timeout=interval):
            self._trigger.request()
