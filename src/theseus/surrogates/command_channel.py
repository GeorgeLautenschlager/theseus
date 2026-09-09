"""The command-channel seam between the host and a surrogate's executor (issue #34).

Commands are the host's own log entries, transmitted wholesale — the surrogate executes
on arrival without evaluating them. The two directions (surrogate→host stimuli, host→
surrogate commands) are independent and asynchronous: hours of silence is the normal
case, so the channel cannot be "reply to the next POST".

Like `StimulusTransport`, this is the seam that makes the carrying transport swappable:
an SSE stream today, a mobile push-as-doorbell transport later. Only bytes-moving
differs between the two, never what the protocol means — which is why `stream` is the
whole interface. A doorbell transport's `stream` drains and returns; a live transport's
blocks; the caller loops over it the same way either way.

Cursor semantics: the surrogate holds the cursor and the **caller** advances it after
hand-off, so delivery is at-least-once — a duplicate spoken sentence is visible on the
tape once execution reporting lands (#35), a silently dropped one is not. This
deliberately inverts the upstream replication rule, where being behind is recoverable
because the host dedupes; nothing dedupes a spoken sentence.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from threading import Condition
from typing import Protocol, runtime_checkable

from theseus.stimulus_log import StimulusEvent


@runtime_checkable
class CommandChannel(Protocol):
    """How commands reach a surrogate. The seam that lets a new transport drop in
    without a caller noticing (issue #34).

    A caller loops over `stream()` the same way for every transport — blocking or
    doorbell — and advances the cursor itself after each hand-off (at-least-once
    delivery; see the module docstring for why).
    """

    def stream(self) -> Iterator[StimulusEvent]:
        """Yield commands from the cursor forward, in seq order.

        Blocks between commands on a live transport and ends only when the transport
        does; returns when a doorbell-style transport has drained. A caller loops over
        it the same way either way — that sameness is the point of the interface.
        """
        ...

    def close(self) -> None:
        """End the channel: an in-progress `stream()` returns promptly.

        Part of the protocol, not an implementation detail: shutting a live stream
        down is something every caller needs (host Ctrl+C, surrogate shutdown), and a
        transport that cannot stop its own stream leaves the caller holding a thread
        that runs forever. Safe to call from another thread and more than once.
        """
        ...


class MemoryCommandChannel:
    """An in-process channel: commands handed to `offer` are yielded by `stream`.

    The fake a composer wires up to exercise an agent with no network — a real
    implementation of the seam, not test scaffolding.

    Not thread-safe against itself, but `close` may be called from another thread and
    ends an in-progress blocking `stream()` promptly. **One stream at a time**: two
    concurrent `stream()` calls on one channel are unsupported; start a second stream
    only after the first has returned.

    `offer` after `close` raises `RuntimeError` — a caller bug, failed loudly at the
    mistake rather than queuing a command nothing will ever deliver.
    """

    def __init__(self, *, block: bool = False) -> None:
        # `block=True` is the live-connection shape: wait for more, end on close().
        # `block=False` is the doorbell shape: drain what is queued, return.
        self._block = block
        self._queue: deque[StimulusEvent] = deque()
        self._closed = False
        # A Condition (not poll-and-sleep) so a cross-thread close() wakes a waiting
        # stream immediately instead of on the next tick.
        self._cond = Condition()

    def offer(self, event: StimulusEvent) -> None:
        with self._cond:
            if self._closed:
                raise RuntimeError("offer after close: the channel can never deliver it")
            self._queue.append(event)
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def stream(self) -> Iterator[StimulusEvent]:
        # Each event is popped before it is yielded, so a consumer that stops
        # mid-stream does not see it again — at-least-once is the caller's cursor's
        # job, not ours. The pop happens under the lock but the yield outside it:
        # holding the lock across the yield would block `offer` from another thread
        # for as long as the consumer paused on the yielded command.
        while True:
            with self._cond:
                while not self._queue:
                    if self._closed or not self._block:
                        return
                    self._cond.wait()
                event = self._queue.popleft()
            yield event
