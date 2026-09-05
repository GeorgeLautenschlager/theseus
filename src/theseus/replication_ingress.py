"""The host's replication ingress: where a surrogate's events actually land.

`replication_batch` decides whether a batch is acceptable, `replication_dedupe` decides
what it adds, and `StimulusLog.append_many` commits it. This module is the only piece that
touches all three, plus the clock, the marks and the network — everything the other two
were kept pure of.
"""

from __future__ import annotations

import threading
import traceback
from typing import Callable


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
        """Begin serving requests. Idempotent; starting twice is a no-op, not two threads."""
        if self._thread is not None:
            return
        self._stopping.clear()
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
        self._thread = None
        self._stopping.set()
        self._pending.set()  # wake the worker so it can notice it is stopping
        thread.join(timeout)

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
