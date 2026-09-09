"""The LAN `CommandChannel`: an SSE subscription to the host's `GET /commands/{target}`.

The downstream counterpart of `HttpTransport`: same seam (`CommandChannel`), other
direction. The wire shape parsed here is exactly `command_feed._format_sse`'s — `id:`
carries the seq, `event:` the log type, `data:` lines the event's JSON, possibly split
across several lines and rejoined with newlines before parsing. Heartbeats are `:`
comment lines: consumed, yielding nothing, never ending the stream.

**This class does not advance the cursor.** A reader who has only seen the replicator
will think that is an oversight — upstream, the receiver's position moves as frames
arrive and the host dedupes replays. Here the cursor means *executed*, not *received*:
the caller advances it after hand-off, and each reconnect re-reads it, so what the
caller has already executed is never replayed. Advancing on receipt would quietly
convert the protocol's at-least-once delivery to at-most-once, and a dropped command is
invisible — nothing dedupes a spoken sentence, and nothing reports it until #35 lands.
A duplicate, by contrast, is visible on the tape. That is the side to fail on.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterator

import httpx

from theseus.stimulus_log import StimulusEvent
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.retry import RetryBudget, backoff_delay

logger = logging.getLogger(__name__)


class SseCommandChannel:
    """A `CommandChannel` over SSE from the host's `CommandFeed` (issue #34, Task 4).

    `url` is the full command endpoint for one surrogate; `cursor` is the
    surrogate's `AckedCursor`, here meaning "how far I have executed" rather than "how
    far the host has acked me" — same mechanism, second use. `client` is injectable so
    tests drive a real host in-process; an injected client's lifecycle belongs to its
    owner, one we create ourselves is closed when the channel is.
    """

    def __init__(
        self,
        url: str,
        cursor: AckedCursor,
        *,
        client: httpx.Client | None = None,
        max_reconnects: int | None = None,
        budget: RetryBudget = RetryBudget(),
        random_fn: Callable[[], float] = random.random,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._url = url
        self._cursor = cursor
        self._max_reconnects = max_reconnects
        self._budget = budget
        self._random_fn = random_fn
        self._sleep_fn = sleep_fn
        self._now_fn = now_fn
        # The read timeout must outlast the host's heartbeat gap (20s by default) or an
        # idle-but-healthy connection would be dropped and reopened on a loop. 60s of
        # silence still reconnects — recoverable, the cursor resumes — and a host that
        # heartbeats slower than that should lower its interval, not raise this.
        # `follow_redirects=False` is the protocol, per `HttpTransport`: a 3xx from a
        # misconfigured host must surface as a failure, not silently retarget the
        # command stream.
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0), follow_redirects=False
        )
        self._own_client = client is None

    def close(self) -> None:
        """End the channel; an in-progress `stream()` ends on the next read."""
        if self._own_client:
            self._client.close()

    def stream(self) -> Iterator[StimulusEvent]:
        """Yield commands from the cursor forward, reconnecting on every drop.

        Each connect re-reads `cursor.acked_seq` — never cached from the first one —
        so a reconnect resumes from what the caller has *executed*, not from what this
        connection last saw. The two differ whenever the caller advanced past only part
        of what a connection delivered, and only the first is correct.
        """
        failures = 0
        while True:
            headers: dict[str, str] = {}
            acked = self._cursor.acked_seq
            # Absent header, not `0`: a never-advanced cursor is "no position", and
            # sending 0 would claim the surrogate has executed everything below 1.
            if acked is not None:
                headers["last-event-id"] = str(acked)
            opened_at: float | None = None
            try:
                with self._client.stream("GET", self._url, headers=headers) as response:
                    response.raise_for_status()
                    opened_at = self._now_fn()
                    for event in self._frames(response):
                        yield event
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if 400 <= status < 500 and status not in (408, 429):
                    # A 4xx is the host saying this request will *never* work — a
                    # mistyped target, a route not mounted. Retrying it forever is how
                    # a broken URL becomes 43k requests a day; `Replicator.drain`
                    # made the same call upstream (4xx stepped over, only 5xx spends
                    # retry budget). 408 and 429 are the exceptions because retrying
                    # is their own definition, not an assumption of ours.
                    logger.error(
                        "command stream from %s refused with HTTP %s; not retrying",
                        self._url,
                        status,
                    )
                    return
                # The host is answering that it is broken (5xx, 408, 429) — transient,
                # same as a drop.
                logger.warning("command stream from %s answered HTTP %s", self._url, status)
            except httpx.HTTPError as exc:
                # A drop — network, timeout, or a host mid-restart — is transient:
                # a command issued during the gap is queued, not lost, and the cursor
                # on reconnect skips what was already executed.
                logger.warning("command stream from %s dropped (%s)", self._url, exc)
            # Reset on a connection that *lasted*, not one that delivered. "Reset
            # when we got a command" is the intuitive wrong answer here: hours of
            # silence is the normal case for a command channel, not a symptom — an
            # idle connection is exactly what it is supposed to be, which is why the
            # host heartbeats. What a flapping host does is close immediately, so the
            # threshold is `budget.base_seconds`: a connection that survived longer
            # than the shortest retry delay was a real connection, while a 200 that
            # closes before saying anything does not clear it and `max_reconnects`
            # still bounds the flapping.
            if opened_at is not None and self._now_fn() - opened_at > self._budget.base_seconds:
                failures = 0
            failures += 1
            if self._max_reconnects is not None and failures > self._max_reconnects:
                return
            # #39's backoff, not a fixed delay: growing with consecutive failures and
            # jittered so a fleet reconnecting after a host restart does not arrive as
            # one thundering herd. `max_attempts` is about a batch upstream and must
            # not bound reconnects — a host down for a day is still worth reaching.
            self._sleep_fn(backoff_delay(failures, self._budget, random_fn=self._random_fn))

    @staticmethod
    def _frames(response: httpx.Response) -> Iterator[StimulusEvent]:
        """Parse SSE frames off the wire, one `StimulusEvent` per event.

        A frame is the `data:` lines since the last blank line, rejoined with newlines
        — the inverse of `_format_sse`'s split. Comment lines (the host's heartbeats)
        yield nothing; `id:`/`event:`/`retry:` carry nothing this side needs, because
        the cursor is the caller's, advanced after execution, not the wire's.
        """
        data: list[str] = []
        for raw in response.iter_lines():
            line = raw.rstrip("\r")
            if line.startswith(":"):
                continue
            if not line.strip():
                if data:
                    payload = "\n".join(data)
                    data = []
                    try:
                        yield StimulusEvent.from_json(payload)
                    except (ValueError, KeyError, TypeError) as exc:
                        # Skip, never raise: one bad frame must not take the channel
                        # down, and the next command is still worth having — but the
                        # skip is loud, because a silently dropped command is exactly
                        # the failure this protocol exists to prevent.
                        logger.warning("skipping malformed command frame: %s", exc)
                continue
            if line.startswith("data:"):
                data.append(line[len("data:") :].lstrip(" "))