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
import time
from collections.abc import Iterator

import httpx

from theseus.stimulus_log import StimulusEvent
from theseus.surrogates.cursor import AckedCursor

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
        reconnect_seconds: float = 2.0,
    ) -> None:
        self._url = url
        self._cursor = cursor
        self._max_reconnects = max_reconnects
        self._reconnect_seconds = reconnect_seconds
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
        attempts = 0
        while True:
            headers: dict[str, str] = {}
            acked = self._cursor.acked_seq
            # Absent header, not `0`: a never-advanced cursor is "no position", and
            # sending 0 would claim the surrogate has executed everything below 1.
            if acked is not None:
                headers["last-event-id"] = str(acked)
            try:
                with self._client.stream("GET", self._url, headers=headers) as response:
                    response.raise_for_status()
                    yield from self._frames(response)
            except httpx.HTTPError as exc:
                # A drop — network, timeout, or a non-2xx from a host mid-restart — is
                # retried, not fatal: a command issued during the gap is queued, not
                # lost, and the cursor on reconnect skips what was already executed.
                logger.warning("command stream from %s dropped (%s)", self._url, exc)
            attempts += 1
            if self._max_reconnects is not None and attempts > self._max_reconnects:
                return
            time.sleep(self._reconnect_seconds)

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