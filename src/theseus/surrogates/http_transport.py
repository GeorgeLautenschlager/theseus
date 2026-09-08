"""The Phase 1 `StimulusTransport`: a POST to the host's `/replicate`.

One request in flight at a time is the replicator's lock, not this class's — a transport
that pipelines would break the stream before the host ever saw it. This class only decides
two things: what "the server answered" looks like (a `TransportResult`), and what "no
server was there" does (it raises). Folding those together is how a surrogate comes to
retry a refused batch as if it were an unreachable one, or vice versa — #32's retry rule
hinges on the difference.
"""

from __future__ import annotations

from typing import Any

import httpx

from theseus.replication_events import MAX_REASON_CHARS
from theseus.surrogates.transport import TransportResult


class HttpTransport:
    """POSTs a batch to a host's `/replicate`. The Phase 1 `StimulusTransport`."""

    def __init__(self, url: str, *, timeout: float = 30.0, client: Any | None = None) -> None:
        self._url = url
        self._timeout = timeout
        # The test seam: hand it a client bound to an in-process ASGI app and no socket
        # or server process is involved. An injected client's lifecycle belongs to its
        # owner, so only one we created ourselves gets closed.
        self._client = client

    def send(self, body: str) -> TransportResult:
        own_client = self._client is None
        # `follow_redirects=False` is the protocol, not a default worth inheriting. A `3xx`
        # from a misconfigured host or a proxy must reach the replicator as a non-2xx so the
        # drain stops; following it silently would POST the batch somewhere nobody chose and
        # report whatever that answered as the ack. httpx already defaults to False — pinned
        # explicitly because `TestClient`, which the tests inject, defaults to True, so the
        # seam would otherwise disagree with production on exactly this axis.
        client = (
            self._client
            if not own_client
            else httpx.Client(timeout=self._timeout, follow_redirects=False)
        )
        try:
            response = client.post(
                self._url,
                content=body.encode("utf-8"),
                headers={"content-type": "application/json"},
            )
        finally:
            if own_client:
                client.close()
        # httpx raises on a network-level failure (unreachable host, DNS, refused
        # connection) and we let it propagate: `5xx` is an answer, "never arrived" is not.
        return TransportResult(status=response.status_code, reason=self._reason(response))

    @staticmethod
    def _reason(response: Any) -> str:
        """The host's own words from a rejection body, read defensively.

        A body that is not JSON — or carries no string `reason` — yields `""` rather than
        raising: a transport that dies parsing an error response turns a clean `4xx` into
        what looks like an unreachable host, the one distinction this seam preserves.
        """
        try:
            reason = response.json().get("reason")
        except Exception:
            return ""
        if not isinstance(reason, str):
            return ""
        return reason[:MAX_REASON_CHARS]
