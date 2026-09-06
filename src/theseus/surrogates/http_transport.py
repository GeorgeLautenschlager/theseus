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
        client = self._client if not own_client else httpx.Client(timeout=self._timeout)
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
        return TransportResult(status=response.status_code)
