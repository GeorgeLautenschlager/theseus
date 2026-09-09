"""Shared test fixtures."""

from __future__ import annotations

import threading
import time

import pytest
import uvicorn


@pytest.fixture
def serve():
    """A threaded uvicorn on an ephemeral port. Starlette's TestClient cannot stream an
    infinite SSE endpoint — `client.stream(...)` never even returns the status line —
    but a real socket with a sync httpx.Client works."""
    servers: list[tuple[uvicorn.Server, threading.Thread]] = []

    def start(app) -> str:
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            log_level="error",
            timeout_graceful_shutdown=1,
            lifespan="off",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "test server did not start in time"
        port = server.servers[0].sockets[0].getsockname()[1]
        servers.append((server, thread))
        return f"http://127.0.0.1:{port}"

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "test server did not shut down in time"
