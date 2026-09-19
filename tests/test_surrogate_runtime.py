"""Tests for `SurrogateRuntime` — the replication drain half (#100).

Fully offline: a fake transport records batches and returns acks; everything else is
the real `StimulusLog`, `AckedCursor`, and `Replicator`. Deterministic — the only
background-thread assertion is an event-wait smoke test with a generous timeout.
"""

from __future__ import annotations

import threading

from theseus.stimulus_log import StimulusLog
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.runtime import SurrogateRuntime
from theseus.surrogates.transport import TransportResult


class FakeTransport:
    """Records each `send(body)` and answers `status`."""

    def __init__(self, status: int = 200, on_send: threading.Event | None = None):
        self.status = status
        self.sent: list[str] = []
        self._on_send = on_send

    def send(self, body: str) -> TransportResult:
        self.sent.append(body)
        if self._on_send is not None:
            self._on_send.set()
        return TransportResult(status=self.status)


def make_runtime(tmp_path, transport) -> tuple[SurrogateRuntime, AckedCursor]:
    log = StimulusLog(tmp_path / "log.jsonl", origin="windows-desktop")
    cursor = AckedCursor(tmp_path / "cursor.json", "windows-desktop")
    return SurrogateRuntime(log, transport, cursor), cursor


def test_submit_appends_chat_message(tmp_path) -> None:
    runtime, _ = make_runtime(tmp_path, FakeTransport())
    runtime.submit_user_message("hi")
    events = runtime._log.read_all()
    assert len(events) == 1
    assert events[0].actor == "user"
    assert events[0].type == "chat_message"
    assert events[0].content == {"message": "hi"}
    assert events[0].origin == "windows-desktop"


def test_drain_once_ships_and_advances_cursor(tmp_path) -> None:
    transport = FakeTransport()
    runtime, cursor = make_runtime(tmp_path, transport)
    runtime.submit_user_message("hi")
    runtime._drain_once()
    assert len(transport.sent) == 1
    assert "hi" in transport.sent[0]
    seq = runtime._log.read_all()[0].seq
    assert cursor.acked_seq == seq


def test_second_drain_ships_nothing_new(tmp_path) -> None:
    transport = FakeTransport()
    runtime, _ = make_runtime(tmp_path, transport)
    runtime.submit_user_message("hi")
    runtime._drain_once()
    runtime._drain_once()
    assert len(transport.sent) == 1


def test_start_ships_append_via_listener(tmp_path) -> None:
    """Smoke test of the live wiring: subscribe + trigger + drain worker."""
    shipped = threading.Event()
    runtime, cursor = make_runtime(
        tmp_path, FakeTransport(on_send=shipped)
    )
    runtime.start()
    try:
        runtime.submit_user_message("hi")
        # Fires in milliseconds; the generous timeout only trips on genuine breakage.
        assert shipped.wait(timeout=10.0)
        assert runtime._log.read_all()[0].seq == cursor.acked_seq
    finally:
        runtime.stop()
