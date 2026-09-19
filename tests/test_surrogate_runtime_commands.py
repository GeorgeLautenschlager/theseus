"""Tests for `SurrogateRuntime`'s command-execution half (#101).

Fully offline: a `MemoryCommandChannel`, a recording renderer, and the same fake
transport as the drain tests. Deterministic — the only background-thread assertion
is the start/stop smoke test, which waits on an event with a generous timeout.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from theseus.command_reports import Executed
from theseus.commands import command_content, command_type
from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.surrogates.command_channel import MemoryCommandChannel
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.runtime import SurrogateRuntime
from theseus.surrogates.transport import TransportResult


class FakeTransport:
    """Records each `send(body)` and answers `status`."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, body: str) -> TransportResult:
        self.sent.append(body)
        return TransportResult(status=200)


def make_command(seq: int = 1, id: str = "c1") -> StimulusEvent:
    return StimulusEvent(
        id=id,
        ts=datetime.now(timezone.utc),
        actor="host",
        type=command_type("say"),
        content=command_content(target="windows-desktop", payload={"text": "hi"}),
        origin="host",
        seq=seq,
    )


def make_runtime(tmp_path):
    log = StimulusLog(tmp_path / "log.jsonl", origin="windows-desktop")
    transport = FakeTransport()
    upstream_cursor = AckedCursor(tmp_path / "upstream.json", "windows-desktop")
    command_cursor = AckedCursor(tmp_path / "commands.json", "windows-desktop")
    rendered: list[StimulusEvent] = []

    def render(event: StimulusEvent) -> Executed:
        rendered.append(event)
        return Executed()

    channel = MemoryCommandChannel(block=False)
    runtime = SurrogateRuntime(
        log,
        transport,
        upstream_cursor,
        command_channel=channel,
        renderer=render,
        command_cursor=command_cursor,
    )
    return runtime, channel, rendered, transport, upstream_cursor, command_cursor


def test_partial_command_wiring_raises(tmp_path) -> None:
    log = StimulusLog(tmp_path / "log.jsonl", origin="windows-desktop")
    cursor = AckedCursor(tmp_path / "cursor.json", "windows-desktop")
    with pytest.raises(ValueError):
        SurrogateRuntime(
            log, FakeTransport(), cursor,
            renderer=lambda e: Executed(),  # missing channel and cursor
        )


def test_command_executes_and_advances_command_cursor(tmp_path) -> None:
    runtime, channel, rendered, _transport, upstream, command_cursor = make_runtime(
        tmp_path
    )
    cmd = make_command()
    channel.offer(cmd)
    runtime._executor.run(channel)  # drains and returns (block=False)
    assert rendered == [cmd]
    reports = [e for e in runtime._log.read_all() if e.type == "command_report.executed"]
    assert len(reports) == 1
    assert command_cursor.acked_seq == 1
    assert upstream.acked_seq is None


def test_report_drains_upstream(tmp_path) -> None:
    runtime, channel, _rendered, transport, _upstream, _cmd = make_runtime(tmp_path)
    channel.offer(make_command())
    runtime._executor.run(channel)
    runtime._drain_once()
    assert any("command_report.executed" in b for b in transport.sent)
    # The host-origin command was never on the surrogate's log, so only its report ships.
    assert not any("command.say" in b for b in transport.sent)


def test_start_stop_smoke(tmp_path) -> None:
    log = StimulusLog(tmp_path / "log.jsonl", origin="windows-desktop")
    rendered = threading.Event()

    def render(event: StimulusEvent) -> Executed:
        rendered.set()
        return Executed()

    channel = MemoryCommandChannel(block=True)
    runtime = SurrogateRuntime(
        log,
        FakeTransport(),
        AckedCursor(tmp_path / "upstream.json", "windows-desktop"),
        command_channel=channel,
        renderer=render,
        command_cursor=AckedCursor(tmp_path / "commands.json", "windows-desktop"),
    )
    runtime.start()
    try:
        channel.offer(make_command())
        # Fires in milliseconds; the generous timeout only trips on genuine breakage.
        assert rendered.wait(timeout=10.0)
    finally:
        runtime.stop()  # close ends stream(); must return without hanging
