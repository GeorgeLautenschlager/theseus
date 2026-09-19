"""Offline tests for the surrogate host wiring (`agents/surrogate_host.py`)."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from theseus.commands import command_target, command_type
from theseus.high_water import HighWaterMarks
from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.agents.surrogate_host import build_surrogate_host


def _host(tmp_path: Path):
    log = StimulusLog(tmp_path / "stimulus_log.jsonl", origin="host")
    marks = HighWaterMarks(log)
    calls: list = []
    fired = threading.Event()

    def orient_callback(*args, **kwargs):
        calls.append((args, kwargs))
        fired.set()

    host = build_surrogate_host(log, marks, orient_callback)
    client = TestClient(host.observer.app)
    return host, client, fired, calls


def _surrogate_event() -> StimulusEvent:
    return StimulusEvent(
        id="ev-1",
        ts=datetime.now(timezone.utc),
        actor="user",
        type="chat_message",
        content={"message": "hi"},
        origin="windows-desktop",
        seq=1,
    )


def test_replicate_appends_and_triggers_orient(tmp_path: Path) -> None:
    host, client, fired, calls = _host(tmp_path)
    host.ingress.start()
    try:
        response = client.post(
            "/replicate",
            content=(_surrogate_event().to_json() + "\n").encode(),
            headers={"content-type": "application/json"},
        )
        assert 200 <= response.status_code < 300
        origins = [event.origin for event in host.observer.stimulus_log.read_all()]
        assert "windows-desktop" in origins
        assert fired.wait(timeout=10.0), "orient_callback never fired"
        assert len(calls) == 1
    finally:
        host.ingress.stop()


def test_say_tool_appends_command_pending_in_feed(tmp_path: Path) -> None:
    host, _client, _fired, _calls = _host(tmp_path)
    host.say_tool.execute(text="hello")
    commands = host.command_feed.pending("windows-desktop", after=None)
    assert len(commands) == 1
    assert commands[0].type == command_type("say")
    assert command_target(commands[0]) == "windows-desktop"


def test_chat_page_served_on_same_app(tmp_path: Path) -> None:
    host, client, _fired, _calls = _host(tmp_path)
    assert client.get("/").status_code == 200


def test_voice_tools_target(tmp_path: Path) -> None:
    host, *_ = _host(tmp_path)
    assert host.say_tool.target == "windows-desktop"
    assert host.notify_tool.target == "windows-desktop"
