from __future__ import annotations

from fastapi.testclient import TestClient

from theseus.agents.alty_mcgee import AltyMcGee
from theseus.commands import command_target, command_type
from theseus.memory_store import MemoryStore
from theseus.stimulus_log import StimulusLog


def make_alty(tmp_path) -> AltyMcGee:
    return AltyMcGee(
        stimulus_log=StimulusLog(path=tmp_path / "log.jsonl"),
        memory_store=MemoryStore(tmp_path / "a_mem.jsonl"),
    )


def test_mount_surrogate_host_swaps_the_mouth(tmp_path):
    alty = make_alty(tmp_path)

    host = alty.mount_surrogate_host(target="windows-desktop")

    assert host.say_tool.target == "windows-desktop"
    assert host.notify_tool.target == "windows-desktop"
    for name in ("say_to_surrogate", "notify_user", "respond_in_web_chat"):
        assert name in alty.core.tools, name
    assert "terminal_chat" not in alty.core.tools


def test_mount_surrogate_host_routes_are_mounted(tmp_path):
    host = make_alty(tmp_path).mount_surrogate_host(target="windows-desktop")

    paths = {getattr(r, "path", None) for r in host.observer.app.routes}
    assert "/replicate" in paths
    assert any(p and p.startswith("/commands") for p in paths)


def test_chat_page_serves_without_cognition(tmp_path):
    host = make_alty(tmp_path).mount_surrogate_host()

    assert TestClient(host.observer.app).get("/").status_code == 200


def test_say_tool_lands_in_the_command_feed(tmp_path):
    host = make_alty(tmp_path).mount_surrogate_host(target="windows-desktop")

    host.say_tool.execute(text="hi")

    pending = host.command_feed.pending("windows-desktop", after=None)
    assert len(pending) == 1
    assert command_type("say") == "command.say"
    assert command_target(pending[0]) == "windows-desktop"
