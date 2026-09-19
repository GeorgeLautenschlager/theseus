"""Issue #97: SaySurrogate / NotifySurrogate append host-origin commands."""

from __future__ import annotations

from theseus.commands import command_content, command_type
from theseus.stimulus_log import StimulusLog
from theseus.tools.surrogate_voice import NotifySurrogate, SaySurrogate


def make_log(tmp_path):
    return StimulusLog(tmp_path / "log.jsonl", origin="host-log")


def test_say_appends_command(tmp_path):
    log = make_log(tmp_path)
    tool = SaySurrogate(target="windows-desktop", stimulus_log=log)
    result = tool.execute(text="hi")
    events = log.read_all()
    assert len(events) == 1
    event = events[0]
    assert event.type == command_type("say")
    assert event.origin == "host-log"  # host's own origin, not foreign
    assert event.seq is not None
    assert event.content == command_content(
        target="windows-desktop", payload={"text": "hi"}
    )
    assert result.is_error is False


def test_notify_appends_command(tmp_path):
    log = make_log(tmp_path)
    tool = NotifySurrogate(target="windows-desktop", stimulus_log=log)
    result = tool.execute(title="T", body="B")
    (event,) = log.read_all()
    assert event.type == command_type("notify")
    assert event.content == command_content(
        target="windows-desktop", payload={"title": "T", "body": "B"}
    )
    assert result.is_error is False


def test_ends_turn_flags(tmp_path):
    log = make_log(tmp_path)
    assert SaySurrogate("t", log).ends_turn is True
    assert NotifySurrogate("t", log).ends_turn is False


def test_schemas(tmp_path):
    log = make_log(tmp_path)
    say = SaySurrogate("t", log)
    notify = NotifySurrogate("t", log)
    for tool, keys in ((say, ["text"]), (notify, ["title", "body"])):
        assert tool.name
        assert tool.description
        assert tool.parameters["type"] == "object"
        assert set(tool.parameters["properties"]) == set(keys)
        assert tool.parameters["required"] == keys
