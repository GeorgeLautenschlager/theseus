"""Claims about what a command is (issue #34, Task 1)."""

from __future__ import annotations

import pytest

from theseus.commands import (
    COMMAND_PREFIX,
    command_content,
    command_target,
    command_type,
    is_command,
)
from theseus.stimulus_log import StimulusEvent


def _event(type: str, content: dict) -> StimulusEvent:
    from datetime import datetime, timezone

    return StimulusEvent(
        id="e1",
        ts=datetime.now(timezone.utc),
        actor="host",
        type=type,
        content=content,
    )


def test_verb_becomes_namespaced_type():
    assert command_type("say") == "command.say"


def test_bad_verbs_refused():
    with pytest.raises(ValueError):
        command_type("")
    with pytest.raises(ValueError):
        command_type("say something")
    # The helpful-caller bug: passing what command_type would return.
    with pytest.raises(ValueError, match="prefix"):
        command_type("command.say")


def test_content_carries_target_and_payload():
    content = command_content(target="tam", payload={"text": "hello"})
    assert content["target"] == "tam"
    assert content["payload"] == {"text": "hello"}
    assert set(content) == {"target", "payload"}
    assert command_content(target="tam", payload={})["payload"] == {}


def test_unaddressed_command_refused():
    with pytest.raises(ValueError):
        command_content(target="", payload={})


def test_bare_prefix_is_not_a_command():
    event = _event(COMMAND_PREFIX, {"target": "tam"})
    assert not is_command(event)


def test_non_command_event_is_not_a_command():
    event = _event("observation", {"note": "the cat is on the mat"})
    assert not is_command(event)
    assert command_target(event) is None


def test_malformed_command_yields_none():
    missing = _event(command_type("say"), {"payload": {}})
    not_a_string = _event(command_type("say"), {"target": 7, "payload": {}})
    assert command_target(missing) is None
    assert command_target(not_a_string) is None
