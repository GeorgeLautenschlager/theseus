"""Unit tests for the WindowsPresence command renderer (issue #96). Fully offline."""

from __future__ import annotations

from datetime import datetime, timezone

from theseus.command_reports import Executed, Failed
from theseus.commands import command_type
from theseus.stimulus_log import StimulusEvent
from theseus.surrogates.presence import WindowsPresence


class FakeChat:
    def __init__(self, focused: bool = True) -> None:
        self.focused = focused
        self.published: list[str] = []

    def publish_agent_message(self, text: str) -> None:
        self.published.append(text)

    def is_focused(self) -> bool:
        return self.focused


class FakeNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def notify(self, title: str, body: str) -> None:
        self.calls.append((title, body))


def event(verb: str, content: dict) -> StimulusEvent:
    return StimulusEvent(
        id="e1",
        ts=datetime.now(timezone.utc),
        actor="host",
        type=command_type(verb),
        content=content,
    )


def test_say_publishes_and_is_executed():
    chat, notifier = FakeChat(), FakeNotifier()
    outcome = WindowsPresence(chat, notifier)(event("say", {"payload": {"text": "hi"}}))
    assert isinstance(outcome, Executed)
    assert chat.published == ["hi"]
    assert notifier.calls == []  # focused by default: no toast


def test_unfocused_say_also_notifies():
    chat, notifier = FakeChat(focused=False), FakeNotifier()
    outcome = WindowsPresence(chat, notifier)(event("say", {"payload": {"text": "hi"}}))
    assert isinstance(outcome, Executed)
    assert chat.published == ["hi"]
    assert notifier.calls == [("New message", "hi")]


def test_notify_calls_notifier():
    chat, notifier = FakeChat(), FakeNotifier()
    outcome = WindowsPresence(chat, notifier)(
        event("notify", {"payload": {"title": "T", "body": "B"}})
    )
    assert isinstance(outcome, Executed)
    assert notifier.calls == [("T", "B")]


def test_unknown_verb_fails():
    chat, notifier = FakeChat(), FakeNotifier()
    outcome = WindowsPresence(chat, notifier)(event("dance", {"payload": {}}))
    assert isinstance(outcome, Failed)
    assert outcome.reason


def test_missing_text_key_fails():
    chat, notifier = FakeChat(), FakeNotifier()
    outcome = WindowsPresence(chat, notifier)(event("say", {"payload": {}}))
    assert isinstance(outcome, Failed)
    assert chat.published == []
    assert notifier.calls == []


def test_non_dict_content_fails():
    chat, notifier = FakeChat(), FakeNotifier()
    outcome = WindowsPresence(chat, notifier)(event("say", "not a dict"))
    assert isinstance(outcome, Failed)
