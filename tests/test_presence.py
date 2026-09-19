"""Tests for the presence seams (issue #95): ChatSurface, Notifier, ConsoleNotifier."""

from __future__ import annotations

from theseus.surrogates import ChatSurface, ConsoleNotifier, Notifier


def test_console_notifier_prints_title_and_body(capsys) -> None:
    ConsoleNotifier().notify("Attention", "Tam needs you")
    out = capsys.readouterr().out
    assert "Attention" in out
    assert "Tam needs you" in out


def test_chat_surface_protocol_accepts_fakes() -> None:
    class FakeSurface:
        def publish_agent_message(self, text: str) -> None: ...
        def is_focused(self) -> bool:
            return True

    assert isinstance(FakeSurface(), ChatSurface)

    # runtime_checkable isinstance checks method presence, not signatures.
    class NotASurface:
        def is_focused(self) -> bool:
            return True

    assert not isinstance(NotASurface(), ChatSurface)


def test_notifier_protocol_accepts_fakes() -> None:
    class FakeNotifier:
        def notify(self, title: str, body: str) -> None: ...

    assert isinstance(FakeNotifier(), Notifier)

    # runtime_checkable isinstance checks method presence, not signatures.
    class NotANotifier:
        def ping(self) -> None: ...

    assert not isinstance(NotANotifier(), Notifier)
