"""Tests for `ToastNotifier` (issue #103) — fully offline, `windows-toasts` absent."""

from __future__ import annotations

import logging

import pytest

import theseus.surrogates.windows.toast as toast_module
from theseus.surrogates.presence import Notifier
from theseus.surrogates.windows.toast import ToastNotifier


def test_imports_and_satisfies_notifier_protocol_without_windows_toasts() -> None:
    notifier = ToastNotifier()
    assert isinstance(notifier, Notifier)


def test_notify_calls_backend_with_app_name_title_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        toast_module, "_show_windows_toast", lambda app, title, body: calls.append((app, title, body))
    )
    ToastNotifier(app_name="Theseus").notify("T", "B")
    assert calls == [("Theseus", "T", "B")]


def test_notify_swallows_backend_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(app: str, title: str, body: str) -> None:
        raise RuntimeError("no toast for you")

    monkeypatch.setattr(toast_module, "_show_windows_toast", boom)
    with caplog.at_level(logging.WARNING):
        ToastNotifier().notify("T", "B")  # must not raise
    assert any(r.levelno == logging.WARNING for r in caplog.records)
