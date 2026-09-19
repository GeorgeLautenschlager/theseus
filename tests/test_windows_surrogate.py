from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from theseus.surrogates.presence import ConsoleNotifier
from theseus.surrogates.windows.toast import ToastNotifier
from theseus.surrogates.windows_surrogate import WindowsSurrogateApp, build_windows_surrogate


def test_build_headless_uses_console_notifier(tmp_path: Path) -> None:
    app = build_windows_surrogate("http://host:8800", data_dir=tmp_path, headless=True)
    assert isinstance(app, WindowsSurrogateApp)
    assert isinstance(app.notifier, ConsoleNotifier)
    assert app.origin == "windows-desktop"
    assert app.headless is True


def test_build_not_headless_uses_toast_notifier(tmp_path: Path) -> None:
    app = build_windows_surrogate("http://host:8800", data_dir=tmp_path, headless=False)
    assert isinstance(app.notifier, ToastNotifier)


def test_chat_submit_reaches_runtime_log(tmp_path: Path) -> None:
    app = build_windows_surrogate("http://host:8800", data_dir=tmp_path, headless=True)
    response = TestClient(app.web_ui.app).post("/chat", data={"message": "hi"})
    assert response.status_code == 200
    events = [e for e in app.web_ui.stimulus_log.read_all() if e.type == "chat_message"]
    assert len(events) == 1
    assert events[0].content["message"] == "hi"


def test_command_half_is_wired(tmp_path: Path) -> None:
    app = build_windows_surrogate("http://host:8800", data_dir=tmp_path, headless=True)
    assert app.runtime._executor is not None


def test_focus_state_wired_to_web_ui(tmp_path: Path) -> None:
    app = build_windows_surrogate("http://host:8800", data_dir=tmp_path, headless=True)
    assert app.web_ui.is_focused() is True
    app.focus_state.set(False)
    assert app.web_ui.is_focused() is False
