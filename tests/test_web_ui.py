from __future__ import annotations

from fastapi.testclient import TestClient

from theseus.surrogates.presence import ChatSurface
from theseus.surrogates.web_ui import SurrogateWebUI


def _make_ui():
    calls: list[str] = []
    ui = SurrogateWebUI(submit_user_message=calls.append)
    return ui, calls


def test_is_chat_surface():
    assert isinstance(SurrogateWebUI(lambda _t: None), ChatSurface)


def test_index_serves_chat_page():
    ui, _ = _make_ui()
    client = TestClient(ui.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_post_chat_calls_callback():
    ui, calls = _make_ui()
    client = TestClient(ui.app)
    response = client.post("/chat", data={"message": "hi"})
    assert response.status_code == 200
    assert calls == ["hi"]


def test_post_chat_empty_message_ignored():
    ui, calls = _make_ui()
    client = TestClient(ui.app)
    for message in ("", "   "):
        response = client.post("/chat", data={"message": message})
        assert response.status_code == 200
    assert calls == []


def test_publish_agent_message_fans_out_and_records():
    ui, _ = _make_ui()
    listener = ui._add_listener(ui._listeners)
    ui.publish_agent_message("hello")
    fragment = listener.get_nowait()
    assert "hello" in fragment
    assert ui.transcript[-1]["role"] == "assistant"
    assert "hello" in ui.transcript[-1]["content_html"]


def test_is_focused_defaults_true():
    assert SurrogateWebUI(lambda _t: None).is_focused() is True


def test_is_focused_uses_provider():
    assert SurrogateWebUI(lambda _t: None, focus_provider=lambda: False).is_focused() is False
    assert SurrogateWebUI(lambda _t: None, focus_provider=lambda: True).is_focused() is True
