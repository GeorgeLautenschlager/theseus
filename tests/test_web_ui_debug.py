"""Tests for the surrogate debug log viewer (issue #99)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from theseus.stimulus_log import StimulusLog
from theseus.surrogates.web_ui import SurrogateWebUI


def _make_log(tmp_path, n=3):
    log = StimulusLog(tmp_path / "stimulus.jsonl")
    for i in range(n):
        log.append(actor="user", type="chat_message", content={"message": f"msg {i}"})
    return log


def test_debug_renders_local_log(tmp_path):
    log = _make_log(tmp_path)
    ui = SurrogateWebUI(submit_user_message=lambda _t: None, stimulus_log=log)
    client = TestClient(ui.app)
    response = client.get("/debug")
    assert response.status_code == 200
    assert "chat_message" in response.text
    assert "msg 2" in response.text


def test_debug_older_pages_prior_batch(tmp_path):
    log = _make_log(tmp_path, n=3)
    ui = SurrogateWebUI(submit_user_message=lambda _t: None, stimulus_log=log)
    client = TestClient(ui.app)
    newest = log.read_all()[-1]
    response = client.get("/debug/older", params={"before": newest.id, "limit": 1})
    assert response.status_code == 200
    assert "msg 1" in response.text
    assert "msg 2" not in response.text


def test_debug_initial_context(tmp_path):
    log = _make_log(tmp_path, n=3)
    ui = SurrogateWebUI(submit_user_message=lambda _t: None, stimulus_log=log)
    context = ui._debug_initial_context()
    events = log.read_all()
    assert [e.id for e in context["events"]] == [e.id for e in events]
    assert context["has_more"] is False
    assert context["oldest_id"] == events[0].id


def test_debug_none_log_empty_view():
    ui = SurrogateWebUI(submit_user_message=lambda _t: None)
    client = TestClient(ui.app)
    response = client.get("/debug")
    assert response.status_code == 200
    assert response.text


def test_debug_older_bad_limit_falls_back_not_500(tmp_path):
    log = _make_log(tmp_path)
    ui = SurrogateWebUI(submit_user_message=lambda _t: None, stimulus_log=log)
    client = TestClient(ui.app)
    response = client.get("/debug/older", params={"before": "", "limit": "abc"})
    assert response.status_code == 200
