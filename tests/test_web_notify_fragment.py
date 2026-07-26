from __future__ import annotations

from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock

from fastapi.templating import Jinja2Templates

import theseus.web
from theseus.web_chat_ui_observer import WebChatUIObserver

_TEMPLATES_DIR = Path(theseus.web.__file__).parent / "templates"


def _render(is_done: bool, notify_preview: str = "Hello George.") -> str:
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    return templates.get_template("_assistant_reply_fragment.html").render(
        is_first=True,
        is_done=is_done,
        bubble_id="abc123",
        content_html="<p>Hello George.</p>",
        time="3:04 PM",
        notify_preview=notify_preview,
    )


class TestNotifySignalElement:
    def test_absent_while_the_reply_is_still_streaming(self):
        assert 'id="chat-notify"' not in _render(is_done=False)

    def test_present_once_the_reply_is_done(self):
        html = _render(is_done=True)

        assert 'id="chat-notify"' in html
        assert 'data-msg-id="abc123"' in html
        assert 'data-preview="Hello George."' in html

    def test_swaps_out_of_band_so_it_reaches_the_page_target(self):
        assert 'hx-swap-oob="true"' in _render(is_done=True)


class TestNotifyPreviewEscaping:
    def test_quotes_cannot_break_out_of_the_attribute(self):
        html = _render(is_done=True, notify_preview='say "hi" now')

        assert 'say "hi" now' not in html
        assert "&#34;hi&#34;" in html or "&quot;hi&quot;" in html

    def test_markup_is_escaped_not_raw(self):
        html = _render(is_done=True, notify_preview="<script>alert(1)</script>")

        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html


class TestObserverSuppliesThePreview:
    def _observer_with_listener(self) -> tuple[WebChatUIObserver, Queue]:
        observer = WebChatUIObserver(orient_chat_message_callback=MagicMock())
        queue: Queue = Queue()
        observer._listeners.append(queue)
        return observer, queue

    def test_completed_reply_carries_a_condensed_preview(self):
        observer, queue = self._observer_with_listener()

        observer.publish_assistant_chunk("msg-1", "## Report\nAll systems normal.", done=True)

        fragment = queue.get_nowait()
        assert 'data-preview="Report All systems normal."' in fragment
        assert 'data-msg-id="msg-1"' in fragment

    def test_streaming_chunk_carries_no_signal(self):
        observer, queue = self._observer_with_listener()

        observer.publish_assistant_chunk("msg-1", "All systems", done=False)

        assert 'id="chat-notify"' not in queue.get_nowait()
