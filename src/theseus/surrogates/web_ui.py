"""SurrogateWebUI — the surrogate's on-screen chat: shows agent messages, captures user input.

A trimmed, surrogate-focused sibling of `WebChatUIObserver` (see
`web_chat_ui_observer.py`): same chat page, same SSE fan-out, same packaged
templates. The crucial difference: a surrogate has no cognitive core of its own,
so on user input it calls the injected `submit_user_message(text)` callback —
the runtime (#100) appends to the LOCAL stimulus log and ships it upstream —
instead of appending and orienting itself. Agent messages arrive whole (no token
streaming) via `publish_agent_message`, the `ChatSurface` seam the renderer calls.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from queue import Empty, Queue
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from theseus.web.markdown import render_markdown
from theseus.web.preview import notification_preview
from theseus.web_chat_ui_observer import _STATIC_DIR, _TEMPLATES_DIR, _format_sse_event

_SSE_POLL_TIMEOUT_SECONDS = 15


class SurrogateWebUI:
    """Serves the surrogate chat UI over HTTP/SSE and implements `ChatSurface`.

    Constructed with the runtime's `submit_user_message` callback (issue #100
    provides it); the HTTP request returns promptly — the callback owns whatever
    happens next, this class never appends to a log or triggers orient.

    `is_focused` is a best-effort `True` for now; the pywebview shell in #104
    refines it.
    """

    def __init__(self, submit_user_message: Callable[[str], None]):
        self.submit_user_message = submit_user_message
        self.transcript: list[dict] = []
        self._listeners: list[Queue] = []
        self._lock = threading.Lock()
        self._templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
        self.app = self._build_app()

    # -- ChatSurface ---------------------------------------------------------

    def publish_agent_message(self, text: str) -> None:
        """Show one agent message: fan it out to every open tab and record it."""
        content_html = render_markdown(text)
        entry = {"role": "assistant", "content_html": content_html, "time": self._time_label()}
        with self._lock:
            self.transcript.append(entry)
            listeners = list(self._listeners)
        fragment = self._templates.get_template("_assistant_reply_fragment.html").render(
            is_first=True,
            is_done=True,
            bubble_id=uuid.uuid4().hex,
            content_html=content_html,
            time=entry["time"],
            notify_preview=notification_preview(text),
        )
        for queue in listeners:
            queue.put(fragment)

    def is_focused(self) -> bool:
        # best-effort default; the pywebview shell (#104) refines it
        return True

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _time_label() -> str:
        return datetime.now().strftime("%I:%M %p").lstrip("0")

    def _handle_chat_submit(self, message: str) -> str:
        entry = {
            "role": "user",
            "content_html": render_markdown(message),
            "time": self._time_label(),
        }
        with self._lock:
            self.transcript.append(entry)
        self.submit_user_message(message)
        return self._templates.get_template("_chat_submit_fragment.html").render(**entry)

    def _add_listener(self) -> Queue:
        queue: Queue = Queue()
        with self._lock:
            self._listeners.append(queue)
        return queue

    async def _sse_stream(self, request: Request):
        queue = self._add_listener()
        try:
            yield "retry: 2000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    fragment = await run_in_threadpool(queue.get, timeout=_SSE_POLL_TIMEOUT_SECONDS)
                except Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield _format_sse_event(fragment)
        finally:
            with self._lock:
                if queue in self._listeners:
                    self._listeners.remove(queue)

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self._templates.TemplateResponse(
                request, "chat.html", {"transcript": self.transcript}
            )

        @app.post("/chat", response_class=HTMLResponse)
        async def chat(request: Request):
            form = await request.form()
            message = str(form.get("message", "")).strip()
            if not message:
                return HTMLResponse("")
            return HTMLResponse(self._handle_chat_submit(message))

        @app.get("/events")
        async def events(request: Request):
            return StreamingResponse(self._sse_stream(request), media_type="text/event-stream")

        return app

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the web server. Blocks; `timeout_graceful_shutdown` keeps Ctrl+C
        from hanging on the infinite /events SSE stream."""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port, timeout_graceful_shutdown=3)
