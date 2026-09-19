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

import json
import threading
import time
import uuid
from datetime import datetime
from queue import Empty, Queue
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from theseus.stimulus_log import StimulusLog
from theseus.web.assets import STATIC_DIR, TEMPLATES_DIR, format_sse_event
from theseus.web.debug_pagination import most_recent_page, older_batch, parse_int_param
from theseus.web.markdown import render_markdown
from theseus.web.preview import notification_preview

_SSE_POLL_TIMEOUT_SECONDS = 15
_DEBUG_PAGE_SIZE = 25
_DEBUG_POLL_INTERVAL_SECONDS = 1.0


class SurrogateWebUI:
    """Serves the surrogate chat UI over HTTP/SSE and implements `ChatSurface`.

    Constructed with the runtime's `submit_user_message` callback (issue #100
    provides it); the HTTP request returns promptly — the callback owns whatever
    happens next, this class never appends to a log or triggers orient.

    `is_focused` answers via the injected `focus_provider` when one was given
    (the Windows shell feeds the pywebview window's focus state in); default True.
    """

    def __init__(
        self,
        submit_user_message: Callable[[str], None],
        stimulus_log: StimulusLog | None = None,
        *,
        focus_provider: Callable[[], bool] | None = None,
    ):
        self.submit_user_message = submit_user_message
        self.stimulus_log = stimulus_log
        self._focus_provider = focus_provider
        self.transcript: list[dict] = []
        self._listeners: list[Queue] = []
        self._lock = threading.Lock()
        self._debug_listeners: list[Queue] = []
        self._debug_last_id: str | None = None
        self._debug_poll_thread: threading.Thread | None = None
        self._templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
        self._templates.env.filters["pretty_json"] = lambda content: json.dumps(
            content, indent=2, ensure_ascii=False
        )
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
        return self._focus_provider() if self._focus_provider else True

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

    def _add_listener(self, listeners: list[Queue]) -> Queue:
        queue: Queue = Queue()
        with self._lock:
            listeners.append(queue)
        return queue

    async def _sse_stream(self, request: Request, listeners: list[Queue]):
        queue = self._add_listener(listeners)
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
                yield format_sse_event(fragment)
        finally:
            with self._lock:
                if queue in listeners:
                    listeners.remove(queue)

    # -- debug mode ------------------------------------------------------

    def _debug_initial_context(self) -> dict:
        events = self.stimulus_log.read_all() if self.stimulus_log is not None else []
        page, has_more = most_recent_page(events, _DEBUG_PAGE_SIZE)
        if page:
            with self._lock:
                self._debug_last_id = max(self._debug_last_id or "", page[-1].id)
        return {"events": page, "has_more": has_more, "oldest_id": page[0].id if page else None}

    def _debug_older_context(self, before: str, limit: int) -> dict:
        limit = max(1, min(limit, 200))
        events = self.stimulus_log.read_all() if self.stimulus_log is not None else []
        batch, has_more = older_batch(events, before_id=before, limit=limit)
        return {"events": batch, "has_more": has_more, "oldest_id": batch[0].id if batch else before}

    def _ensure_debug_poll_thread(self) -> None:
        with self._lock:
            if self._debug_poll_thread is not None and self._debug_poll_thread.is_alive():
                return
            self._debug_poll_thread = threading.Thread(target=self._debug_poll_loop, daemon=True)
            self._debug_poll_thread.start()

    def _debug_poll_loop(self) -> None:
        """Broadcast newly appended StimulusEvents to open /debug tabs.

        Polls read_all() every interval (StimulusLog has no subscriber
        mechanism); runs only while at least one debug tab is connected and
        self-terminates otherwise, restarted lazily by _ensure_debug_poll_thread.
        Mirrors WebChatUIObserver._debug_poll_loop.
        """
        while True:
            time.sleep(_DEBUG_POLL_INTERVAL_SECONDS)
            with self._lock:
                listeners = list(self._debug_listeners)
                cursor = self._debug_last_id
            if not listeners:
                return
            if self.stimulus_log is None:
                continue
            events = self.stimulus_log.read_all()
            if cursor is None:
                with self._lock:
                    self._debug_last_id = events[-1].id if events else None
                continue
            new_events = [e for e in events if e.id > cursor]
            if not new_events:
                continue
            fragment = self._templates.get_template("_debug_new_rows_fragment.html").render(
                events=new_events
            )
            with self._lock:
                self._debug_last_id = max(self._debug_last_id or "", new_events[-1].id)
                listeners = list(self._debug_listeners)
            for queue in listeners:
                queue.put(fragment)

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
            return StreamingResponse(
                self._sse_stream(request, self._listeners), media_type="text/event-stream"
            )

        @app.get("/debug", response_class=HTMLResponse)
        async def debug_page(request: Request):
            return self._templates.TemplateResponse(
                request, "debug.html", self._debug_initial_context()
            )

        @app.get("/debug/older", response_class=HTMLResponse)
        async def debug_older(request: Request):
            before = request.query_params.get("before", "")
            limit = parse_int_param(request.query_params.get("limit"), _DEBUG_PAGE_SIZE)
            return self._templates.TemplateResponse(
                request, "_debug_older_fragment.html", self._debug_older_context(before, limit)
            )

        @app.get("/debug/events")
        async def debug_events(request: Request):
            self._ensure_debug_poll_thread()
            return StreamingResponse(
                self._sse_stream(request, self._debug_listeners), media_type="text/event-stream"
            )

        return app

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the web server. Blocks; `timeout_graceful_shutdown` keeps Ctrl+C
        from hanging on the infinite /events SSE stream."""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port, timeout_graceful_shutdown=3)
