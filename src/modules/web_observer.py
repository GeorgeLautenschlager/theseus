"""WebObserver — serves the Theseus Chat web UI and feeds user messages into the agent.

Pairs with `WebEffector` (see `web_effector.py`) the same way `ChatObserver`
pairs with `ChatEffector`, but over HTTP/SSE instead of stdin/stdout.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from src.modules.web.markdown import render_markdown

_WEB_DIR = Path(__file__).parent / "web"
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"

_SSE_POLL_TIMEOUT_SECONDS = 15


class WebObserver:
    """Serves the Theseus Chat web UI and feeds user messages into the agent.

    Mirrors ChatObserver's role — append the incoming stimulus, then hand off
    to whatever triggers orientation — but the "blocking read" is an HTTP
    request instead of stdin, and replies are pushed back out over
    Server-Sent Events instead of a blocking function return.

    `orient_chat_message_callback` is invoked with the user's message text on
    a background thread, so the HTTP request returns immediately (the page
    shows a typing indicator while the agent works — the browser is never
    left hanging on a slow LLM call). Wire it directly to a
    `ChatCognitiveCore.orient(message)`-style core. For a zero-argument core
    such as `CognitiveCore.orient()`, wrap it in the agent file:
    `orient_chat_message_callback=lambda message: core.orient()` — the
    message has already been appended to the stimulus log by the time the
    callback fires, so a log-driven core still sees it.

    Only one `orient` call is ever in flight at a time: the composer is
    disabled for the duration of a reply, both because the design calls for
    it and because the cognitive cores in this codebase keep loop-scoped
    state (`loop_memory`, `WorkingMemory`) that isn't safe for concurrent
    orientation.
    """

    def __init__(
        self,
        orient_chat_message_callback: Callable[[str], None],
        stimulus_log=None,
    ):
        self.orient_chat_message_callback = orient_chat_message_callback
        self.stimulus_log = stimulus_log
        self.transcript: list[dict] = []
        self._listeners: list[Queue] = []
        self._streaming: dict[str, str] = {}
        self._lock = threading.Lock()
        self._templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
        self.app = self._build_app()

    # -- called by WebEffector ---------------------------------------------

    def publish_assistant_chunk(self, message_id: str, text: str, done: bool) -> None:
        """Broadcast one step of an in-progress (or completed) agent reply.

        `message_id` identifies one reply across repeated calls: the first
        call for a given id creates the assistant's bubble in every open
        browser tab, later calls replace its contents, and the call with
        `done=True` finalizes it (restores "Online" status, re-enables the
        composer, and records the reply in the transcript for future page
        loads).
        """
        content_html = render_markdown(text)
        with self._lock:
            is_first = message_id not in self._streaming
            time_label = self._streaming.setdefault(message_id, self._time_label())
            if done:
                self._streaming.pop(message_id, None)
                self.transcript.append(
                    {"role": "assistant", "content_html": content_html, "time": time_label}
                )
            listeners = list(self._listeners)
        fragment = self._templates.get_template("_assistant_reply_fragment.html").render(
            is_first=is_first,
            is_done=done,
            bubble_id=message_id,
            content_html=content_html,
            time=time_label,
        )
        for queue in listeners:
            queue.put(fragment)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _time_label() -> str:
        return datetime.now().strftime("%I:%M %p").lstrip("0")

    def _make_entry(self, role: str, content: str) -> dict:
        return {
            "role": role,
            "content_html": render_markdown(content),
            "time": self._time_label(),
        }

    def _handle_chat_submit(self, message: str) -> str:
        entry = self._make_entry("user", message)
        with self._lock:
            self.transcript.append(entry)
        if self.stimulus_log is not None:
            self.stimulus_log.append(
                actor="user", type="chat_message", content={"message": message}
            )
        threading.Thread(target=self._run_core, args=(message,), daemon=True).start()
        return self._templates.get_template("_chat_submit_fragment.html").render(**entry)

    def _run_core(self, message: str) -> None:
        try:
            self.orient_chat_message_callback(message)
        except Exception:
            # Otherwise a failed orient() (LLM error, bad JSON, ...) leaves
            # the UI stuck mid-"Thinking…" forever with the composer
            # disabled and no way to recover short of a page reload.
            traceback.print_exc()
            self.publish_assistant_chunk(
                uuid.uuid4().hex,
                "Something went wrong while thinking that over — check the server logs.",
                done=True,
            )

    async def _event_stream(self, request: Request):
        queue: Queue = Queue()
        with self._lock:
            self._listeners.append(queue)
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
            return StreamingResponse(self._event_stream(request), media_type="text/event-stream")

        return app

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the web server. Blocks, the way ChatObserver's stdin loop
        blocks — call this from the agent's run() instead of looping
        `observe_chat_message()`-style calls."""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)


def _format_sse_event(html_fragment: str) -> str:
    lines = html_fragment.splitlines() or [""]
    payload = "\n".join(f"data: {line}" for line in lines)
    return f"event: message\n{payload}\n\n"
