"""The runnable Windows surrogate: assemble the full runtime, serve the UI, open the shell.

The entry point (issue #105) for Phase 1's surrogate↔host loop. `build_windows_surrogate`
wires everything offline-testably — local log, replication transport, command channel,
web UI, presence renderer — without serving or opening a window; `main` is the thin live
entry that adds the HTTP server and (off-headless) the pywebview/tray shell. Same shape
as `theseus.agents.surrogate_host`.
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path

from theseus.stimulus_log import StimulusLog
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.http_transport import HttpTransport
from theseus.surrogates.presence import ConsoleNotifier, WindowsPresence
from theseus.surrogates.runtime import SurrogateRuntime
from theseus.surrogates.sse_command_channel import SseCommandChannel
from theseus.surrogates.web_ui import SurrogateWebUI
from theseus.surrogates.windows.toast import ToastNotifier


@dataclass(frozen=True)
class WindowsSurrogateApp:
    """Everything the surrogate assembled, so `main` and tests can reach the parts."""

    runtime: SurrogateRuntime
    web_ui: SurrogateWebUI
    notifier: ConsoleNotifier | ToastNotifier
    origin: str
    headless: bool


def build_windows_surrogate(
    host_url: str,
    *,
    origin: str = "windows-desktop",
    data_dir: Path,
    headless: bool = False,
) -> WindowsSurrogateApp:
    """Wire the surrogate's full runtime. No server, no window — offline-testable.

    The wiring cycle (renderer needs the web UI, runtime needs the renderer, web UI
    needs the runtime's submit) is broken by a closure that reads `runtime` at call
    time — always after it is assigned, since a chat POST cannot arrive before this
    function has returned.
    """
    log = StimulusLog(data_dir / "stimulus_log.jsonl", origin=origin)
    upstream_cursor = AckedCursor(data_dir / "upstream_cursor.json", origin)
    command_cursor = AckedCursor(data_dir / "command_cursor.json", origin)
    base_url = host_url.rstrip("/")
    transport = HttpTransport(base_url + "/replicate")
    command_channel = SseCommandChannel(base_url + "/commands/" + origin, command_cursor)
    notifier = ConsoleNotifier() if headless else ToastNotifier()

    runtime: SurrogateRuntime  # assigned in step 4; closure reads it at call time
    web_ui = SurrogateWebUI(
        submit_user_message=lambda text: runtime.submit_user_message(text),
        stimulus_log=log,
    )
    renderer = WindowsPresence(web_ui, notifier)
    runtime = SurrogateRuntime(
        log,
        transport,
        upstream_cursor,
        command_channel=command_channel,
        renderer=renderer,
        command_cursor=command_cursor,
    )
    return WindowsSurrogateApp(
        runtime=runtime,
        web_ui=web_ui,
        notifier=notifier,
        origin=origin,
        headless=headless,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Windows surrogate")
    parser.add_argument("--host-url", required=True, help="Surrogate host base URL")
    parser.add_argument("--origin", default="windows-desktop", help="Command target")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8765)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    app = build_windows_surrogate(
        args.host_url, origin=args.origin, data_dir=args.data_dir, headless=args.headless
    )
    app.runtime.start()
    url = f"http://{args.ui_host}:{args.ui_port}"

    if args.headless:
        # ponytail: serve on the main thread headlessly; shell mode needs it free for pywebview
        print(f"Surrogate UI: {url}")
        # Serve-and-return: this blocks until the process is killed, and every runtime
        # thread is a daemon, so there is no shutdown path here to call runtime.stop() on
        # (mirrors surrogate_host.main). Graceful headless shutdown is a follow-up.
        app.web_ui.serve(args.ui_host, args.ui_port)
        return

    from theseus.surrogates.windows.shell import run_shell

    ui_thread = threading.Thread(
        target=app.web_ui.serve,
        args=(args.ui_host, args.ui_port),
        name="surrogate-ui",
        daemon=True,
    )
    ui_thread.start()
    run_shell(url, on_quit=app.runtime.stop, title="Theseus")  # blocks on the main thread
    # Belt-and-suspenders: tray Quit already fired on_quit=runtime.stop on the tray thread,
    # but a window close (no Quit) does not — so stop again here. runtime.stop() is
    # idempotent and safe to call twice/concurrently (guarded, None-out, idempotent close).
    app.runtime.stop()


if __name__ == "__main__":
    main()
