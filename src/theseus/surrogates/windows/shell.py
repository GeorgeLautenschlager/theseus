"""The Windows app shell: the local UI in a native pywebview window plus a pystray tray.

Phase 1's shell (issue #104) is what makes the surrogate feel like a real Windows app
rather than a browser tab: a native window over the local web UI and a tray icon with
show/hide/quit. `run_shell` blocks — pywebview's event loop must own the calling
thread — so the entry point (#105) calls it on the MAIN thread and treats its return
as app exit.

Like `toast.py`, both Windows-only dependencies (pywebview, pystray) are imported
lazily inside `run_shell`, so importing this module requires nothing beyond the
standard library and unit tests run on Linux with neither dep installed.

The window's focus/blur events (where pywebview exposes them) are fed to the
optional `on_focus_change` callback, so an unfocused `command.say` also toasts.
The subscription is best-effort: pywebview's focus events are version/OS-dependent,
so if they are unavailable the wiring simply does not subscribe and focus stays at
the `FocusState` default (focused), a safe fallback.
`command.notify` toasts unconditionally, so notifications work regardless.
"""

from __future__ import annotations

from collections.abc import Callable


def run_shell(
    url: str,
    *,
    on_quit: Callable[[], None] = lambda: None,
    title: str = "Theseus",
    on_focus_change: Callable[[bool], None] | None = None,
) -> None:
    """Open `url` in a pywebview window with a tray icon; block until the app closes.

    Called on the MAIN thread by the entry point. Returns when the window is closed
    (or Quit is chosen from the tray, which also stops pywebview).
    """
    import io
    import threading

    import pystray
    import webview

    def _focus(focused: bool) -> None:
        if on_focus_change is not None:
            on_focus_change(focused)

    def _quit(icon, item) -> None:
        if webview.windows:
            webview.windows[0].destroy()
        on_quit()

    def _toggle(icon, item) -> None:
        if webview.windows:
            window = webview.windows[0]
            if window.hidden:
                window.show()
            else:
                window.hide()

    def _run_tray() -> None:
        # PIL is pystray's default backend requirement; imported lazily with it
        from PIL import Image

        # 1x1 transparent placeholder; a real icon asset is a follow-up polish
        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c626001000000ffff03000006000557bfabd40000000049454e44ae426082"
        )
        menu = pystray.Menu(
            pystray.MenuItem("Show/Hide", _toggle),
            pystray.MenuItem("Quit", _quit),
        )
        icon = pystray.Icon("theseus", Image.open(io.BytesIO(png)), title, menu)
        icon.run()

    window = webview.create_window(title, url, width=900, height=700, on_top=False)
    # Best-effort focus wiring: pywebview's focus/blur events are version/OS-dependent,
    # so only subscribe when they exist; otherwise focus stays at the FocusState default.
    if on_focus_change is not None and getattr(window.events, "focused", None) is not None:
        window.events.focused += lambda: _focus(True)
        if getattr(window.events, "blurred", None) is not None:
            window.events.blurred += lambda: _focus(False)
    threading.Thread(target=_run_tray, name="surrogate-tray", daemon=True).start()
    webview.start()
