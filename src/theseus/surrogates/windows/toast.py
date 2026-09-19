"""The real Windows toast notifier: `ToastNotifier` backed by `windows-toasts`.

`WindowsPresence` (issue #96) needs a native notification when the chat window
is unfocused or a `command.notify` arrives. This module is that `Notifier`
implementation for Windows, using the `windows-toasts` library (WinRT). The
dependency is optional (the `windows` dependency group, #105) and therefore
imported lazily inside `_show_windows_toast` — importing this module on Linux
requires nothing beyond the standard library, and failures to show a toast are
logged and swallowed so a notification can never take down the command loop.
"""

from __future__ import annotations

import logging

from theseus.surrogates.presence import Notifier

logger = logging.getLogger(__name__)


def _show_windows_toast(app_name: str, title: str, body: str) -> None:
    """Raise one toast via `windows-toasts`. Imports the dep lazily; Windows-only."""
    # ponytail: exact windows-toasts calls verified only in the #105 live check
    import windows_toasts

    toaster = windows_toasts.WindowsToaster(app_name)
    toast = windows_toasts.Toast(text=(title, body))
    toaster.show_toast(toast)


class ToastNotifier:
    """Raises native Windows toasts — the real `Notifier` behind `WindowsPresence`.

    A toast failure is never worth an exception: `notify` logs a warning and
    returns, so the command executor's render path keeps running.
    """

    def __init__(self, app_name: str = "Theseus") -> None:
        self._app_name = app_name

    def notify(self, title: str, body: str) -> None:
        try:
            _show_windows_toast(self._app_name, title, body)
        except Exception:  # noqa: BLE001 — a failed toast must never crash the loop
            logger.warning("Failed to show Windows toast for %r", title, exc_info=True)
