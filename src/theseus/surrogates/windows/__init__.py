"""Windows-specific surrogate organs (real `Notifier`/OS surfaces, issues #94/#103).

Everything here is import-safe on any OS: the Windows-only dependencies
(`windows-toasts`, pywebview, pystray) are imported lazily inside the code
paths that need them, so these modules can be imported and unit-tested on
Linux with none of the optional deps installed.
"""

from theseus.surrogates.windows.toast import ToastNotifier

__all__ = ["ToastNotifier"]
