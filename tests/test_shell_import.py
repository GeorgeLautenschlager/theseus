from __future__ import annotations

import inspect


def test_shell_imports_without_optional_deps() -> None:
    from theseus.surrogates.windows import shell

    assert callable(shell.run_shell)


def test_run_shell_signature() -> None:
    from theseus.surrogates.windows.shell import run_shell

    params = inspect.signature(run_shell).parameters
    assert "url" in params
    assert "on_quit" in params
    assert "title" in params
    assert "on_focus_change" in params
