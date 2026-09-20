"""`import theseus` (and the windows-surrogate entry point) must not require `fcntl`.

`fcntl` is a Unix-only stdlib module; on Windows importing it raises
`ModuleNotFoundError`. The deployment/ops code uses it only for advisory file
locking, but `theseus/__init__` eagerly imports those modules, so a stray
module-level `import fcntl` breaks EVERY `from theseus... import` on Windows —
including `windows-surrogate`, whose whole job is to run there.

Each case runs a FRESH interpreter subprocess with `fcntl` blocked from the
start (a fresh process is the only faithful way to reproduce Windows on Linux —
`fcntl` and C-extensions like numpy are already loaded in the test process and
cannot be unloaded/reloaded). The import must succeed; the lock functions that
need `fcntl` import it lazily when actually called.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_MODULES = [
    "theseus",
    "theseus.deployment_store",
    "theseus.deployment_control",
    "theseus.surrogates.windows_surrogate",
]


@pytest.mark.parametrize("module", _MODULES)
def test_imports_without_fcntl(module: str) -> None:
    code = textwrap.dedent(
        f"""
        import sys

        class _NoFcntlFinder:
            # Reproduce Windows: `import fcntl` raises ModuleNotFoundError.
            def find_spec(self, name, path=None, target=None):
                if name == "fcntl":
                    raise ModuleNotFoundError("No module named 'fcntl'")
                return None

        sys.meta_path.insert(0, _NoFcntlFinder())
        import importlib
        importlib.import_module({module!r})
        print("IMPORT_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"{module} failed to import with fcntl blocked:\n{result.stderr}"
    )
    assert "IMPORT_OK" in result.stdout
