from __future__ import annotations

from typing import Protocol


class Memory(Protocol):
    """A pluggable long-term memory module for a OODACore.

    The core only ever signals a memory module; everything else — what to
    consolidate, how to store it, when it was last formed — is the module's
    own business and must not leak into the core.
    """

    def form(self) -> None:
        """Consolidate recent experience into memory. Called by the core when a
        cognitive loop terminates; the module decides what (if anything) is new
        enough to form. Must never raise into the loop."""
        ...

    def retrieve(self, query: str) -> str:
        """Return a rendered block of memories relevant to `query`, ready to be
        placed in a prompt. Empty string when nothing relevant is found."""
        ...
