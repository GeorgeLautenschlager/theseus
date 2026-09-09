"""Ordinary Python reuse: vary the minimal definition without copying its settings."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy

from theseus.assembly import InterfaceSpec

base = runpy.run_path(str(Path(__file__).with_name("minimal.py")))["SPEC"]
SPEC = replace(
    base,
    name="Research Experiment",
    persona="Ask precise questions and distinguish observations from guesses.",
    tools=("read", "ls", "find", "grep"),
    interface=InterfaceSpec("web", port=8001),
    window_size=60,
)
