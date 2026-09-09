"""Assemble Tam using his separately installed local integration modules.

Set TAM_HOME for identity/cadence inputs and make tam + tam_memory importable.
The generated agent snapshots identity/cadence; it never copies runtime state.
"""
from __future__ import annotations

import os
from pathlib import Path

from theseus.assembly import AgentSpec, InterfaceSpec, MemorySpec, ModelSpec
from theseus.cadence import Cadence
from theseus.tools.recall import RecallTool
from theseus.tools.registry import all_tools

if not os.environ.get("TAM_HOME"):
    raise ValueError("Set TAM_HOME to the directory containing Tam’s identity and CADENCE.md")
source_home = Path(os.environ["TAM_HOME"]).expanduser()
cadence = Cadence.parse((source_home / "CADENCE.md").read_text(encoding="utf-8"))
if cadence.lint_lines():
    raise ValueError(f"Invalid Tam cadence: {cadence.lint_lines()}")
SPEC = AgentSpec(
    name="Tam",
    constitution=(source_home / "CONSTITUTION.md").read_text(encoding="utf-8"),
    persona=(source_home / "PERSONA.md").read_text(encoding="utf-8"),
    core_factory="tam:TamCore",
    models=tuple(
        ModelSpec(
            rule.provider_key, rule.model,
            context=rule.context_tokens or 4096,
            tick=rule.tick_seconds,
            during=f"{rule.start:%H:%M}-{rule.end:%H:%M}" if rule.start is not None else None,
        )
        for rule in cadence.rules
    ),
    tools=tuple(all_tools()),
    memory=MemorySpec(
        kind="custom", factory="tam_memory:TamMemory",
        recall_description=RecallTool.description + (
            " Use a focused query about the specific agreement or event; "
            "if results are unrelated, retry with different terms."
        ),
    ),
    interface=InterfaceSpec("web", host="0.0.0.0", port=1337),
    window_size=60,
)
