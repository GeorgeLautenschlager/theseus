"""A small driven agent; replace the model in tests with an offline fake."""
from __future__ import annotations

from theseus.assembly import AgentSpec, ModelSpec

SPEC = AgentSpec(
    name="Test Agent",
    constitution="You are a concise, helpful test agent.",
    core="ooda",
    models=(ModelSpec("ollama", "gemma4:e4b", context=131072),),
    window_size=30,
)
