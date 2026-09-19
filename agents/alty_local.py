"""Alty's fresh, headless container test bed; inference stays on local Ollama."""
from theseus.agents.alty_mcgee import ALTY_CONSTITUTION, PERSONA
from theseus.assembly import AgentSpec, InterfaceSpec, MemorySpec, ModelSpec
from theseus.deployment import DeploymentSpec, ResourceSpec

DEPLOYMENT = DeploymentSpec(
    id="alty-local",
    agents={
        "alty": AgentSpec(
            name="Alty McGee",
            constitution=ALTY_CONSTITUTION + """
Your current assignment is a small container deployment smoke test.
Use read and write to maintain a short DEPLOYMENT_CHECK.md in your home:
identify yourself, record that this is a local Ollama container trial, and
describe which previous test artifacts you can read. On later turns, preserve
the existing record and note that your state survived. Keep responses short.
Work only in your own home. Once the check is complete, remain idle until the
next autonomous turn. Do not invent additional projects or change CADENCE.md.
""",
            persona=PERSONA,
            core="auto",
            models=(ModelSpec("ollama", "gemma4:e4b", context=8192, tick=900),),
            tools=("read", "write", "ls"),
            memory=MemorySpec("module"),
            interface=InterfaceSpec("none"),
            window_size=30,
        ),
    },
    resources={"alty": ResourceSpec(cpus=1.0, memory_mb=768)},
)
