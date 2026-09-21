"""Managed Tam deployment: existing identity, OpenRouter, and Telegram.

This is the supported Autocore/Telegram subset of deployment_manifest_wip.toml.
It reads identity and configuration only; stopped state is migrated separately.
"""
from pathlib import Path
import os
import tomllib

from theseus.assembly import AgentSpec, InterfaceSpec, MemorySpec, ModelSpec
from theseus.deployment import DeploymentSpec, ResourceSpec
from theseus.tools.registry import all_tools

manifest_path = Path(os.environ.get(
    "TAM_MANIFEST", str(Path(__file__).resolve().parents[1] / "deployment_manifest_wip.toml")
))
manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
source_home = Path(os.environ.get("TAM_HOME", "/home/aldric/tam")).expanduser()


def allowed_ids(variable: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in os.environ.get(variable, "").split(",") if value.strip())


telegram = next(item for item in manifest["assembly"]["interfaces"]
                if item["implementation"] == "TelegramObserver")["config"]
cognition = manifest["models"]["cognition"]
embeddings = manifest["models"]["embeddings"]
DEPLOYMENT = DeploymentSpec(
    id="tam",
    agents={
        "tam": AgentSpec(
            name=manifest["agent"]["name"],
            constitution=(source_home / "CONSTITUTION.md").read_text(encoding="utf-8"),
            persona=(source_home / "PERSONA.md").read_text(encoding="utf-8"),
            core="auto",
            models=(ModelSpec(cognition["provider"], cognition["model"], context=131072, tick=3600),),
            tools=tuple(all_tools()),
            memory=MemorySpec(
                "module",
                model=ModelSpec(cognition["provider"], cognition["model"]),
                embedding=ModelSpec(embeddings["provider"], embeddings["model"]),
                consolidate_every_seconds=900,
                recall_description="Recall durable knowledge and memory. Use a focused query about the specific agreement or event; if results are unrelated, retry with different terms.",
            ),
            interface=InterfaceSpec(
                "telegram",
                bot_token_env=telegram["bot_token_env"],
                allowed_user_ids=allowed_ids(telegram["allowed_user_ids_env"]),
                allowed_chat_ids=allowed_ids(telegram["allowed_chat_ids_env"]),
            ),
            window_size=60,
        ),
    },
    resources={"tam": ResourceSpec(cpus=1.0, memory_mb=1024)},
    secrets=("OPENROUTER_API_KEY", telegram["bot_token_env"], "OPENROUTER_REASONING_EFFORT"),
)
