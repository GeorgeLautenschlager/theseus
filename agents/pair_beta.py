"""Beta of the observable Telegram pair; assemble into build/pair-beta."""
from __future__ import annotations

import os

from theseus.assembly import AgentSpec, InterfaceSpec, ModelSpec, PairingSpec, MemorySpec


SPEC = AgentSpec(
    name="Beta",
    constitution=(
        "You are Beta. Work with Alpha in the shared Telegram group. Speak clearly "
        "to Alpha and the human observer. Use the group chat for collaboration; "
        "your partner's recent activity appears in peer_stimulus_log."
    ),
    models=(ModelSpec("ollama", os.getenv("PAIR_MODEL", "gemma4:e4b"),
                      context=131072, tick=300),),
    tools=("read", "ls", "find", "grep"),
    interface=InterfaceSpec(
        "telegram", bot_token_env="TELEGRAM_BETA_TOKEN",
        allowed_chat_ids=(int(os.environ["PAIR_GROUP_ID"]),),
        allowed_user_ids=(int(os.environ["PAIR_HUMAN_ID"]),
                          int(os.environ["PAIR_ALPHA_BOT_ID"])),
        poll_timeout_seconds=2,
        outgoing_interval_seconds=5.0, outgoing_chat_ids_only=True,
    ),
    pairing=PairingSpec("../../pair-alpha/state/stimulus_log.jsonl", "Alpha"),
    memory=MemorySpec(
        "module",
        model=ModelSpec(os.getenv("PAIR_MEMORY_PROVIDER", "ollama"),
                        os.getenv("PAIR_MEMORY_MODEL", "gemma4:e4b")),
        embedding=ModelSpec(os.getenv("PAIR_EMBEDDING_PROVIDER", "ollama"),
                            os.getenv("PAIR_EMBEDDING_MODEL", "nomic-embed-text")),
        consolidate_every_seconds=300,
    ),
)
