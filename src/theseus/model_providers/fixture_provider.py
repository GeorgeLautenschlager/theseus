"""Deterministic, network-free provider for deployment acceptance tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from theseus.model_providers.model_provider import ModelProvider
from theseus.tools.tool import AssistantTurn, Tool


class FixtureProvider(ModelProvider):
    """A quiet provider enabled only by an explicit test-only environment flag.

    Managed deployment acceptance tests need a real agent process without paid or
    host-local inference. Calls can be recorded on the agent's private state mount
    so the harness can prove that preflight made no inference request.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._enabled = os.environ.get("THESEUS_ENABLE_FIXTURE_PROVIDER") == "1"
        value = os.environ.get("THESEUS_FIXTURE_PROVIDER_LOG")
        self._log_path = Path(value) if value else None

    def is_available(self) -> bool:
        return self._enabled

    def _record(self, method: str) -> None:
        if not self._enabled:
            raise RuntimeError("fixture provider requires THESEUS_ENABLE_FIXTURE_PROVIDER=1")
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"method": method, "model": self.model}, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        max_tokens: int = 8196,
        temperature: float = 0.7,
    ) -> AssistantTurn:
        self._record("complete_with_tools")
        return AssistantTurn()

    def chat(self, prompt: str, **kwargs: Any) -> str:
        self._record("chat")
        return json.dumps({"summary": "fixture", "assertions": []})

    def embed(self, text: str) -> list[float]:
        self._record("embed")
        return [1.0, 0.0]
