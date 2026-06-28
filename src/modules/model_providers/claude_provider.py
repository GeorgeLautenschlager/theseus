from __future__ import annotations

import subprocess

from .model_provider import ModelProvider


class ClaudeProvider(ModelProvider):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 8196,
        temperature: float = 0.7,
    ) -> str:
        cmd = ["claude", "-p", prompt, "--model", self.model, "--no-memory"]
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
