from __future__ import annotations

import json
import shutil
import subprocess

from .model_provider import ModelProvider


class ClaudeProvider(ModelProvider):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 8196,
        temperature: float = 0.7,
        json_schema: dict | None = None,
    ) -> str:
        # --safe-mode and --tools "" keep this call a plain completion: without
        # them the CLI loads this project's CLAUDE.md/skills and the model
        # responds as an interactive Claude Code session (e.g. offering to
        # invoke skills) instead of just answering the prompt.
        cmd = ["claude", "-p", prompt, "--model", self.model, "--safe-mode", "--tools", ""]
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]
        if json_schema is not None:
            cmd += ["--json-schema", json.dumps(json_schema)]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
