from __future__ import annotations

from model_provider import ModelProvider


class LlamaProvider(ModelProvider):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "gemma",
        api_key: str = "none",
    ):
        super().__init__(base_url=base_url, model=model, api_key=api_key)
