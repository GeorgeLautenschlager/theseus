from __future__ import annotations

from openai import OpenAIError

from .model_provider import ModelProvider


class OllamaProvider(ModelProvider):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "gemma",
        api_key: str = "none",
    ):
        super().__init__(base_url=base_url, model=model, api_key=api_key)

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except OpenAIError:
            return False
