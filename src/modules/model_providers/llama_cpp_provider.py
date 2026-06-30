from __future__ import annotations

from .model_provider import ModelProvider


class LlamaCppProvider(ModelProvider):
    def __init__(
        self,
        base_url: str = "http://100.126.84.49:8080/v1",
        model: str = "local-model",
        api_key: str = "local-llama",
        local: bool = False,
    ):
        super().__init__(base_url=base_url, model=model, api_key=api_key)