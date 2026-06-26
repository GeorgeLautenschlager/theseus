from __future__ import annotations

from model_provider import ModelProvider


class LmStudioProvider(ModelProvider):
    def __init__(
        self,
        base_url: str = "http://100.126.84.49:1234/v1",
        model: str = "local-model",
        api_key: str = "lm-studio",
    ):
        super().__init__(base_url=base_url, model=model, api_key=api_key)