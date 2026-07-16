from __future__ import annotations

import os

from openai import OpenAIError

from .model_provider import ModelProvider


class OpenRouterProvider(ModelProvider):
    """Hosted inference via OpenRouter (https://openrouter.ai).

    The API key comes from the OPENROUTER_API_KEY environment variable unless
    passed explicitly. Model names are OpenRouter slugs, e.g.
    "anthropic/claude-sonnet-4.5".
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
    ):
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenRouter API key: pass api_key or set the "
                "OPENROUTER_API_KEY environment variable."
            )
        super().__init__(base_url=base_url, model=model, api_key=api_key)

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except OpenAIError:
            return False
