from __future__ import annotations

import os

from openai import OpenAI, OpenAIError

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
        reasoning_effort: str | None = None,
    ):
        if reasoning_effort is None:
            reasoning_effort = os.environ.get("OPENROUTER_REASONING_EFFORT") or None
        if reasoning_effort is not None and reasoning_effort not in (
            "none", "minimal", "low", "medium", "high", "xhigh", "max",
        ):
            raise ValueError("Invalid OPENROUTER_REASONING_EFFORT")
        self.reasoning_effort = reasoning_effort
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenRouter API key: pass api_key or set the "
                "OPENROUTER_API_KEY environment variable."
            )
        super().__init__(base_url=base_url, model=model, api_key=api_key)
        # ponytail: flash models answer in ~30s; fail fast on a hung backend instance
        # and retry (the 600s default timeout once sank a whole eval run).
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0, max_retries=5)

    def _chat_request_options(self) -> dict:
        # Omission preserves the model/provider default. Supported efforts are
        # model-specific and must be selected from OpenRouter's model catalog.
        if self.reasoning_effort is None:
            return {}
        return {"extra_body": {"reasoning": {"effort": self.reasoning_effort}}}

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except OpenAIError:
            return False
