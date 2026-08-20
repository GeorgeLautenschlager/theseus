from __future__ import annotations

import os

from openai import OpenAIError

from .model_provider import ModelProvider


class UnslothProvider(ModelProvider):
    """Local inference via Unsloth Desktop.

    Speaks the same OpenAI-compatible API as LM Studio, but its server requires
    a bearer token. The OpenAI client sends `api_key` as `Authorization: Bearer
    ...`, so the token is all that differs. It comes from the UNSLOTH_API_KEY
    environment variable unless passed explicitly.

    `base_url` defaults to a local server on the loopback interface; pass it
    explicitly when Unsloth Desktop runs on another host or port.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://100.126.84.49:8888/v1",
        api_key: str | None = None,
    ):
        if api_key is None:
            api_key = os.environ.get("UNSLOTH_API_KEY")
        if not api_key:
            raise ValueError(
                "No Unsloth API key: pass api_key or set the "
                "UNSLOTH_API_KEY environment variable."
            )
        super().__init__(base_url=base_url, model=model, api_key=api_key)

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except OpenAIError:
            return False
