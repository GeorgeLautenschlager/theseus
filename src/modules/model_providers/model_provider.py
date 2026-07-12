from __future__ import annotations

from abc import ABC, abstractmethod
from openai import OpenAI


class ModelProvider(ABC):
    """Base class for LLM providers using OpenAI-compatible APIs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
    ):
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 8196,
        temperature: float = 0.7,
        json_schema: dict | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        extra_kwargs = {}
        if json_schema is not None:
            extra_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **extra_kwargs,
        )
        return response.choices[0].message.content
