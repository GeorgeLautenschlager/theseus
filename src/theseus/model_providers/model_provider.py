from __future__ import annotations

from abc import ABC, abstractmethod
from openai import OpenAI


class ModelProvider(ABC):
    """Base class for LLM providers using OpenAI-compatible APIs."""

    # Class-level default so subclasses that skip __init__ (e.g. ClaudeProvider)
    # still fail cleanly in embed() rather than with AttributeError.
    embedding_model: str | None = None

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        embedding_model: str | None = None,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if this provider is currently reachable."""
        ...

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

    def embed(self, text: str) -> list[float]:
        if self.embedding_model is None:
            raise RuntimeError(
                f"{type(self).__name__} has no embedding_model configured; "
                "pass embedding_model=... to use embeddings."
            )
        response = self._client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding
