from __future__ import annotations

from abc import ABC, abstractmethod
from openai import OpenAI


class ModelProvider(ABC):
    """Base class for LLM providers using OpenAI-compatible APIs.

    A provider class is a *place we get models* (LM Studio, Ollama, ...); each
    instance is exactly one model. An embedding provider is therefore just
    another instance whose model is an embedding model, e.g.
    OllamaProvider(model="nomic-embed-text").
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
    ):
        self.model = model
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
        images: list[str] | None = None,
    ) -> str:
        """`images` are data URIs (e.g. "data:image/jpeg;base64,...") attached
        to the user message; the model must support vision."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if images:
            content = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": url}} for url in images
            )
            messages.append({"role": "user", "content": content})
        else:
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
        response = self._client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding
