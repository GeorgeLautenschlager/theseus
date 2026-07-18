from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI

from theseus.tools.tool import AssistantTurn, Tool, ToolCall, to_openai_tool


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

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        max_tokens: int = 8196,
        temperature: float = 0.7,
    ) -> AssistantTurn:
        """One turn of an OpenAI-style tool-calling exchange.

        `messages` is a raw OpenAI-format history (the caller owns it — see `ToolRunner`).
        Tools are serialized with `to_openai_tool`; the request is deliberately
        non-streamed, which both simplifies parsing and sidesteps Ollama's streamed
        multi-tool-call `index` bug. Any tool calls in the response are normalized to
        `ToolCall` (arguments JSON-decoded to a dict).
        """
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = [to_openai_tool(tool) for tool in tools]
            extra_kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **extra_kwargs,
        )
        message = response.choices[0].message
        calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            raw_args = tc.function.arguments
            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                arguments = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
        return AssistantTurn(text=message.content, tool_calls=tuple(calls))

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding
