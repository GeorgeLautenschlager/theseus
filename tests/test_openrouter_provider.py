from __future__ import annotations

import pytest
from openai import OpenAIError

from theseus.model_providers.openrouter_provider import OpenRouterProvider


@pytest.mark.parametrize("method", ["chat", "complete_with_tools"])
def test_reasoning_effort_is_sent_on_chat_requests(monkeypatch, method):
    from types import SimpleNamespace
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "low")
    provider = OpenRouterProvider(model="z-ai/glm-5.3-flash")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello", tool_calls=[]))],
            usage=None,
        )

    monkeypatch.setattr(provider._client.chat.completions, "create", create)
    if method == "chat":
        provider.chat("hello")
    else:
        provider.complete_with_tools([{"role": "user", "content": "hello"}])
    assert calls[0]["extra_body"] == {"reasoning": {"effort": "low"}}


def test_reasoning_defaults_unchanged_and_embeddings_unaffected(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_REASONING_EFFORT", raising=False)
    provider = OpenRouterProvider(model="qwen/qwen3-embedding-8b")
    assert provider._chat_request_options() == {}
    provider.reasoning_effort = "low"
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])], usage=None)

    monkeypatch.setattr(provider._client.embeddings, "create", create)
    assert provider.embed("hello") == [1.0, 0.0]
    assert "extra_body" not in calls[0]


def test_invalid_reasoning_effort_rejected(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "typo")
    with pytest.raises(ValueError, match="REASONING_EFFORT"):
        OpenRouterProvider(model="z-ai/glm-5.3-flash")


def test_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    provider = OpenRouterProvider(model="anthropic/claude-sonnet-4.5")
    assert provider._client.api_key == "sk-or-test-key"


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
    provider = OpenRouterProvider(model="anthropic/claude-sonnet-4.5", api_key="sk-or-explicit")
    assert provider._client.api_key == "sk-or-explicit"


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(model="anthropic/claude-sonnet-4.5")


def test_default_base_url_is_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    provider = OpenRouterProvider(model="anthropic/claude-sonnet-4.5")
    assert str(provider._client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_is_available_false_when_unreachable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    provider = OpenRouterProvider(model="anthropic/claude-sonnet-4.5")

    class FailingModels:
        def list(self):
            raise OpenAIError("unreachable")

    monkeypatch.setattr(provider._client, "models", FailingModels())
    assert provider.is_available() is False


def test_is_available_true_when_reachable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    provider = OpenRouterProvider(model="anthropic/claude-sonnet-4.5")

    class WorkingModels:
        def list(self):
            return []

    monkeypatch.setattr(provider._client, "models", WorkingModels())
    assert provider.is_available() is True
