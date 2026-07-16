from __future__ import annotations

import pytest
from openai import OpenAIError

from theseus.model_providers.openrouter_provider import OpenRouterProvider


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
