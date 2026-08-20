from __future__ import annotations

import pytest
from openai import OpenAIError

from theseus.model_providers.unsloth_provider import UnslothProvider


def test_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("UNSLOTH_API_KEY", "unsloth-test-key")
    provider = UnslothProvider(model="gemma-4-e4b-it-qat-nvfp4")
    assert provider._client.api_key == "unsloth-test-key"


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("UNSLOTH_API_KEY", "unsloth-env-key")
    provider = UnslothProvider(model="gemma-4-e4b-it-qat-nvfp4", api_key="unsloth-explicit")
    assert provider._client.api_key == "unsloth-explicit"


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("UNSLOTH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="UNSLOTH_API_KEY"):
        UnslothProvider(model="gemma-4-e4b-it-qat-nvfp4")


def test_sends_bearer_token_on_requests(monkeypatch):
    monkeypatch.setenv("UNSLOTH_API_KEY", "unsloth-test-key")
    provider = UnslothProvider(model="gemma-4-e4b-it-qat-nvfp4")
    assert provider._client.auth_headers == {"Authorization": "Bearer unsloth-test-key"}


def test_base_url_is_overridable(monkeypatch):
    monkeypatch.setenv("UNSLOTH_API_KEY", "unsloth-test-key")
    provider = UnslothProvider(
        model="gemma-4-e4b-it-qat-nvfp4", base_url="http://100.126.84.49:2345/v1"
    )
    assert str(provider._client.base_url).rstrip("/") == "http://100.126.84.49:2345/v1"


def test_is_available_false_when_unreachable(monkeypatch):
    monkeypatch.setenv("UNSLOTH_API_KEY", "unsloth-test-key")
    provider = UnslothProvider(model="gemma-4-e4b-it-qat-nvfp4")

    class FailingModels:
        def list(self):
            raise OpenAIError("unreachable")

    monkeypatch.setattr(provider._client, "models", FailingModels())
    assert provider.is_available() is False


def test_is_available_true_when_reachable(monkeypatch):
    monkeypatch.setenv("UNSLOTH_API_KEY", "unsloth-test-key")
    provider = UnslothProvider(model="gemma-4-e4b-it-qat-nvfp4")

    class WorkingModels:
        def list(self):
            return []

    monkeypatch.setattr(provider._client, "models", WorkingModels())
    assert provider.is_available() is True
