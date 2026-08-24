from __future__ import annotations

from theseus.model_providers.claude_provider import ClaudeProvider
from theseus.model_providers.llama_cpp_provider import LlamaCppProvider
from theseus.model_providers.lm_studio_provider import LmStudioProvider
from theseus.model_providers.model_provider import ModelProvider
from theseus.model_providers.ollama_provider import OllamaProvider
from theseus.model_providers.openrouter_provider import OpenRouterProvider
from theseus.model_providers.unsloth_provider import UnslothProvider

# Short names used by CADENCE.md rules (see theseus.cadence). Every class here is
# constructible as cls(model=<model>); endpoints and API keys come from each
# provider's own defaults and environment variables.
PROVIDER_REGISTRY: dict[str, type[ModelProvider]] = {
    "claude": ClaudeProvider,
    "llama_cpp": LlamaCppProvider,
    "lm_studio": LmStudioProvider,
    "ollama": OllamaProvider,
    "openrouter": OpenRouterProvider,
    "unsloth": UnslothProvider,
}

__all__ = [
    "ClaudeProvider",
    "LlamaCppProvider",
    "LmStudioProvider",
    "ModelProvider",
    "OllamaProvider",
    "OpenRouterProvider",
    "PROVIDER_REGISTRY",
    "UnslothProvider",
]
