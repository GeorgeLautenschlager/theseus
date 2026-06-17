from __future__ import annotations

from src.modules.model_providers.llama_provider import LlamaProvider
from src.modules.theseus_agent import Action, MemoryModule, Orientation


class LLMDecider:
    def __init__(self, provider: LlamaProvider) -> None:
        self.provider = provider

    def decide(self, orientation: Orientation, memory: list[MemoryModule]) -> Action:
        response = self.provider.chat(orientation.context)
        return Action(emit=True, response=response)
