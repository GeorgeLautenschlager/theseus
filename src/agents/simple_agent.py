from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.modules.memory.naive_context_assembler import NaiveContextAssembler
from src.modules.user_interfaces.textual_ui import AgentApp, TextualUI
from src.modules.model_providers.llama_provider import LlamaProvider
from src.modules.theseus_agent import (
    Action,
    MemoryModule,
    Observation,
    Orientation,
    TheseusAgent,
)


class NaiveObserver:
    def observe(self, user_input: str, memory: list[MemoryModule], cycle: int) -> Observation:
        parts = [m.retrieve(user_input) for m in memory]
        context = "\n".join(p for p in parts if p)
        for m in memory:
            m.store(f"User: {user_input}")
        return Observation(user_input=user_input, memory_context=context, cycle=cycle)


class NaiveOrienter:
    def orient(self, observation: Observation, memory: list[MemoryModule]) -> Orientation:
        prefix = f"{observation.memory_context}\n" if observation.memory_context else ""
        context = f"{prefix}User: {observation.user_input}\nAgent:"
        return Orientation(observation=observation, context=context)


class LLMDecider:
    def __init__(self, provider: LlamaProvider) -> None:
        self.provider = provider

    def decide(self, orientation: Orientation, memory: list[MemoryModule]) -> Action:
        response = self.provider.chat(orientation.context)
        return Action(emit=True, response=response)


class NaiveActor:
    def act(self, action: Action, memory: list[MemoryModule]) -> str:
        for m in memory:
            m.store(f"Agent: {action.response}")
        return action.response


_memory = NaiveContextAssembler()
_provider = LlamaProvider()
_ui = TextualUI()

simple_agent = TheseusAgent(
    observer=NaiveObserver(),
    orienter=NaiveOrienter(),
    decider=LLMDecider(_provider),
    actor=NaiveActor(),
    memory=[_memory],
    ui=_ui,
)


if __name__ == "__main__":
    AgentApp(simple_agent, _ui).run()
