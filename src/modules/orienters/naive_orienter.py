from __future__ import annotations

from src.modules.theseus_agent import MemoryModule, Observation, Orientation


class NaiveOrienter:
    def orient(self, observation: Observation, memory: list[MemoryModule]) -> Orientation:
        prefix = f"{observation.memory_context}\n" if observation.memory_context else ""
        context = f"{prefix}User: {observation.user_input}\nAgent:"
        return Orientation(observation=observation, context=context)
