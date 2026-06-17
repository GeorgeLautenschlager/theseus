from __future__ import annotations

from src.modules.theseus_agent import Action, MemoryModule


class NaiveActor:
    def act(self, action: Action, memory: list[MemoryModule]) -> str:
        for m in memory:
            m.store(f"Agent: {action.response}")
        return action.response
