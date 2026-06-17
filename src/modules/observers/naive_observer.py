from __future__ import annotations

from src.modules.theseus_agent import MemoryModule, Observation


class NaiveObserver:
    def observe(self, user_input: str, memory: list[MemoryModule], cycle: int) -> Observation:
        parts = [m.retrieve(user_input) for m in memory]
        context = "\n".join(p for p in parts if p)
        for m in memory:
            m.store(f"User: {user_input}")
        return Observation(user_input=user_input, memory_context=context, cycle=cycle)
