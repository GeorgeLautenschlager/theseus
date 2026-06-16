from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Observation:
    user_input: str
    memory_context: str
    cycle: int


@dataclass
class Orientation:
    observation: Observation
    context: str


@dataclass
class Action:
    emit: bool
    response: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    thought: str | None = None


class MaxCyclesExceeded(Exception):
    def __init__(self, cycles: int, last_action: Action):
        self.cycles = cycles
        self.last_action = last_action
        super().__init__(f"Agent exceeded {cycles} OODA cycles without emitting")


class MemoryModule(Protocol):
    def retrieve(self, query: str) -> str: ...
    def store(self, content: str) -> None: ...


class UI(Protocol):
    def render(self, content: str) -> None: ...


class Observer(Protocol):
    def observe(self, user_input: str, memory: list[MemoryModule], cycle: int) -> Observation: ...


class Orienter(Protocol):
    def orient(self, observation: Observation, memory: list[MemoryModule]) -> Orientation: ...


class Decider(Protocol):
    def decide(self, orientation: Orientation, memory: list[MemoryModule]) -> Action: ...


class Actor(Protocol):
    def act(self, action: Action, memory: list[MemoryModule]) -> str: ...
