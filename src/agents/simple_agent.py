from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog

from src.modules.memory.naive_context_assembler import NaiveContextAssembler
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


class TextualUI:
    def __init__(self) -> None:
        self._log: RichLog | None = None

    def set_log(self, log: RichLog) -> None:
        self._log = log

    def render(self, content: str) -> None:
        if self._log is not None:
            self._log.write(f"[bold]agent:[/bold] {content}")


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


class AgentApp(App):
    def compose(self) -> ComposeResult:
        yield RichLog(highlight=True, markup=True)
        yield Input(placeholder="words go here")

    def on_mount(self) -> None:
        _ui.set_log(self.query_one(RichLog))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        log = self.query_one(RichLog)
        log.write(f"[bold]you:[/bold] {event.value}")
        simple_agent.process(event.value)
        event.input.clear()


if __name__ == "__main__":
    AgentApp().run()
