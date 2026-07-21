from __future__ import annotations

from datetime import datetime
from typing import List

from theseus.cognitive_prompts import (
    build_decide_system_prompt,
    build_decide_user_prompt,
)
from theseus.context_assembler import ContextAssembler
from theseus.memory import Memory
from theseus.stimulus_log import StimulusLog
from theseus.tools.tool import Tool

from .model_providers.model_provider import ModelProvider


class CognitiveCore:
    """A truncated OODA loop that serves as the primary cognitive process for a Theseus agent.

    The core does Orient, Decide and Act, steps and internalizes the LLM. Steps never pass
    data to each other directly — everything a loop accumulates lives in `loop_memory`,
    because any step may kick back to Orient and re-enter the cycle.

    Args:
        model_providers: One or more model provider. These are listed in priority order, with the cognitive core ultimately
        deciding which model provider is used.
        tools: List of available Tool objects.
        memory: Optional memory module, seen only through the Memory protocol: Orient pulls
        `retrieve(...)` results into context (via the ContextAssembler), and loop termination
        signals `form()`. What and how the module consolidates is its own business.
        name: The core's own actor name, used when logging its decisions and actions.
    """
    def __init__(
        self,
        constitution: str,
        model_providers: List[ModelProvider],
        tools: dict[str, Tool],
        stimulus_log: StimulusLog,
        memory: Memory | None = None,
        name: str = "Tam",
    ):
        self.constitution = constitution
        self.model_providers = model_providers
        self.loop_memory = {}
        self.tools = tools
        self.stimulus_log = stimulus_log
        self.memory = memory
        self.context_assembler = ContextAssembler(stimulus_log=stimulus_log, memory=memory)
        self.name = name

    def _select_model_provider(self) -> ModelProvider:
        """Selects the first available provider, in priority order."""
        for provider in self.model_providers:
            if provider.is_available():
                return provider
        raise RuntimeError("No model providers are currently available.")

    def orient(self):
        """Callback to be invoked by chat UI"""
        assembled = self.context_assembler.assemble_context()
        self.loop_memory["recent_events"] = assembled.recent_events
        self.loop_memory["memories"] = assembled.memories

        self.decide()

    def decide(self):
        model_provider = self._select_model_provider()

        system_prompt = build_decide_system_prompt(self.constitution, self.tools.values())
        prompt = build_decide_user_prompt(
            self.loop_memory["recent_events"],
            str(datetime.now()),
            memories=self.loop_memory["memories"],
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # One native tool-calling turn: the model's chosen tool(s) and their arguments
        # come back together, so there is no separate Act model call.
        turn = model_provider.complete_with_tools(messages, list(self.tools.values()))
        self.loop_memory["decision"] = turn

        self.stimulus_log.append(
            actor=self.name,
            type="decision",
            content={
                "text": turn.text,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in turn.tool_calls
                ],
            },
        )

        self.act()

    def act(self):
        turn = self.loop_memory["decision"]

        # No tool call is the native-tools equivalent of "wait": the model chose to
        # act on nothing this cycle.
        if not turn.tool_calls:
            self.loop_termination()
            return

        for call in turn.tool_calls:
            tool = self.tools.get(call.name)
            if tool is None:
                print(f"Unknown tool: {call.name}. No tool available to handle this call.")
                continue

            result = tool.execute(**call.arguments)
            self.stimulus_log.append(
                actor=self.name,
                type="action",
                content={
                    "action": call.name,
                    "arguments": call.arguments,
                    "output": result.content,
                    "is_error": result.is_error,
                },
            )

        # Nothing can request another cycle yet; when tools gain that power, this
        # branch re-enters Orient and loop_termination is skipped.
        loop_again = False
        if loop_again:
            self.orient()
        else:
            self.loop_termination()

    def loop_termination(self):
        """Runs once per cognitive loop, only when Act is actually terminating it —
        skipped whenever a step kicks off another cycle. For now the memory-formation
        signal is hard coded; later this should accumulate callbacks and iterate over
        them."""
        if self.memory is not None:
            self.memory.form()
        self.loop_memory = {}
