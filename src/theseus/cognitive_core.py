from __future__ import annotations

from datetime import datetime
from typing import List

from theseus.cognitive_prompts import (
    WAIT_ACTION,
    build_act_system_prompt,
    build_act_user_prompt,
    build_decide_system_prompt,
    build_decide_user_prompt,
    decision_json_schema,
)
from theseus.context_assembler import ContextAssembler
from theseus.effector import Effector
from theseus.json_utils import parse_json_response
from theseus.memory import Memory
from theseus.stimulus_log import StimulusLog

from .model_providers.model_provider import ModelProvider


class CognitiveCore:
    """A truncated OODA loop that serves as the primary cognitive process for a Theseus agent.

    The core does Orient, Decide and Act, steps and internalizes the LLM. Steps never pass
    data to each other directly — everything a loop accumulates lives in `loop_memory`,
    because any step may kick back to Orient and re-enter the cycle.

    Args:
        model_providers: One or more model provider. These are listed in priority order, with the cognitive core ultimately
        deciding which model provider is used.
        effectors: Available effectors keyed by their `name`.
        memory: Optional memory module, seen only through the Memory protocol: Orient pulls
        `retrieve(...)` results into context (via the ContextAssembler), and loop termination
        signals `form()`. What and how the module consolidates is its own business.
        name: The core's own actor name, used when logging its decisions and actions.
    """
    def __init__(
        self,
        constitution: str,
        model_providers: List[ModelProvider],
        effectors: dict[str, Effector],
        stimulus_log: StimulusLog,
        memory: Memory | None = None,
        name: str = "Tam",
    ):
        self.constitution = constitution
        self.model_providers = model_providers
        self.loop_memory = {}
        self.effectors = effectors
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

        options = [(effector.name, effector.description) for effector in self.effectors.values()]
        action_names = [name for name, _ in options] + [WAIT_ACTION]

        system_prompt = build_decide_system_prompt(self.constitution, options)
        prompt = build_decide_user_prompt(
            self.loop_memory["recent_events"],
            str(datetime.now()),
            memories=self.loop_memory["memories"],
        )

        raw_response = model_provider.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            json_schema=decision_json_schema(action_names),
        )

        decision = parse_json_response(raw_response)
        self.loop_memory["decision"] = decision

        self.stimulus_log.append(
            actor=self.name,
            type="decision",
            content={
                "action": decision["action"],
                "rationale": decision["rationale"],
                "raw_response": raw_response,
            },
        )

        self.act()

    def act(self):
        decision = self.loop_memory["decision"]
        action_name = decision["action"]

        if action_name == WAIT_ACTION:
            self.loop_termination()
            return

        effector = self.effectors.get(action_name)
        if effector is None:
            print(f"Unknown action: {action_name}. No effector available to handle this action.")
            self.loop_termination()
            return

        system_prompt = build_act_system_prompt(self.constitution)
        prompt = build_act_user_prompt(
            self.loop_memory["recent_events"],
            action_name,
            decision["rationale"],
            effector.act_instruction,
            memories=self.loop_memory["memories"],
        )

        model_provider = self._select_model_provider()
        response = model_provider.chat(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        effector.execute(response)

        self.stimulus_log.append(
            actor=self.name,
            type="action",
            content={"action": action_name, "output": response},
        )

        # Nothing can request another cycle yet; when effectors gain that power,
        # this branch re-enters Orient and loop_termination is skipped.
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
