from __future__ import annotations

from datetime import datetime
from typing import List

from theseus.agentic_memory import AgenticMemory
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
from theseus.stimulus_log import StimulusLog

from .model_providers.model_provider import ModelProvider


class CognitiveCore:
    """A truncated OODA loop that serves as the primary cognitive process for a Theseus agent.

    The core does Orient, Decide and Act, steps and internalizes the LLM.
    Args:
        model_providers: One or more model provider. These are listed in priority order, with the cognitive core ultimately
        deciding which model provider is used.
        effectors: Available effectors keyed by their `name`.
        memory: Optional long-term memory. When present, Orient retrieves relevant notes into
        the Decide/Act context, and each cycle ends by consolidating new stimulus events into
        memory (post-cycle, so the agent's response is never blocked by memory writes).
        name: The core's own actor name, used when logging its decisions and actions.
    """
    def __init__(
        self,
        constitution: str,
        model_providers: List[ModelProvider],
        effectors: dict[str, Effector],
        stimulus_log: StimulusLog,
        memory: AgenticMemory | None = None,
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

        self.decide(assembled.recent_events, memories=assembled.memories)

        self._form_memories()

    def _form_memories(self):
        """Post-cycle memory formation: consolidate every stimulus event newer than
        the store's high-water mark into a note. Runs after Act so it never delays
        the agent's outward response."""
        if self.memory is None:
            return
        high_water = self.memory.store.last_consolidated_id()
        events = [
            e for e in self.stimulus_log.read_all()
            if high_water is None or e.id > high_water
        ]
        self.memory.form(events)

    def decide(self, context: str, memories: str = ""):
        model_provider = self._select_model_provider()

        options = [(effector.name, effector.description) for effector in self.effectors.values()]
        action_names = [name for name, _ in options] + [WAIT_ACTION]

        system_prompt = build_decide_system_prompt(self.constitution, options)
        prompt = build_decide_user_prompt(context, str(datetime.now()), memories=memories)

        raw_response = model_provider.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            json_schema=decision_json_schema(action_names),
        )

        decision = parse_json_response(raw_response)

        self.stimulus_log.append(
            actor=self.name,
            type="decision",
            content={
                "action": decision["action"],
                "rationale": decision["rationale"],
                "raw_response": raw_response,
            },
        )

        self.act(decision, context, memories=memories)

    def act(self, decision: dict, context: str, memories: str = ""):
        action_name = decision["action"]

        if action_name == WAIT_ACTION:
            return

        effector = self.effectors.get(action_name)
        if effector is None:
            print(f"Unknown action: {action_name}. No effector available to handle this action.")
            return

        system_prompt = build_act_system_prompt(self.constitution)
        prompt = build_act_user_prompt(
            context, action_name, decision["rationale"], effector.act_instruction, memories=memories
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
