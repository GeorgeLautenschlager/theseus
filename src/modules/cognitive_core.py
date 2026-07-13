from __future__ import annotations

import json
import re
from datetime import datetime
from typing import List

from src.modules.cognitive_prompts import (
    WAIT_ACTION,
    build_act_system_prompt,
    build_act_user_prompt,
    build_decide_system_prompt,
    build_decide_user_prompt,
    decision_json_schema,
)
from src.modules.context_assembler import ContextAssembler
from src.modules.effector import Effector
from src.modules.stimulus_log import StimulusLog

from .model_providers.model_provider import ModelProvider


class CognitiveCore:
    """A truncated OODA loop that serves as the primary cognitive process for a Theseus agent.

    The core does Orient, Decide and Act, steps and internalizes the LLM.
    Args:
        model_providers: One or more model provider. These are listed in priority order, with the cognitive core ultimately
        deciding which model provider is used.
        effectors: Available effectors keyed by their `name`.
        name: The core's own actor name, used when logging its decisions and actions.
    """
    def __init__(
        self,
        constitution: str,
        model_providers: List[ModelProvider],
        effectors: dict[str, Effector],
        stimulus_log: StimulusLog,
        name: str = "Tam",
    ):
        self.constitution = constitution
        self.model_providers = model_providers
        self.loop_memory = {}
        self.effectors = effectors
        self.stimulus_log = stimulus_log
        self.context_assembler = ContextAssembler(stimulus_log=stimulus_log)
        self.name = name

    @staticmethod
    def _parse_json_response(raw_response: str) -> dict:
        """Parses a model's JSON reply, tolerating ```json ... ``` code fences."""
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        json_str = match.group(0) if match else raw_response
        return json.loads(json_str)

    def _select_model_provider(self) -> ModelProvider:
        """Selects the first available provider, in priority order."""
        for provider in self.model_providers:
            if provider.is_available():
                return provider
        raise RuntimeError("No model providers are currently available.")

    def orient(self):
        """Callback to be invoked by chat UI"""
        context = self.context_assembler.assemble_context()

        self.decide(context)

    def decide(self, context: str):
        model_provider = self._select_model_provider()

        options = [(effector.name, effector.description) for effector in self.effectors.values()]
        action_names = [name for name, _ in options] + [WAIT_ACTION]

        system_prompt = build_decide_system_prompt(self.constitution, options)
        prompt = build_decide_user_prompt(context, str(datetime.now()))

        raw_response = model_provider.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            json_schema=decision_json_schema(action_names),
        )

        decision = self._parse_json_response(raw_response)

        self.stimulus_log.append(
            actor=self.name,
            type="decision",
            content={
                "action": decision["action"],
                "rationale": decision["rationale"],
                "raw_response": raw_response,
            },
        )

        self.act(decision, context)

    def act(self, decision: dict, context: str):
        action_name = decision["action"]

        if action_name == WAIT_ACTION:
            return

        effector = self.effectors.get(action_name)
        if effector is None:
            print(f"Unknown action: {action_name}. No effector available to handle this action.")
            return

        system_prompt = build_act_system_prompt(self.constitution)
        prompt = build_act_user_prompt(context, action_name, decision["rationale"], effector.act_instruction)

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
