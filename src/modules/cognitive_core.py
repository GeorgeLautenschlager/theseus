import json
import re
from datetime import datetime
from typing import List

from typer import prompt

from src.modules.stimulus_log import StimulusLog
from src.modules.context_assembler import ContextAssembler

from .model_providers.model_provider import ModelProvider


class CognitiveCore(ModelProvider):
    """A truncated OODA loop that serves as the primary cognitive process for a Theseus agent.

    The core does Orient, Decide and Act, steps and internalizes the LLM.
    Args:
        model_providers: One or more model provider. These are listed in priority order, with the cognitive core ultimately
        deciding which model provider is used.
    """
    def __init__(
        self,
        constitution: str,
        model_providers: List[ModelProvider],
        effector_callbacks: dict,
        stimulus_log: StimulusLog,
    ):
        self.constitution = constitution
        self.model_providers = model_providers
        self.loop_memory = {}
        self.effector_callbacks = effector_callbacks
        self.stimulus_log = stimulus_log
        self.context_assembler = ContextAssembler(stimulus_log=stimulus_log)

    @staticmethod
    def _parse_json_response(raw_response: str) -> dict:
        """Parses a model's JSON reply, tolerating ```json ... ``` code fences."""
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        json_str = match.group(0) if match else raw_response
        return json.loads(json_str)

    def _select_model_provider(self) -> ModelProvider:
        """Selects the most appropriate model provider based on the current context and available providers."""
        # For now, just return the first provider in the list.
        return self.model_providers[0]

    def orient(self):
        """Callback to be invoked by chat UI"""
        context = self.context_assembler.assemble_context()

        self.decide(context)

    def decide(self, context: str):
        model_provider = self._select_model_provider()

        prompt = (
            "Your StimulusLog records events you have recently experienced. This is your recent history:\n" +
            str(context) +
            "\n" +
            "The current system time is " + str(datetime.now()) + "." +
            "Given recent stimuli and the capabilities described in your system prompt, " + 
            "decide what to do next and emit that in the following JSON format: " +
            '{"action": "name of effector to use", "rationale": "reasoning behind the action"}'
        )

        system_prompt = (
            str(self.constitution) +
            "You have access to the following tools to help you decide:" +
            "You can effect the world through the following effectors: [ChatEffector]" +
            "Emit your decision as JSON in the following format with NO OTHER PARAMETERS: " +
            '{"action": "name of effector to use", "rationale": "reasoning behind the action"}'
        )

        raw_response = model_provider.chat(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        action = self._parse_json_response(raw_response)

        self.stimulus_log.append(
            actor="Tam",
            type="decision",
            content={"action": action["action"], "rationale": action["rationale"]},
        )

        print(
            "########### ########### ########### \n" +
            f"Decided on action: {action['action']}\n" +
            f"for rationale: {action['rationale']}\n" +
            "########### ########### ########### \n"
        )

        self.act(action, context)

    def act(self, action: dict, context: str):
        if action["action"] == "ChatEffector":
            self.chat_effector_callback = self.effector_callbacks.get("chat_effector_callback")

            prompt = (
                "Your StimulusLog records events you have recently experienced. This is your recent history:\n" +
                str(context) +
                "\n" +
                "Given that recent stimuli " + 
                "compose a response for the chat interface."
            )

            system_prompt = self.constitution

            model_provider = self._select_model_provider()
            response = model_provider.chat(
                prompt=prompt,
                system_prompt=system_prompt,
            )

            self.chat_effector_callback(response)
        else:
            print(f"Unknown action: {action['action']}. No effector available to handle this action.")

