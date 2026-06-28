from typing import List

from .model_providers.model_provider import ModelProvider
from .working_memory import WorkingMemory


class CognitiveCore(ModelProvider):
    """A truncated OODA loop that serves as the primary cognitive process for a Theseus agent.

    The core does Orient, Decide and Act, steps and internalizes the LLM.
    Args:
        model_providers: One or more model provider. These are listed in priority order, with the cognitive core ultimately
        deciding which model provider is used.
    """
    def __init__(
        self,
        model_providers: List[ModelProvider],
        effector_callbacks: dict,
    ):
        self.model_providers = model_providers
        self.working_memory = WorkingMemory()
        self.loop_memory = {"current_message": None}
        self.effector_callbacks = effector_callbacks

    def _select_model_provider(self) -> ModelProvider:
        """Selects the most appropriate model provider based on the current context and available providers."""
        # For now, just return the first provider in the list.
        return self.model_providers[0]

    def orient(self, message: str):
        """Callback to be invoked by chat UI"""
        self.working_memory.remember(message)
        self.loop_memory["current_message"] = message
        self.decide()

    def decide(self):
        action = "Generate Response"
        self.act(action)

    def act(self, action: str):
        if action == "Generate Response":
            model_provider = self._select_model_provider()
            chat_history = self.working_memory.recall()

            response = model_provider.chat(
                prompt="Given the following chat history and user message, generate a response to the user's message. Chat history: " + str(chat_history) + " User message: " + str(self.loop_memory["current_message"]),
            )

            self.working_memory.remember(response)
            self.effector_callbacks["chat_effector_callback"](response)
        else:
            raise ValueError(f"Unknown action: {action}")

