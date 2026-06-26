from typing import List

from modules.model_providers.model_provider import ModelProvider


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
    ):
        self.model_providers = model_providers

    def cognitive_loop(self):
        """The cognitive loop of the agent. This is a truncated OODA loop that does Orient, Decide and Act."""
        # Simple Orient
        # load chat history

        # Simple Decide
        # essentially a no-op, the user always gets a response

        # Simple Act
        #  - Assemble chat history and user message into a prompt
        #  - submit that to the LLM
        #  - relay response to user
        #  - append response to chat history
        #  - terminate

