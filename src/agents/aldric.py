from src.modules.chat_effector import ChatEffector
from src.modules.chat_observer import ChatObserver
from src.modules.cognitive_core import CognitiveCore
from src.modules.model_providers.ollama_provider import OllamaProvider


class Aldric:
    """Aldric is an agent built with the Theseus architecture.

    Aldric is build in concentric layers. In the centre is an LLM, surrounded by a
    truncated OODA loop. Orient, Decide and Act make up what is called the Cognitive Core.
    Observers, are outside of that, along with memory systems and more complex subagents that
    function as sensory surrogates, feeding pre-processed information into the core.

    Args:
        core: this is the cognition of the agent. It operates as a truncated OODA loop which can be
        entered at any point, but loops until it decides to terminate.
        observers: One or more modules responsible for collecting and  where appropriate, pre-processing data
        Memories: One or more memory systems
        surrogates: One or more subagents that function as sensory surrogates, feeding pre-processed information into the core.
        model_providers: One or more model provider. These are listed in priority order, with the cognitive core ultimately
        deciding which model provider is used.
    """
    def __init__(
        self
    ):
        self.chat_effector = ChatEffector()
        self.core = CognitiveCore(
            model_providers=[
                OllamaProvider(model="gemma4:e4b"),
            ],
            effector_callbacks={"chat_effector_callback": self.chat_effector.respond_callback},
        )
        self.chat_observer = ChatObserver(
            orient_chat_message_callback=self.core.orient
        )

    def run(self):
        """Run the agent. This is the main entry point for the agent."""
        while True:
            self.chat_observer.observe_chat_message()


