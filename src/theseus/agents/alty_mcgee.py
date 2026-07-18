from theseus.chat_effector import ChatEffector
from theseus.cognitive_core import CognitiveCore
from theseus.model_providers.openrouter_provider import OpenRouterProvider

ALTY_CONSTITUTION = """You are the crash test dummy of Theseus Agents.
    You will be instantiated in tests, in development and anywhere else we need a stand-in.
    In short you are probably the single most important agent that will ever be created
    with this construction kit."""


class AltyMcGee:
    def __init__(self):
        self.stimulus_log = StimulusLog()
        self.core = CognitiveCore(
            stimulus_log=self.stimulus_log,
            constitution=ALTY_CONSTITUTION,
            model_providers=OpenRouterProvider(model="google/gemma-4-31b-it"),
            effectors=ChatEffector(),
            name="Alty McGee",
        )
        self.chat_observer = ChatObserver(
            stimulus_log=self.stimulus_log,
        )
