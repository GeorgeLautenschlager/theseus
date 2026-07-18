from theseus.stimulus_log import StimulusLog
from theseus.cognitive_core import CognitiveCore
from theseus.chat_observer import ChatObserver
from theseus.chat_effector import ChatEffector
from theseus.model_providers.lm_studio_provider import LmStudioProvider

ALTY_CONSTITUTION = """You are the crash test dummy of Theseus Agents.
    You will be instantiated in tests, in development and anywhere else we need a stand-in.
    In short you are probably the single most important agent that will ever be created
    with this construction kit."""


class AltyMcGee:
    def __init__(self):
        chat_effector = ChatEffector()
        self.stimulus_log = StimulusLog('stimulus_log.jsonl')
        self.core = CognitiveCore(
            stimulus_log=self.stimulus_log,
            constitution=ALTY_CONSTITUTION,
            model_providers=[LmStudioProvider(model="gemma-4-e4b-it-qat-nvfp4")],
            effectors={chat_effector.name: chat_effector},
            name="Alty McGee",
        )
        self.chat_observer = ChatObserver(
            stimulus_log=self.stimulus_log,
            orient_chat_message_callback=self.core.orient
        )
    
    def run(self):
        """Run the agent. This is the main entry point for the agent."""
        while True:
            self.chat_observer.observe_chat_message()

def main() -> None:
    AltyMcGee().run()

if __name__ == "__main__":
    main()