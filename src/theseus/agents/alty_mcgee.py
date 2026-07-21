from __future__ import annotations

from theseus.stimulus_log import StimulusLog
from theseus.cognitive_core import CognitiveCore
from theseus.chat_observer import ChatObserver
from theseus.model_providers.lm_studio_provider import LmStudioProvider
from theseus.model_providers.ollama_provider import OllamaProvider
from theseus.tools.registry import all_tools
from theseus.tools.terminal_chat import TerminalChat

ALTY_CONSTITUTION = """You are the crash test dummy of Theseus Agents.
    You will be instantiated in tests, in development and anywhere else we need a stand-in.
    In short you are probably the single most important agent that will ever be created
    with this construction kit."""


class AltyMcGee:
    def __init__(self, stimulus_log: StimulusLog | None = None):
        self.terminal_chat = TerminalChat()
        self.stimulus_log = stimulus_log or StimulusLog('stimulus_log.jsonl')
        self.core = CognitiveCore(
            stimulus_log=self.stimulus_log,
            constitution=ALTY_CONSTITUTION,
            model_providers=[
                OllamaProvider(model="gemma4:e4b"),
                LmStudioProvider(model="gemma-4-e4b-it-qat-nvfp4")
            ],
            tools=all_tools() | {self.terminal_chat.name: self.terminal_chat},
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
