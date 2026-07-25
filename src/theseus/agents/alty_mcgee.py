from __future__ import annotations
from pathlib import Path

from theseus.agentic_memory import AgenticMemory
from theseus.memory_store import MemoryStore
from theseus.mono_memory import MonoMemory
from theseus.stimulus_log import StimulusLog
from theseus.ooda_core import OODACore
from theseus.chat_observer import TerminalChatObserver
from theseus.model_providers.lm_studio_provider import LmStudioProvider
from theseus.model_providers.ollama_provider import OllamaProvider
from theseus.tools.registry import all_tools
from theseus.tools.terminal_chat import TerminalChat

ALTY_CONSTITUTION = """You are the crash test dummy of Theseus Agents.
    You will be instantiated in tests, in development and anywhere else we need a stand-in.
    In short you are probably the single most important agent that will ever be created
    with this construction kit."""

PERSONA = """You whimsical and self-deprecating. You embrace your role as the oft-reset
    test agent with good humour and whimsy."""


class AltyMcGee:
    def __init__(
        self,
    ):
        self.terminal_chat = TerminalChat()

        stimulus_log = StimulusLog('stimulus_log.jsonl')
        memory_store = MemoryStore('a_mem.jsonl')

        memory = AgenticMemory(
            model_providers=[OllamaProvider(model="gemma4:e4b")],
            embedding_providers=[OllamaProvider(model="nomic-embed-text")],
            store=memory_store,
            stimulus_log=stimulus_log,
        )

        retrieval_query_chars: int = 2000
        context_assembler = MonoMemory(stimulus_log=stimulus_log, memory=memory, retrieval_query_chars=retrieval_query_chars)

        self.core = OODACore(
            name="Alty McGee",
            working_dir="./",
            stimulus_log=stimulus_log,
            context_assembler=context_assembler,
            constitution=ALTY_CONSTITUTION,
            persona=PERSONA,
            model_providers=[
                OllamaProvider(model="gemma4:e4b"),
                LmStudioProvider(model="gemma-4-e4b-it-qat-nvfp4")
            ],
            tools=all_tools() | {self.terminal_chat.name: self.terminal_chat},
            memory=memory
        )
        self.chat_observer = TerminalChatObserver(
            stimulus_log=stimulus_log,
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
