from __future__ import annotations
from pathlib import Path

from theseus.agentic_memory import AgenticMemory
from theseus.auto_core import Autocore
from theseus.memory_store import MemoryStore
from theseus.context_assembler import ContextAssembler
from theseus.stimulus_log import StimulusLog
from theseus.ooda_core import OODACore
from theseus.chat_observer import TerminalChatObserver
from theseus.model_providers.lm_studio_provider import LmStudioProvider
from theseus.model_providers.ollama_provider import OllamaProvider
from theseus.tools.recall import RecallTool
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
        stimulus_log: StimulusLog | None = None,
        memory_store: MemoryStore | None = None,
    ):
        """Both stores default to files in the working directory, which is what `poetry
        run alty` wants. Tests inject their own so a run cannot scribble on the repo — and
        so a "session" can be modelled as a fresh instance over the same log."""
        self.terminal_chat = TerminalChat()

        # `is None`, not `or`: MemoryStore defines __len__, so an empty injected store is
        # falsy and `or` would silently swap it for the working-directory file.
        stimulus_log = StimulusLog('stimulus_log.jsonl') if stimulus_log is None else stimulus_log
        memory_store = MemoryStore('a_mem.jsonl') if memory_store is None else memory_store
        self.stimulus_log = stimulus_log

        memory = AgenticMemory(
            model_providers=[OllamaProvider(model="gemma4:e4b")],
            embedding_providers=[OllamaProvider(model="nomic-embed-text")],
            store=memory_store,
            stimulus_log=stimulus_log,
        )

        context_assembler = ContextAssembler(stimulus_log=stimulus_log)

        # Recall is a tool rather than something the assembler does behind Alty's back,
        # so it needs the memory module and is composed here, not in the registry.
        recall = RecallTool(memory)

        self.core = OODACore(
            name="Alty McGee",
            stimulus_log=stimulus_log,
            context_assembler=context_assembler,
            constitution=ALTY_CONSTITUTION,
            persona=PERSONA,
            model_providers=[
                OllamaProvider(model="gemma4:e4b"),
                LmStudioProvider(model="gemma-4-e4b-it-qat-nvfp4")
            ],
            tools=all_tools() | {
                self.terminal_chat.name: self.terminal_chat,
                recall.name: recall,
            },
            memory=memory
        )
        self.memory = memory
        self.chat_observer = TerminalChatObserver(
            stimulus_log=stimulus_log,
            orient_chat_message_callback=self.core.orient_and_wait
        )

    def run(self):
        """Run the agent. This is the main entry point for the agent."""
        while True:
            self.chat_observer.observe_chat_message()


class AutoAlty:
    """Alty's autonomous sibling: the crash test dummy for `Autocore`.

    Where AltyMcGee waits for chat messages, AutoAlty ticks on its own cadence.
    TerminalChat is its mouth — replies land on stdout."""

    def __init__(self, home_directory: Path | None = None):
        """The home directory defaults to `auto_alty/` in the working directory,
        which is what running it by hand wants. Tests inject a tmp path so a run
        cannot scribble on the repo."""
        home_directory = Path("auto_alty") if home_directory is None else home_directory

        self.terminal_chat = TerminalChat()
        self.core = Autocore(
            name="Auto Alty",
            home_directory=home_directory,
            tools=all_tools() | {self.terminal_chat.name: self.terminal_chat},
        )

        # Autocore touches these empty on first boot; seed Alty's identity into
        # them without clobbering a file the agent (or George) has since edited.
        for name, text in (
            ("CONSTITUTION.md", ALTY_CONSTITUTION),
            ("persona.md", PERSONA),
        ):
            path = home_directory / name
            if not path.read_text(encoding="utf-8").strip():
                path.write_text(text, encoding="utf-8")
                setattr(self.core, name.removesuffix(".md"), text)

        self.stimulus_log = self.core.stimulus_log

    def run(self):
        """Run the agent's autonomous loop. Blocks forever."""
        self.core.loop()


def main() -> None:
    AutoAlty().run()

if __name__ == "__main__":
    main()
