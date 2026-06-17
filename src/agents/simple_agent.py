from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.modules.actors.naive_actor import NaiveActor
from src.modules.deciders.llm_decider import LLMDecider
from src.modules.memory.naive_context_assembler import NaiveContextAssembler
from src.modules.model_providers.llama_provider import LlamaProvider
from src.modules.observers.naive_observer import NaiveObserver
from src.modules.orienters.naive_orienter import NaiveOrienter
from src.modules.theseus_agent import TheseusAgent
from src.modules.user_interfaces.textual_ui import TextualUI


class SimpleAgent(TheseusAgent):
    observer = NaiveObserver()
    orienter = NaiveOrienter()
    decider = LLMDecider(LlamaProvider())
    actor = NaiveActor()
    memory = [NaiveContextAssembler()]
    ui = TextualUI()


if __name__ == "__main__":
    SimpleAgent().run()
