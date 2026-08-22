from __future__ import annotations

import threading
from datetime import datetime
from typing import List

from theseus.agentic_memory import AgenticMemory
from theseus.context_assembler import ContextAssembler
from theseus.stimulus_log import StimulusLog
from theseus.tools.tool import Tool

from .model_providers.model_provider import ModelProvider

class TaskCore:
    def __init__(
        self,
        name: str,
        constitution: str,
        persona: str,
        context_assembler: ContextAssembler,
        stimulus_log: StimulusLog,
        model_providers: Dict[str, List[ModelProvider]],
        tools: dict[str, Tool],
        memory: AgenticMemory | None = None,
        max_loops: int = 25
    ):
        self.name = name
        self.stimulus_log = stimulus_log
        self.memory = memory
        self.constitution = constitution
        self.persona = persona
        self.context_assembler = context_assembler
        self.model_providers = model_providers
        self.loop_memory = {}
        self.tools = tools
        self.max_loops = max_loops

    def wake(self):
        """Wake up and look at the stimulus log before deciding what to do"""
        # TODO: Type Task?
        self.loop_memory["task"] = self.identify_task()

        if self.loop_memory["task"] is None:
            return

        # Begin a cognitive loop in which this task will be completed
        self.loop()

        self.loop_termination()
        

    def loop(self):
        """Execute the identified task, using as many agent turns as necessary"""

        # Prompt the primary cognition model to 


    def loop_termination(self):
        """When the cognitive loop is complete, anything that needs to happen immediately after
        goes here"""
        if self.memory is not None:
            self.memory.form()
        self.loop_memory = {}