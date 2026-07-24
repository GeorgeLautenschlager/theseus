from __future__ import annotations

from dataclasses import dataclass

from theseus.memory import Memory
from theseus.stimulus_log import StimulusLog


@dataclass(frozen=True)
class AssembledContext:
    recent_events: str   # tail of the stimulus log, one JSON event per line
    memories: str        # rendered memories from the memory module; "" when none


class MonoMemory:
    """assembles context for Decide from the following sources:
    - StimulusLog: the last `window_size` events, verbatim
    - the agent's memory systems, if any are present: whatever the module's
      `retrieve` returns against the recent-events window
    - persona, if present
    - constitution, if present
    """

    def __init__(
        self,
        stimulus_log: StimulusLog,
        memory: Memory | None = None,
        window_size: int = 50,
        persona: str | None = None,
        constitution: str | None = None,
        retrieval_query_chars: int = 2000,
    ):
        self.stimulus_log = stimulus_log
        self.memory = memory
        self.window_size = window_size
        self.constitution = constitution
        self.retrieval_query_chars = retrieval_query_chars

    def assemble_context(self) -> AssembledContext:
        events = self.stimulus_log.read_all()[-self.window_size:]
        recent_events = "\n".join(event.to_json() for event in events)

        memories = ""
        if self.memory is not None and recent_events:
            # Retrieval embeds this query, so cap it to the most-recent tail — the full
            # window is for the LLM prompt, not the (much smaller) embedding context.
            memories = self.memory.retrieve(recent_events[-self.retrieval_query_chars:])

        return AssembledContext(recent_events=recent_events, memories=memories)
