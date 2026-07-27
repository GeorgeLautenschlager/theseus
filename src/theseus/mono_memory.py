from __future__ import annotations

from dataclasses import dataclass

from theseus.stimulus_log import StimulusLog


@dataclass(frozen=True)
class AssembledContext:
    recent_events: str   # tail of the stimulus log, one JSON event per line


class MonoMemory:
    """Assembles context for Decide from the last `window_size` stimulus events, verbatim.

    Deliberately the *only* source. Long-term memory used to be pulled in here, behind
    the agent's back, and rendered as a second prompt section — which meant recall was
    something that happened *to* the agent rather than something it did. Recall is now a
    tool (`tools/recall.py`): the agent asks, and the recollection lands in the stimulus
    log as a tool_result, so it arrives through this same window on the next pass.
    """

    def __init__(
        self,
        stimulus_log: StimulusLog,
        window_size: int = 50,
    ):
        self.stimulus_log = stimulus_log
        self.window_size = window_size

    def assemble_context(self) -> AssembledContext:
        events = self.stimulus_log.read_all()[-self.window_size:]
        return AssembledContext(
            recent_events="\n".join(event.to_json() for event in events)
        )
