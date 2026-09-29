from __future__ import annotations

from theseus.stimulus_log import StimulusEvent

DESKTOP_OBSERVATION = "observation.desktop"


def wakes_agent(event: StimulusEvent) -> bool:
    """Ambient state is remembered without requesting an immediate cognitive turn."""
    return event.type != DESKTOP_OBSERVATION
