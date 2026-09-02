"""IntelligenceLayer — the agent's live context, served as a read strategy.

The fourth "layer" has no file of its own: it is the tail of the StimulusLog,
ranked by recency. That is the point — what the agent is currently doing and
saying is already durable in the log; giving this layer its own store would
duplicate the bedrock and invite drift. It exists as a collaborator so the
module can fan a recall out over it like any other layer, and so "no fourth
file" is an invariant the layout makes visible rather than a comment.
"""

from __future__ import annotations

import json

from theseus.layer_store import LayerHit
from theseus.stimulus_log import StimulusLog


class IntelligenceLayer:
    def __init__(self, stimulus_log: StimulusLog, tail: int = 20) -> None:
        self._log = stimulus_log
        self._tail = max(1, tail)

    def read(self) -> list[LayerHit]:
        """The most recent `tail` events, newest first. Score is rank-based
        (newest = highest), which is all a recency ordering needs."""
        events = self._log.read_all()[-self._tail :]
        hits: list[LayerHit] = []
        for rank, event in enumerate(reversed(events)):
            hits.append(
                LayerHit(
                    id=event.id,
                    text=f"[{event.id}] {event.actor} {event.type}: "
                    + json.dumps(event.content, ensure_ascii=False),
                    score=float(len(events) - rank),
                )
            )
        return hits
