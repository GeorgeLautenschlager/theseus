"""The transport seam between a surrogate's replicator and whatever carries its batches.

Pure interface: no implementation, no package imports. Swapping HTTP for something else
must not touch the log, the Assembler, or the agent — and the offline suite drives a fake
of this protocol rather than a live server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TransportResult:
    status: int


class StimulusTransport(Protocol):
    """Where a batch goes. The seam that makes the transport swappable and the suite offline.

    `send` returns a `TransportResult` for anything the far end actually answered, and
    **raises** when it could not be reached at all. The replicator does not catch that:
    deciding what a failure means is #32's job, not the transport's and not the loop's.
    """

    def send(self, body: str) -> TransportResult: ...
