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
    # The host's own words, when it gave any. Bounded here — this is a remote party's
    # output and the transport should not carry an unbounded string around; the tape
    # bounds it again on write (`replication_events.MAX_REASON_CHARS`).
    reason: str = ""


class StimulusTransport(Protocol):
    """Where a batch goes. The seam that makes the transport swappable and the suite offline.

    `send` returns a `TransportResult` for anything the far end actually answered, and
    **raises** when it could not be reached at all. The replicator reads the distinction:
    a status spends retry budget, a raise stops the drain cleanly and spends none (#32).
    """

    def send(self, body: str) -> TransportResult: ...
