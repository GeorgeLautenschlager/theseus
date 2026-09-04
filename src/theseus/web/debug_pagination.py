"""Pure pagination logic for the StimulusLog debug view.

No I/O, no FastAPI. `StimulusLog.read_all()` returns events in append order, and ids are
minted from arrival, so plain slicing and id comparison over that list agree with each
other and are cheap relative to the file read itself.

Note that this is *arrival* order, which since #26 is no longer the same as chronological
order: a replicated event carries its producer's own `event_ts` and can have happened long
before the line above it. The debug view therefore shows the tape as written, while the
model sees the same events sorted by when they happened (see `ContextAssembler`).
"""

from __future__ import annotations

import bisect

from theseus.stimulus_log import StimulusEvent


def most_recent_page(
    events: list[StimulusEvent], page_size: int
) -> tuple[list[StimulusEvent], bool]:
    page = events[-page_size:]
    return page, len(events) > len(page)


def older_batch(
    events: list[StimulusEvent], before_id: str, limit: int
) -> tuple[list[StimulusEvent], bool]:
    ids = [e.id for e in events]
    idx = bisect.bisect_left(ids, before_id)
    start = max(0, idx - limit)
    return events[start:idx], start > 0
