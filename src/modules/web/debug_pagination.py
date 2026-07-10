"""Pure pagination logic for the StimulusLog debug view.

No I/O, no FastAPI. `StimulusLog.read_all()` already returns events in
append/chronological/lexical-ID-ascending order, so plain slicing over that
list is correct and cheap relative to the file read itself.
"""

from __future__ import annotations

import bisect

from src.modules.stimulus_log import StimulusEvent


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
