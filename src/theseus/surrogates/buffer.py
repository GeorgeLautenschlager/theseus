"""How large the surrogate's local buffer may grow, and how far back eviction cuts.

The surrogate's log is a buffer, not a tape: under storage pressure it may evict
oldest-first ahead of the acked cursor, declaring the hole with a `storage_pressure`
gap marker. This policy is the knob for that eviction; the evicting log itself
consumes it. A plain `StimulusLog` never enters this mode and has no knob that
would let it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BufferPolicy:
    """How large the surrogate's buffer may grow, and how far back eviction cuts."""

    # 256 MB is a buffer a modest edge device can spare, and holds days of
    # text-rate observation. A deployment override, not a law — a starting
    # point chosen so the default never evicts in normal operation.
    max_bytes: int = 256 * 1024 * 1024
    # Where eviction cuts back to, as a fraction of max_bytes — not where it
    # starts. Eviction triggers at max_bytes and then keeps going until the buffer
    # is at or under max_bytes * low_water, so the two lines are a hysteresis band.
    # Strictly inside (0, 1): at 1.0 the band is empty and the buffer rewrites
    # itself on every append past the line; at 0.0 one eviction empties it.
    low_water: float = 0.8

    def __post_init__(self) -> None:
        """Fail at composition, not under pressure: bad config must never reach a live surrogate."""
        if self.max_bytes < 1:
            raise ValueError(
                f"max_bytes must be at least 1 (got {self.max_bytes!r}); a non-positive "
                f"buffer can hold nothing and evicts on every append"
            )
        if not 0 < self.low_water < 1:
            raise ValueError(
                f"low_water must be strictly between 0 and 1 (got {self.low_water!r}); at "
                f"1.0 there is no hysteresis and at 0.0 eviction empties the buffer"
            )
