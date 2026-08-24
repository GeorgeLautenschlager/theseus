"""Cadence — the CADENCE.md grammar: which model provider to use at a given time of
day, and how fast to tick.

A CADENCE.md file is ordinary markdown; freeform prose is allowed anywhere and is inert.
Only lines matching one of two rule forms do anything:

    - 08:00-22:00: ollama gemma3:12b, tick every 2 minutes
    - 22:00-08:00: lm_studio qwen/qwen3-32b, tick every 15 minutes
    - default: claude claude-sonnet-4-6, tick every 5 minutes

Window rules match on time of day, start-inclusive, end-exclusive, and may wrap midnight
(`22:00-08:00` means "from 22:00, through midnight, until 08:00"); `start == end` covers
the whole day. When several windows contain a time, file order is priority — first match
wins. The `default` rule applies outside all windows and doubles as the availability
fallback: callers walk `candidates_for` in order and take the first provider that is
actually reachable. `tick every N seconds|minutes|hours` sets how long the agent sleeps
between autonomous turns inside that window; omitted, it is DEFAULT_TICK_SECONDS.

This module is pure parsing and selection over datetimes it is handed — it never reads
the clock and knows nothing about timezones or provider availability. Lines that look
like rules but fail the grammar are collected by `lint_lines` so the agent can be asked
to rewrite them (see Autocore's schedule_lint stimulus).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime

DEFAULT_TICK_SECONDS = 300

_TICK_FRAGMENT = r"(?:,\s*tick every (\d+) (seconds?|minutes?|hours?))?"
_WINDOW_RULE_RE = re.compile(
    r"^- (\d{2}):(\d{2})-(\d{2}):(\d{2}): (\S+) (\S+)" + _TICK_FRAGMENT + r"$"
)
_DEFAULT_RULE_RE = re.compile(r"^- default: (\S+) (\S+)" + _TICK_FRAGMENT + r"$")
_TICK_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600}


@dataclass(frozen=True)
class CadenceRule:
    provider_key: str
    model: str
    tick_seconds: int
    start: dtime | None = None  # None on the default rule
    end: dtime | None = None

    def contains(self, t: dtime) -> bool:
        """Whether time-of-day `t` falls in this window. False for the default rule —
        it is not a window; Cadence appends it explicitly."""
        if self.start is None or self.end is None:
            return False
        if self.start == self.end:
            return True
        if self.start < self.end:
            return self.start <= t < self.end
        return t >= self.start or t < self.end


class Cadence:
    def __init__(self, windows: tuple[CadenceRule, ...], default: CadenceRule | None, lint: tuple[str, ...]):
        self._windows = windows
        self._default = default
        self._lint = lint

    @classmethod
    def parse(cls, text: str) -> Cadence:
        windows: list[CadenceRule] = []
        default: CadenceRule | None = None
        lint: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            rule = cls._parse_rule(stripped)
            if rule is not None:
                if rule.start is None:
                    default = rule
                else:
                    windows.append(rule)
            elif ":" in stripped:
                # Looks like it was meant to be a rule (a `- ` bullet with a colon)
                # but the grammar rejected it; plain bullets stay prose.
                lint.append(stripped)
        return cls(tuple(windows), default, tuple(lint))

    @property
    def rules(self) -> tuple[CadenceRule, ...]:
        """All parsed rules in file order, default rule (if any) last."""
        if self._default is None:
            return self._windows
        return self._windows + (self._default,)

    def rule_for(self, now: datetime) -> CadenceRule | None:
        """First window containing now's time of day, else the default rule, else None."""
        candidates = self.candidates_for(now)
        return candidates[0] if candidates else None

    def candidates_for(self, now: datetime) -> list[CadenceRule]:
        """All windows containing now's time of day in file order, then the default
        rule if present. The caller tries each in order until one's provider is
        actually available."""
        t = now.time()
        matched = [rule for rule in self._windows if rule.contains(t)]
        if self._default is not None:
            matched.append(self._default)
        return matched

    def lint_lines(self) -> list[str]:
        """Rule-looking lines (a `- ` bullet containing a colon) that fail the grammar,
        in file order."""
        return list(self._lint)

    @staticmethod
    def _parse_rule(stripped: str) -> CadenceRule | None:
        match = _WINDOW_RULE_RE.match(stripped)
        if match:
            sh, sm, eh, em, provider_key, model, n, unit = match.groups()
            try:
                start = dtime(int(sh), int(sm))
                end = dtime(int(eh), int(em))
                tick = _parse_tick(n, unit)
            except ValueError:
                return None
            return CadenceRule(provider_key, model, tick, start, end)
        match = _DEFAULT_RULE_RE.match(stripped)
        if match:
            provider_key, model, n, unit = match.groups()
            try:
                tick = _parse_tick(n, unit)
            except ValueError:
                return None
            return CadenceRule(provider_key, model, tick, None, None)
        return None


def _parse_tick(n: str | None, unit: str | None) -> int:
    if n is None:
        return DEFAULT_TICK_SECONDS
    count = int(n)
    if count < 1:
        raise ValueError(f"invalid tick count: {n!r}")
    return count * _TICK_UNIT_SECONDS[unit.rstrip("s")]
