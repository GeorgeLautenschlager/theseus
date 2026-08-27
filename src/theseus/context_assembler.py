from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

from theseus.stimulus_log import StimulusEvent, StimulusLog

# Seed ratio for English-ish JSON, used until the first real measurement lands. The error
# is asymmetric: too low merely underfills the window, too high under-charges every event
# and overruns the model, losing the turn. And under a provider that reports no usage (the
# Claude CLI) `observe` never corrects it, so the seed is the permanent answer rather than
# just the first one.
#
# It was 3.4, from a 3.46 measurement over an early agent log. That log was not dense
# enough to be representative: Tam's, measured 2026-08-26 against a backend that does
# report usage, ran 417,546 chars to 150,305 prompt tokens — 2.78. Real logs are mostly
# ULIDs, ISO timestamps, escaped JSON and file paths, none of which tokenize like prose,
# so 3.4 was under-charging every event by ~22% in exactly the unsafe direction.
DEFAULT_CHARS_PER_TOKEN = 2.78

# The smallest context window we can actually expect to be running against: Ollama's
# default `num_ctx`. Deliberately conservative — a rule that knows its model declares
# `context N` in CADENCE.md, and one that doesn't will at least not overflow the weakest
# backend.
DEFAULT_TOKEN_BUDGET = 4096

# Kept in step with `ModelProvider.complete_with_tools`'s `max_tokens`. The context window
# holds the prompt *and* the completion, so a budget that spends the whole window on the
# prompt has already overrun by the time the model opens its mouth.
DEFAULT_RESERVED_OUTPUT_TOKENS = 8196

# Skimmed off the top for what we provably cannot see: `prompt_chars` counts message
# content only, while the server also charges for the chat template's scaffolding and for
# the native tool schemas (~1.2k tokens for a modest toolset), which go up the wire as
# `tools=` rather than as message text. It also absorbs the gap between our tokenizer
# estimate and the model's actual tokenizer, which we never get to see. A tenth of the
# window buys ~13k tokens of slack at 128k, which is worth more than the history it
# costs: overrunning loses the whole turn, underfilling loses the oldest few events.
DEFAULT_SAFETY_FRACTION = 0.10

# No single event may eat more than this share of the window, so one enormous `read`
# result cannot crowd out every other thing the agent just did.
DEFAULT_MAX_EVENT_FRACTION = 0.25

# ...but never trim an event below this, however tight the budget. Under a very small
# budget the clamp would otherwise shave the one event we are guaranteeing down to nothing,
# which is the empty window this class exists to avoid.
MIN_EVENT_TOKENS = 256

_TRUNCATION_MARKER = "…[truncated {} chars — re-read the source if you need the rest]"


@dataclass(frozen=True)
class AssembledContext:
    recent_events: str   # tail of the stimulus log, one JSON event per line
    window_chars: int = 0
    budget_tokens: float | None = None  # what the window was fitted against, for debugging


class ContextAssembler:
    """Assembles context for Decide from the tail of the stimulus log, verbatim.

    Deliberately the *only* source. Long-term memory used to be pulled in here, behind
    the agent's back, and rendered as a second prompt section — which meant recall was
    something that happened *to* the agent rather than something it did. Recall is now a
    tool (`tools/recall.py`): the agent asks, and the recollection lands in the stimulus
    log as a tool_result, so it arrives through this same window on the next pass.

    The window is sized in *tokens*, not events. An event count is arbitrary in the wrong
    unit — fifty events is a couple of thousand tokens of chat, or a hundred thousand
    tokens if one of them is a `read` result carrying a whole file.

    ### Where the size comes from

    Sizing properly wants the model's context window, and that turns out not to be
    knowable by asking: the OpenAI-compatible `/v1/models` endpoint carries no context
    length, and the vendor-native endpoints that do carry one mostly report the model's
    *trained* maximum rather than the window the server actually loaded (Ollama will
    happily tell you 131072 while serving 4096). So it is declared instead, per rule, in
    CADENCE.md — `context 128k` — and handed here via `set_context_limit`. That keeps the
    budget attached to the model that actually won the turn, which matters because Cadence
    changes models by time of day: a budget fixed once at startup is a budget that is
    wrong for most of the day.

    Given a limit, the window budget is what's left after everything else that must fit:

        budget = (limit - reserved_output) * (1 - safety) - overhead

    `overhead` is the constitution, persona and tool schemas. Pass it per-assembly as
    `overhead_chars` when the caller has already rendered the system prompt and can just
    measure it; otherwise the value learned from the last `observe` is used, which is
    necessarily one turn stale and is zero on the very first turn.

    `chars_per_token` is closed-loop rather than predicted: assemble against an estimate,
    then let `observe` correct the estimate from what the backend says the prompt really
    cost. `window_size` remains as a hard cap on event count, bounding the first pass —
    before any measurement exists — and keeping assembly cheap on a long log.
    """

    def __init__(
        self,
        stimulus_log: StimulusLog,
        window_size: int = 200,
        token_budget: int | None = DEFAULT_TOKEN_BUDGET,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
        context_tokens: int | None = None,
        reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
        safety_fraction: float = DEFAULT_SAFETY_FRACTION,
        max_event_fraction: float = DEFAULT_MAX_EVENT_FRACTION,
    ):
        self.stimulus_log = stimulus_log
        self.window_size = window_size
        self.token_budget = token_budget
        self.chars_per_token = chars_per_token
        self.context_tokens = context_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.safety_fraction = safety_fraction
        self.max_event_fraction = max_event_fraction
        # Everything in the prompt that isn't the window — constitution, persona, tool
        # schemas. Derived from measurement rather than configured, so the budget stays
        # honest as tools are added without the caller re-tuning it.
        self.overhead_tokens = 0.0

    def set_context_limit(self, context_tokens: int | None) -> None:
        """Declare the window of the model about to be prompted.

        `None` is "this rule didn't say", and leaves whatever was already in force — the
        conservative `token_budget` default, or the last limit set. Silently guessing a
        large window for an undeclared model is the failure this whole change exists to
        remove.
        """
        if context_tokens is not None:
            self.context_tokens = context_tokens

    def assemble_context(self, overhead_chars: int | None = None) -> AssembledContext:
        """Fit the tail of the log into the budget.

        `overhead_chars` is the measured length of everything else going into this
        prompt. Supplying it is strictly better than letting `observe` infer it after the
        fact: the inferred value describes the *previous* turn and is 0.0 on the first,
        which is exactly when a fixed budget is most likely to overrun.
        """
        events = self.stimulus_log.read_all()[-self.window_size:]
        budget = self._budget_tokens(overhead_chars)

        if budget is None:
            lines = [event.to_json() for event in events]
        else:
            lines = self._fit_to_budget(events, budget)

        recent_events = "\n".join(lines)
        return AssembledContext(
            recent_events=recent_events,
            window_chars=len(recent_events),
            budget_tokens=budget,
        )

    def observe(
        self,
        prompt_tokens: int | None,
        prompt_chars: int,
        window_chars: int,
    ) -> None:
        """Calibrate from one real exchange: what we sent, and what the backend charged.

        A no-op when the provider reports no usage — `ClaudeProvider` shells out to the
        CLI and gets none, and not every OpenAI-compatible server populates the field.
        The seeded estimate then simply stands, which is why the seed is conservative:
        under Claude it is never corrected at all.
        """
        if not prompt_tokens or prompt_chars <= 0:
            return

        self.chars_per_token = prompt_chars / prompt_tokens
        self.overhead_tokens = max(0.0, (prompt_chars - window_chars) / self.chars_per_token)

    def _budget_tokens(self, overhead_chars: int | None = None) -> float | None:
        """Tokens available for the event window itself. None means "unbounded" — only
        reachable by explicitly passing `token_budget=None` with no declared context."""
        if self.context_tokens is not None:
            usable = (self.context_tokens - self.reserved_output_tokens) * (
                1.0 - self.safety_fraction
            )
        elif self.token_budget is not None:
            usable = float(self.token_budget)
        else:
            return None

        if overhead_chars is None:
            overhead = self.overhead_tokens
        else:
            overhead = overhead_chars / self.chars_per_token
        return usable - overhead

    def _fit_to_budget(self, events: list[StimulusEvent], budget: float) -> list[str]:
        """Take events newest-first until the budget runs out, then restore log order.

        Always returns at least one event. A window that overshoots the budget is
        recoverable — the backend truncates, or errors, and the next pass is calibrated —
        whereas an empty one asks the model to decide with no stimulus at all. The
        per-event clamp is what makes that guarantee affordable: without it the one event
        we promise to emit could itself be a whole file.
        """
        max_event_chars = self._max_event_chars(budget)
        kept: list[str] = []
        used = 0.0

        for event in reversed(events):
            line = self._serialize(event, max_event_chars)
            cost = len(line) / self.chars_per_token
            if kept and used + cost > budget:
                break
            kept.append(line)
            used += cost

        kept.reverse()
        return kept

    def _max_event_chars(self, budget: float) -> int:
        tokens = max(budget * self.max_event_fraction, float(MIN_EVENT_TOKENS))
        return int(tokens * self.chars_per_token)

    def _serialize(self, event: StimulusEvent, max_chars: int) -> str:
        """The event as one JSON line, trimmed if it is outsized.

        Trimming happens here, on the way into the prompt, and never in the StimulusLog —
        the log is append-only bedrock and the full text stays in it. The result is still
        a valid, parseable event carrying its own id, actor and type: the agent can see
        *that* it was cut, and go re-read the source if it needs the rest.

        What gets cut is the longest string *anywhere* in `content`, repeatedly, until the
        line fits. Searching the whole tree rather than the top level matters: a
        `tool_result` keeps its bulk in a top-level `output`, but a `decision` keeps it
        nested in `tool_calls[].arguments`, and a top-level-only scan would shave that
        event's short `text` — throwing away the agent's stated reasoning while leaving
        the actual bulk untouched.
        """
        line = event.to_json()
        if len(line) <= max_chars:
            return line

        content = copy.deepcopy(event.content)
        # Several medium strings can each need a pass; bounded so a pathological payload
        # cannot spin. Falling out still over budget is fine — the next turn calibrates.
        for _ in range(8):
            path, length = _longest_string(content)
            if path is None or length == 0:
                break  # nothing trimmable left; over budget beats malformed
            parent = content
            for key in path[:-1]:
                parent = parent[key]
            original = parent[path[-1]]
            overshoot = len(line) - max_chars
            marker_len = len(_TRUNCATION_MARKER.format(length))
            keep = max(0, len(original) - overshoot - marker_len)
            if keep >= len(original):
                break
            parent[path[-1]] = original[:keep] + _TRUNCATION_MARKER.format(
                len(original) - keep
            )
            line = replace(event, content=content).to_json()
            if len(line) <= max_chars:
                break
        return line


def _longest_string(node: Any, path: tuple = ()) -> tuple[tuple | None, int]:
    """Path to the longest string anywhere in a JSON-shaped tree, and its length."""
    if isinstance(node, str):
        return path, len(node)
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return None, 0

    best_path: tuple | None = None
    best_len = 0
    for key, value in items:
        found, length = _longest_string(value, path + (key,))
        if length > best_len:
            best_path, best_len = found, length
    return best_path, best_len
