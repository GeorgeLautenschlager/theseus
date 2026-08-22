from __future__ import annotations

from dataclasses import dataclass

from theseus.stimulus_log import StimulusLog

# Seed ratio for English-ish JSON, used until the first real measurement lands. Wrong by
# a little for everyone and catastrophically for no one, which is all a seed has to be.
DEFAULT_CHARS_PER_TOKEN = 4.0

# The smallest context window we can actually expect to be running against: Ollama's
# default `num_ctx`. Deliberately conservative — an agent that knows its model should
# raise this, and one that doesn't will at least not overflow the weakest backend.
DEFAULT_TOKEN_BUDGET = 4096


@dataclass(frozen=True)
class AssembledContext:
    recent_events: str   # tail of the stimulus log, one JSON event per line
    window_chars: int = 0


class ContextAssembler:
    """Assembles context for Decide from the tail of the stimulus log, verbatim.

    Deliberately the *only* source. Long-term memory used to be pulled in here, behind
    the agent's back, and rendered as a second prompt section — which meant recall was
    something that happened *to* the agent rather than something it did. Recall is now a
    tool (`tools/recall.py`): the agent asks, and the recollection lands in the stimulus
    log as a tool_result, so it arrives through this same window on the next pass.

    The window is sized in *tokens*, not events. An event count is arbitrary in the wrong
    unit — fifty events is a couple of thousand tokens of chat, or a hundred thousand
    tokens if one of them is a `read` result carrying a whole file. Sizing it properly
    would want the model's context window, and that turns out not to be knowable: the
    OpenAI-compatible `/v1/models` endpoint carries no context length, and the
    vendor-native endpoints that do carry one mostly report the model's *trained* maximum
    rather than the window the server actually loaded (Ollama will happily tell you
    131072 while serving 4096).

    What *is* universally available is `usage.prompt_tokens` on every response. So the
    budget is closed-loop rather than predicted: assemble against an estimate, then let
    `observe` correct the estimate from what the backend says the prompt really cost.
    `window_size` remains as a hard cap on event count, bounding the first pass — before
    any measurement exists — and keeping assembly cheap on a long log.
    """

    def __init__(
        self,
        stimulus_log: StimulusLog,
        window_size: int = 200,
        token_budget: int | None = DEFAULT_TOKEN_BUDGET,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    ):
        self.stimulus_log = stimulus_log
        self.window_size = window_size
        self.token_budget = token_budget
        self.chars_per_token = chars_per_token
        # Everything in the prompt that isn't the window — constitution, persona, tool
        # schemas. Derived from measurement rather than configured, so the budget stays
        # honest as tools are added without the caller re-tuning it.
        self.overhead_tokens = 0.0

    def assemble_context(self) -> AssembledContext:
        events = self.stimulus_log.read_all()[-self.window_size:]
        lines = [event.to_json() for event in events]

        if self.token_budget is not None:
            lines = self._fit_to_budget(lines)

        recent_events = "\n".join(lines)
        return AssembledContext(
            recent_events=recent_events,
            window_chars=len(recent_events),
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
        The seeded estimate then simply stands.
        """
        if not prompt_tokens or prompt_chars <= 0:
            return

        self.chars_per_token = prompt_chars / prompt_tokens
        self.overhead_tokens = max(0.0, (prompt_chars - window_chars) / self.chars_per_token)

    def _fit_to_budget(self, lines: list[str]) -> list[str]:
        """Take events newest-first until the budget runs out, then restore log order.

        Always returns at least one event. A window that overshoots the budget is
        recoverable — the backend truncates, or errors, and the next pass is calibrated —
        whereas an empty one asks the model to decide with no stimulus at all.
        """
        budget = self.token_budget - self.overhead_tokens
        kept: list[str] = []
        used = 0.0

        for line in reversed(lines):
            cost = len(line) / self.chars_per_token
            if kept and used + cost > budget:
                break
            kept.append(line)
            used += cost

        kept.reverse()
        return kept
