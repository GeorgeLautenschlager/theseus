from __future__ import annotations

import inspect
import json

from theseus.context_assembler import (
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    ContextAssembler,
)
from theseus.stimulus_log import StimulusLog


def fill_log(tmp_path, n, message=None) -> StimulusLog:
    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    for i in range(n):
        body = f"msg {i}" if message is None else f"msg {i} {message}"
        log.append(actor="george", type="exchange", content={"message": body})
    return log


def event_tokens(log, chars_per_token=DEFAULT_CHARS_PER_TOKEN) -> float:
    """Estimated cost of one event in `log`, by the same rule the assembler uses.

    Tests derive budgets from this rather than hardcoding token counts, so they keep
    meaning if the event serialization grows a field.
    """
    return len(log.read_all()[-1].to_json()) / chars_per_token


class TestRecentEventsWindow:
    def test_includes_all_events_when_under_window(self, tmp_path):
        log = fill_log(tmp_path, 3)
        assembled = ContextAssembler(stimulus_log=log, window_size=50).assemble_context()

        assert assembled.recent_events.count("\n") == 2
        assert "msg 0" in assembled.recent_events
        assert "msg 2" in assembled.recent_events

    def test_truncates_to_most_recent_window(self, tmp_path):
        log = fill_log(tmp_path, 5)
        assembled = ContextAssembler(stimulus_log=log, window_size=2).assemble_context()

        assert "msg 2" not in assembled.recent_events
        assert "msg 3" in assembled.recent_events
        assert "msg 4" in assembled.recent_events


class TestTokenBudget:
    """The window's real constraint is the model's context, not an event count.

    A count is arbitrary in the wrong unit — fifty events is a couple of thousand tokens
    of chat or a hundred thousand tokens of file contents, depending entirely on what
    landed in the log. The budget is expressed in tokens; `window_size` stays on as a
    hard cap so assembly is bounded before any measurement exists.
    """

    def test_budget_trims_before_the_count_cap(self, tmp_path):
        log = fill_log(tmp_path, 10, message="x" * 400)
        # Room for three events and change; the count cap is nowhere near binding.
        budget = int(event_tokens(log) * 3.5)

        assembled = ContextAssembler(
            stimulus_log=log, window_size=200, token_budget=budget
        ).assemble_context()

        assert assembled.recent_events.count("\n") + 1 == 3
        assert "msg 9" in assembled.recent_events
        assert "msg 6" not in assembled.recent_events

    def test_count_cap_still_binds_when_events_are_small(self, tmp_path):
        log = fill_log(tmp_path, 10)

        assembled = ContextAssembler(
            stimulus_log=log, window_size=2, token_budget=100_000
        ).assemble_context()

        assert assembled.recent_events.count("\n") + 1 == 2
        assert "msg 9" in assembled.recent_events

    def test_single_oversized_event_is_still_emitted(self, tmp_path):
        # An empty window is strictly worse than an over-budget one: the model would be
        # asked to decide with no stimulus at all.
        log = fill_log(tmp_path, 3, message="x" * 4000)

        assembled = ContextAssembler(
            stimulus_log=log, window_size=200, token_budget=1
        ).assemble_context()

        assert "msg 2" in assembled.recent_events
        assert "msg 1" not in assembled.recent_events

    def test_no_budget_is_pure_count_behaviour(self, tmp_path):
        log = fill_log(tmp_path, 10, message="x" * 4000)

        assembled = ContextAssembler(
            stimulus_log=log, window_size=4, token_budget=None
        ).assemble_context()

        assert assembled.recent_events.count("\n") + 1 == 4

    def test_window_chars_reports_the_assembled_size(self, tmp_path):
        log = fill_log(tmp_path, 5)

        assembled = ContextAssembler(stimulus_log=log).assemble_context()

        assert assembled.window_chars == len(assembled.recent_events)


class TestCalibration:
    """Closed loop: the assembler predicts, the backend measures, the prediction corrects.

    No OpenAI-compatible endpoint reports the context window it actually loaded, but every
    one of them reports `usage.prompt_tokens`. So the ratio is never guessed twice.
    """

    def test_observe_corrects_chars_per_token(self, tmp_path):
        log = fill_log(tmp_path, 1)
        assembler = ContextAssembler(stimulus_log=log, chars_per_token=4.0)

        # 900 chars actually cost 300 tokens -> 3.0 chars per token, not the 4.0 seed.
        assembler.observe(prompt_tokens=300, prompt_chars=900, window_chars=900)

        assert assembler.chars_per_token == 3.0

    def test_observe_derives_prompt_overhead(self, tmp_path):
        log = fill_log(tmp_path, 1)
        assembler = ContextAssembler(stimulus_log=log)

        # 900 chars at 300 tokens, of which only 300 chars were the window: the other
        # 600 chars (200 tokens) are constitution + persona + tool schemas.
        assembler.observe(prompt_tokens=300, prompt_chars=900, window_chars=300)

        assert assembler.overhead_tokens == 200

    def test_derived_overhead_shrinks_the_window_budget(self, tmp_path):
        log = fill_log(tmp_path, 10, message="x" * 400)
        per_event = event_tokens(log)
        assembler = ContextAssembler(
            stimulus_log=log, window_size=200, token_budget=int(per_event * 5.5)
        )

        assert assembler.assemble_context().recent_events.count("\n") + 1 == 5

        # Two events' worth of fixed overhead now has to come out of the same budget.
        overhead_chars = int(per_event * 2 * DEFAULT_CHARS_PER_TOKEN)
        assembler.observe(
            prompt_tokens=int(overhead_chars / DEFAULT_CHARS_PER_TOKEN),
            prompt_chars=overhead_chars,
            window_chars=0,
        )

        assert assembler.assemble_context().recent_events.count("\n") + 1 == 3

    def test_observe_ignores_a_provider_that_reports_no_usage(self, tmp_path):
        # ClaudeProvider shells out to the CLI and gets no usage back; the seed estimate
        # must survive rather than blow up on a division by zero.
        log = fill_log(tmp_path, 1)
        assembler = ContextAssembler(stimulus_log=log, chars_per_token=4.0)

        assembler.observe(prompt_tokens=None, prompt_chars=900, window_chars=300)

        assert assembler.chars_per_token == 4.0
        assert assembler.overhead_tokens == 0


class TestNoInvoluntaryRetrieval:
    """The assembler no longer reaches into memory behind the agent's back. Recall is a
    tool the agent calls; its result reaches the next prompt through the stimulus log
    like any other tool_result, so there is nothing for the assembler to do."""

    def test_assembler_takes_no_memory_or_retrieval_parameters(self):
        # `constitution` used to sit here unread — the same dead-parameter trap that hid
        # persona from the system prompt. The assembler assembles the window and sizes it;
        # nothing else. Deliberately not an equality check on the whole signature: the
        # guard is against retrieval creeping back in, not against the window gaining
        # knobs for how big it should be.
        params = set(inspect.signature(ContextAssembler.__init__).parameters) - {"self"}
        assert params.isdisjoint({"memory", "retrieval_query_chars", "embedding_provider"})
        assert "stimulus_log" in params

    def test_assembled_context_carries_only_the_event_window(self, tmp_path):
        log = fill_log(tmp_path, 1)

        assembled = ContextAssembler(stimulus_log=log).assemble_context()

        assert not hasattr(assembled, "memories")
        assert "msg 0" in assembled.recent_events

    def test_empty_log_assembles_empty_context(self, tmp_path):
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")

        assert ContextAssembler(stimulus_log=log).assemble_context().recent_events == ""


class TestContextLimitBudget:
    """The budget is derived from the model's declared window, not hand-set per agent.

    The bug this replaces: a 120k budget hardcoded for Claude stayed in force after
    CADENCE.md started routing the same agent to a 128k local model. The whole log fit
    the budget, so nothing trimmed it, and every turn overran.
    """

    def test_context_limit_reserves_room_for_the_completion(self, tmp_path):
        log = fill_log(tmp_path, 1)
        assembler = ContextAssembler(stimulus_log=log, context_tokens=128_000)

        budget = assembler._budget_tokens()

        # The window holds prompt *and* completion; the prompt may not have all of it.
        assert budget < 128_000 - DEFAULT_RESERVED_OUTPUT_TOKENS
        assert budget > 100_000

    def test_context_limit_overrides_the_conservative_default(self, tmp_path):
        log = fill_log(tmp_path, 200, message="x" * 400)
        assembler = ContextAssembler(stimulus_log=log, window_size=500)

        small = assembler.assemble_context().recent_events.count("\n")
        assembler.set_context_limit(200_000)
        large = assembler.assemble_context().recent_events.count("\n")

        assert large > small

    def test_set_context_limit_ignores_none(self, tmp_path):
        """A cadence rule that declares no `context` must leave the budget where it was,
        not inherit whatever the previous rule happened to be."""
        log = fill_log(tmp_path, 1)
        assembler = ContextAssembler(stimulus_log=log, context_tokens=128_000)

        assembler.set_context_limit(None)

        assert assembler.context_tokens == 128_000

    def test_measured_overhead_shrinks_the_window_on_the_very_first_turn(self, tmp_path):
        # Enough history to overflow the budget either way, so the only thing separating
        # the two assemblies is whether the overhead was accounted for.
        log = fill_log(tmp_path, 200, message="x" * 400)
        assembler = ContextAssembler(
            stimulus_log=log, window_size=500, context_tokens=32_000
        )

        # No observe() has ever run: the inferred overhead is 0.0 and always will be
        # under a provider that reports no usage. Passing it explicitly is what keeps
        # the first turn honest.
        uninformed = assembler.assemble_context().recent_events.count("\n")
        informed = assembler.assemble_context(overhead_chars=40_000).recent_events.count("\n")

        assert informed < uninformed

    def test_a_declared_window_actually_fits_a_real_sized_log(self, tmp_path):
        """End-to-end shape of the original failure, in miniature."""
        log = fill_log(tmp_path, 300, message="x" * 1200)
        assembler = ContextAssembler(
            stimulus_log=log, window_size=1000, context_tokens=32_000
        )

        assembled = assembler.assemble_context(overhead_chars=8_000)

        assert len(assembled.recent_events) / DEFAULT_CHARS_PER_TOKEN < 32_000
        assert assembled.recent_events  # but not empty


class TestOversizedEventClamp:
    """One event must not be able to eat the window.

    `_fit_to_budget` guarantees at least one event, which is right — an empty window asks
    the model to decide with no stimulus. But the guarantee is only affordable if a single
    event is bounded: a `read` result carrying a whole file is one event.
    """

    def test_huge_event_is_truncated_and_marked(self, tmp_path):
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        log.append(actor="tam", type="tool_result",
                   content={"tool": "read", "output": "y" * 200_000})

        assembled = ContextAssembler(
            stimulus_log=log, context_tokens=32_000
        ).assemble_context()

        assert "truncated" in assembled.recent_events
        assert len(assembled.recent_events) < 200_000

    def test_truncated_event_is_still_valid_json_with_its_metadata(self, tmp_path):
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        log.append(actor="tam", type="tool_result",
                   content={"tool": "read", "output": "y" * 200_000, "is_error": False})

        line = ContextAssembler(
            stimulus_log=log, context_tokens=32_000
        ).assemble_context().recent_events

        event = json.loads(line)
        assert event["actor"] == "tam"
        assert event["type"] == "tool_result"
        assert event["content"]["tool"] == "read"      # small fields survive intact
        assert event["content"]["is_error"] is False
        assert "truncated" in event["content"]["output"]

    def test_the_log_itself_is_never_modified(self, tmp_path):
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        log.append(actor="tam", type="tool_result",
                   content={"tool": "read", "output": "y" * 200_000})

        ContextAssembler(stimulus_log=log, context_tokens=32_000).assemble_context()

        assert len(log.read_all()[0].content["output"]) == 200_000

    def test_one_huge_event_does_not_crowd_out_recent_history(self, tmp_path):
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        log.append(actor="tam", type="tool_result",
                   content={"tool": "read", "output": "y" * 400_000})
        for i in range(5):
            log.append(actor="george", type="exchange", content={"message": f"msg {i}"})

        assembled = ContextAssembler(
            stimulus_log=log, context_tokens=32_000
        ).assemble_context()

        for i in range(5):
            assert f"msg {i}" in assembled.recent_events

    def test_clamp_never_shaves_an_event_to_nothing(self, tmp_path):
        # Budget of 1 token: without the floor the guaranteed event would be trimmed
        # away entirely, which is the empty window the guarantee exists to prevent.
        log = fill_log(tmp_path, 3, message="x" * 4000)

        assembled = ContextAssembler(
            stimulus_log=log, window_size=200, token_budget=1
        ).assemble_context()

        assert "msg 2" in assembled.recent_events

    def test_trims_the_bulk_not_the_reasoning_on_a_decision_event(self, tmp_path):
        """A `decision` keeps its bulk nested in `tool_calls[].arguments`, not at the top
        level. A top-level-only scan cut the short `text` — the agent's stated reasoning —
        and left the actual payload untouched, so the line stayed over budget too."""
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        log.append(actor="tam", type="decision", content={
            "text": "Writing the file now.",
            "tool_calls": [{"name": "write", "arguments": {"content": "z" * 200_000}}],
        })

        line = ContextAssembler(
            stimulus_log=log, context_tokens=32_000
        ).assemble_context().recent_events
        event = json.loads(line)

        assert event["content"]["text"] == "Writing the file now."   # reasoning intact
        assert "truncated" in event["content"]["tool_calls"][0]["arguments"]["content"]
        assert len(line) < 200_000

    def test_many_medium_strings_are_trimmed_until_the_line_fits(self, tmp_path):
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        log.append(actor="tam", type="decision", content={
            "tool_calls": [
                {"name": "write", "arguments": {"content": "z" * 30_000}}
                for _ in range(6)
            ],
        })

        line = ContextAssembler(
            stimulus_log=log, context_tokens=16_000
        ).assemble_context().recent_events

        assert json.loads(line)["type"] == "decision"
        assert len(line) < 30_000
