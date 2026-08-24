from __future__ import annotations

from datetime import datetime, time

from theseus.cadence import DEFAULT_TICK_SECONDS, Cadence, CadenceRule
from theseus.model_providers import PROVIDER_REGISTRY
from theseus.model_providers.claude_provider import ClaudeProvider
from theseus.model_providers.llama_cpp_provider import LlamaCppProvider
from theseus.model_providers.lm_studio_provider import LmStudioProvider
from theseus.model_providers.ollama_provider import OllamaProvider
from theseus.model_providers.openrouter_provider import OpenRouterProvider
from theseus.model_providers.unsloth_provider import UnslothProvider

EXAMPLE = """\
# Cadence

Overnight the GPU box is free, so run the big model and think slowly.

- 08:00-22:00: ollama gemma3:12b, tick every 2 minutes
- 22:00-08:00: lm_studio qwen/qwen3-32b, tick every 15 minutes
- default: claude claude-sonnet-4-6, tick every 5 minutes
"""


def at(hour, minute=0):
    return datetime(2026, 8, 23, hour, minute)


def test_parses_example_file():
    cadence = Cadence.parse(EXAMPLE)
    assert cadence.rules == (
        CadenceRule("ollama", "gemma3:12b", 120, time(8, 0), time(22, 0)),
        CadenceRule("lm_studio", "qwen/qwen3-32b", 900, time(22, 0), time(8, 0)),
        CadenceRule("claude", "claude-sonnet-4-6", 300, None, None),
    )


def test_tick_defaults_when_omitted():
    cadence = Cadence.parse("- 08:00-22:00: ollama gemma3\n")
    assert cadence.rules[0].tick_seconds == DEFAULT_TICK_SECONDS


def test_tick_units_singular_and_plural():
    cadence = Cadence.parse(
        "- 00:00-01:00: ollama a, tick every 45 seconds\n"
        "- 01:00-02:00: ollama b, tick every 1 second\n"
        "- 02:00-03:00: ollama c, tick every 1 minute\n"
        "- 03:00-04:00: ollama d, tick every 2 hours\n"
        "- 04:00-05:00: ollama e, tick every 1 hour\n"
    )
    assert [rule.tick_seconds for rule in cadence.rules] == [45, 1, 60, 7200, 3600]


def test_window_boundaries_start_inclusive_end_exclusive():
    cadence = Cadence.parse(EXAMPLE)
    assert cadence.rule_for(at(8, 0)).provider_key == "ollama"
    assert cadence.rule_for(at(21, 59)).provider_key == "ollama"
    assert cadence.rule_for(at(22, 0)).provider_key == "lm_studio"


def test_midnight_wrap():
    cadence = Cadence.parse(EXAMPLE)
    assert cadence.rule_for(at(23, 0)).provider_key == "lm_studio"
    assert cadence.rule_for(at(7, 59)).provider_key == "lm_studio"
    assert cadence.rule_for(at(8, 0)).provider_key == "ollama"


def test_equal_start_and_end_matches_all_day():
    cadence = Cadence.parse("- 09:00-09:00: ollama allday\n")
    assert cadence.rule_for(at(3, 0)).model == "allday"
    assert cadence.rule_for(at(15, 0)).model == "allday"


def test_overlapping_windows_first_match_wins_and_candidates_ordered():
    cadence = Cadence.parse(
        "- 08:00-12:00: ollama first\n"
        "- 06:00-14:00: lm_studio second\n"
        "- default: claude fallback\n"
    )
    assert cadence.rule_for(at(9, 0)).model == "first"
    assert [rule.model for rule in cadence.candidates_for(at(9, 0))] == [
        "first",
        "second",
        "fallback",
    ]


def test_no_matching_window_falls_back_to_default():
    cadence = Cadence.parse(EXAMPLE)
    only_default = Cadence.parse("- default: claude claude-sonnet-4-6\n")
    assert only_default.rule_for(at(9, 0)).provider_key == "claude"
    assert cadence.candidates_for(at(23, 0))[-1].provider_key == "claude"


def test_no_default_and_no_match():
    cadence = Cadence.parse("- 08:00-09:00: ollama gemma3\n")
    assert cadence.rule_for(at(12, 0)) is None
    assert cadence.candidates_for(at(12, 0)) == []


def test_empty_or_prose_only_file():
    cadence = Cadence.parse("just thoughts, no rules\n")
    assert cadence.rules == ()
    assert cadence.rule_for(at(9, 0)) is None


def test_lint_flags_rule_looking_lines_only():
    cadence = Cadence.parse(
        "# Cadence\n"
        "prose without a colon stays prose\n"
        "- just a prose bullet without colon\n"
        "- mornings: use the small model\n"
        "- 8:00-22:00: ollama gemma3\n"
        "- 25:00-26:00: ollama gemma3\n"
        "- 08:00-22:00: ollama gemma3\n"
        "- default: claude claude-sonnet-4-6\n"
    )
    assert cadence.lint_lines() == [
        "- mornings: use the small model",
        "- 8:00-22:00: ollama gemma3",
        "- 25:00-26:00: ollama gemma3",
    ]


def test_provider_registry_maps_short_names():
    assert PROVIDER_REGISTRY == {
        "claude": ClaudeProvider,
        "llama_cpp": LlamaCppProvider,
        "lm_studio": LmStudioProvider,
        "ollama": OllamaProvider,
        "openrouter": OpenRouterProvider,
        "unsloth": UnslothProvider,
    }
