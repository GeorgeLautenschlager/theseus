from __future__ import annotations

import json

import pytest

from theseus.memory_accuracy_eval import (
    SCENARIOS,
    REQUIRED_CATEGORIES,
    EpisodeSpec,
    Scenario,
    Transition,
    validate_scenarios,
)


def test_dataset_is_well_formed_and_covers_every_category():
    validate_scenarios(SCENARIOS)
    categories = {s.category for s in SCENARIOS}
    assert REQUIRED_CATEGORIES <= categories
    for s in SCENARIOS:
        assert len(s.events) >= 2 or len(s.episodes) >= 2, s.name
    for s in SCENARIOS:
        covered = [i for ep in s.episodes for i in ep.event_indices]
        assert sorted(covered) == list(range(len(s.events))), s.name


def test_validator_rejects_a_transition_without_a_supporting_fact():
    broken = Scenario(
        name="broken", category="plan_then_failure",
        events=(("human", "chat_message", {"message": "x"}, "evidence"),
                ("human", "chat_message", {"message": "y"}, "evidence")),
        episodes=(EpisodeSpec((0, 1), "summary", ()),),
        transitions=(Transition("Atlas", "payment status", "failed"),),
    )
    with pytest.raises(ValueError, match="transition"):
        validate_scenarios((broken,))


def test_validator_rejects_noncontiguous_or_out_of_range_episode():
    broken = Scenario(
        name="broken2", category="plan_then_failure",
        events=(("human", "chat_message", {"message": "x"}, "evidence"),),
        episodes=(EpisodeSpec((0, 5), "summary", ()),),
    )
    with pytest.raises(ValueError):
        validate_scenarios((broken,))
