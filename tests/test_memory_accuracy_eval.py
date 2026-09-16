from __future__ import annotations

import json

import pytest

from theseus.memory_accuracy_eval import (
    SCENARIOS,
    REQUIRED_CATEGORIES,
    EpisodeSpec,
    Scenario,
    Transition,
    ReferenceExtractor,
    build_memory,
    drive_scenarios,
    measure_correct_updates,
    measure_supported_retained,
    measure_unsupported_present,
    validate_scenarios,
    run_offline,
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


from datetime import datetime, timezone



def _drive_all(tmp_path):
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    drive_scenarios(memory, log, SCENARIOS, reference=True, extractor=extractor,
                    start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    return memory, log


def test_metrics_all_pass_on_the_reference_run(tmp_path):
    memory, _ = _drive_all(tmp_path)
    updates = measure_correct_updates(memory, SCENARIOS)
    retained = measure_supported_retained(memory, SCENARIOS, budget_tokens=2000)
    unsupported = measure_unsupported_present(memory, SCENARIOS)
    assert updates["total"] > 0 and updates["passed"] == updates["total"]
    assert retained["total"] > 0 and retained["passed"] == retained["total"]
    assert unsupported["total"] > 0 and unsupported["violations"] == 0


def test_unsupported_metric_flags_a_leaked_current_fact(tmp_path):
    from theseus.knowledge_layer import KnowledgeRecord
    memory, _ = _drive_all(tmp_path)
    memory.knowledge.add(KnowledgeRecord(
        "leak", datetime(2030, 1, 1, tzinfo=timezone.utc), "Atlas", "phase", "pilot"))
    unsupported = measure_unsupported_present(memory, SCENARIOS)
    assert unsupported["violations"] >= 1
    assert any("pilot" in v["fragment"] for v in unsupported["failures"])


def _scenario(name):
    return next(s for s in SCENARIOS if s.name == name)


def test_driver_applies_correction_across_episodes(tmp_path):
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    drive_scenarios(memory, log, (_scenario("atlas-deadline"),),
                    reference=True, extractor=extractor,
                    start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    current = memory.knowledge.current("Atlas", "prototype deadline")
    assert current and current[0].value == "Monday"
    assert all(r.value != "Friday" for r in memory.knowledge.current())


def test_driver_never_promotes_recall_context_to_a_fact(tmp_path):
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    trace = drive_scenarios(memory, log, (_scenario("raven-code"),),
                            reference=True, extractor=extractor,
                            start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    values = [r.value for r in memory.knowledge.current()]
    assert "RAVEN-42" in values
    assert "QUAIL-7" not in values
    prompt = next(iter(trace["raven-code"]["episode_prompts"].values()))
    assert "QUAIL-7" in prompt


def test_run_offline_reports_separated_metrics_and_survives_restart(tmp_path):
    report = run_offline(tmp_path)
    assert report["mode"] == "reference-extraction-offline"
    metrics = report["metrics"]
    assert metrics["correct_updates"]["passed"] == metrics["correct_updates"]["total"] > 0
    assert metrics["supported_retained"]["passed"] == metrics["supported_retained"]["total"] > 0
    assert metrics["unsupported_present"]["violations"] == 0
    assert report["restart_recall"]["passed"] == report["restart_recall"]["total"] > 0
    assert report["context_departure_recall"]["passed"] == report["context_departure_recall"]["total"] > 0
    assert report["provider"] is None
    assert report["costs"]["answer_chat_calls"] == 0
    assert report["costs"]["embedding_calls_at_consolidation"] == 0
    assert json.loads((tmp_path / "report.json").read_text())["mode"] == report["mode"]


def test_run_offline_detects_the_dropped_decisive_tail(tmp_path):
    oversized = run_offline(tmp_path)["oversized"]
    assert oversized["decisive_tail_in_budget"] is False
    assert oversized["full_event_searchable"] is True


def test_run_offline_refuses_a_nonempty_workdir(tmp_path):
    run_offline(tmp_path)
    with pytest.raises(ValueError, match="empty workdir"):
        run_offline(tmp_path)
