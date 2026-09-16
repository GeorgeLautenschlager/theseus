from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

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
    measure_source_validity,
    measure_labeled_assertions,
    validate_scenarios,
    run_offline,
    run_live,
    main,
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
        for ep in s.episodes:
            for assertion in ep.assertions:
                assert assertion["support_event_indices"]
                assert all(s.events[i][3] == "evidence" for i in assertion["support_event_indices"])


def test_dataset_rejects_context_only_reference_support():
    original = next(s for s in SCENARIOS if s.name == "raven-code")
    bad_label = {**original.episodes[0].assertions[0], "support_event_indices": (1,)}
    broken = replace(original, episodes=(replace(original.episodes[0], assertions=(bad_label,)),))
    with pytest.raises(ValueError, match="eligible supporting events"):
        validate_scenarios((broken,))


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


def _drive_all(tmp_path):
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    drive_scenarios(memory, log, SCENARIOS, reference=True, extractor=extractor,
                    start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    return memory, log


def test_metrics_all_pass_on_the_reference_run(tmp_path):
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    trace = drive_scenarios(memory, log, SCENARIOS, reference=True, extractor=extractor,
                            start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    updates = measure_correct_updates(memory, SCENARIOS)
    retained = measure_supported_retained(memory, SCENARIOS, budget_tokens=2000)
    unsupported = measure_unsupported_present(memory, SCENARIOS)
    assert updates["total"] > 0 and updates["passed"] == updates["total"]
    assert retained["total"] > 0 and retained["passed"] == retained["total"]
    assert unsupported["total"] > 0 and unsupported["violations"] == 0
    assert measure_source_validity(memory, SCENARIOS, trace)["passed"] > 0
    labeled = measure_labeled_assertions(memory, SCENARIOS, trace)
    assert labeled["passed"] == labeled["total"] > 0


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
    """#66: the correction is worded under a different predicate than the
    original ("delivery date" vs "prototype deadline") — reconciliation, not
    a shared key, is what has to resolve it as the same attribute."""
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    drive_scenarios(memory, log, (_scenario("atlas-deadline"),),
                    reference=True, extractor=extractor,
                    start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert memory.knowledge.current("Atlas", "prototype deadline") == []
    current = memory.knowledge.current("Atlas", "delivery date")
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


def test_valid_source_id_does_not_make_a_false_outcome_semantically_correct(tmp_path):
    class FalseOutcomeExtractor:
        def chat(self, prompt, **kwargs):
            evidence = prompt.split("<evidence>\n", 1)[1].split("\n</evidence>", 1)[0]
            tool_id = json.loads(evidence.splitlines()[1])["id"]
            return json.dumps({"summary": "Atlas payment completed.", "assertions": [{
                "kind": "fact", "subject": "Atlas", "predicate": "payment status",
                "value": "completed", "statement": "Atlas payment completed.",
                "support_event_ids": [tool_id], "attribution": "direct_observation",
                "action_status": "confirmed_outcome",
            }]})

    scenario = next(s for s in SCENARIOS if s.name == "atlas-payment")
    extractor = FalseOutcomeExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    trace = drive_scenarios(memory, log, (scenario,), reference=False, extractor=extractor,
                            start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    validity = measure_source_validity(memory, (scenario,), trace)
    semantics = measure_labeled_assertions(memory, (scenario,), trace)
    assert validity["passed"] == validity["total"] == 1
    assert semantics["passed"] == 0 and semantics["total"] == 1
    assert measure_correct_updates(memory, (scenario,))["passed"] == 0


def test_run_offline_reports_separated_metrics_and_survives_restart(tmp_path):
    report = run_offline(tmp_path)
    assert report["mode"] == "reference-extraction-offline"
    metrics = report["metrics"]
    assert metrics["correct_updates"]["passed"] == metrics["correct_updates"]["total"] > 0
    assert metrics["supported_retained"]["passed"] == metrics["supported_retained"]["total"] > 0
    assert metrics["unsupported_present"]["violations"] == 0
    assert metrics["source_validity"]["passed"] == metrics["source_validity"]["total"] > 0
    assert metrics["labeled_assertions"]["passed"] == metrics["labeled_assertions"]["total"] > 0
    assert report["restart_recall"]["passed"] == report["restart_recall"]["total"] > 0
    assert report["context_departure_recall"]["passed"] == report["context_departure_recall"]["total"] > 0
    assert report["provider"] is None
    assert report["costs"]["answer_chat_calls"] == 0
    assert report["costs"]["embedding_calls_at_consolidation"] == 0
    assert json.loads((tmp_path / "report.json").read_text())["mode"] == report["mode"]


def test_driver_keeps_coexisting_preferences_under_one_broad_predicate(tmp_path):
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    drive_scenarios(memory, log, (_scenario("user-prefs"),),
                    reference=True, extractor=extractor,
                    start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    current = memory.knowledge.current("George", "preference")
    assert {r.value for r in current} == {"morning meetings", "concise written summaries"}
    added = next(r for r in current if r.value == "concise written summaries")
    assert added.reconciliation == "coexist" and added.supersedes is None


def test_driver_keeps_an_unresolved_contradiction_visible_and_attributed(tmp_path):
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    drive_scenarios(memory, log, (_scenario("boreal-conflict"),),
                    reference=True, extractor=extractor,
                    start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    current = {r.reported_by: r for r in memory.knowledge.current("Boreal", "deadline")}
    assert {r.value for r in current.values()} == {"Thursday", "Friday"}
    assert current["Beta"].contradicts == (current["Alpha"].id,)
    assert current["Beta"].reconciliation == "contradiction"


def test_driver_promotes_an_inferred_principle_after_independent_support(tmp_path):
    extractor = ReferenceExtractor()
    memory, log = build_memory(tmp_path, extractor=extractor, embedder=None)
    drive_scenarios(memory, log, (_scenario("brevity-pattern"),),
                    reference=True, extractor=extractor,
                    start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    current = memory.wisdom.current()
    assert len(current) == 1
    assert current[0].status == "established"
    assert len(current[0].supporting_episode_ids) == 2
    assert current[0].reconciliation == "reinforce"


def test_run_offline_confirms_the_decisive_tail_reaches_extraction(tmp_path):
    """#65: oversized evidence is chunked, not head-truncated, so the decisive
    fact at the end of a huge event reaches the model instead of being cut
    away by the extraction budget."""
    oversized = run_offline(tmp_path)["oversized"]
    assert oversized["decisive_tail_in_budget"] is True
    assert oversized["full_event_searchable"] is True


def test_run_offline_refuses_a_nonempty_workdir(tmp_path):
    run_offline(tmp_path)
    with pytest.raises(ValueError, match="empty workdir"):
        run_offline(tmp_path)


class _ScriptedProvider:
    model = "scripted-live"
    last_chat_usage = {"total_tokens": 7}

    def chat(self, prompt, **kwargs):
        if "assertions" in prompt or "consolidation" in prompt:
            event_id = json.loads(prompt.split("<evidence>\n", 1)[1].splitlines()[0])["id"]
            return json.dumps({"summary": "ok", "assertions": [{"kind": "fact", "subject": "Atlas", "predicate": "phase", "value": "production", "statement": "Atlas phase: production.", "support_event_ids": [event_id], "attribution": "inference", "action_status": "not_applicable"}]})
        return json.dumps({"answer": "production", "evidence_ids": ["invented-id"]})


def test_run_live_stamps_mode_and_rejects_unknown_citations(tmp_path):
    report = run_live(tmp_path, extractor=_ScriptedProvider(), answerer=_ScriptedProvider())
    assert report["mode"] == "live-extraction"
    assert report["costs"]["answer_chat_calls"] > 0
    assert report["costs"]["reported_answer_usage"]
    assert any(not answer["valid_citations"] for answer in report["answers"])
    assert "not treated as proof" in report["limitations"]


def test_run_live_survives_a_nonstring_answer(tmp_path):
    class BadAnswerProvider:
        model = "bad-answer"

        def chat(self, prompt, **kwargs):
            if "assertions" in prompt or "consolidation" in prompt:
                return json.dumps({"summary": "ok", "assertions": []})
            return json.dumps({"answer": 42, "evidence_ids": []})

    report = run_live(tmp_path, extractor=BadAnswerProvider(), answerer=BadAnswerProvider())
    assert report["answers"]
    assert all(not a["valid_citations"] for a in report["answers"])


def test_main_offline_writes_a_report(tmp_path, capsys):
    main(["--workdir", str(tmp_path / "run")])
    assert json.loads((tmp_path / "run" / "report.json").read_text())["mode"] == "reference-extraction-offline"
