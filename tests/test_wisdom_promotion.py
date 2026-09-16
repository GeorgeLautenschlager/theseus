"""Provisional-to-established promotion for inferred principles (#68).

A single extracted principle used to go straight to wisdom with no
aggregation across episodes and no distinction between something the agent
was explicitly told and something it merely inferred. Reconciliation (#66)
now covers principle-kind candidates too: it tracks distinct supporting
episodes, and an explicitly attributed preference is established the moment
it's written while an inferred generalization stays visibly provisional
until independent episodes have reinforced it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from theseus.memory_module import Episode, MemoryModule
from theseus.stimulus_log import StimulusLog
from theseus.wisdom_layer import WisdomLayer, WisdomRecord


class ScriptedProvider:
    """A model double that answers extraction and reconciliation calls
    differently, distinguishing them the same way the real prompts do (an
    extraction prompt has <evidence>; a reconciliation prompt has both
    <candidates> and <existing_knowledge>)."""

    def __init__(self, extraction_response=None):
        self.extraction_response = extraction_response
        self.reconciliation_response = None  # None => empty decisions (falls back)
        self.calls = 0
        self.prompts = []

    def chat(self, prompt, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if "<candidates>" in prompt and "<existing_knowledge>" in prompt:
            return self.reconciliation_response or '{"decisions": []}'
        return self.extraction_response


def principle(statement, *, attribution="inference", reported_by=None):
    assertion = {"kind": "principle", "statement": statement,
                "support_event_ids": ["<EVIDENCE_ID>"], "attribution": attribution,
                "action_status": "not_applicable"}
    if reported_by is not None:
        assertion["reported_by"] = reported_by
    return assertion


def extraction_response(summary, *assertions):
    return json.dumps({"summary": summary, "assertions": list(assertions)})


def consolidate_principle(module, log, episode_id, actor, message, provider, response,
                          reconciliation_response=None):
    event = log.append(actor, "chat_message", {"message": message})
    provider.extraction_response = response.replace("<EVIDENCE_ID>", event.id)
    provider.reconciliation_response = (
        reconciliation_response.replace("<EVIDENCE_ID>", event.id) if reconciliation_response else None
    )
    return module.consolidate(Episode(episode_id, event.id, event.id)), event


def setup(tmp_path):
    log = StimulusLog(tmp_path / "log.jsonl")
    provider = ScriptedProvider()
    module = MemoryModule(tmp_path / "memory", log, model_providers=[provider])
    return module, log, provider


def test_explicit_preference_is_established_immediately(tmp_path):
    """Criterion: explicit preferences don't need repeated observation."""
    module, log, provider = setup(tmp_path)
    consolidate_principle(
        module, log, "ep1", "human", "I always want dark mode.", provider,
        extraction_response("Preference stated.",
                            principle("The user prefers dark mode.", attribution="partner_report",
                                     reported_by="human")),
    )
    current = module.wisdom.current()
    assert len(current) == 1
    assert current[0].status == "established"
    assert current[0].supporting_episode_ids == ("ep1",)
    assert "(provisional)" not in current[0].render()


def test_single_inferred_observation_stays_provisional(tmp_path):
    """Criterion: new inferred generalizations remain visibly provisional."""
    module, log, provider = setup(tmp_path)
    consolidate_principle(
        module, log, "ep1", "agent", "noted a pattern", provider,
        extraction_response("Pattern noted.",
                            principle("The user seems to prefer terse replies.", attribution="inference")),
    )
    current = module.wisdom.current()
    assert len(current) == 1
    assert current[0].status == "provisional"
    assert current[0].evidence_count == 1
    assert "(provisional)" in current[0].render()


def test_independent_repeated_support_promotes_to_established(tmp_path):
    """Criterion: independent supporting episodes are tracked and drive promotion."""
    module, log, provider = setup(tmp_path)
    consolidate_principle(
        module, log, "ep1", "agent", "first observation", provider,
        extraction_response("Noted.", principle("The user seems to prefer terse replies.",
                                                 attribution="inference")),
    )
    provisional = module.wisdom.current()[0]
    assert provisional.status == "provisional"

    consolidate_principle(
        module, log, "ep2", "agent", "second observation", provider,
        extraction_response("Noted again.", principle(
            "User replies suggest a preference for terse answers.", attribution="inference")),
        reconciliation_response=json.dumps({
            "decisions": [{"candidate_index": 0, "decision": "reinforce", "target_id": provisional.id}],
        }),
    )
    current = module.wisdom.current()
    assert len(current) == 1
    assert current[0].status == "established"
    assert set(current[0].supporting_episode_ids) == {"ep1", "ep2"}
    assert current[0].evidence_count == 2
    assert current[0].supersedes == provisional.id
    # The original provisional record is still on file, just no longer current.
    assert {r.id for r in module.wisdom.read_all()} == {provisional.id, current[0].id}


def test_duplicate_extraction_within_one_episode_does_not_inflate_support(tmp_path):
    """Criterion: duplicate extraction cannot increase support. Two differently
    worded principle candidates in the SAME episode both reconciling to the
    same existing provisional principle must count as one episode of support,
    not two, and must not leave two competing "current" reinforcements."""
    module, log, provider = setup(tmp_path)
    consolidate_principle(
        module, log, "ep1", "agent", "first observation", provider,
        extraction_response("Noted.", principle("The user seems to prefer terse replies.",
                                                 attribution="inference")),
    )
    provisional = module.wisdom.current()[0]

    event = log.append("agent", "chat_message", {"message": "second observation, said twice"})
    provider.extraction_response = extraction_response(
        "Noted twice.",
        principle("User replies suggest brevity is preferred.", attribution="inference"),
        principle("The user favors short answers.", attribution="inference"),
    ).replace("<EVIDENCE_ID>", event.id)
    provider.reconciliation_response = json.dumps({"decisions": [
        {"candidate_index": 0, "decision": "reinforce", "target_id": provisional.id},
        {"candidate_index": 1, "decision": "reinforce", "target_id": provisional.id},
    ]})
    module.consolidate(Episode("ep2", event.id, event.id))

    current = module.wisdom.current()
    assert len(current) == 1  # not two competing "current" reinforcements
    assert set(current[0].supporting_episode_ids) == {"ep1", "ep2"}
    assert current[0].status == "established"


def test_retrying_a_failed_episode_does_not_double_count_support(tmp_path):
    """Criterion: retries cannot increase support — consolidate() is already
    idempotent per episode_id, so a genuine retry never re-runs reconciliation
    for an episode that already landed."""
    module, log, provider = setup(tmp_path)
    consolidate_principle(
        module, log, "ep1", "agent", "first observation", provider,
        extraction_response("Noted.", principle("The user seems to prefer terse replies.",
                                                 attribution="inference")),
    )
    provisional = module.wisdom.current()[0]
    result, event = consolidate_principle(
        module, log, "ep2", "agent", "second observation", provider,
        extraction_response("Noted again.", principle("Terse replies preferred.", attribution="inference")),
        reconciliation_response=json.dumps({
            "decisions": [{"candidate_index": 0, "decision": "reinforce", "target_id": provisional.id}],
        }),
    )
    assert not result.skipped
    established = module.wisdom.current()[0]
    assert established.status == "established"

    retry = module.consolidate(Episode("ep2", event.id, event.id))
    assert retry.skipped
    assert module.wisdom.current() == [established]  # unchanged, no new record


def test_unresolved_contradiction_stays_visible_not_unqualified_established(tmp_path):
    """Criterion: contradictory evidence is retained and a disputed principle
    is never presented as an unqualified established rule."""
    module, log, provider = setup(tmp_path)
    consolidate_principle(
        module, log, "ep1", "human", "Always confirm risky actions before executing.", provider,
        extraction_response("Rule stated.", principle("Always confirm risky actions before executing.",
                                                       attribution="partner_report", reported_by="human")),
    )
    established = module.wisdom.current()[0]
    assert established.status == "established"

    consolidate_principle(
        module, log, "ep2", "human", "Actually, never ask — just proceed with risky actions.", provider,
        extraction_response("Conflicting rule stated.", principle(
            "Never ask before risky actions — just proceed.", attribution="partner_report",
            reported_by="human")),
        reconciliation_response=json.dumps({
            "decisions": [{"candidate_index": 0, "decision": "contradiction", "target_id": established.id}],
        }),
    )
    current = module.wisdom.current()
    assert len(current) == 2  # both stay visible; neither silently wins
    disputed = next(r for r in current if r.contradicts)
    assert disputed.contradicts == (established.id,)
    assert f"contradicts {established.id}" in disputed.render()
    # Still "established" by attribution, but visibly disputed — not unqualified.
    assert disputed.status == "established"
    assert "[contradicts" in disputed.render()


def test_status_and_support_persist_across_restart(tmp_path):
    module, log, provider = setup(tmp_path)
    consolidate_principle(
        module, log, "ep1", "agent", "first observation", provider,
        extraction_response("Noted.", principle("The user seems to prefer terse replies.",
                                                 attribution="inference")),
    )
    provisional = module.wisdom.current()[0]
    consolidate_principle(
        module, log, "ep2", "agent", "second observation", provider,
        extraction_response("Noted again.", principle("Terse replies preferred.", attribution="inference")),
        reconciliation_response=json.dumps({
            "decisions": [{"candidate_index": 0, "decision": "reinforce", "target_id": provisional.id}],
        }),
    )
    established = module.wisdom.current()[0]

    reopened = MemoryModule(module.memory_dir, log, model_providers=[provider])
    current = reopened.wisdom.current()
    assert len(current) == 1
    assert current[0].id == established.id
    assert current[0].status == "established"
    assert set(current[0].supporting_episode_ids) == {"ep1", "ep2"}


def test_legacy_record_without_status_reads_without_invented_evidence():
    """Criterion: old records remain readable without invented evidence."""
    old = {"id": "old", "ts": "2026-01-01T00:00:00+00:00", "statement": "Legacy principle."}
    record = WisdomRecord.from_json(json.dumps(old))
    assert record.status is None
    assert record.supporting_episode_ids == ()
    assert record.supersedes is None
    assert record.contradicts is None
    assert "(provisional)" not in record.render()
    assert "[contradicts" not in record.render()


def test_legacy_record_is_current_and_readable_in_a_fresh_layer(tmp_path):
    layer = WisdomLayer(tmp_path / "wisdom.jsonl")
    layer.add(WisdomRecord("old", datetime.now(timezone.utc), "Legacy principle."))
    reopened = WisdomLayer(tmp_path / "wisdom.jsonl")
    assert [r.id for r in reopened.current()] == ["old"]
