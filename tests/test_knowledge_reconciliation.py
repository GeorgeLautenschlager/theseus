"""Reconciliation of newly extracted facts against existing knowledge (#66).

KnowledgeLayer itself no longer decides supersession by matching subject+
predicate keys and picking the newest timestamp — that let an alternate-
worded correction evade the fact it should have updated, let two independent
claims under one broad predicate clobber each other, and let a later-
processed but temporally-stale report silently win. Reconciliation is a
separate step in consolidate() that judges each new fact against a bounded
set of the subject's existing current knowledge and decides whether it
reinforces, replaces, coexists beside, contradicts, or is merely historical
relative to what's already known.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from theseus.memory_module import Episode, MemoryModule
from theseus.stimulus_log import StimulusLog


class ScriptedProvider:
    """A model double that answers extraction and reconciliation calls
    differently, distinguishing them the same way the real prompts do (an
    extraction prompt has <evidence>; a reconciliation prompt has both
    <candidates> and <existing_knowledge>)."""

    def __init__(self, extraction_response=None):
        self.extraction_response = extraction_response
        self.reconciliation_response = None  # None => explicit new-fact decisions
        self.calls = 0
        self.prompts = []

    def chat(self, prompt, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if "<candidates>" in prompt and "<existing_knowledge>" in prompt:
            return self.reconciliation_response or json.dumps({"decisions": [
                {"candidate_index": i, "decision": "new", "target_id": None}
                for i, _ in enumerate(json.loads(self.extraction_response)["assertions"])
            ]})
        return self.extraction_response


def fact(subject, predicate, value, *, statement=None, attribution="direct_observation",
        reported_by=None, action_status="not_applicable"):
    assertion = {"kind": "fact", "subject": subject, "predicate": predicate, "value": value,
                "statement": statement or f"{subject} {predicate}: {value}",
                "support_event_ids": ["<EVIDENCE_ID>"], "attribution": attribution,
                "action_status": action_status}
    if reported_by is not None:
        assertion["reported_by"] = reported_by
    return assertion


def extraction_response(summary, *assertions):
    return json.dumps({"summary": summary, "assertions": list(assertions)})


def consolidate_fact(module, log, episode_id, actor, message, provider, response,
                     reconciliation_response=None):
    """Append one event, script the extraction (and optionally reconciliation)
    response, and consolidate it as its own episode. Substitutes the literal
    placeholder "<EVIDENCE_ID>" in both responses with the new event's real id,
    the same trick the module-level fixtures use elsewhere in this suite."""
    event = log.append(actor, "chat_message", {"message": message})
    provider.extraction_response = response.replace("<EVIDENCE_ID>", event.id)
    if reconciliation_response is not None:
        provider.reconciliation_response = reconciliation_response.replace("<EVIDENCE_ID>", event.id)
    else:
        provider.reconciliation_response = None
    return module.consolidate(Episode(episode_id, event.id, event.id)), event


def setup(tmp_path):
    log = StimulusLog(tmp_path / "log.jsonl")
    provider = ScriptedProvider()
    module = MemoryModule(tmp_path / "memory", log, model_providers=[provider])
    return module, log, provider


def test_alternate_wording_correction_updates_the_intended_fact(tmp_path):
    """A correction phrased with a different predicate than the original still
    replaces it — reconciliation resolves this by subject and attribute
    meaning, not by matching the predicate string."""
    module, log, provider = setup(tmp_path)
    consolidate_fact(
        module, log, "ep1", "human", "Atlas prototype deadline is Friday.", provider,
        extraction_response("Deadline set.", fact("Atlas", "prototype deadline", "Friday")),
    )
    original = module.knowledge.current(subject="Atlas")[0]

    consolidate_fact(
        module, log, "ep2", "human", "Atlas delivery date moved to Monday.", provider,
        extraction_response("Deadline corrected.", fact("Atlas", "delivery date", "Monday")),
        reconciliation_response=json.dumps({
            "decisions": [{"candidate_index": 0, "decision": "replace", "target_id": original.id}],
        }),
    )

    assert module.knowledge.current(subject="Atlas", predicate="prototype deadline") == []
    current = module.knowledge.current(subject="Atlas")
    assert [r.value for r in current] == ["Monday"]
    assert current[0].reconciliation == "replace"
    assert current[0].supersedes == original.id


def test_coexisting_preferences_survive_a_shared_broad_predicate(tmp_path):
    """Two independent preferences filed under the same generic predicate
    don't clobber each other — reconciliation recognizes them as different
    attributes and both stay current."""
    module, log, provider = setup(tmp_path)
    consolidate_fact(
        module, log, "ep1", "human", "I prefer dark mode.", provider,
        extraction_response("Preference noted.", fact("George", "preference", "dark mode")),
    )
    first = module.knowledge.current(subject="George")[0]

    consolidate_fact(
        module, log, "ep2", "human", "I prefer concise replies.", provider,
        extraction_response("Preference noted.", fact("George", "preference", "concise replies")),
        reconciliation_response=json.dumps({"decisions": [{"candidate_index": 0, "decision": "coexist"}]}),
    )

    current = module.knowledge.current(subject="George", predicate="preference")
    assert {r.value for r in current} == {"dark mode", "concise replies"}
    assert first.id in {r.id for r in current}
    added = next(r for r in current if r.value == "concise replies")
    assert added.reconciliation == "coexist" and added.supersedes is None


def test_repeated_support_reinforces_without_a_spurious_value_change(tmp_path):
    module, log, provider = setup(tmp_path)
    consolidate_fact(
        module, log, "ep1", "human", "The staging DB is Postgres 16.", provider,
        extraction_response("Noted.", fact("staging DB", "engine", "Postgres 16")),
    )
    original = module.knowledge.current(subject="staging DB")[0]

    consolidate_fact(
        module, log, "ep2", "human", "Confirmed: staging DB runs Postgres 16.", provider,
        extraction_response("Reconfirmed.", fact("staging DB", "engine", "Postgres 16")),
        reconciliation_response=json.dumps({
            "decisions": [{"candidate_index": 0, "decision": "reinforce", "target_id": original.id}],
        }),
    )

    current = module.knowledge.current(subject="staging DB")
    assert [r.value for r in current] == ["Postgres 16"]  # one current value, not two
    assert current[0].reconciliation == "reinforce"
    assert current[0].supersedes == original.id
    # Both the original and the reinforcing evidence remain in the append-only file.
    assert len(module.knowledge.read_all()) == 2


def test_reinforce_mislabeled_over_a_real_change_is_still_applied(tmp_path):
    """A defensive guard: if reconciliation calls something "reinforce" but
    the value actually differs from the target, that's a missed update, not a
    confirmation — it must not be silently dropped."""
    module, log, provider = setup(tmp_path)
    consolidate_fact(
        module, log, "ep1", "human", "Atlas quote is 180 CAD.", provider,
        extraction_response("Noted.", fact("Atlas", "quote", "180 CAD")),
    )
    original = module.knowledge.current(subject="Atlas")[0]

    consolidate_fact(
        module, log, "ep2", "human", "Atlas quote is actually 210 CAD.", provider,
        extraction_response("Updated.", fact("Atlas", "quote", "210 CAD")),
        reconciliation_response=json.dumps({
            "decisions": [{"candidate_index": 0, "decision": "reinforce", "target_id": original.id}],
        }),
    )

    current = module.knowledge.current(subject="Atlas")
    assert [r.value for r in current] == ["210 CAD"]
    assert current[0].reconciliation == "replace"  # re-labeled, not trusted as-is


def test_unresolved_contradiction_keeps_both_reports_visible_and_attributed(tmp_path):
    module, log, provider = setup(tmp_path)
    consolidate_fact(
        module, log, "ep1", "Alpha", "Boreal deadline is Thursday.", provider,
        extraction_response("Noted.", fact("Boreal", "deadline", "Thursday", attribution="partner_report",
                                          reported_by="Alpha")),
    )
    first = module.knowledge.current(subject="Boreal")[0]

    consolidate_fact(
        module, log, "ep2", "Beta", "Boreal deadline is Friday.", provider,
        extraction_response("Noted.", fact("Boreal", "deadline", "Friday", attribution="partner_report",
                                          reported_by="Beta")),
        reconciliation_response=json.dumps({
            "decisions": [{"candidate_index": 0, "decision": "contradiction", "target_id": first.id}],
        }),
    )

    current = {r.reported_by: r for r in module.knowledge.current(subject="Boreal")}
    assert set(current) == {"Alpha", "Beta"}
    assert current["Beta"].contradicts == (first.id,)
    assert current["Alpha"].contradicts is None
    assert current["Beta"].supersedes is None
    rendered = current["Beta"].render()
    assert f"contradicts {first.id}" in rendered


def test_a_historical_report_cannot_displace_a_current_fact_by_processing_later(tmp_path):
    """The report is consolidated (processed) after the current fact, but
    describes an earlier state — reconciliation must judge effective time
    from the claim itself, not from which episode ran later."""
    module, log, provider = setup(tmp_path)
    consolidate_fact(
        module, log, "ep1", "tool", "Atlas moved to production.", provider,
        extraction_response("Noted.", fact("Atlas", "phase", "production",
                                          attribution="direct_observation",
                                          action_status="confirmed_outcome")),
    )
    current_record = module.knowledge.current(subject="Atlas")[0]

    consolidate_fact(
        module, log, "ep2", "Alpha", "Last quarter Atlas was in the pilot phase.", provider,
        extraction_response("Noted.", fact("Atlas", "phase", "pilot",
                                          statement="Last quarter Atlas was in the pilot phase.",
                                          attribution="partner_report", reported_by="Alpha")),
        reconciliation_response=json.dumps({
            "decisions": [{"candidate_index": 0, "decision": "historical", "target_id": current_record.id}],
        }),
    )

    current = module.knowledge.current(subject="Atlas")
    assert [r.value for r in current] == ["production"]
    all_records = module.knowledge.read_all()
    historical = next(r for r in all_records if r.value == "pilot")
    assert historical.reconciliation == "historical"
    assert historical not in current


def test_reconciliation_is_skipped_for_a_subject_with_no_existing_knowledge(tmp_path):
    """No existing knowledge for the subject means nothing to reconcile
    against — the module doesn't spend a call finding that out."""
    module, log, provider = setup(tmp_path)
    result, _ = consolidate_fact(
        module, log, "ep1", "human", "Atlas prototype deadline is Friday.", provider,
        extraction_response("Noted.", fact("Atlas", "prototype deadline", "Friday")),
    )
    assert provider.calls == 1  # extraction only, no reconciliation call
    trace = json.loads((module.memory_dir / "traces" / "consolidation.jsonl").read_text().splitlines()[0])
    assert trace["reconciliation_calls"] == 0


def test_reconciliation_failure_preserves_current_fact_without_blocking(tmp_path):
    """If no provider can answer the reconciliation call, the episode still
    completes, but uncertain claims must not replace current knowledge."""
    module, log, provider = setup(tmp_path)
    consolidate_fact(
        module, log, "ep1", "human", "Atlas prototype deadline is Friday.", provider,
        extraction_response("Noted.", fact("Atlas", "prototype deadline", "Friday")),
    )

    event = log.append("human", "chat_message", {"message": "Atlas prototype deadline moved to Monday."})
    provider.extraction_response = extraction_response(
        "Corrected.", fact("Atlas", "prototype deadline", "Monday"),
    ).replace("<EVIDENCE_ID>", event.id)

    class ForwardingProvider:
        def chat(self, prompt, **kwargs):
            if "<candidates>" in prompt:
                raise RuntimeError("reconciliation endpoint down")
            return provider.extraction_response

    module._model_providers = [ForwardingProvider()]
    result = module.consolidate(Episode("ep2", event.id, event.id))
    assert not result.skipped
    assert [r.value for r in module.knowledge.current(subject="Atlas")] == ["Friday"]
    pending = module.knowledge.read_all()[-1]
    assert pending.value == "Monday" and pending.reconciliation == "unresolved"
    assert pending.supersedes is None
    assert (module.memory_dir / "reconciliation_failures.jsonl").exists()


def test_a_hallucinated_target_id_is_rejected_and_falls_back(tmp_path):
    """A target_id that doesn't resolve to a record actually offered as
    context is dropped rather than trusted — the candidate falls back to the
    unresolved state instead of superseding something it was never shown."""
    module, log, provider = setup(tmp_path)
    consolidate_fact(
        module, log, "ep1", "human", "Atlas prototype deadline is Friday.", provider,
        extraction_response("Noted.", fact("Atlas", "prototype deadline", "Friday")),
    )
    original = module.knowledge.current(subject="Atlas")[0]

    consolidate_fact(
        module, log, "ep2", "human", "Atlas prototype deadline moved to Monday.", provider,
        extraction_response("Corrected.", fact("Atlas", "prototype deadline", "Monday")),
        reconciliation_response=json.dumps({
            "decisions": [{"candidate_index": 0, "decision": "replace", "target_id": "invented-id"}],
        }),
    )

    current = module.knowledge.current(subject="Atlas")
    assert current == [original]
    assert module.knowledge.read_all()[-1].reconciliation == "unresolved"


def test_two_corrections_in_one_episode_chain_instead_of_both_staying_current(tmp_path):
    """Two candidates in the same episode both targeting the original record
    (a fact corrected twice in one breath) must not both end up current."""
    module, log, provider = setup(tmp_path)
    consolidate_fact(
        module, log, "ep1", "human", "Atlas prototype deadline is Friday.", provider,
        extraction_response("Noted.", fact("Atlas", "prototype deadline", "Friday")),
    )
    original = module.knowledge.current(subject="Atlas")[0]

    event = log.append("human", "chat_message",
                       {"message": "Actually Monday. No wait, Tuesday."})
    provider.extraction_response = extraction_response(
        "Corrected twice.",
        fact("Atlas", "prototype deadline", "Monday"),
        fact("Atlas", "prototype deadline", "Tuesday"),
    ).replace("<EVIDENCE_ID>", event.id)
    provider.reconciliation_response = json.dumps({"decisions": [
        {"candidate_index": 0, "decision": "replace", "target_id": original.id},
        {"candidate_index": 1, "decision": "replace", "target_id": original.id},
    ]})
    module.consolidate(Episode("ep2", event.id, event.id))

    current = module.knowledge.current(subject="Atlas")
    assert [r.value for r in current] == ["Tuesday"]


def test_reconciliation_context_per_subject_is_bounded(tmp_path):
    log = StimulusLog(tmp_path / "log.jsonl")
    provider = ScriptedProvider()
    module = MemoryModule(tmp_path / "memory", log, model_providers=[provider],
                          reconciliation_context_k=2)
    for i in range(4):
        consolidate_fact(
            module, log, f"ep{i}", "human", f"Atlas fact {i} is set.", provider,
            extraction_response("Noted.", fact("Atlas", f"attribute {i}", f"value {i}")),
        )
    assert len(module.knowledge.current(subject="Atlas")) == 4

    event = log.append("human", "chat_message", {"message": "Atlas has one more fact."})
    provider.extraction_response = extraction_response(
        "Noted.", fact("Atlas", "attribute 4", "value 4"),
    ).replace("<EVIDENCE_ID>", event.id)
    provider.reconciliation_response = None
    module.consolidate(Episode("ep-last", event.id, event.id))

    prompt = provider.prompts[-1]
    assert prompt.count("Current fact: Atlas") == 2


@pytest.mark.parametrize('response', ['broken', '{"decisions": []}',
    '{"decisions": [{"candidate_index": 0, "decision": [], "target_id": null}]}'])
@pytest.mark.parametrize('predicate', ['deadline', 'delivery date'])
def test_unusable_reconciliation_retains_stale_claim_without_promoting_it(tmp_path, response, predicate):
    module, log, provider = setup(tmp_path)
    for episode_id, day, attr, value in [('recent', 16, 'deadline', 'Monday'), ('old', 1, predicate, 'Friday')]:
        event = log.append('human', 'chat_message', {'message': f'Atlas {attr}: {value}'},
                           ts=datetime(2026, 9, day, tzinfo=timezone.utc))
        provider.extraction_response = extraction_response('Deadline report', fact('Atlas', attr, value)).replace('<EVIDENCE_ID>', event.id)
        provider.reconciliation_response = response
        module.consolidate(Episode(episode_id, event.id, event.id))
    reopened = MemoryModule(module.memory_dir, log)
    assert [r.value for r in reopened.knowledge.current('Atlas')] == ['Monday']
    retained = reopened.knowledge.read_all()[-1]
    assert retained.value == 'Friday'
    assert retained.reconciliation == 'unresolved' and retained.supersedes is None
    assert retained.support_event_ids == (event.id,)
    assert 'Unresolved claim' in retained.render()
    assert all(hit.id != retained.id for hit in reopened.knowledge.search({'atlas'}))


@pytest.mark.parametrize('exact', [False, True])
def test_reconciliation_selects_relevant_target_beyond_first_six_facts(tmp_path, exact):
    module, log, provider = setup(tmp_path)
    for i in range(8):
        attr = 'prototype delivery deadline' if i == 7 else f'unrelated attribute {i}'
        consolidate_fact(module, log, f'ep{i}', 'human', f'Atlas {attr}', provider,
                         extraction_response('Attribute', fact('Atlas', attr, 'Friday')))
    target = module.knowledge.current('Atlas', 'prototype delivery deadline')[0]
    correction = 'prototype delivery deadline' if exact else 'prototype delivery date'
    consolidate_fact(module, log, 'update', 'human', 'Atlas prototype delivery moved to Monday', provider,
                     extraction_response('Correction', fact('Atlas', correction, 'Monday')),
                     reconciliation_response=json.dumps({'decisions': [
                         {'candidate_index': 0, 'decision': 'replace', 'target_id': target.id}]}))
    prompt = provider.prompts[-1]
    assert prompt.count('Current fact: Atlas') == 6
    assert target.id in prompt
    current = module.knowledge.current('Atlas')
    assert len(current) == 8
    assert target.id not in {r.id for r in current}
    assert any(r.value == 'Monday' and r.supersedes == target.id for r in current)


def test_exact_attribute_precedes_newer_lexically_similar_facts(tmp_path):
    module, log, provider = setup(tmp_path)
    module._reconciliation_context_k = 1
    for i, predicate in enumerate(['deadline', 'deadline for prototype', 'deadline for invoice']):
        consolidate_fact(module, log, f'ep{i}', 'human', 'Atlas deadline', provider,
                         extraction_response('Noted', fact('Atlas', predicate, 'Friday')))
    target = module.knowledge.current('Atlas', 'deadline')[0]
    consolidate_fact(module, log, 'update', 'human', 'Atlas deadline is Monday', provider,
                     extraction_response('Correction', fact('Atlas', ' DEADLINE ', 'Monday')),
                     reconciliation_response=json.dumps({'decisions': [
                         {'candidate_index': 0, 'decision': 'replace', 'target_id': target.id}]}))
    assert provider.prompts[-1].count('Current fact: Atlas') == 1
    assert target.id in provider.prompts[-1]
    assert [r.value for r in module.knowledge.current('Atlas', 'deadline')] == ['Monday']
