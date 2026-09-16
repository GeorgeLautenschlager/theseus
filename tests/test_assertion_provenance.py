"""Multi-event provenance and semantic counterexamples for consolidation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from theseus.knowledge_layer import KnowledgeRecord
from theseus.memory_layer import MemoryRecord
from theseus.memory_module import Episode, MemoryModule
from theseus.stimulus_log import StimulusLog
from theseus.wisdom_layer import WisdomRecord


class Extractor:
    def __init__(self, assertions):
        self.assertions = assertions
        self.calls = 0

    def chat(self, prompt, **kwargs):
        self.calls += 1
        return json.dumps({"summary": "Atlas payment and backup were discussed.",
                           "assertions": self.assertions})


def multi_event_case(tmp_path, assertions):
    log = StimulusLog(tmp_path / "stimulus.jsonl")
    report = log.append("Beta", "chat_message", {"message": "I plan to back up Atlas."})
    recall = log.append("Alty", "tool_result",
                        {"tool": "recall", "output": "Atlas payment was confirmed earlier."})
    failure = log.append("Alty", "tool_result",
                         {"tool": "payment", "output": "Atlas payment attempt failed."})
    extractor = Extractor(assertions(report.id, recall.id, failure.id))
    memory = MemoryModule(tmp_path / "memory", log, model_providers=[extractor])
    result = memory.consolidate(Episode("atlas", report.id, failure.id))
    return memory, result, (report, recall, failure), extractor


def claim(event_id, **changes):
    assertion = {"kind": "fact", "subject": "Atlas", "predicate": "payment status",
                 "value": "failed", "statement": "Atlas payment attempt failed.",
                 "support_event_ids": [event_id], "attribution": "direct_observation",
                 "action_status": "failure"}
    assertion.update(changes)
    return assertion


@pytest.mark.parametrize("support,reason", [
    (lambda report, recall, failure: None, "missing or invalid support_event_ids"),
    (lambda report, recall, failure: [], "missing or invalid support_event_ids"),
    (lambda report, recall, failure: ["not-in-episode"], "unknown support event"),
    (lambda report, recall, failure: [recall], "context-only support event"),
    (lambda report, recall, failure: [failure, failure], "missing or invalid support_event_ids"),
])
def test_multi_event_eval_rejects_ineligible_source_ids(tmp_path, support, reason):
    def assertions(report, recall, failure):
        invalid = claim(failure)
        value = support(report, recall, failure)
        if value is None:
            del invalid["support_event_ids"]
        else:
            invalid["support_event_ids"] = value
        return [claim(failure), invalid]

    memory, result, events, _ = multi_event_case(tmp_path, assertions)
    assert result.extracted == 2 and result.schema_failures == 1
    assert len(memory.knowledge.current()) == 1
    assert memory.knowledge.current()[0].support_event_ids == (events[2].id,)
    dead = json.loads((memory.memory_dir / "dead_letter.jsonl").read_text().splitlines()[0])
    assert reason in dead["reason"]
    assert events[1].id not in memory.memory.read_all()[-1].content
    assert events[0].id in memory.memory.read_all()[-1].content
    assert events[2].id in memory.memory.read_all()[-1].content


def test_multi_event_eval_keeps_report_and_action_states_in_recall(tmp_path):
    def assertions(report, recall, failure):
        return [
            {"kind": "fact", "subject": "Beta", "predicate": "Atlas backup plan",
             "value": "planned", "statement": "Beta plans to back up Atlas.",
             "support_event_ids": [report], "attribution": "partner_report",
             "reported_by": "Beta", "action_status": "intention"},
            claim(failure),
        ]

    memory, result, events, _ = multi_event_case(tmp_path, assertions)
    assert result.schema_failures == 0
    records = {r.subject: r for r in memory.knowledge.current()}
    assert records["Beta"].reported_by == "Beta"
    assert records["Beta"].action_status == "intention"
    assert records["Atlas"].action_status == "failure"
    assert records["Beta"].support_event_ids == (events[0].id,)
    assert records["Atlas"].support_event_ids == (events[2].id,)
    texts = [entry.text for entry in memory.recall("Atlas backup payment", 3000).entries]
    assert any("partner_report (Beta)" in text and "Action status: intention" in text for text in texts)
    assert any("direct_observation" in text and "Action status: failure" in text for text in texts)


def test_multi_event_eval_preserves_every_action_stage(tmp_path):
    log = StimulusLog(tmp_path / "stimulus.jsonl")
    report = log.append("Beta", "chat_message", {"message": "I report Atlas scope is a prototype."})
    plan = log.append("Beta", "chat_message", {"message": "I intend to make a backup."})
    attempt = log.append("Alty", "decision", {"text": "Started the backup."})
    failure = log.append("Alty", "tool_result", {"tool": "backup", "output": "Backup failed."})
    outcome = log.append("Alty", "tool_result", {"tool": "export", "output": "Prototype export saved."})
    rows = (
        (report, "scope", "prototype", "not_applicable", "partner_report"),
        (plan, "backup plan", "planned", "intention", "partner_report"),
        (attempt, "backup attempt", "started", "attempt", "direct_observation"),
        (failure, "backup result", "failed", "failure", "direct_observation"),
        (outcome, "export result", "saved", "confirmed_outcome", "direct_observation"),
    )
    assertions = [
        {"kind": "fact", "subject": "Atlas", "predicate": predicate,
         "value": value, "statement": f"Atlas {predicate}: {value}",
         "support_event_ids": [event.id], "attribution": attribution,
         "action_status": status,
         **({"reported_by": "Beta"} if attribution == "partner_report" else {})}
        for event, predicate, value, status, attribution in rows
    ]
    memory = MemoryModule(tmp_path / "memory", log, model_providers=[Extractor(assertions)])
    result = memory.consolidate(Episode("stages", report.id, outcome.id))
    assert result.schema_failures == 0
    current = {r.predicate: r for r in memory.knowledge.current()}
    assert {r.action_status for r in current.values()} == {
        "not_applicable", "intention", "attempt", "failure", "confirmed_outcome"}
    assert current["scope"].reported_by == "Beta"
    recalled = "\n".join(entry.text for entry in memory.recall("Atlas", 6000).entries)
    for status in ("intention", "attempt", "failure", "confirmed_outcome"):
        assert f"Action status: {status}" in recalled
    assert "Attribution: partner_report (Beta)" in recalled


def test_multi_event_eval_detects_semantic_counterexample_with_valid_ids(tmp_path):
    # The source ID is real, but Beta's plan and a failed tool result do not
    # establish a confirmed payment. The structural gate cannot prove truth.
    def assertions(report, recall, failure):
        return [claim(report, value="paid", statement="Atlas payment is confirmed.",
                      attribution="partner_report", reported_by="Beta",
                      action_status="confirmed_outcome")]

    memory, result, events, _ = multi_event_case(tmp_path, assertions)
    assert result.schema_failures == 0
    stored = memory.knowledge.current()[0]
    assert stored.support_event_ids == (events[0].id,)
    expected = {"value": "failed", "action_status": "failure",
                "support_event_ids": (events[2].id,)}
    assert any(getattr(stored, field) != value for field, value in expected.items())


@pytest.mark.parametrize("change,reason", [
    ({"attribution": "partner_report"}, "partner report missing reported_by"),
    ({"attribution": "partner_report", "reported_by": "Alpha"},
     "reported_by is not a supporting event actor"),
    ({"action_status": "success"}, "missing or invalid action_status"),
    ({"attribution": []}, "missing or invalid attribution"),
    ({"action_status": {}}, "missing or invalid action_status"),
])
def test_multi_event_eval_rejects_invalid_status_metadata(tmp_path, change, reason):
    def assertions(report, recall, failure):
        return [claim(failure, **change)]

    memory, result, _, _ = multi_event_case(tmp_path, assertions)
    assert result.schema_failures == 1 and not memory.knowledge.current()
    dead = json.loads((memory.memory_dir / "dead_letter.jsonl").read_text().splitlines()[0])
    assert reason == dead["reason"]


@pytest.mark.parametrize("record_type,payload", [
    (KnowledgeRecord, {"subject": "Atlas", "predicate": "status", "value": "unknown"}),
    (MemoryRecord, {"content": "original", "summary": "Old note."}),
    (WisdomRecord, {"statement": "Verify payments."}),
])
def test_legacy_records_show_unknown_provenance_without_source_ids(record_type, payload):
    old = {"id": "old", "ts": datetime.now(timezone.utc).isoformat(),
           "source_episode_id": "legacy-episode", **payload}
    record = record_type.from_json(json.dumps(old))
    assert record.support_event_ids is None
    assert record.attribution is None and record.action_status is None
    assert "Supporting events: unknown" in record.render()
    assert json.loads(record.to_json())["support_event_ids"] is None
