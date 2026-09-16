"""Multi-event consolidation accuracy eval.

Consolidates labeled multi-event scenarios (plans followed by failure or
success, corrections, coexisting preferences, historical reports, oversized-tail
evidence, recall repetition, split action/result pairs, principles) and reports
correct knowledge updates, supported-claim retention, and unsupported claims
separately. Provides a deterministic offline reference-extraction run (part of
the offline suite), an opt-in live run, and a CLI (``python -m
theseus.memory_accuracy_eval``).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from theseus.assertion_metadata import ACTION_STATUSES, ATTRIBUTIONS
from theseus.json_utils import parse_json_response
from theseus.model_providers import PROVIDER_REGISTRY

from theseus.memory_module import Episode, MemoryModule
from theseus.layer_store import load_lines
from theseus.stimulus_log import StimulusLog
from theseus.tools.recall import RECALL_TOOL_NAME


class ReferenceExtractor:
    model = "ReferenceExtractor"

    def __init__(self):
        self.response = ""
        self.reconciliation_response = None  # None => new facts; default principle handling
        self.last_prompt = None                  # last *extraction* prompt only
        self.last_reconciliation_prompt = None

    def chat(self, prompt, **kwargs):
        if "<candidates>" in prompt and "<existing_knowledge>" in prompt:
            self.last_reconciliation_prompt = prompt
            if self.reconciliation_response is not None:
                return self.reconciliation_response
            if "wisdom-reconciliation" in prompt:
                return '{"decisions": []}'
            candidates = prompt.split("<candidates>\n", 1)[1].split("\n</candidates>", 1)[0].splitlines()
            return json.dumps({"decisions": [
                {"candidate_index": i, "decision": "new", "target_id": None}
                for i in range(len(candidates))
            ]})
        self.last_prompt = prompt
        return self.response


def build_memory(workdir, *, extractor, embedder):
    log = StimulusLog(Path(workdir) / "stimulus.jsonl")
    memory = MemoryModule(
        Path(workdir) / "memory", log,
        model_providers=[extractor],
        embedding_providers=[embedder] if embedder else [],
    )
    return memory, log


def drive_scenarios(memory, log, scenarios, *, reference, extractor, start):
    """Append each scenario's events to the log and consolidate its episodes.

    Events are stamped one per calendar day starting at ``start``; the day
    counter advances across all scenarios (never reset), so every event gets a
    distinct, monotonically increasing timestamp. Events whose role is
    ``recall_context`` are appended as ``tool_result`` records tagged with the
    recall tool name, so recalled text is never eligible to become a fact. When
    ``reference`` is true the extractor is scripted with each episode's labeled
    summary/assertions before consolidation.

    Returns a dict keyed by scenario name; each value is
    ``{"event_ids": [...], "episode_prompts": {episode_id: prompt}}``.
    """
    trace = {}
    day = 0
    for scenario in scenarios:
        ids = []
        for actor, event_type, content, role in scenario.events:
            timestamp = start + timedelta(days=day)
            day += 1
            if role == "recall_context":
                content = {**content, "tool": RECALL_TOOL_NAME}
                event = log.append(actor, "tool_result", content, ts=timestamp)
            else:
                event = log.append(actor, event_type, content, ts=timestamp)
            ids.append(event.id)
        prompts = {}
        for index, episode_spec in enumerate(scenario.episodes):
            episode_id = f"{scenario.name}-ep{index}"
            if reference:
                assertions = []
                for label in episode_spec.assertions:
                    assertion = dict(label)
                    indices = assertion.pop("support_event_indices")
                    assertion["support_event_ids"] = [ids[i] for i in indices]
                    if assertion["attribution"] == "partner_report":
                        assertion["reported_by"] = scenario.events[indices[0]][0]
                    assertions.append(assertion)
                extractor.response = json.dumps({
                    "summary": episode_spec.summary,
                    "assertions": assertions,
                })
                extractor.reconciliation_response = (
                    episode_spec.reconciliation(memory) if episode_spec.reconciliation is not None else None
                )
            memory.consolidate(Episode(
                episode_id, ids[episode_spec.event_indices[0]],
                ids[episode_spec.event_indices[-1]],
            ))
            prompts[episode_id] = getattr(extractor, "last_prompt", None)
        trace[scenario.name] = {"event_ids": ids, "episode_prompts": prompts}
    return trace


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    event_indices: tuple[int, ...]
    summary: str
    assertions: tuple[dict, ...] = ()
    # Scripts the reference reconciliation response for this episode, given the
    # live MemoryModule (queried right before consolidation, so it can name a
    # real prior record's id — those are assigned during consolidation, not
    # known ahead of time). None labels fact candidates as new independent attributes; corrections
    # must explicitly script their reconciliation decision.
    reconciliation: Callable[[MemoryModule], str] | None = None


@dataclass(frozen=True, slots=True)
class Transition:
    subject: str
    predicate: str
    current_value: str
    superseded_values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Query:
    question: str
    expected: str | None
    forbidden: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    category: str
    events: tuple[tuple[str, str, dict, str], ...]
    episodes: tuple[EpisodeSpec, ...]
    transitions: tuple[Transition, ...] = ()
    supported: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    queries: tuple[Query, ...] = ()


def _ev(actor, message, role="evidence"):
    return (actor, "chat_message", {"message": message}, role)


def _fact(subject, predicate, value, statement=None, *, support, attribution="partner_report",
          action_status="not_applicable"):
    return {"kind": "fact", "subject": subject, "predicate": predicate,
            "value": value, "statement": statement or f"{subject} {predicate}: {value}",
            "support_event_indices": support, "attribution": attribution,
            "action_status": action_status}


def _principle(statement, *, support, attribution="partner_report"):
    return {"kind": "principle", "statement": statement,
            "support_event_indices": support, "attribution": attribution,
            "action_status": "not_applicable"}


def _note(statement, *, support):
    # kind "event" = a non-fact episodic note; unrelated to the StimulusEvent event_type
    return {"kind": "event", "statement": statement,
            "support_event_indices": support, "attribution": "partner_report",
            "action_status": "not_applicable"}


REQUIRED_CATEGORIES = frozenset({
    "plan_then_failure", "plan_then_success", "alternate_wording_correction",
    "coexisting_preferences", "historical_report", "recall_repetition",
    "split_action_result", "principles", "oversized_tail", "contradiction",
    "principle_promotion",
})

_OVERSIZED_MESSAGE = ("padding. " * 8000) + "DECISIVE: The Atlas security token is ZULU-9."

SCENARIOS = (
    Scenario("atlas-payment", "plan_then_failure",
        (_ev("Alpha", "I will attempt the Atlas payment now."), _ev("tool", "Atlas payment attempt failed; no funds were received.")),
        (EpisodeSpec((0, 1), "I attempted the Atlas payment; it failed and no funds arrived.", (_fact("Atlas", "payment status", "failed", "Atlas payment attempt failed; no funds received.", support=(1,), attribution="direct_observation", action_status="failure"),)),),
        (Transition("Atlas", "payment status", "failed"),), ("failed",), ("completed", "received"),
        (Query("What is Atlas's payment status?", "failed", ("completed", "received", "paid")), Query("What is Atlas's bank account number?", None))),
    Scenario("project-backup", "plan_then_success",
        (_ev("Beta", "I plan to back up the project database."), _ev("tool", "Project database backup completed successfully.")),
        (EpisodeSpec((0, 1), "Beta planned the backup and it completed successfully.", (_fact("Project", "backup status", "completed", support=(1,), attribution="direct_observation", action_status="confirmed_outcome"),)),),
        (Transition("Project", "backup status", "completed"),), ("completed",), ("planned", "not yet"),
        (Query("What is the project backup status?", "completed", ("planned", "not yet")),)),
    Scenario("atlas-deadline", "alternate_wording_correction",
        (_ev("human", "The Atlas prototype is due Friday."), _ev("human", "Scratch that — Atlas delivery date moved to Monday.")),
        (EpisodeSpec((0,), "Atlas prototype is due Friday.", (_fact("Atlas", "prototype deadline", "Friday", support=(0,)),)),
         EpisodeSpec((1,), "Atlas delivery was pushed to Monday.", (_fact("Atlas", "delivery date", "Monday", support=(1,)),),
                     # Genuinely different predicate wording — reconciliation, not a
                     # shared key, is what has to recognize this as the same attribute.
                     reconciliation=lambda memory: json.dumps({"decisions": [
                         {"candidate_index": 0, "decision": "replace", "target_id":
                          memory.knowledge.current(subject="Atlas", predicate="prototype deadline")[0].id},
                     ]}))),
        (Transition("Atlas", "delivery date", "Monday"),), ("Monday",), ("Friday",),
        (Query("What is Atlas's prototype deadline?", "Monday", ("Friday",)),)),
    Scenario("user-prefs", "coexisting_preferences",
        (_ev("human", "I prefer morning meetings."), _ev("human", "I also prefer concise written summaries.")),
        (EpisodeSpec((0,), "The user prefers morning meetings.", (_fact("George", "preference", "morning meetings", support=(0,)),)),
         EpisodeSpec((1,), "The user also prefers concise written summaries.",
                     (_fact("George", "preference", "concise written summaries", support=(1,)),),
                     # Same broad predicate on purpose — reconciliation has to tell these
                     # are independent attributes, not competing values of one.
                     reconciliation=lambda memory: json.dumps({
                         "decisions": [{"candidate_index": 0, "decision": "coexist"}],
                     }))),
        (Transition("George", "preference", "morning meetings"),),
        ("morning meetings", "concise written summaries"), (),
        (Query("What meeting time does the user prefer?", "morning"), Query("What summary format does the user prefer?", "concise"))),
    Scenario("boreal-conflict", "contradiction",
        (_ev("Alpha", "Boreal deadline is Thursday."), _ev("Beta", "Boreal deadline is Friday.")),
        (EpisodeSpec((0,), "Alpha reported the Boreal deadline is Thursday.",
                     (_fact("Boreal", "deadline", "Thursday", support=(0,)),)),
         EpisodeSpec((1,), "Beta reported a conflicting Boreal deadline of Friday.",
                     (_fact("Boreal", "deadline", "Friday", support=(1,)),),
                     # Two partner reports, no authoritative signal either way —
                     # reconciliation must not silently pick a winner.
                     reconciliation=lambda memory: json.dumps({"decisions": [
                         {"candidate_index": 0, "decision": "contradiction", "target_id":
                          memory.knowledge.current(subject="Boreal", predicate="deadline")[0].id},
                     ]}))),
        (), ("Thursday", "Friday"), (),
        (Query("What is the Boreal deadline?", None),)),
    Scenario("atlas-phase", "historical_report",
        (_ev("Alpha", "Last quarter Atlas was in the pilot phase."), _ev("human", "Atlas is now in the production phase.")),
        (EpisodeSpec((0,), "Alpha reported that last quarter Atlas was in the pilot phase.", (_note("Last quarter Atlas was in the pilot phase.", support=(0,)),)), EpisodeSpec((1,), "Atlas is now in the production phase.", (_fact("Atlas", "phase", "production", support=(1,)),))),
        (Transition("Atlas", "phase", "production"),), ("production", "pilot"), ("pilot",),
        (Query("What phase is Atlas in now?", "production", ("pilot",)),)),
    Scenario("raven-code", "recall_repetition",
        (_ev("human", "The Atlas access code is RAVEN-42."), _ev("agent", "(recalled) The Atlas access code was once QUAIL-7.", "recall_context")),
        (EpisodeSpec((0, 1), "The Atlas access code is RAVEN-42.", (_fact("Atlas", "access code", "RAVEN-42", support=(0,)),)),),
        (Transition("Atlas", "access code", "RAVEN-42"),), ("RAVEN-42",), ("QUAIL-7",),
        (Query("What is the Atlas access code?", "RAVEN-42"),) * 3),
    Scenario("boreal-invoice", "split_action_result",
        (_ev("Alpha", "Sending the Boreal invoice now."), _ev("tool", "Boreal invoice sent; confirmation BOR-77.")),
        (EpisodeSpec((0, 1), "Alpha sent the Boreal invoice; confirmation BOR-77.", (_fact("Boreal", "invoice status", "sent", "Boreal invoice sent; confirmation BOR-77.", support=(1,), attribution="direct_observation", action_status="confirmed_outcome"),)),),
        (Transition("Boreal", "invoice status", "sent"),), ("sent", "BOR-77"), ("draft",),
        (Query("What is Boreal's invoice status?", "sent", ("draft", "not sent")),)),
    Scenario("confirm-policy", "principles",
        (_ev("human", "Always confirm risky actions before executing."), _ev("human", "Reminder: confirm risky actions before executing."), _ev("human", "For low-risk actions, skip the confirmation step.")),
        (EpisodeSpec((0,), "The user set a rule to always confirm risky actions.", (_principle("Always confirm risky actions before executing.", support=(0,)),)), EpisodeSpec((1,), "The user repeated the rule to confirm risky actions.", (_principle("Always confirm risky actions before executing.", support=(1,)),)), EpisodeSpec((2,), "The user added that low-risk actions can skip confirmation.", (_principle("For low-risk actions, skip the confirmation step.", support=(2,)),))),
        (), ("confirm risky actions", "low-risk"), (),
        (Query("What is the policy for risky actions?", "confirm"), Query("What is the policy for low-risk actions?", "low-risk"))),
    Scenario("brevity-pattern", "principle_promotion",
        (_ev("agent", "First noticed the user replies tersely."),
         _ev("agent", "Noticed it again in a second, unrelated exchange.")),
        (EpisodeSpec((0,), "The agent noticed a pattern in how the user replies.",
                     (_principle("The user seems to prefer terse replies.", support=(0,),
                                attribution="inference"),)),
         EpisodeSpec((1,), "The agent noticed the same pattern again, independently.",
                     (_principle("User replies again suggest a preference for terse answers.",
                                support=(1,), attribution="inference"),),
                     # Independent evidence for the same inferred generalization —
                     # reconciliation has to recognize it despite the reworded statement.
                     # Scenarios share cumulative memory, so pick this scenario's own
                     # principle by recency (current() is ts-sorted) rather than
                     # index 0, which could belong to an earlier scenario.
                     reconciliation=lambda memory: json.dumps({"decisions": [
                         {"candidate_index": 0, "decision": "reinforce", "target_id":
                          memory.wisdom.current()[-1].id},
                     ]}))),
        (), ("terse",), (),
        (Query("What does the agent believe about the user's reply style?", "terse"),)),
    Scenario("atlas-token", "oversized_tail",
        (("tool", "tool_result", {"message": _OVERSIZED_MESSAGE}, "evidence"), _ev("human", "Store the Atlas security token safely.")),
        (EpisodeSpec((0, 1), "Atlas security token noted.", (_fact("Atlas", "security token", "ZULU-9", support=(0,), attribution="direct_observation"),)),),
        (Transition("Atlas", "security token", "ZULU-9"),), ("ZULU-9",), (),
        (Query("What is the Atlas security token?", "ZULU-9"),)),
)


def measure_correct_updates(memory, scenarios):
    failures = []
    total = passed = 0
    for scenario in scenarios:
        for transition in scenario.transitions:
            total += 1
            current = memory.knowledge.current(transition.subject, transition.predicate)
            values = [record.value for record in current]
            expected = transition.current_value.casefold()
            superseded = {value.casefold() for value in transition.superseded_values}
            if values and values[0].casefold() == expected and not superseded.intersection(value.casefold() for value in values):
                passed += 1
            else:
                failures.append({
                    "scenario": scenario.name,
                    "subject": transition.subject,
                    "predicate": transition.predicate,
                    "expected": transition.current_value,
                    "found": values,
                })
    return {"passed": passed, "total": total, "failures": failures}


def measure_supported_retained(memory, scenarios, *, budget_tokens=2000):
    failures = []
    total = passed = 0
    for scenario in scenarios:
        text = " ".join(entry.text for query in scenario.queries
                          for entry in memory.recall(query.question, budget_tokens).entries)
        text += " " + " ".join(record.value for record in memory.knowledge.current())
        folded = text.casefold()
        for fragment in scenario.supported:
            total += 1
            if fragment.casefold() in folded:
                passed += 1
            else:
                failures.append({"scenario": scenario.name, "fragment": fragment})
    return {"passed": passed, "total": total, "failures": failures}


def measure_unsupported_present(memory, scenarios):
    failures = []
    total = 0
    for scenario in scenarios:
        subjects = {transition.subject.casefold() for transition in scenario.transitions}
        records = memory.knowledge.current()
        if subjects:
            records = [record for record in records if record.subject.casefold() in subjects]
        current_values = [record.value.casefold() for record in records]
        for fragment in scenario.unsupported:
            total += 1
            if any(fragment.casefold() in value for value in current_values):
                failures.append({"scenario": scenario.name, "fragment": fragment})
    return {"violations": len(failures), "total": total, "failures": failures}


def measure_source_validity(memory, scenarios, traces):
    """Check accepted claims against the evidence IDs of their own episode."""
    eligible = {}
    for scenario in scenarios:
        ids = traces[scenario.name]["event_ids"]
        for index, episode in enumerate(scenario.episodes):
            eligible[f"{scenario.name}-ep{index}"] = {
                ids[i] for i in episode.event_indices
                if scenario.events[i][3] == "evidence"
            }
    failures = []
    total = passed = 0
    for layer in (memory.knowledge, memory.memory, memory.wisdom):
        for record in layer.read_all():
            if record.source_episode_id not in eligible or record.attribution is None:
                continue  # episode summaries and unrelated/legacy records are not assertions
            total += 1
            support = record.support_event_ids
            if support and len(set(support)) == len(support) and set(support) <= eligible[record.source_episode_id]:
                passed += 1
            else:
                failures.append({"episode_id": record.source_episode_id, "record_id": record.id,
                                 "support_event_ids": support})
    return {"passed": passed, "total": total, "failures": failures}


def measure_labeled_assertions(memory, scenarios, traces):
    """Compare stored claims with semantic labels, separately from ID validity."""
    failures = []
    total = passed = 0
    for scenario in scenarios:
        ids = traces[scenario.name]["event_ids"]
        for index, episode in enumerate(scenario.episodes):
            episode_id = f"{scenario.name}-ep{index}"
            for label in episode.assertions:
                total += 1
                kind = label["kind"]
                if kind == "fact":
                    records = memory.knowledge.read_all()
                    matches = [r for r in records if r.source_episode_id == episode_id
                               and (r.subject, r.predicate, r.value) ==
                               (label["subject"], label["predicate"], label["value"])]
                elif kind == "principle":
                    matches = [r for r in memory.wisdom.read_all()
                               if r.source_episode_id == episode_id and r.statement == label["statement"]]
                else:
                    matches = [r for r in memory.memory.read_all()
                               if r.source_episode_id == episode_id and r.content == label["statement"]]
                support = tuple(ids[i] for i in label["support_event_indices"])
                expected_reporter = (scenario.events[label["support_event_indices"][0]][0]
                                     if label["attribution"] == "partner_report" else None)
                if any(r.support_event_ids == support and r.attribution == label["attribution"]
                       and r.reported_by == expected_reporter
                       and r.action_status == label["action_status"] for r in matches):
                    passed += 1
                else:
                    failures.append({"scenario": scenario.name, "episode_id": episode_id,
                                     "kind": kind, "statement": label["statement"]})
    return {"passed": passed, "total": total, "failures": failures}


def _evaluate(memory, log, scenarios, traces, workdir, *, budget_tokens):
    metrics = {
        "correct_updates": measure_correct_updates(memory, scenarios),
        "supported_retained": measure_supported_retained(memory, scenarios, budget_tokens=budget_tokens),
        "unsupported_present": measure_unsupported_present(memory, scenarios),
        "source_validity": measure_source_validity(memory, scenarios, traces),
        "labeled_assertions": measure_labeled_assertions(memory, scenarios, traces),
    }
    restarted = MemoryModule(workdir / "memory", log, model_providers=memory._model_providers, embedding_providers=[])
    restart_recall = measure_supported_retained(restarted, scenarios, budget_tokens=budget_tokens)
    for i in range(30):
        log.append("agent", "decision", {"text": f"Routine housekeeping {i}"})
    departed = MemoryModule(workdir / "memory", log, model_providers=memory._model_providers, embedding_providers=[])
    context_departure_recall = measure_supported_retained(departed, scenarios, budget_tokens=budget_tokens)
    prompt = next(iter(traces["atlas-token"]["episode_prompts"].values()))
    oversized = {"decisive_tail_in_budget": bool(prompt) and "ZULU-9" in prompt,
                 "full_event_searchable": any("ZULU-9" in record.content for record in memory.memory.read_all())}
    trace_rows = [json.loads(line) for line in load_lines(workdir / "memory" / "traces" / "consolidation.jsonl")]
    costs = {"extraction_chat_calls": sum(t["chat_calls"] for t in trace_rows), "answer_chat_calls": 0,
             "embedding_calls_at_consolidation": sum(t["embedding_calls"] for t in trace_rows),
             "embedding_calls_at_recall": memory._embedding_calls,
             "tokens_in_estimated": sum(t["tokens_in"] for t in trace_rows),
             "tokens_out_estimated": sum(t["tokens_out"] for t in trace_rows),
             "reported_chat_usage": [u for t in trace_rows for u in t["reported_chat_usage"]]}
    return metrics, restart_recall, context_departure_recall, oversized, costs


def _scenario_rows(memory, scenarios, budget_tokens):
    rows = []
    for scenario in scenarios:
        keys = {(t.subject.casefold(), t.predicate.casefold()) for t in scenario.transitions}
        current = [f"{r.subject} {r.predicate}: {r.value}" for r in memory.knowledge.current()
                   if (r.subject.casefold(), r.predicate.casefold()) in keys]
        recall = {q.question: [e.text for e in memory.recall(q.question, budget_tokens).entries[:3]] for q in scenario.queries}
        rows.append({"name": scenario.name, "category": scenario.category, "current_knowledge": current, "recall": recall})
    return rows


def run_offline(workdir, *, budget_tokens=2000) -> dict:
    workdir = Path(workdir)
    if workdir.exists() and any(workdir.iterdir()):
        raise ValueError("use an empty workdir so trials cannot reuse earlier memories")
    workdir.mkdir(parents=True, exist_ok=True)
    validate_scenarios(SCENARIOS)

    extractor = ReferenceExtractor()
    memory, log = build_memory(workdir, extractor=extractor, embedder=None)
    traces = drive_scenarios(
        memory, log, SCENARIOS, reference=True, extractor=extractor,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    metrics, restart_recall, context_departure_recall, oversized, costs = _evaluate(
        memory, log, SCENARIOS, traces, workdir, budget_tokens=budget_tokens)
    scenarios = _scenario_rows(memory, SCENARIOS, budget_tokens)

    report = {
        "mode": "reference-extraction-offline", "provider": None,
        "model": "ReferenceExtractor", "embedding": None,
        "scenarios_count": len(SCENARIOS),
        "categories": sorted({s.category for s in SCENARIOS}),
        "metrics": metrics, "restart_recall": restart_recall,
        "context_departure_recall": context_departure_recall,
        "oversized": oversized, "costs": costs,
        "limitations": (
            "Reference extractions are labels not model output; exact-text checks are "
            "retrieval diagnostics not semantic correctness; reported usage may omit failed "
            "requests and provider retries; live semantic accuracy requires run_live with an "
            "explicit model; valid supporting event IDs alone do not prove a claim is true."),
        "scenarios": scenarios,
    }
    (workdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_live(workdir, *, extractor, embedder=None, answerer=None, budget_tokens=2000) -> dict:
    workdir = Path(workdir)
    if workdir.exists() and any(workdir.iterdir()):
        raise ValueError("use an empty workdir so trials cannot reuse earlier memories")
    workdir.mkdir(parents=True, exist_ok=True)
    validate_scenarios(SCENARIOS)
    memory, log = build_memory(workdir, extractor=extractor, embedder=embedder)
    traces = drive_scenarios(memory, log, SCENARIOS, reference=False, extractor=extractor,
                             start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    metrics, restart, departure, oversized, costs = _evaluate(memory, log, SCENARIOS, traces, workdir, budget_tokens=budget_tokens)
    answers, usage = [], []
    if answerer is not None:
        for scenario in SCENARIOS:
            for query in scenario.queries:
                records = [{"id": e.provenance.record_id, "text": e.text} for e in memory.recall(query.question, budget_tokens).entries[:3]]
                raw = answerer.chat("Answer from these memory records. Prefer current facts over historical reports. Plans and attempts do not establish completed actions. If unsupported, answer unknown. Return JSON with answer and evidence_ids.\n" + json.dumps({"query": query.question, "records": records}), max_tokens=512)
                reported = getattr(answerer, "last_chat_usage", None)
                if isinstance(reported, dict):
                    usage.append(dict(reported))
                try:
                    parsed = parse_json_response(raw)
                    answer = parsed["answer"]
                    if not isinstance(answer, str):
                        raise ValueError("non-string answer")
                    cited = parsed.get("evidence_ids", [])
                    valid = isinstance(answer, str) and isinstance(cited, list) and all(i in {r["id"] for r in records} for i in cited)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    answer, valid = "<invalid JSON answer>", False
                answers.append({"scenario": scenario.name, "question": query.question, "answer": answer, "valid_citations": valid,
                    "expected_present": query.expected.casefold() in answer.casefold() if query.expected else answer.strip().casefold() == "unknown",
                    "forbidden_present": any(x.casefold() in answer.casefold() for x in query.forbidden)})
    costs.update(answer_chat_calls=len(answers), reported_answer_usage=usage)
    report = {
        "mode": "live-extraction",
        "provider": getattr(extractor, "model", type(extractor).__name__),
        "model": getattr(extractor, "model", None),
        "embedding": getattr(embedder, "model", None),
        "answers": answers,
        "scenarios_count": len(SCENARIOS),
        "categories": sorted({s.category for s in SCENARIOS}),
        "metrics": metrics,
        "restart_recall": restart,
        "context_departure_recall": departure,
        "oversized": oversized,
        "costs": costs,
        "scenarios": _scenario_rows(memory, SCENARIOS, budget_tokens),
        "limitations": (
            "Substring matches and valid citation IDs are recorded for review "
            "but are not treated as proof of semantic support. Source validity "
            "checks eligible IDs; labeled assertions separately check what those "
            "events establish. Live model "
            "quality and provider usage may vary."
        ),
    }
    (workdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDER_REGISTRY))
    parser.add_argument("--model")
    parser.add_argument("--embedding-provider", choices=sorted(PROVIDER_REGISTRY))
    parser.add_argument("--embedding-model")
    parser.add_argument("--answers", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.provider) != bool(args.model) or bool(args.embedding_provider) != bool(args.embedding_model):
        parser.error("provide both a provider and its model")
    if (args.answers or args.embedding_provider) and not args.provider:
        parser.error("live embeddings/answers require a live extraction provider")
    if not args.provider:
        report = run_offline(args.workdir)
    else:
        extractor = PROVIDER_REGISTRY[args.provider](model=args.model)
        embedder = PROVIDER_REGISTRY[args.embedding_provider](model=args.embedding_model) if args.embedding_provider else None
        report = run_live(args.workdir, extractor=extractor, embedder=embedder, answerer=extractor if args.answers else None)
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "scenarios")}, indent=2))


def validate_scenarios(scenarios):
    """Validate the scenario dataset, raising ValueError on the first defect.

    Per scenario: each episode's ``event_indices`` must be non-empty, sorted,
    and contiguous and in range; the episodes must partition the scenario's
    events exactly; fact assertions must carry subject/predicate/value and every
    other assertion must carry a ``statement``; every event role must be
    ``evidence`` or ``recall_context``; and every transition must be backed by a
    matching fact assertion (subject, predicate, and current value).
    """
    for scenario in scenarios:
        facts = []
        covered = []
        for episode in scenario.episodes:
            indices = episode.event_indices
            if not indices or tuple(sorted(indices)) != indices or any(b != a + 1 for a, b in zip(indices, indices[1:])):
                raise ValueError(f"{scenario.name}: episode indices must be sorted and contiguous")
            if any(i < 0 or i >= len(scenario.events) for i in indices):
                raise ValueError(f"{scenario.name}: episode index out of range")
            covered.extend(indices)
            for assertion in episode.assertions:
                support = assertion.get("support_event_indices")
                if (not isinstance(support, tuple) or not support
                    or any(type(i) is not int for i in support)
                    or len(set(support)) != len(support)
                    or any(i not in indices or scenario.events[i][3] != "evidence"
                           for i in support)):
                    raise ValueError(f"{scenario.name}: assertion lacks eligible supporting events")
                attribution = assertion.get("attribution")
                status = assertion.get("action_status")
                if (not isinstance(attribution, str) or attribution not in ATTRIBUTIONS
                    or not isinstance(status, str) or status not in ACTION_STATUSES):
                    raise ValueError(f"{scenario.name}: invalid assertion metadata")
                if assertion.get("kind") == "fact":
                    if not all(assertion.get(k) for k in ("subject", "predicate", "value")):
                        raise ValueError(f"{scenario.name}: malformed fact assertion")
                    facts.append(assertion)
                elif not assertion.get("statement"):
                    raise ValueError(f"{scenario.name}: malformed assertion")
        if sorted(covered) != list(range(len(scenario.events))):
            raise ValueError(f"{scenario.name}: episodes do not partition events")
        for event in scenario.events:
            if len(event) != 4 or event[3] not in {"evidence", "recall_context"}:
                raise ValueError(f"{scenario.name}: invalid event role")
        for transition in scenario.transitions:
            if not any(f["subject"].casefold() == transition.subject.casefold() and f["predicate"].casefold() == transition.predicate.casefold() and f["value"] == transition.current_value for f in facts):
                raise ValueError(f"{scenario.name}: transition lacks supporting fact")


if __name__ == "__main__":
    main()
