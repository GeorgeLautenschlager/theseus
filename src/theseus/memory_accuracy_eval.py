"""Labeled multi-event scenarios for consolidation accuracy evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

from theseus.memory_module import Episode, MemoryModule
from theseus.stimulus_log import StimulusLog
from theseus.tools.recall import RECALL_TOOL_NAME


class ReferenceExtractor:
    model = "ReferenceExtractor"

    def __init__(self):
        self.response = ""
        self.last_prompt = None

    def chat(self, prompt, **kwargs):
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
                extractor.response = json.dumps({
                    "summary": episode_spec.summary,
                    "assertions": [dict(assertion) for assertion in episode_spec.assertions],
                })
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


def _fact(subject, predicate, value, statement=None):
    return {"kind": "fact", "subject": subject, "predicate": predicate,
            "value": value, "statement": statement or f"{subject} {predicate}: {value}"}


def _principle(statement):
    return {"kind": "principle", "statement": statement}


def _note(statement):
    return {"kind": "event", "statement": statement}


REQUIRED_CATEGORIES = frozenset({
    "plan_then_failure", "plan_then_success", "alternate_wording_correction",
    "coexisting_preferences", "historical_report", "recall_repetition",
    "split_action_result", "principles", "oversized_tail",
})

_OVERSIZED_MESSAGE = ("padding. " * 8000) + "DECISIVE: The Atlas security token is ZULU-9."

SCENARIOS = (
    Scenario("atlas-payment", "plan_then_failure",
        (_ev("Alpha", "I will attempt the Atlas payment now."), _ev("tool", "Atlas payment attempt failed; no funds were received.")),
        (EpisodeSpec((0, 1), "I attempted the Atlas payment; it failed and no funds arrived.", (_fact("Atlas", "payment status", "failed", "Atlas payment attempt failed; no funds received."),)),),
        (Transition("Atlas", "payment status", "failed"),), ("failed",), ("completed", "received"),
        (Query("What is Atlas's payment status?", "failed", ("completed", "received", "paid")), Query("What is Atlas's bank account number?", None))),
    Scenario("project-backup", "plan_then_success",
        (_ev("Beta", "I plan to back up the project database."), _ev("tool", "Project database backup completed successfully.")),
        (EpisodeSpec((0, 1), "Beta planned the backup and it completed successfully.", (_fact("Project", "backup status", "completed"),)),),
        (Transition("Project", "backup status", "completed"),), ("completed",), ("planned", "not yet"),
        (Query("What is the project backup status?", "completed", ("planned", "not yet")),)),
    Scenario("atlas-deadline", "alternate_wording_correction",
        (_ev("human", "The Atlas prototype is due Friday."), _ev("human", "Scratch that — we pushed Atlas delivery to Monday.")),
        (EpisodeSpec((0,), "Atlas prototype is due Friday.", (_fact("Atlas", "prototype deadline", "Friday"),)), EpisodeSpec((1,), "Atlas delivery was pushed to Monday.", (_fact("Atlas", "prototype deadline", "Monday"),))),
        (Transition("Atlas", "prototype deadline", "Monday", ("Friday",)),), ("Monday",), ("Friday",),
        (Query("What is Atlas's prototype deadline?", "Monday", ("Friday",)),)),
    Scenario("user-prefs", "coexisting_preferences",
        (_ev("human", "I prefer morning meetings."), _ev("human", "I prefer concise written summaries.")),
        (EpisodeSpec((0,), "The user prefers morning meetings.", (_fact("George", "meeting time preference", "morning"),)), EpisodeSpec((1,), "The user prefers concise written summaries.", (_fact("George", "summary format preference", "concise written"),))),
        (Transition("George", "meeting time preference", "morning"), Transition("George", "summary format preference", "concise written")), ("morning", "concise"), (),
        (Query("What meeting time does the user prefer?", "morning"), Query("What summary format does the user prefer?", "concise"))),
    Scenario("atlas-phase", "historical_report",
        (_ev("Alpha", "Last quarter Atlas was in the pilot phase."), _ev("human", "Atlas is now in the production phase.")),
        (EpisodeSpec((0,), "Alpha reported that last quarter Atlas was in the pilot phase.", (_note("Last quarter Atlas was in the pilot phase."),)), EpisodeSpec((1,), "Atlas is now in the production phase.", (_fact("Atlas", "phase", "production"),))),
        (Transition("Atlas", "phase", "production"),), ("production", "pilot"), ("pilot",),
        (Query("What phase is Atlas in now?", "production", ("pilot",)),)),
    Scenario("raven-code", "recall_repetition",
        (_ev("human", "The Atlas access code is RAVEN-42."), _ev("agent", "(recalled) The Atlas access code was once QUAIL-7.", "recall_context")),
        (EpisodeSpec((0, 1), "The Atlas access code is RAVEN-42.", (_fact("Atlas", "access code", "RAVEN-42"),)),),
        (Transition("Atlas", "access code", "RAVEN-42"),), ("RAVEN-42",), ("QUAIL-7",),
        (Query("What is the Atlas access code?", "RAVEN-42"),) * 3),
    Scenario("boreal-invoice", "split_action_result",
        (_ev("Alpha", "Sending the Boreal invoice now."), _ev("tool", "Boreal invoice sent; confirmation BOR-77.")),
        (EpisodeSpec((0, 1), "Alpha sent the Boreal invoice; confirmation BOR-77.", (_fact("Boreal", "invoice status", "sent", "Boreal invoice sent; confirmation BOR-77."),)),),
        (Transition("Boreal", "invoice status", "sent"),), ("sent", "BOR-77"), ("draft",),
        (Query("What is Boreal's invoice status?", "sent", ("draft", "not sent")),)),
    Scenario("confirm-policy", "principles",
        (_ev("human", "Always confirm risky actions before executing."), _ev("human", "Reminder: confirm risky actions before executing."), _ev("human", "For low-risk actions, skip the confirmation step.")),
        (EpisodeSpec((0,), "The user set a rule to always confirm risky actions.", (_principle("Always confirm risky actions before executing."),)), EpisodeSpec((1,), "The user repeated the rule to confirm risky actions.", (_principle("Always confirm risky actions before executing."),)), EpisodeSpec((2,), "The user added that low-risk actions can skip confirmation.", (_principle("For low-risk actions, skip the confirmation step."),))),
        (), ("confirm risky actions", "low-risk"), (),
        (Query("What is the policy for risky actions?", "confirm"), Query("What is the policy for low-risk actions?", "low-risk"))),
    Scenario("atlas-token", "oversized_tail",
        (("tool", "tool_result", {"message": _OVERSIZED_MESSAGE}, "evidence"), _ev("human", "Store the Atlas security token safely.")),
        (EpisodeSpec((0, 1), "Atlas security token noted.", (_fact("Atlas", "security token", "ZULU-9"),)),),
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


def validate_scenarios(scenarios):
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
