"""Pure prompt builders and JSON schemas for the memory pipelines.

Mirrors cognitive_prompts.py: no I/O, no state — just strings and schemas, so
the whole module is offline-testable. Steps:

1. Note construction (AgenticMemory) — distill a batch of stimulus events into
   an enriched note (context, keywords, tags).
2. Link decision (AgenticMemory) — given the new note and its nearest
   neighbors, choose which (if any) existing notes it should link to.
3. Assertion extraction (layered MemoryModule) — distill one episode's evidence
   into a summary plus candidate assertions for write routing.
"""

from __future__ import annotations

from theseus.memory_note import MemoryNote


def note_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "context": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["context", "keywords", "tags"],
        "additionalProperties": False,
    }


def build_note_construction_prompt(events_text: str) -> str:
    return (
        "You are the memory-formation step of a cognitive agent. Below is a batch of "
        "recent events from the agent's stimulus log (one JSON event per line). Distill "
        "them into a single memory note.\n\n"
        "<events>\n"
        f"{events_text}\n"
        "</events>\n\n"
        "You are writing as the agent, in its voice. The agent is the actor on the "
        "'decision' and 'tool_result' events; everyone else in the log — the user above "
        "all — is someone else. \"I\" always means the agent. Refer to the user in the "
        "third person and by name where you know it. Never write as the user: \"I learned "
        "the user's name is George\" is right, \"I learned that my name is George\" is "
        "wrong.\n\n"
        "Produce:\n"
        "- context: 2-4 sentences in the first person, as the agent recalling this later "
        "— \"I met a user named George...\", \"George told me...\". This is the whole "
        "memory: the raw events are not kept alongside it, so carry the substance here. "
        "Prefer durable facts (names, preferences, decisions, commitments) over "
        "conversational filler.\n"
        "- keywords: the specific entities and terms involved.\n"
        "- tags: a few broad category labels.\n\n"
        "Reply with a single JSON object and nothing else — no code fences, no commentary. "
        'Use double quotes: {"context": "...", "keywords": ["..."], "tags": ["..."]}'
    )


def link_json_schema(candidate_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "links": {
                "type": "array",
                "items": {"type": "string", "enum": candidate_ids},
            },
        },
        "required": ["links"],
        "additionalProperties": False,
    }


def extraction_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "assertions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["fact", "principle", "event"]},
                        "subject": {"type": "string"},
                        "predicate": {"type": "string"},
                        "value": {"type": "string"},
                        "statement": {"type": "string"},
                        "support_event_ids": {"type": "array", "items": {"type": "string"},
                                              "minItems": 1, "uniqueItems": True},
                        "attribution": {"enum": ["direct_observation", "partner_report", "inference"]},
                        "reported_by": {"type": "string"},
                        "action_status": {"enum": ["not_applicable", "intention", "attempt",
                                                   "failure", "confirmed_outcome"]},
                    },
                    "required": ["kind", "statement", "support_event_ids", "attribution",
                                 "action_status"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "assertions"],
        "additionalProperties": False,
    }


def build_extraction_prompt(evidence_text: str, context_text: str = "") -> str:
    context_block = (
        f"\n<context_only>\n{context_text}\n</context_only>\n\n"
        "The context_only block shows what the agent recalled while this episode was "
        "happening. It is there so you can interpret the evidence; it is NOT evidence — "
        "never source an assertion from it or list one of its IDs as support.\n\n" if context_text else ""
    )
    return (
        "You are the memory-consolidation step of a cognitive agent. Below is one episode: "
        "the stimulus events that just happened (one JSON event per line). Distill it.\n\n"
        "<evidence>\n"
        f"{evidence_text}\n"
        "</evidence>\n"
        + context_block
        + "Produce:\n"
        "- summary: 1-3 sentences, in the agent's first person, of what happened in this "
        "episode. This is the whole episode record: carry the substance here.\n"
        "- assertions: the durable claims worth keeping, each with:\n"
        "Keep each claim atomic and preserve names, amounts, dates, and task owners. "
        "A decision or plan to act is not evidence that the action succeeded. "
        "Keep intentions, attempts, failures, and confirmed outcomes distinct. "
        "A partner's message is a report from that partner; preserve that attribution. "
        "Use consistent subjects and predicates for updates to existing facts. "
        "Do not infer successful payments or completed work without confirming evidence. "
        "Some entries are a bounded excerpt of one larger event, marked with chunk_of, "
        "chunk_index, and chunk_count fields rather than being the whole event — treat "
        "each as one piece of a longer record. Do not conclude an event failed or is "
        "incomplete just because a given excerpt stops before showing the outcome; the "
        "decisive result may be in a later chunk.\n"
        '  - kind: "fact" (a checkable claim about a subject), "principle" (a generalized '
        'rule or preference that would guide future behavior), or "event" (something that '
        "happened, not durable enough to be a fact).\n"
        '  - for kind "fact": subject, predicate, value — e.g. subject "George", predicate '
        '"prefers", value "dark mode". Only use kind "fact" when you can state all three.\n'
        "  - statement: one plain sentence stating the claim, for every kind.\n"
        "  - support_event_ids: one or more exact id strings from events in the evidence "
        "block that support this claim. Do not use context_only IDs or invent IDs. "
        "If no evidence event supports a claim, omit the claim.\n"
        '  - attribution: "partner_report" for a partner or user saying something; '
        'include reported_by with that event actor. Use "direct_observation" for '
        'an observed tool outcome and "inference" for a deduction from evidence. '
        "A report is not independent confirmation of its contents.\n"
        '  - action_status: "not_applicable" for non-actions, "intention" for a plan, '
        '"attempt" for an unfinished action, "failure" for a failed action, or '
        '"confirmed_outcome" only when an evidence event confirms completion.\n'
        "Skip conversational filler and anything that is only true of this exact moment.\n\n"
        "Reply with a single JSON object and nothing else — no code fences, no commentary. "
        'Use double quotes: {"summary": "...", "assertions": [{"kind": "fact", '
        '"subject": "...", "predicate": "...", "value": "...", "statement": "...", '
        '"support_event_ids": ["<evidence event id>"], "attribution": "partner_report", '
        '"reported_by": "<event actor>", "action_status": "not_applicable"}]}'
    )


def reconciliation_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_index": {"type": "integer"},
                        "decision": {"enum": ["new", "reinforce", "replace", "coexist",
                                              "contradiction", "historical"]},
                        "target_id": {"type": "string"},
                    },
                    "required": ["candidate_index", "decision"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def build_reconciliation_prompt(candidates: list[dict], existing: list) -> str:
    rendered_candidates = "\n".join(
        f"[{i}] subject={c['subject']!r} predicate={c['predicate']!r} value={c['value']!r} "
        f"statement={c['statement']!r} attribution={c.get('attribution')!r} "
        f"reported_by={c.get('reported_by')!r} action_status={c.get('action_status')!r}"
        for i, c in enumerate(candidates)
    )
    rendered_existing = "\n".join(record.render() for record in existing) or "(none)"
    return (
        "You are the knowledge-reconciliation step of a cognitive agent. Below are newly "
        "extracted candidate facts and the agent's existing current knowledge that might "
        "relate to them, by shared subject. Decide, for each candidate, how it relates to "
        "what the agent already believes — do not just match on wording.\n\n"
        "<candidates>\n"
        f"{rendered_candidates}\n"
        "</candidates>\n\n"
        "<existing_knowledge>\n"
        f"{rendered_existing}\n"
        "</existing_knowledge>\n\n"
        "For each candidate, choose exactly one decision:\n"
        '- "new": nothing existing describes the same attribute; no target_id.\n'
        '- "reinforce": an existing record already states the same value for the same '
        "attribute, just worded differently or restated — this does not change what the "
        "agent believes. target_id: that record's id.\n"
        '- "replace": an existing record describes the same specific attribute but the '
        "value has genuinely changed — a correction or update, however differently it is "
        "worded (a different predicate string does not by itself mean a different "
        "attribute). target_id: the record it replaces.\n"
        '- "coexist": an existing record shares the subject and a similarly broad '
        "predicate, but describes a different attribute or aspect — e.g. two independent "
        "preferences filed under one general \"preference\" predicate. Both remain true at "
        "once; no target_id.\n"
        '- "contradiction": an existing record and this candidate describe the same '
        "attribute with conflicting values, and neither is clearly more authoritative or "
        "more current than the other — keep both visible rather than guessing which is "
        "right. target_id: the conflicting record.\n"
        '- "historical": this candidate itself describes a past or since-superseded state '
        "(a prior value, something that used to be true, an earlier report), not the "
        "current one — judge this from what the claim itself describes, never from which "
        "episode was processed more recently. Record it, but it must not become current. "
        "target_id: the existing record it is historical relative to, if any.\n\n"
        "A later processing time never by itself justifies replacing a current record — "
        "only the claim's own effective time and content do. When unsure between replace "
        "and contradiction, prefer contradiction: an unresolved conflict should stay "
        "visible, not be silently decided for the agent.\n\n"
        "Reply with a single JSON object and nothing else — no code fences, no commentary. "
        'Use double quotes: {"decisions": [{"candidate_index": 0, "decision": "new"}, '
        '{"candidate_index": 1, "decision": "replace", "target_id": "<existing id>"}]}'
    )


def build_principle_reconciliation_prompt(candidates: list[dict], existing: list) -> str:
    rendered_candidates = "\n".join(
        f"[{i}] statement={c['statement']!r} attribution={c.get('attribution')!r} "
        f"reported_by={c.get('reported_by')!r}"
        for i, c in enumerate(candidates)
    )
    rendered_existing = "\n".join(record.render() for record in existing) or "(none)"
    return (
        "You are the wisdom-reconciliation step of a cognitive agent. Below are newly "
        "extracted candidate principles — generalized rules or preferences — and the "
        "agent's existing current principles that might relate to them. Decide, for each "
        "candidate, how it relates to what the agent already believes.\n\n"
        "<candidates>\n"
        f"{rendered_candidates}\n"
        "</candidates>\n\n"
        "<existing_knowledge>\n"
        f"{rendered_existing}\n"
        "</existing_knowledge>\n\n"
        "For each candidate, choose exactly one decision:\n"
        '- "new": nothing existing expresses the same generalization; no target_id.\n'
        '- "reinforce": an existing record already expresses essentially the same rule or '
        "preference, however differently worded — independent evidence for the same "
        "principle, not a new one. target_id: that record's id.\n"
        '- "contradiction": an existing record and this candidate genuinely conflict (e.g. '
        '"always confirm risky actions" vs "never ask, just proceed" for the same kind of '
        "action) and neither is clearly the current rule — keep both visible rather than "
        "guessing which governs. A narrower rule for a specific case is not a contradiction "
        "of a general one (\"skip confirmation for low-risk actions\" refines, it does not "
        "conflict with, \"confirm risky actions\") — that is \"new\". target_id: the "
        "conflicting record.\n"
        '- "historical": this candidate describes a rule or preference that no longer '
        "applies (explicitly superseded, reversed, or stated as past practice), not a "
        "living principle — record it, but it must not become current. target_id: the "
        "existing record it is historical relative to, if any.\n\n"
        "An inferred generalization (attribution \"inference\") stays provisional until "
        "independent episodes reinforce it — that policy is applied outside this decision, "
        "but it means restating the same inference twice in one breath is not two episodes "
        "of support, and only genuinely independent reinforcement should be marked as such. "
        "An explicitly stated preference or rule (attribution \"partner_report\" or "
        "\"direct_observation\") does not need reinforcement to matter; judge it as new or "
        "reinforcing on its content, not on how many times it has been said.\n\n"
        "Reply with a single JSON object and nothing else — no code fences, no commentary. "
        'Use double quotes: {"decisions": [{"candidate_index": 0, "decision": "new"}, '
        '{"candidate_index": 1, "decision": "reinforce", "target_id": "<existing id>"}]}'
    )


def build_link_decision_prompt(new_note: MemoryNote, candidates: list[MemoryNote]) -> str:
    rendered = "\n\n".join(c.render() for c in candidates)
    return (
        "You are the memory-linking step of a cognitive agent. A new memory note was just "
        "formed. Below are its nearest existing memories by similarity. Decide which of "
        "them (if any) are genuinely related to the new note — shared subject matter, a "
        "continuation of the same thread, or context that would help interpret it later. "
        "Mere surface similarity is not a reason to link.\n\n"
        "<new_note>\n"
        f"{new_note.render()}\n"
        "</new_note>\n\n"
        "<candidates>\n"
        f"{rendered}\n"
        "</candidates>\n\n"
        "Reply with a single JSON object and nothing else — no code fences, no commentary. "
        "List the ids (shown in [brackets]) of the candidates to link, or an empty list: "
        '{"links": ["<id>", ...]}'
    )
