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
                    },
                    "required": ["kind", "statement"],
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
        "never source an assertion from it.\n\n" if context_text else ""
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
        '  - kind: "fact" (a checkable claim about a subject), "principle" (a generalized '
        'rule or preference that would guide future behavior), or "event" (something that '
        "happened, not durable enough to be a fact).\n"
        '  - for kind "fact": subject, predicate, value — e.g. subject "George", predicate '
        '"prefers", value "dark mode". Only use kind "fact" when you can state all three.\n'
        "  - statement: one plain sentence stating the claim, for every kind.\n"
        "Skip conversational filler and anything that is only true of this exact moment.\n\n"
        "Reply with a single JSON object and nothing else — no code fences, no commentary. "
        'Use double quotes: {"summary": "...", "assertions": [{"kind": "fact", '
        '"subject": "...", "predicate": "...", "value": "...", "statement": "..."}]}'
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
