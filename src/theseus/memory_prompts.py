"""Pure prompt builders and JSON schemas for the AgenticMemory pipeline.

Mirrors cognitive_prompts.py: no I/O, no state — just strings and schemas, so
the whole module is offline-testable. Two LLM steps:

1. Note construction — distill a batch of stimulus events into an enriched
   note (context, keywords, tags).
2. Link decision — given the new note and its nearest neighbors, choose which
   (if any) existing notes it should link to.
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
