from __future__ import annotations

from datetime import datetime, timezone

from theseus.memory_note import MemoryNote
from theseus.memory_prompts import (
    build_link_decision_prompt,
    build_note_construction_prompt,
    link_json_schema,
    note_json_schema,
)


def make_note(note_id: str, context: str) -> MemoryNote:
    return MemoryNote(
        id=note_id,
        ts=datetime.now(timezone.utc),
        content="raw content",
        context=context,
        keywords=["kw"],
        tags=["tag"],
    )


class TestNoteConstruction:
    def test_schema_requires_all_enrichment_fields(self):
        schema = note_json_schema()
        assert set(schema["required"]) == {"context", "keywords", "tags"}
        assert schema["additionalProperties"] is False

    def test_prompt_contains_the_events(self):
        prompt = build_note_construction_prompt('{"actor":"george","type":"exchange"}')
        assert '{"actor":"george","type":"exchange"}' in prompt
        assert "<events>" in prompt

    def test_prompt_asks_for_a_first_person_account(self):
        # `context` is the only field rendered into a recollection and the only field
        # retrieval embeds, so it has to carry the substance — in the agent's own voice,
        # since it is what the agent will read back as its own memory.
        prompt = build_note_construction_prompt("events")
        assert "first person" in prompt.lower()

    def test_prompt_says_whose_voice_the_first_person_is(self):
        """Asking for "the first person" alone makes the model adopt whichever voice is
        loudest in the events — it wrote "I learned that my name is George" about the
        *user's* name. The prompt has to pin "I" to the agent and push the user into the
        third person, using a cue the model can check against the events themselves."""
        prompt = build_note_construction_prompt("events").lower()

        assert "third person" in prompt, "the user must be named, not voiced"
        assert "decision" in prompt, "the actor on 'decision' events identifies the agent"


class TestLinkDecision:
    def test_schema_constrains_links_to_candidate_ids(self):
        schema = link_json_schema(["id1", "id2"])
        assert schema["properties"]["links"]["items"]["enum"] == ["id1", "id2"]

    def test_prompt_contains_new_note_and_candidates(self):
        new = make_note("new1", "George prefers tea.")
        candidates = [make_note("old1", "George dislikes coffee.")]

        prompt = build_link_decision_prompt(new, candidates)

        assert "George prefers tea." in prompt
        assert "George dislikes coffee." in prompt
        assert "[old1]" in prompt
