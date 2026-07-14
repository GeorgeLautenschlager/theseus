from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from theseus.agentic_memory import AgenticMemory
from theseus.memory_note import MemoryNote
from theseus.memory_store import MemoryStore
from theseus.stimulus_log import StimulusLog


def make_provider(chat_responses):
    provider = MagicMock()
    provider.is_available.return_value = True
    provider.chat.side_effect = list(chat_responses)
    return provider


def make_embedder(embedding=None):
    embedder = MagicMock()
    embedder.is_available.return_value = True
    embedder.embed.return_value = embedding or [1.0, 0.0]
    return embedder


ENRICHMENT = json.dumps(
    {"context": "George introduced himself.", "keywords": ["George"], "tags": ["identity"]}
)


def make_memory(tmp_path, chat_responses, embedding=None):
    provider = make_provider(chat_responses)
    embedder = make_embedder(embedding)
    log = StimulusLog(tmp_path / "stimulus_log.jsonl")
    memory = AgenticMemory(
        model_providers=[provider],
        embedding_providers=[embedder],
        store=MemoryStore(tmp_path / "memory.jsonl"),
        stimulus_log=log,
    )
    return memory, provider, embedder


def seed_note(store, note_id="seed1", embedding=None, links=None) -> MemoryNote:
    return store.add(
        MemoryNote(
            id=note_id,
            ts=datetime.now(timezone.utc),
            content="earlier content",
            context="An earlier memory.",
            links=links or [],
            embedding=embedding or [1.0, 0.0],
        )
    )


class TestForm:
    def test_forms_enriched_embedded_note_from_new_events(self, tmp_path):
        memory, provider, embedder = make_memory(tmp_path, [ENRICHMENT], embedding=[0.5, 0.5])
        first = memory.stimulus_log.append(actor="george", type="exchange", content={"message": "Hi."})
        last = memory.stimulus_log.append(actor="george", type="exchange", content={"message": "I'm George."})

        memory.form()

        notes = memory.store.read_all()
        assert len(notes) == 1
        note = notes[0]
        assert note.context == "George introduced himself."
        assert note.keywords == ["George"]
        assert note.tags == ["identity"]
        assert note.embedding == [0.5, 0.5]
        assert note.source_span == (first.id, last.id)
        assert first.to_json() in note.content
        # Enrichment came from the chat provider; the embedding from the embedding provider.
        embedder.embed.assert_called_once_with("George introduced himself.")
        assert embedder.chat.call_count == 0

    def test_empty_log_forms_nothing(self, tmp_path):
        memory, provider, _ = make_memory(tmp_path, [])

        memory.form()

        assert len(memory.store) == 0
        assert provider.chat.call_count == 0

    def test_resumes_from_high_water_mark(self, tmp_path):
        memory, provider, _ = make_memory(
            tmp_path, [ENRICHMENT, ENRICHMENT, json.dumps({"links": []})]
        )
        memory.stimulus_log.append(actor="george", type="exchange", content={"message": "First."})
        memory.form()
        second = memory.stimulus_log.append(
            actor="george", type="exchange", content={"message": "Second."}
        )
        memory.form()

        notes = memory.store.read_all()
        assert len(notes) == 2
        # The second note starts exactly where the first left off — no re-consolidation.
        assert notes[1].source_span[0] == second.id
        assert "First." not in notes[1].content

    def test_nothing_new_is_a_no_op(self, tmp_path):
        memory, provider, _ = make_memory(tmp_path, [ENRICHMENT])
        memory.stimulus_log.append(actor="george", type="exchange", content={"message": "Hi."})
        memory.form()

        memory.form()  # no new events since the last formation

        assert len(memory.store) == 1
        assert provider.chat.call_count == 1

    def test_first_note_skips_link_decision(self, tmp_path):
        memory, provider, _ = make_memory(tmp_path, [ENRICHMENT])
        memory.stimulus_log.append(actor="george", type="exchange", content={"message": "Hi."})

        memory.form()

        assert provider.chat.call_count == 1  # construction only, no candidates to link
        assert memory.store.read_all()[0].links == []

    def test_links_note_to_llm_chosen_neighbors(self, tmp_path):
        memory, provider, _ = make_memory(
            tmp_path, [ENRICHMENT, json.dumps({"links": ["seed1"]})]
        )
        seed_note(memory.store, "seed1")
        memory.stimulus_log.append(actor="george", type="exchange", content={"message": "Hi."})

        memory.form()

        note = memory.store.read_all()[-1]
        assert note.links == ["seed1"]
        link_prompt = provider.chat.call_args_list[1].kwargs["prompt"]
        assert "[seed1]" in link_prompt

    def test_hallucinated_link_ids_are_dropped(self, tmp_path):
        memory, provider, _ = make_memory(
            tmp_path, [ENRICHMENT, json.dumps({"links": ["not_a_note"]})]
        )
        seed_note(memory.store, "seed1")
        memory.stimulus_log.append(actor="george", type="exchange", content={"message": "Hi."})

        memory.form()

        assert memory.store.read_all()[-1].links == []

    def test_malformed_llm_json_does_not_raise(self, tmp_path):
        memory, provider, _ = make_memory(tmp_path, ["this is not json"])
        memory.stimulus_log.append(actor="george", type="exchange", content={"message": "Hi."})

        memory.form()

        assert len(memory.store) == 0


class TestRetrieve:
    def test_empty_store_retrieves_nothing_without_embedding(self, tmp_path):
        memory, _, embedder = make_memory(tmp_path, [])

        assert memory.retrieve("anything") == ""
        assert embedder.embed.call_count == 0

    def test_returns_rendered_top_k_by_similarity(self, tmp_path):
        memory, _, _ = make_memory(tmp_path, [], embedding=[1.0, 0.0])
        memory.k_retrieve = 1
        near = seed_note(memory.store, "near", embedding=[1.0, 0.0])
        seed_note(memory.store, "far", embedding=[0.0, 1.0])

        rendered = memory.retrieve("query")

        assert "[near]" in rendered
        assert "[far]" not in rendered
        assert near.context in rendered

    def test_expands_one_hop_along_links(self, tmp_path):
        memory, _, _ = make_memory(tmp_path, [], embedding=[1.0, 0.0])
        memory.k_retrieve = 1
        seed_note(memory.store, "linked", embedding=[0.0, 1.0])
        seed_note(memory.store, "hit", embedding=[1.0, 0.0], links=["linked"])

        rendered = memory.retrieve("query")

        assert "[hit]" in rendered
        assert "[linked]" in rendered

    def test_embed_failure_returns_no_memories(self, tmp_path):
        memory, _, embedder = make_memory(tmp_path, [])
        seed_note(memory.store)
        embedder.embed.side_effect = RuntimeError("embedding endpoint down")

        assert memory.retrieve("query") == ""
