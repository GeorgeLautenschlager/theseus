from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from theseus.agentic_memory import AgenticMemory
from theseus.memory_note import MemoryNote
from theseus.memory_store import MemoryStore
from theseus.stimulus_log import StimulusEvent, new_id


def make_event(content=None) -> StimulusEvent:
    return StimulusEvent(
        id=new_id(),
        ts=datetime.now(timezone.utc),
        actor="george",
        type="exchange",
        content=content or {"message": "Hello, my name is George."},
    )


def make_provider(chat_responses, embedding=None):
    provider = MagicMock()
    provider.is_available.return_value = True
    provider.chat.side_effect = list(chat_responses)
    provider.embed.return_value = embedding or [1.0, 0.0]
    return provider


ENRICHMENT = json.dumps(
    {"context": "George introduced himself.", "keywords": ["George"], "tags": ["identity"]}
)


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
    def test_forms_enriched_embedded_note_from_events(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        provider = make_provider([ENRICHMENT], embedding=[0.5, 0.5])
        memory = AgenticMemory(model_providers=[provider], store=store)
        events = [make_event(), make_event()]

        note = memory.form(events)

        assert note.context == "George introduced himself."
        assert note.keywords == ["George"]
        assert note.tags == ["identity"]
        assert note.embedding == [0.5, 0.5]
        assert note.source_span == (events[0].id, events[-1].id)
        assert events[0].to_json() in note.content
        assert store.read_all() == [note]

    def test_empty_batch_forms_nothing(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        provider = make_provider([])
        memory = AgenticMemory(model_providers=[provider], store=store)

        assert memory.form([]) is None
        assert provider.chat.call_count == 0

    def test_first_note_skips_link_decision(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        provider = make_provider([ENRICHMENT])
        memory = AgenticMemory(model_providers=[provider], store=store)

        note = memory.form([make_event()])

        assert provider.chat.call_count == 1  # construction only, no candidates to link
        assert note.links == []

    def test_links_note_to_llm_chosen_neighbors(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        seed_note(store, "seed1")
        provider = make_provider([ENRICHMENT, json.dumps({"links": ["seed1"]})])
        memory = AgenticMemory(model_providers=[provider], store=store)

        note = memory.form([make_event()])

        assert note.links == ["seed1"]
        link_prompt = provider.chat.call_args_list[1].kwargs["prompt"]
        assert "[seed1]" in link_prompt

    def test_hallucinated_link_ids_are_dropped(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        seed_note(store, "seed1")
        provider = make_provider([ENRICHMENT, json.dumps({"links": ["not_a_note"]})])
        memory = AgenticMemory(model_providers=[provider], store=store)

        note = memory.form([make_event()])

        assert note.links == []

    def test_malformed_llm_json_does_not_raise(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        provider = make_provider(["this is not json"])
        memory = AgenticMemory(model_providers=[provider], store=store)

        assert memory.form([make_event()]) is None
        assert len(store) == 0


class TestRetrieve:
    def test_empty_store_retrieves_nothing_without_embedding(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        provider = make_provider([])
        memory = AgenticMemory(model_providers=[provider], store=store)

        assert memory.retrieve("anything") == []
        assert provider.embed.call_count == 0

    def test_returns_top_k_by_similarity(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        near = seed_note(store, "near", embedding=[1.0, 0.0])
        seed_note(store, "far", embedding=[0.0, 1.0])
        provider = make_provider([], embedding=[1.0, 0.0])
        memory = AgenticMemory(model_providers=[provider], store=store, k_retrieve=1)

        assert memory.retrieve("query") == [near]

    def test_expands_one_hop_along_links(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        linked = seed_note(store, "linked", embedding=[0.0, 1.0])
        hit = seed_note(store, "hit", embedding=[1.0, 0.0], links=["linked"])
        provider = make_provider([], embedding=[1.0, 0.0])
        memory = AgenticMemory(model_providers=[provider], store=store, k_retrieve=1)

        assert memory.retrieve("query") == [hit, linked]

    def test_embed_failure_returns_no_memories(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        seed_note(store)
        provider = make_provider([])
        provider.embed.side_effect = RuntimeError("no embedding model")
        memory = AgenticMemory(model_providers=[provider], store=store)

        assert memory.retrieve("query") == []
