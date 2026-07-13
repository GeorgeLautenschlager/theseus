from __future__ import annotations

from datetime import datetime, timezone

from theseus.memory_note import MemoryNote
from theseus.memory_store import MemoryStore
from theseus.stimulus_log import new_id


def make_note(**overrides) -> MemoryNote:
    defaults = dict(
        id=new_id(),
        ts=datetime.now(timezone.utc),
        content="George said hello.",
        context="George greeted the agent.",
        keywords=["George"],
        tags=["greeting"],
        links=[],
        source_span=("A", "B"),
        embedding=[1.0, 0.0],
    )
    defaults.update(overrides)
    return MemoryNote(**defaults)


class TestPersistence:
    def test_round_trips_notes_across_instances(self, tmp_path):
        path = tmp_path / "memory.jsonl"
        note = make_note()
        MemoryStore(path).add(note)

        reloaded = MemoryStore(path).read_all()

        assert len(reloaded) == 1
        assert reloaded[0] == note

    def test_tolerates_torn_final_line(self, tmp_path):
        path = tmp_path / "memory.jsonl"
        store = MemoryStore(path)
        note = make_note()
        store.add(note)
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"id": "torn')  # crash mid-write: no trailing newline

        reloaded = MemoryStore(path).read_all()

        assert [n.id for n in reloaded] == [note.id]

    def test_get_finds_note_by_id(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        note = store.add(make_note())

        assert store.get(note.id) == note
        assert store.get("missing") is None


class TestTopK:
    def test_ranks_by_cosine_similarity(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        far = store.add(make_note(embedding=[0.0, 1.0]))
        near = store.add(make_note(embedding=[1.0, 0.1]))

        results = store.top_k([1.0, 0.0], k=2)

        assert results == [near, far]

    def test_returns_at_most_k(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        for _ in range(3):
            store.add(make_note())

        assert len(store.top_k([1.0, 0.0], k=2)) == 2


class TestLastConsolidatedId:
    def test_none_when_empty(self, tmp_path):
        assert MemoryStore(tmp_path / "memory.jsonl").last_consolidated_id() is None

    def test_returns_max_span_end(self, tmp_path):
        store = MemoryStore(tmp_path / "memory.jsonl")
        store.add(make_note(source_span=("A", "C")))
        store.add(make_note(source_span=("D", "F")))
        store.add(make_note(source_span=("B", "E")))

        assert store.last_consolidated_id() == "F"
