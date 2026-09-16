"""Failure injection and restart tests for memory, using offline providers."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from theseus.layer_store import append_record, load_lines
from theseus.memory_module import MemoryModule, Episode
from theseus.memory_consolidator import MemoryConsolidator
from theseus.memory_layer import MemoryRecord, MemoryLayer
from theseus.wisdom_layer import WisdomRecord, WisdomLayer
from theseus.knowledge_layer import KnowledgeRecord, KnowledgeLayer
from theseus.stimulus_log import StimulusLog


class Embedder:
    model = "test-v1"

    def __init__(self):
        self.fail = False
        self.vector = [1.0, 0.0]
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding outage")
        return self.vector


class Extractor:
    def __init__(self, response=None):
        self.calls = 0
        self.response = response or json.dumps({
            "summary": "Client deadline is Friday; Beta owns delivery.",
            "assertions": [
                {"kind": "fact", "subject": "Client", "predicate": "deadline", "value": "Friday",
                 "statement": "Client deadline is Friday."},
                {"kind": "event", "statement": "Beta accepted the client delivery task."},
                {"kind": "principle", "statement": "Confirm client deadlines."},
            ],
        })
        self.prompts = []

    def chat(self, prompt, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        return self.response


def setup(tmp_path, embedder=None, extractor=None):
    log = StimulusLog(tmp_path / "log.jsonl")
    event = log.append("human", "chat_message", {"message": "Client deadline is Friday; Beta owns delivery."})
    chat = extractor or Extractor()
    module = MemoryModule(tmp_path / "memory", log, model_providers=[chat],
                          embedding_providers=[embedder] if embedder else [])
    return module, log, Episode("one", event.id, event.id), chat


@pytest.mark.parametrize("tail", [b'{"id":', b'{"text":"\xf0\x9f', b'{"id":"valid"}'])
def test_append_after_torn_or_unterminated_record(tmp_path, tail):
    path = tmp_path / "log"
    append_record(path, '{"id":"first"}')
    with path.open("ab") as stream:
        stream.write(tail)
    load_lines(path)
    append_record(path, '{"id":"last"}')
    assert json.loads(load_lines(path)[-1])["id"] == "last"
    assert len(load_lines(path)) == (3 if b"valid" in tail else 2)


def test_interior_corruption_is_not_erased(tmp_path):
    path = tmp_path / "log"
    path.write_text('broken\n{"valid":true}\n')
    with pytest.raises(ValueError, match="corrupt interior"):
        load_lines(path)


@pytest.mark.parametrize("stage", ["knowledge", "memory", "wisdom", "ledger", "trace"])
@pytest.mark.parametrize("after_write", [False, True])
def test_interrupted_commit_recovers_without_reextracting_or_duplicating(tmp_path, monkeypatch, stage, after_write):
    import theseus.memory_module as implementation
    module, log, episode, chat = setup(tmp_path)
    if stage in ("knowledge", "memory", "wisdom"):
        layer_type = type(getattr(module, stage))
        original = layer_type.add

        def interrupted(self, record):
            if after_write:
                original(self, record)
            raise OSError("power lost")

        with monkeypatch.context() as patch:
            patch.setattr(layer_type, "add", interrupted)
            with pytest.raises(OSError):
                module.consolidate(episode)
    else:
        original = implementation.append_record
        filename = "consolidation_ledger.jsonl" if stage == "ledger" else "consolidation.jsonl"

        def interrupted(path, line):
            if path.name == filename:
                if after_write:
                    original(path, line)
                raise OSError("power lost")
            original(path, line)

        with monkeypatch.context() as patch:
            patch.setattr(implementation, "append_record", interrupted)
            with pytest.raises(OSError):
                module.consolidate(episode)
    reopened = MemoryModule(module.memory_dir, log, model_providers=[chat])
    assert reopened.consolidate(episode).skipped
    assert chat.calls == 1
    assert [len(reopened.knowledge), len(reopened.memory), len(reopened.wisdom)] == [1, 2, 1]
    assert len(load_lines(module.memory_dir / "consolidation_ledger.jsonl")) == 1
    assert len(load_lines(module.memory_dir / "traces" / "consolidation.jsonl")) == 1
    assert not (module.memory_dir / "pending.json").exists()


@pytest.mark.parametrize("response", ['{}', '[]', 'null', '{"summary":"s","assertions":{}}',
                                      '{"summary":null,"assertions":[]}', 'not JSON'])
def test_invalid_envelope_does_not_complete_episode(tmp_path, response):
    module, _, episode, chat = setup(tmp_path, extractor=Extractor(response))
    with pytest.raises(RuntimeError, match="valid extraction"):
        module.consolidate(episode)
    assert len(module.memory) == 0
    assert not (module.memory_dir / "consolidation_ledger.jsonl").exists()
    chat.response = Extractor().response
    assert not module.consolidate(episode).skipped


def test_invalid_first_provider_falls_back(tmp_path):
    module, _, episode, _ = setup(tmp_path, extractor=Extractor('{}'))
    backup = Extractor()
    module._model_providers.append(backup)
    result = module.consolidate(episode)
    assert result.extracted == 3 and backup.calls == 1


def test_embedding_outage_then_recovery_keeps_recall_usable_and_repairs_index(tmp_path):
    embed = Embedder()
    embed.fail = True
    module, log, episode, _ = setup(tmp_path, embed)
    module.consolidate(episode)
    assert module.recall("client deadline", 2000).entries
    embed.fail = False
    event = log.append("human", "chat_message", {"message": "another event"})
    module.consolidate(Episode("two", event.id, event.id))
    assert module.recall("client deadline", 2000).entries
    original = (module.memory_dir / "memory.jsonl").read_bytes()
    assert module.repair_embeddings(limit=10) == 3  # episode, event, principle from the outage
    assert module.repair_embeddings(limit=10) == 0
    assert (module.memory_dir / "memory.jsonl").read_bytes() == original
    embed.model = "test-v2"  # even equal dimensions represent a different vector space
    assert module.repair_embeddings(limit=2) == 2
    assert module.recall("client deadline", 2000).entries


@pytest.mark.parametrize("bad", [[], [1.0], [float('nan'), 1.0], [float('inf'), 0.0], [0.0, 0.0]])
def test_vector_layers_ignore_invalid_or_wrong_dimension_vectors(tmp_path, bad):
    now = datetime.now(timezone.utc)
    memory = MemoryLayer(tmp_path / "memory")
    wisdom = WisdomLayer(tmp_path / "wisdom")
    memory.add(MemoryRecord("bad", now, "bad", "bad", bad))
    memory.add(MemoryRecord("good", now, "good", "good", [1.0, 0.0]))
    wisdom.add(WisdomRecord("bad", now, "bad", bad))
    wisdom.add(WisdomRecord("good", now, "good", [1.0, 0.0]))
    assert [hit.id for hit in memory.query([1.0, 0.0])] == ["good"]
    assert [hit.id for hit in wisdom.query([1.0, 0.0])] == ["good"]


def test_late_old_fact_cannot_replace_new_fact(tmp_path):
    layer = KnowledgeLayer(tmp_path / "knowledge")
    now = datetime.now(timezone.utc)
    layer.add(KnowledgeRecord("new", now, "Client", "deadline", "Monday"))
    layer.add(KnowledgeRecord("old", now - timedelta(days=1), "Client", "deadline", "Friday"))
    assert layer.current()[0].value == "Monday"
    assert KnowledgeLayer(layer.path).current()[0].value == "Monday"


def test_recall_filters_unrelated_and_previous_recall_events(tmp_path):
    module, log, episode, _ = setup(tmp_path)
    module.consolidate(episode)
    for _ in range(25):
        log.append("agent", "decision", {"text": "weather gardening"})
    log.append("agent", "tool_result", {"tool": "recall", "output": "Client deadline SECRET_ECHO"})
    result = module.recall("client deadline", 2000)
    assert result.entries[0].provenance.layer == "knowledge"
    assert all("SECRET_ECHO" not in entry.text for entry in result.entries)


def test_restart_keeps_failed_episode_boundaries_even_after_more_events(tmp_path):
    module, log, _, chat = setup(tmp_path, extractor=Extractor('{}'))
    runner = MemoryConsolidator(module)
    with pytest.raises(RuntimeError):
        runner.tick()
    pending = json.loads((module.memory_dir / "formation" / "cursor.json").read_text())["pending"]
    log.append("human", "chat_message", {"message": "NEXT_EPISODE"})
    chat.response = Extractor().response
    restarted = MemoryConsolidator(MemoryModule(module.memory_dir, log, model_providers=[chat]))
    assert restarted.tick().episode_id == pending["episode_id"]
    assert "NEXT_EPISODE" not in chat.prompts[-1]
    assert restarted.pending_events == 1


def test_oversized_event_is_chunked_and_decisive_ending_reaches_extraction(tmp_path):
    """An event too large for one request is split into several bounded chunks
    that together cover it completely, so the decisive marker at its very end
    reaches extraction instead of being cut away by a prefix truncation — and
    identical assertions the same canned response yields per chunk collapse
    into one via dedup."""
    module, log, _, chat = setup(tmp_path)
    event = log.append("tool", "tool_result", {"output": "large " * 20000 + "END_MARKER"})
    result = module.consolidate(Episode("big", event.id, event.id))
    assert len(chat.prompts) > 1
    assert all(len(p) <= module._max_input_chars for p in chat.prompts)
    assert all(event.id in p for p in chat.prompts)  # original event id carried through every chunk
    assert "END_MARKER" in chat.prompts[-1]
    assert result.extracted == 3  # repeated per-chunk assertions deduped, not multiplied
    assert "END_MARKER" in module.memory.read_all()[-1].content
    assert "END_MARKER" in module.recall("END_MARKER", 2000).entries[0].text


def test_mixed_short_and_long_events_pack_short_ones_whole_and_chunk_the_long_one(tmp_path):
    """Short events aren't truncated down to an equal per-event share just
    because one event in the same episode is huge — unused capacity from the
    short events is not imposed as a ceiling on them."""
    module, log, _, chat = setup(tmp_path)
    first = log.append("human", "chat_message", {"message": "short one"})
    log.append("human", "chat_message", {"message": "short two"})
    big = log.append("tool", "tool_result", {"output": "large " * 20000 + "END_MARKER"})
    last = log.append("human", "chat_message", {"message": "short three"})
    module.consolidate(Episode("mix", first.id, last.id))

    pack_prompts = [p for p in chat.prompts if "short one" in p]
    assert len(pack_prompts) == 1
    assert "short two" in pack_prompts[0] and "short three" in pack_prompts[0]
    assert big.id not in pack_prompts[0]  # the oversized event stayed out of the shared pack

    chunk_prompts = [p for p in chat.prompts if big.id in p]
    assert chunk_prompts and all(len(p) <= module._max_input_chars for p in chunk_prompts)
    assert "END_MARKER" in chunk_prompts[-1]


def test_failed_chunk_extraction_leaves_episode_pending_and_writes_nothing(tmp_path):
    """A required chunk that no provider can extract must not complete the
    episode partially — same transaction guarantee a single-request failure
    already gets, extended to every unit chunking can produce."""

    class FlakyExtractor:
        def __init__(self, fail_after):
            self.calls = 0
            self.prompts = []
            self.fail_after = fail_after

        def chat(self, prompt, **kwargs):
            self.calls += 1
            self.prompts.append(prompt)
            if self.calls > self.fail_after:
                raise RuntimeError("boom")
            return Extractor().response

    chat = FlakyExtractor(fail_after=1)
    module, log, _, _ = setup(tmp_path, extractor=chat)
    event = log.append("tool", "tool_result", {"output": "large " * 20000 + "END_MARKER"})
    episode = Episode("big", event.id, event.id)
    with pytest.raises(RuntimeError, match="valid extraction"):
        module.consolidate(episode)
    assert len(module.memory) == 0
    assert not (module.memory_dir / "consolidation_ledger.jsonl").exists()
    assert not (module.memory_dir / "pending.json").exists()

    chat.fail_after = 999  # provider recovers; retry from scratch succeeds
    result = module.consolidate(episode)
    assert not result.skipped
    assert "END_MARKER" in module.memory.read_all()[-1].content


def test_recall_and_consolidation_are_serialized_across_module_instances(tmp_path, monkeypatch):
    first, log, episode, _ = setup(tmp_path)
    second = MemoryModule(first.memory_dir, log)
    entered, release, reading = threading.Event(), threading.Event(), threading.Event()
    original = MemoryLayer.add

    def paused(self, record):
        entered.set()
        assert release.wait(3)
        return original(self, record)

    def recall():
        reading.set()
        return second.recall("client deadline", 2000)

    monkeypatch.setattr(MemoryLayer, "add", paused)
    with ThreadPoolExecutor(2) as pool:
        write = pool.submit(first.consolidate, episode)
        assert entered.wait(3)
        read = pool.submit(recall)
        assert reading.wait(3)
        assert not read.done()
        release.set()
        write.result(timeout=3)
        assert read.result(timeout=3).entries
        assert len(second.memory) == 2


def test_committed_episode_survives_cursor_write_failure(tmp_path, monkeypatch):
    import theseus.memory_consolidator as policy_module

    module, log, episode, extractor = setup(tmp_path)
    policy = MemoryConsolidator(module, max_events=1)
    original = policy_module.atomic_json

    def fail_cursor_commit(path, value):
        if 'last_id' in value and 'pending' not in value:
            raise OSError('cursor disk failure')
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(policy_module, 'atomic_json', fail_cursor_commit)
        with pytest.raises(OSError, match='cursor disk failure'):
            policy.tick()
    assert extractor.calls == 1
    log.append('human', 'chat_message', {'message': 'later event'})
    reopened = MemoryModule(module.memory_dir, log, model_providers=[extractor])
    retry = MemoryConsolidator(reopened, max_events=1)
    assert retry.tick().skipped
    assert extractor.calls == 1
    assert retry.pending_events == 1
    state = json.loads((module.memory_dir / 'formation' / 'cursor.json').read_text())
    assert state['last_id'] == episode.end_id
    assert len(reopened.memory.read_all()) == 2


def test_formation_limits_batch_and_defers_next_attempt(tmp_path):
    module, log, episode, extractor = setup(tmp_path)
    for i in range(3):
        log.append('human', 'chat_message', {'message': f'next {i}'})
    clock = [0.0]
    policy = MemoryConsolidator(module, max_events=2, every_seconds=10, now=lambda: clock[0])
    policy.tick()
    assert policy.pending_events == 2
    assert extractor.calls == 1
    assert policy.tick() is None
    assert extractor.calls == 1
    clock[0] = 10.0
    policy.tick()
    assert extractor.calls == 2
    assert policy.pending_events == 0


@pytest.mark.parametrize('interval', [float('nan'), float('inf'), -1])
def test_formation_rejects_invalid_interval(tmp_path, interval):
    module, *_ = setup(tmp_path)
    with pytest.raises(ValueError):
        MemoryConsolidator(module, every_seconds=interval)


def test_fact_lookup_normalizes_whitespace_and_keeps_search_keyword(tmp_path):
    layer = KnowledgeLayer(tmp_path / 'knowledge.jsonl')
    record = KnowledgeRecord('fact', datetime.now(timezone.utc), ' Atlas  Team ', ' Delivery Owner ', 'Beta')
    layer.add(record)
    assert layer.current('atlas team', 'delivery owner') == [record]
    assert layer.search(terms={'ATLAS'})[0].id == 'fact'
