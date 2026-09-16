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
                 "statement": "Client deadline is Friday.", "support_event_ids": ["<EVIDENCE_ID>"],
                 "attribution": "partner_report", "reported_by": "human",
                 "action_status": "not_applicable"},
                {"kind": "event", "statement": "Beta accepted the client delivery task.",
                 "support_event_ids": ["<EVIDENCE_ID>"], "attribution": "partner_report",
                 "reported_by": "human", "action_status": "intention"},
                {"kind": "principle", "statement": "Confirm client deadlines.",
                 "support_event_ids": ["<EVIDENCE_ID>"], "attribution": "inference",
                 "action_status": "not_applicable"},
            ],
        })
        self.prompts = []

    def chat(self, prompt, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        evidence = prompt.split("<evidence>\n", 1)[1].split("\n</evidence>", 1)[0]
        event_id = json.loads(evidence.splitlines()[0])["id"]
        return self.response.replace("<EVIDENCE_ID>", event_id)


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
    for record in (reopened.knowledge.current()[0], *reopened.memory.read_all(),
                   reopened.wisdom.read_all()[0]):
        assert record.support_event_ids == (episode.start_id,)
    assert reopened.knowledge.current()[0].attribution == "partner_report"
    assert reopened.knowledge.current()[0].reported_by == "human"
    assert reopened.memory.read_all()[0].action_status == "intention"
    assert reopened.wisdom.read_all()[0].attribution == "inference"
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


def test_add_only_supersedes_what_the_caller_explicitly_names(tmp_path):
    """#66: KnowledgeLayer no longer infers supersession from a shared
    subject+predicate key or a newer timestamp (that let a late, out-of-order
    write silently win, or an unrelated fact under a broad predicate clobber
    another) — reconciliation decides that, and this layer just applies it."""
    layer = KnowledgeLayer(tmp_path / "knowledge")
    now = datetime.now(timezone.utc)
    layer.add(KnowledgeRecord("new", now, "Client", "deadline", "Monday"))
    layer.add(KnowledgeRecord("old", now - timedelta(days=1), "Client", "deadline", "Friday"))
    # Neither record named the other in `supersedes`, so both stay current —
    # a shared key and timestamp order are no longer enough to replace one.
    assert {r.value for r in layer.current()} == {"Monday", "Friday"}
    assert {r.value for r in KnowledgeLayer(layer.path).current()} == {"Monday", "Friday"}


def test_add_applies_explicit_supersession_regardless_of_write_order(tmp_path):
    layer = KnowledgeLayer(tmp_path / "knowledge")
    now = datetime.now(timezone.utc)
    old = layer.add(KnowledgeRecord("old", now - timedelta(days=1), "Client", "deadline", "Friday"))
    layer.add(KnowledgeRecord("new", now, "Client", "deadline", "Monday", supersedes=old.id))
    assert [r.value for r in layer.current()] == ["Monday"]
    assert [r.value for r in KnowledgeLayer(layer.path).current()] == ["Monday"]


def test_add_rejects_supersedes_of_an_unknown_record(tmp_path):
    layer = KnowledgeLayer(tmp_path / "knowledge")
    with pytest.raises(ValueError, match="unknown record"):
        layer.add(KnowledgeRecord("new", datetime.now(timezone.utc), "Client", "deadline", "Monday",
                                  supersedes="never-written"))


def test_historical_record_is_kept_but_never_current(tmp_path):
    layer = KnowledgeLayer(tmp_path / "knowledge")
    now = datetime.now(timezone.utc)
    layer.add(KnowledgeRecord("current", now, "Atlas", "phase", "production"))
    layer.add(KnowledgeRecord("past", now - timedelta(days=90), "Atlas", "phase", "pilot",
                              reconciliation="historical"))
    assert [r.id for r in layer.current(subject="Atlas")] == ["current"]
    assert {r.id for r in layer.read_all()} == {"current", "past"}
    assert KnowledgeLayer(layer.path).current(subject="Atlas")[0].id == "current"


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


def _last_extraction_prompt(chat):
    # A tick can also fire a #66 reconciliation call, whose prompt has no
    # <evidence> block — skip those to find the actual extraction prompt.
    return next(p for p in reversed(chat.prompts) if "<evidence>" in p)


def test_action_result_pair_is_not_split_by_a_forced_batch_limit(tmp_path):
    """#67: the naive `events[:max_events]` slice would land exactly between
    a decision and its own tool_result here — the batch stops one event
    earlier instead, and the pair rides together into one episode."""
    module, log, _, chat = setup(tmp_path)
    d1 = log.append("agent", "decision", {"text": "attempt Atlas payment"})
    r1 = log.append("agent", "tool_result", {"tool": "pay", "output": "failed"})
    log.append("agent", "decision", {"text": "notify the user"})
    r2 = log.append("agent", "tool_result", {"tool": "reply", "output": "sent"})
    clock = [0.0]
    policy = MemoryConsolidator(module, max_events=3, every_seconds=10, now=lambda: clock[0])

    first = policy.tick()
    assert not first.skipped
    state = json.loads((module.memory_dir / "formation" / "cursor.json").read_text())
    assert state["last_id"] == r1.id  # stopped after the pair, not mid-pair

    clock[0] = 10.0
    second = policy.tick()
    assert not second.skipped
    state = json.loads((module.memory_dir / "formation" / "cursor.json").read_text())
    assert state["last_id"] == r2.id
    # The second episode's extraction carries the first pair as bounded context.
    prompt = _last_extraction_prompt(chat)
    assert "<context_only>" in prompt
    assert d1.id in prompt and r1.id in prompt


def test_oversized_interaction_unit_forces_through_together(tmp_path):
    """A decision with more tool_results than max_events alone allows still
    lands in one episode — keeping an action with its own outcome takes
    priority over the configured batch size; #65's chunking still bounds
    what actually reaches extraction."""
    log = StimulusLog(tmp_path / "log.jsonl")
    chat = Extractor()
    module = MemoryModule(tmp_path / "memory", log, model_providers=[chat])
    log.append("agent", "decision", {"text": "run three checks"})
    log.append("agent", "tool_result", {"tool": "check1", "output": "ok"})
    log.append("agent", "tool_result", {"tool": "check2", "output": "ok"})
    r3 = log.append("agent", "tool_result", {"tool": "check3", "output": "ok"})
    policy = MemoryConsolidator(module, max_events=2)

    result = policy.tick()
    assert not result.skipped
    state = json.loads((module.memory_dir / "formation" / "cursor.json").read_text())
    assert state["last_id"] == r3.id


def test_logs_without_decision_result_typing_batch_per_event(tmp_path):
    """Deterministic fallback: a log with no `decision`/`tool_result` events at
    all (an egocentric capture stream, say) batches one event per unit, same
    as the pre-#67 per-event slicing — bounded work continues without
    needing any richer interaction metadata."""
    log = StimulusLog(tmp_path / "log.jsonl")
    chat = Extractor()
    module = MemoryModule(tmp_path / "memory", log, model_providers=[chat])
    events = [log.append("sensor", "observation", {"i": i}) for i in range(5)]
    policy = MemoryConsolidator(module, max_events=3)

    policy.tick()
    state = json.loads((module.memory_dir / "formation" / "cursor.json").read_text())
    assert state["last_id"] == events[2].id


def test_context_event_ids_survive_a_failed_attempt_and_restart(tmp_path):
    log = StimulusLog(tmp_path / "log.jsonl")
    chat = Extractor()
    module = MemoryModule(tmp_path / "memory", log, model_providers=[chat])
    d1 = log.append("agent", "decision", {"text": "step one"})
    r1 = log.append("agent", "tool_result", {"tool": "x", "output": "ok"})
    policy = MemoryConsolidator(module, max_events=2)
    assert not policy.tick().skipped

    log.append("agent", "decision", {"text": "step two"})
    log.append("agent", "tool_result", {"tool": "y", "output": "fail"})
    broken = MemoryModule(module.memory_dir, log, model_providers=[Extractor('{}')])
    broken_policy = MemoryConsolidator(broken, max_events=2)
    with pytest.raises(RuntimeError):
        broken_policy.tick()
    pending = json.loads((module.memory_dir / "formation" / "cursor.json").read_text())["pending"]
    assert set(pending["context_event_ids"]) == {d1.id, r1.id}

    log.append("human", "chat_message", {"message": "NEW_EVENT_SHOULD_NOT_APPEAR"})
    fixed_chat = Extractor()
    restarted = MemoryConsolidator(MemoryModule(module.memory_dir, log, model_providers=[fixed_chat]))
    result = restarted.tick()
    assert result.episode_id == pending["episode_id"]
    prompt = _last_extraction_prompt(fixed_chat)
    assert "NEW_EVENT_SHOULD_NOT_APPEAR" not in prompt
    assert "<context_only>" in prompt
    assert d1.id in prompt and r1.id in prompt


def test_pending_state_without_context_event_ids_still_replays(tmp_path):
    """A cursor.json pending block written before #67 (no context_event_ids
    key) still replays under the new code — the field is additive."""
    module, log, episode, chat = setup(tmp_path)
    policy = MemoryConsolidator(module)
    path = module.memory_dir / "formation" / "cursor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pending": {
        "episode_id": "legacy-ep", "start_id": episode.start_id, "end_id": episode.end_id,
    }}))
    result = policy.tick()
    assert result.episode_id == "legacy-ep"


def test_context_event_ids_are_readable_but_never_citable_as_support(tmp_path):
    module, log, _, chat = setup(tmp_path)
    prior = log.append("agent", "tool_result", {"tool": "pay", "output": "PRIOR_CONTEXT_MARKER"})
    new_event = log.append("human", "chat_message", {"message": "what happened with the payment?"})

    module.consolidate(Episode("ep1", new_event.id, new_event.id, context_event_ids=(prior.id,)))

    prompt = _last_extraction_prompt(chat)
    assert "PRIOR_CONTEXT_MARKER" in prompt
    assert "<context_only>" in prompt
    for record in module.knowledge.read_all() + module.memory.read_all() + module.wisdom.read_all():
        assert prior.id not in (record.support_event_ids or ())


def test_context_event_id_cited_as_support_is_dead_lettered(tmp_path):
    class CitingExtractor:
        def __init__(self, context_id):
            self.context_id = context_id
            self.calls = 0

        def chat(self, prompt, **kwargs):
            self.calls += 1
            return json.dumps({"summary": "s", "assertions": [{
                "kind": "event", "statement": "cites prior context",
                "support_event_ids": [self.context_id],
                "attribution": "direct_observation", "action_status": "not_applicable",
            }]})

    log = StimulusLog(tmp_path / "log.jsonl")
    prior = log.append("agent", "tool_result", {"tool": "pay", "output": "ok"})
    new_event = log.append("human", "chat_message", {"message": "next"})
    chat = CitingExtractor(prior.id)
    module = MemoryModule(tmp_path / "memory", log, model_providers=[chat])

    result = module.consolidate(Episode("ep1", new_event.id, new_event.id, context_event_ids=(prior.id,)))

    assert result.schema_failures == 1
    dead = json.loads((module.memory_dir / "dead_letter.jsonl").read_text().splitlines()[0])
    assert "context-only support event" in dead["reason"]


# Direct-observation/inference attribution only, so validity doesn't depend on
# which actor produced the (tool-authored) oversized event.
_CHUNK_TEST_RESPONSE = json.dumps({
    "summary": "A large tool output was consolidated.",
    "assertions": [
        {"kind": "fact", "subject": "Output", "predicate": "marker", "value": "END_MARKER",
         "statement": "The tool output contains an END_MARKER.", "support_event_ids": ["<EVIDENCE_ID>"],
         "attribution": "direct_observation", "action_status": "confirmed_outcome"},
        {"kind": "event", "statement": "The tool produced a large output.",
         "support_event_ids": ["<EVIDENCE_ID>"], "attribution": "direct_observation",
         "action_status": "not_applicable"},
        {"kind": "principle", "statement": "Watch for decisive markers in tool output.",
         "support_event_ids": ["<EVIDENCE_ID>"], "attribution": "inference",
         "action_status": "not_applicable"},
    ],
})


def test_oversized_event_is_chunked_and_decisive_ending_reaches_extraction(tmp_path):
    """An event too large for one request is split into several bounded chunks
    that together cover it completely, so the decisive marker at its very end
    reaches extraction instead of being cut away by a prefix truncation — and
    identical assertions the same canned response yields per chunk collapse
    into one via dedup."""
    module, log, _, chat = setup(tmp_path, extractor=Extractor(_CHUNK_TEST_RESPONSE))
    event = log.append("tool", "tool_result", {"output": "large " * 20000 + "END_MARKER"})
    result = module.consolidate(Episode("big", event.id, event.id))
    assert len(chat.prompts) > 1
    assert all(len(p) <= module._max_input_chars for p in chat.prompts)
    assert all(event.id in p for p in chat.prompts)  # original event id carried through every chunk
    assert "END_MARKER" in chat.prompts[-1]
    assert result.schema_failures == 0
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


def test_assertion_cannot_cite_support_from_an_event_its_own_request_never_saw(tmp_path):
    """support_event_ids is validated against what THIS request was given, not
    the whole episode: an oversized event pulled out of the shared pack means
    the pack's own request never saw it, so a pack-unit assertion citing that
    event's (real, in-episode) id is dead-lettered — while the same id cited by
    that event's own chunk requests, which did see it, validates fine."""

    class CrossCitingExtractor:
        def __init__(self, oversized_event_id):
            self.oversized_event_id = oversized_event_id
            self.calls = 0
            self.prompts = []

        def chat(self, prompt, **kwargs):
            self.calls += 1
            self.prompts.append(prompt)
            evidence = prompt.split("<evidence>\n", 1)[1].split("\n</evidence>", 1)[0]
            this_id = json.loads(evidence.splitlines()[0])["id"]
            return json.dumps({
                "summary": "s",
                "assertions": [{"kind": "event", "statement": "cross-cited claim",
                               "support_event_ids": [self.oversized_event_id],
                               "attribution": "direct_observation", "action_status": "not_applicable"}],
            })

    module, log, _, _ = setup(tmp_path)
    first = log.append("human", "chat_message", {"message": "short one"})
    big = log.append("tool", "tool_result", {"output": "large " * 20000 + "END_MARKER"})
    chat = CrossCitingExtractor(big.id)
    module._model_providers = [chat]

    result = module.consolidate(Episode("cross", first.id, big.id))
    assert chat.calls > 1  # a pack call plus at least one chunk call
    assert result.schema_failures == 1  # the pack-unit occurrence, citing an id it never saw
    dead = json.loads((module.memory_dir / "dead_letter.jsonl").read_text().splitlines()[0])
    assert "unknown support event" in dead["reason"]
    # every chunk-unit occurrence legitimately self-cites the same id and is
    # identical, so dedup collapses them to the one accepted record.
    assert result.routed.get("memory", 0) == 2  # the accepted assertion + the episode summary


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
    # +2 over the first tick: the second episode's canned fact and principle
    # both match something already on file, so fact and wisdom reconciliation
    # each have something to check against and make their own call.
    assert extractor.calls == 4
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


@pytest.mark.parametrize('restart', [False, True])
def test_large_prior_context_does_not_stall_next_scheduled_episode(tmp_path, restart):
    log = StimulusLog(tmp_path / 'log.jsonl')
    chat = Extractor(json.dumps({'summary': 'Source retained.', 'assertions': []}))
    module = MemoryModule(tmp_path / 'memory', log, model_providers=[chat], max_input_chars=24000)
    large = log.append('tool', 'tool_result', {'output': 'x' * 60000 + ' FINAL_OUTCOME'})
    now = [0.0]
    policy = MemoryConsolidator(module, max_events=1, now=lambda: now[0])
    policy.tick()
    small = log.append('human', 'chat_message', {'message': 'Next small task'})
    if restart:
        module = MemoryModule(module.memory_dir, log, model_providers=[chat], max_input_chars=24000)
        policy = MemoryConsolidator(module, max_events=1, now=lambda: now[0])
    now[0] = 301
    assert not policy.tick().skipped
    state = json.loads((module.memory_dir / 'formation' / 'cursor.json').read_text())
    assert state['last_id'] == small.id and 'pending' not in state
    assert policy.pending_events == 0
    assert all(len(prompt) <= 24000 for prompt in chat.prompts)
    prompt = _last_extraction_prompt(chat)
    context = prompt.split('<context_only>\n')[1].split('\n</context_only>')[0]
    assert len(context) <= 4096
    assert 'context_truncated' in context and 'FINAL_OUTCOME' in context
    assert large.id in context and small.id in prompt
    assert any('x' * 60000 in record.content for record in module.memory.read_all())
