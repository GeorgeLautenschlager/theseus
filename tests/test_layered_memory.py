"""Layered memory module — acceptance-criterion tests.

One named test per brief acceptance criterion (see
docs/superpowers/specs/2026-08-29-layered-memory-module.md), plus the leak
test. All offline: providers are fakes, no live LLM required.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from theseus.intelligence_layer import IntelligenceLayer
from theseus.knowledge_layer import KnowledgeLayer, KnowledgeRecord
from theseus.memory_layer import MemoryLayer, MemoryRecord
from theseus.memory_module import Episode, MemoryModule, RecallResult
from theseus.stimulus_log import StimulusLog
from theseus.wisdom_layer import WisdomLayer, WisdomRecord

LAYER_NAMES = {"knowledge", "memory", "wisdom", "intelligence"}


def make_embedder(embedding: list[float] | None = None) -> MagicMock:
    embedder = MagicMock()
    embedder.is_available.return_value = True
    embedder.embed.return_value = embedding if embedding is not None else [1.0, 0.0]
    return embedder


def make_chat_provider(response_json: str) -> MagicMock:
    provider = MagicMock()
    provider.is_available.return_value = True
    provider.chat.return_value = response_json
    return provider


EXTRACTION = json.dumps(
    {
        "summary": "George introduced himself and stated a preference.",
        "assertions": [
            {"kind": "fact", "subject": "George", "predicate": "prefers",
             "value": "dark mode", "statement": "George prefers dark mode."},
            {"kind": "event", "statement": "George greeted the agent."},
        ],
    }
)


def make_module(tmp_path, with_embedding: bool = True, chat_response: str | None = EXTRACTION,
                lenient: bool = False) -> MemoryModule:
    log = StimulusLog(tmp_path / "stimulus.jsonl")
    providers = [make_embedder()] if with_embedding else []
    model_providers = [make_chat_provider(chat_response)] if chat_response is not None else []
    return MemoryModule(
        tmp_path / "mem", log,
        embedding_providers=providers, model_providers=model_providers,
        lenient_fact_routing=lenient,
    )


def _ts(offset_days: float = 0.0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=offset_days)


class TestLayers:
    def test_layers_persist_independently_without_mutation(self, tmp_path):
        """Criterion 1: separate persistence semantics; no layer decays or
        mutates its own records. Reopening each store from disk yields the
        same records; query-time recency shifts scores, never data."""
        k = KnowledgeLayer(tmp_path / "knowledge.jsonl")
        m = MemoryLayer(tmp_path / "memory.jsonl")
        w = WisdomLayer(tmp_path / "wisdom.jsonl")

        k.add(KnowledgeRecord(id="k1", ts=_ts(), subject="Alty", predicate="is", value="a test agent"))
        m.add(
            MemoryRecord(
                id="m1", ts=_ts(10), content="raw evidence", summary="A session happened.",
                embedding=[1.0, 0.0], source_episode_id="ep1",
            )
        )
        w.add(WisdomRecord(id="w1", ts=_ts(), statement="Verify before asserting.", embedding=[0.0, 1.0]))

        # Reopen from disk: what was written is exactly what comes back.
        k2 = KnowledgeLayer(tmp_path / "knowledge.jsonl")
        m2 = MemoryLayer(tmp_path / "memory.jsonl")
        w2 = WisdomLayer(tmp_path / "wisdom.jsonl")
        assert [r.to_json() for r in k2.read_all()] == [k.get("k1").to_json()]
        assert [r.to_json() for r in m2.read_all()] == [m.get("m1").to_json()]
        assert [r.to_json() for r in w2.read_all()] == [w.get("w1").to_json()]

        # Recency is a query-time weight: same stored record, different now →
        # different score, identical record.
        now = _ts(0)
        fresh = m2.query([1.0, 0.0], k=5, now=now)
        aged = m2.query([1.0, 0.0], k=5, now=now + timedelta(days=30))
        assert fresh[0].score > aged[0].score
        assert m2.get("m1").to_json() == m.get("m1").to_json()

    def test_knowledge_supersession_is_explicit_and_logged(self, tmp_path):
        """Criterion 6: supersession is explicit (new record names the old) and
        logged (both records stay in the append-only file)."""
        k = KnowledgeLayer(tmp_path / "knowledge.jsonl")
        first = k.add(KnowledgeRecord(id="k1", ts=_ts(2), subject="George", predicate="prefers", value="dark mode"))
        second = k.add(KnowledgeRecord(id="k2", ts=_ts(0), subject="George", predicate="prefers", value="light mode"))

        assert second.supersedes == first.id
        current = k.current(subject="George", predicate="prefers")
        assert [r.id for r in current] == ["k2"]
        # The file holds the whole history; nothing was rewritten.
        assert [r.id for r in k.read_all()] == ["k1", "k2"]
        reopened = KnowledgeLayer(tmp_path / "knowledge.jsonl")
        assert [r.id for r in reopened.current(subject="George")] == ["k2"]
        assert reopened.get("k2").supersedes == "k1"

    def test_wisdom_retrieval_filters_by_evidence_count(self, tmp_path):
        """Criterion 7: wisdom is retrievable and honors the evidence_count filter."""
        w = WisdomLayer(tmp_path / "wisdom.jsonl")
        weak = WisdomRecord(id="w1", ts=_ts(), statement="Test early.", embedding=[1.0, 0.0], evidence_count=1)
        strong = WisdomRecord(id="w2", ts=_ts(), statement="Test often.", embedding=[1.0, 0.0], evidence_count=5)
        w.add(weak)
        w.add(strong)

        unfiltered = {h.id for h in w.query([1.0, 0.0], k=5)}
        assert unfiltered == {"w1", "w2"}
        top = [h.id for h in w.query([1.0, 0.0], k=5, min_evidence=3)]
        assert top == ["w2"]

    def test_intelligence_reads_log_tail_without_a_file(self, tmp_path):
        """Criterion 9 (layer half): intelligence is the StimulusLog tail —
        no store file of its own."""
        log = StimulusLog(tmp_path / "stimulus.jsonl")
        for i in range(5):
            log.append(actor="george", type="exchange", content={"message": f"msg {i}"})
        layer = IntelligenceLayer(log, tail=3)

        hits = layer.read()
        assert len(hits) == 3
        assert "msg 4" in hits[0].text  # newest first
        assert not hasattr(layer, "path") and not (tmp_path / "intelligence.jsonl").exists()


class TestRecall:
    def test_recall_reports_misses_as_data(self, tmp_path):
        """Criteria 2+3: recall returns misses explicitly, as data on the
        result — never an exception, never a silent empty."""
        module = make_module(tmp_path)
        result = module.recall("anything at all", budget_tokens=100)

        assert isinstance(result, RecallResult)
        assert result.entries == ()
        expected_misses = {
            "knowledge: no matches",
            "memory: no matches",
            "wisdom: no matches",
            "intelligence: no matches",
        }
        assert expected_misses <= set(result.misses)
        # Misses are plain data a caller can log or render.
        assert all(isinstance(m, str) for m in result.misses)
        assert result.total_tokens == 0

        # No embedder at all: the vector layers report it as miss data too.
        bare = MemoryModule(tmp_path / "bare", StimulusLog(tmp_path / "s2.jsonl"))
        bare_result = bare.recall("anything", budget_tokens=100)
        assert "embedding unavailable" in bare_result.misses
        assert "memory: no matches" in bare_result.misses
        assert "wisdom: no matches" in bare_result.misses

    def test_recall_respects_token_budget(self, tmp_path):
        """Criterion 8: the token budget fills the window — fused order kept,
        entries that don't fit are skipped (not truncated), smaller ones after
        them still fill in."""
        module = make_module(tmp_path)
        base = _ts(0)
        # Distinct predicates so nothing supersedes; equal term-overlap scores,
        # so the ts tiebreak (newest first) sets the order: k3, k2, k1.
        for rid, value_len, secs in (("k1", 400, 1), ("k2", 400, 2), ("k3", 40, 3)):
            module.knowledge.add(
                KnowledgeRecord(
                    id=rid, ts=base + timedelta(seconds=secs),
                    subject="alpha", predicate=f"{rid} predicate",
                    value="x" * value_len, source_episode_id="ep",
                )
            )

        # Full budget: everything fits, fused order preserved.
        full = module.recall("alpha beta", budget_tokens=10_000)
        assert [e.provenance.record_id for e in full.entries] == ["k3", "k2", "k1"]
        assert full.total_tokens <= 10_000

        # Tiny budget: only the top fused entry fits; nothing after it can fill
        # because what remains is bigger than the window left.
        tiny_budget = full.entries[0].tokens
        tiny = module.recall("alpha beta", budget_tokens=tiny_budget)
        assert [e.provenance.record_id for e in tiny.entries] == ["k3"]
        assert tiny.total_tokens <= tiny_budget

    def test_no_layer_names_leak_into_public_surface(self, tmp_path):
        """Leak contract: no layer name in a public signature or in a
        RecallResult field the core must interpret. (Provenance.layer and
        costs values are opaque data — allowed, logged not interpreted.)"""
        sig = inspect.signature(MemoryModule.recall)
        assert set(sig.parameters) == {"self", "query", "budget_tokens"}
        surface = [repr(sig.return_annotation)] + [
            repr(p.annotation)
            for p in sig.parameters.values()
            if p.annotation is not inspect.Parameter.empty
        ]
        for text in surface:
            words = {w.lower() for w in text.replace("_", " ").split()}
            assert not (LAYER_NAMES & words), text

        result_fields = {f.name for f in fields(RecallResult)}
        assert result_fields == {"query", "subqueries", "entries", "misses", "total_tokens", "costs"}

    def test_traces_emitted_per_recall_and_consolidation(self, tmp_path):
        """Criterion 11: one JSONL trace record per recall (and per
        consolidation — asserted once issue 4 lands). Records carry the brief's
        fields: query, subqueries, layers queried, per-layer hit counts,
        pre/post-fusion ranks, returned set, misses, latencies."""
        module = make_module(tmp_path)
        base = _ts(0)
        module.knowledge.add(
            KnowledgeRecord(id="k1", ts=base, subject="alpha", predicate="beta one",
                            value="v", source_episode_id="ep")
        )

        result = module.recall("alpha beta", budget_tokens=100)
        traces = (module.memory_dir / "traces" / "recall.jsonl").read_text().strip().splitlines()
        assert len(traces) == 1
        record = json.loads(traces[0])

        assert record["query"] == "alpha beta"
        assert record["subqueries"] == ["alpha beta"]
        assert set(record["layers_queried"]) == LAYER_NAMES
        assert record["layer_hit_counts"]["knowledge"] == 1
        assert record["returned"] == [
            {"record_id": "k1", "layer": "knowledge",
             "pre_fusion_rank": 1, "post_fusion_rank": 1}
        ]
        assert isinstance(record["misses"], list)
        assert set(record["costs"]) == LAYER_NAMES | {"total"}
        assert record["re_queried_within_n_turns"] is None  # derived offline

        module.recall("alpha beta", budget_tokens=100)
        traces = (module.memory_dir / "traces" / "recall.jsonl").read_text().strip().splitlines()
        assert len(traces) == 2  # one record per recall, append-only

        # Consolidation half: one trace record per consolidation too.
        first = module._stimulus_log.append(actor="george", type="exchange",
                                            content={"message": "hi"})
        result = module.consolidate(Episode("ep1", first.id, first.id))
        c_traces = (module.memory_dir / "traces" / "consolidation.jsonl").read_text().strip().splitlines()
        assert len(c_traces) == 1
        c_record = json.loads(c_traces[0])
        assert c_record["episode_id"] == "ep1"
        assert c_record["candidates_extracted"] == result.extracted
        assert set(c_record["routed"]) >= {"knowledge", "memory"}
        for field in ("supersessions", "schema_failures", "tokens_in", "tokens_out", "wall_time_s"):
            assert field in c_record

    def test_consolidate_is_idempotent_per_episode(self, tmp_path):
        """Criterion 4: re-consolidating an episode is a no-op — keyed on the
        episode (assertion ids are assigned per run, so the ledger is what makes
        replay safe)."""
        module = make_module(tmp_path)
        t0 = datetime.now(timezone.utc)
        # Distinct-millisecond timestamps: intra-ms ULIDs are random-suffixed,
        # so read_range's lexical span needs ms separation (human-rate logs).
        first = module._stimulus_log.append(actor="george", type="exchange",
                                            content={"message": "I prefer dark mode."}, ts=t0)
        last = module._stimulus_log.append(actor="Alty", type="decision",
                                           content={"text": "noted"},
                                           ts=t0 + timedelta(milliseconds=5))
        episode = Episode("ep1", first.id, last.id)

        r1 = module.consolidate(episode)
        assert not r1.skipped and r1.extracted == 2
        assert r1.routed["knowledge"] == 1 and r1.routed["memory"] == 2  # event + episode record

        before = (len(module.knowledge), len(module.memory), len(module.wisdom))
        r2 = module.consolidate(episode)
        assert r2.skipped is True
        assert (len(module.knowledge), len(module.memory), len(module.wisdom)) == before
        ledger = (module.memory_dir / "consolidation_ledger.jsonl").read_text().strip().splitlines()
        assert [json.loads(l)["episode_id"] for l in ledger] == ["ep1"]
        # A fresh module over the same dir honors the ledger too.
        reopened = MemoryModule(
            tmp_path / "mem", module._stimulus_log,
            embedding_providers=[make_embedder()], model_providers=[make_chat_provider(EXTRACTION)],
        )
        assert reopened.consolidate(episode).skipped is True

    def test_recall_flagged_stimuli_excluded_from_evidence(self, tmp_path):
        """Criterion 5: recall-flagged stimuli may be read for context but are
        never evidence and never provenance."""
        chat = make_chat_provider(EXTRACTION)
        log = StimulusLog(tmp_path / "stimulus.jsonl")
        module = MemoryModule(
            tmp_path / "mem", log,
            embedding_providers=[make_embedder()], model_providers=[chat],
        )
        t0 = datetime.now(timezone.utc)
        first = log.append(actor="george", type="exchange",
                          content={"message": "I prefer dark mode."}, ts=t0)
        log.append(actor="Alty", type="tool_result",
                  content={"tool": "recall", "arguments": {"query": "prefs"},
                           "output": "SECRET-RECALLED-TEXT", "is_error": False},
                  ts=t0 + timedelta(milliseconds=5))
        last = log.append(actor="george", type="exchange",
                         content={"message": "and light text"},
                         ts=t0 + timedelta(milliseconds=10))

        module.consolidate(Episode("ep1", first.id, last.id))

        prompt = chat.chat.call_args[0][0]
        assert "SECRET-RECALLED-TEXT" in prompt  # readable as context...
        for record in module.memory.read_all():
            assert "SECRET-RECALLED-TEXT" not in record.content  # ...never evidence

    def test_consolidate_spans_same_millisecond_bursts(self, tmp_path):
        """Real logs burst events faster than 1ms; ULIDs are random within a
        millisecond, so the episode span must be positional (file order), not a
        lexical id range."""
        module = make_module(tmp_path)
        t0 = datetime.now(timezone.utc)  # all three share one millisecond
        first = module._stimulus_log.append(actor="george", type="exchange",
                                            content={"message": "burst-one"}, ts=t0)
        module._stimulus_log.append(actor="Alty", type="decision",
                                   content={"text": "burst-two"}, ts=t0)
        last = module._stimulus_log.append(actor="george", type="exchange",
                                           content={"message": "burst-three"}, ts=t0)

        module.consolidate(Episode("ep1", first.id, last.id))

        prompt = module._model_providers[0].chat.call_args[0][0]
        for text in ("burst-one", "burst-two", "burst-three"):
            assert text in prompt  # every event in the span reached extraction

    def test_schema_invalid_extractions_dead_lettered_and_counted(self, tmp_path):
        """Criterion 12: schema-invalid candidates are dead-lettered AND counted
        — the valid ones still land."""
        bad = json.dumps(
            {
                "summary": "s",
                "assertions": [
                    {"kind": "fact", "subject": "George", "predicate": "prefers",
                     "value": "dark mode", "statement": "George prefers dark mode."},
                    {"kind": "bogus", "statement": "unknown kind"},
                    {"kind": "fact", "subject": "X", "predicate": "y"},  # missing value+statement
                ],
            }
        )
        module = make_module(tmp_path, chat_response=bad)
        first = module._stimulus_log.append(actor="george", type="exchange",
                                            content={"message": "hi"})
        result = module.consolidate(Episode("ep1", first.id, first.id))

        assert result.extracted == 3
        assert result.schema_failures == 2
        assert result.routed["knowledge"] == 1
        dead = (module.memory_dir / "dead_letter.jsonl").read_text().strip().splitlines()
        assert len(dead) == 2
        for line in dead:
            record = json.loads(line)
            assert record["episode_id"] == "ep1"
            assert record["assertion_id"] and record["reason"]

    def test_lenient_routing_keeps_tripleless_facts_in_memory(self, tmp_path):
        """Lenient A/B variant: a fact without its triple is not dead-lettered —
        it routes to Memory with the statement intact (strict default unchanged)."""
        candidate = {"kind": "fact",
                     "statement": "The staging database uses Postgres 16."}
        chat_response = json.dumps({"summary": "s", "assertions": [candidate]})
        module = make_module(tmp_path, chat_response=chat_response, lenient=True)
        first = module._stimulus_log.append(actor="mara", type="exchange",
                                            content={"message": "staging is Postgres 16"})
        result = module.consolidate(Episode("ep1", first.id, first.id))

        assert result.schema_failures == 0
        assert result.routed["memory"] == 2  # the assertion + the episode record
        assert "knowledge" not in result.routed
        contents = [r.content for r in module.memory.read_all()]
        assert any(candidate["statement"] in c for c in contents)
        assert not (module.memory_dir / "dead_letter.jsonl").exists()

    def test_no_episode_detection_in_public_surface(self, tmp_path):
        """Criterion 10: no Segmenter — the caller builds Episodes; the module
        has no episode-detection surface at all."""
        public = [name for name in dir(MemoryModule) if not name.startswith("_")]
        assert not {n for n in public if "segment" in n.lower() or "detect" in n.lower()}
        assert {f.name for f in fields(Episode)} == {"episode_id", "start_id", "end_id"}


class TestTornLineRecovery:
    def test_torn_final_line_dropped_interior_corruption_raises(self, tmp_path):
        """Store discipline matches StimulusLog/MemoryStore: torn tail recovers,
        interior corruption raises."""
        from theseus.layer_store import append_record, load_lines

        p = tmp_path / "s.jsonl"
        append_record(p, json.dumps({"id": "a"}))
        with open(p, "a") as f:
            f.write('{"id": "torn"')  # no newline: crash mid-write
        assert [json.loads(l)["id"] for l in load_lines(p)] == ["a"]
