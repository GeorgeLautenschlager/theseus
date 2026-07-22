"""AgenticMemory — A-MEM-inspired long-term memory over the StimulusLog.

Pipeline (per Xu et al., arXiv 2502.12110, minus memory evolution, which is
deferred): new stimulus events are distilled into an enriched MemoryNote
(context/keywords/tags via one LLM call), embedded, then agentically linked to
its nearest existing notes (a second LLM call chooses which neighbors are
genuinely related). Retrieval is embedding similarity plus one hop of link
expansion.

One memory module among (eventually) many: the core knows it only through the
Memory protocol's form()/retrieve() and signals form() when a cognitive loop
terminates. Everything else — which events are new (the high-water mark
derived from note source spans), storage, linking — stays in here. Formation
must never take the loop down: every failure is caught, reported to stdout,
and swallowed.

Chat and embedding models come from separate provider lists; an embedding
provider is just a ModelProvider instance whose one model is an embedding
model, e.g. OllamaProvider(model="nomic-embed-text").
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from theseus.json_utils import parse_json_response
from theseus.memory_note import MemoryNote
from theseus.memory_prompts import (
    build_link_decision_prompt,
    build_note_construction_prompt,
    link_json_schema,
    note_json_schema,
)
from theseus.memory_store import MemoryStore
from theseus.model_providers.model_provider import ModelProvider
from theseus.stimulus_log import StimulusEvent, StimulusLog, new_id


class AgenticMemory:
    def __init__(
        self,
        model_providers: List[ModelProvider],
        embedding_providers: List[ModelProvider],
        store: MemoryStore,
        stimulus_log: StimulusLog,
        k_neighbors: int = 5,
        k_retrieve: int = 5,
        retrieval_query_chars: int = 2000,
    ):
        self.model_providers = model_providers
        self.embedding_providers = embedding_providers
        self.store = store
        self.stimulus_log = stimulus_log
        self.k_neighbors = k_neighbors
        self.k_retrieve = k_retrieve
        self.retrieval_query_chars = retrieval_query_chars

    @staticmethod
    def _first_available(providers: List[ModelProvider]) -> ModelProvider:
        for provider in providers:
            if provider.is_available():
                return provider
        raise RuntimeError("No model providers are currently available.")

    def _select_model_provider(self) -> ModelProvider:
        return self._first_available(self.model_providers)

    def _select_embedding_provider(self) -> ModelProvider:
        return self._first_available(self.embedding_providers)

    def form(self) -> None:
        """Consolidate every stimulus event newer than the store's high-water
        mark into one linked, embedded note. Signalled by the core at loop
        termination; a no-op when nothing new has happened."""
        try:
            high_water = self.store.last_consolidated_id()
            events = [
                e for e in self.stimulus_log.read_all()
                if high_water is None or e.id > high_water
            ]
            if events:
                self._form(events)
        except Exception as exc:
            print(f"Memory formation failed; skipping this batch: {exc}")

    def _form(self, events: list[StimulusEvent]) -> MemoryNote:
        provider = self._select_model_provider()
        events_text = "\n".join(e.to_json() for e in events)

        raw = provider.chat(
            prompt=build_note_construction_prompt(events_text),
            json_schema=note_json_schema(),
        )
        enrichment = parse_json_response(raw)

        note = MemoryNote(
            id=new_id(),
            ts=datetime.now(timezone.utc),
            content=events_text,
            context=enrichment["context"],
            keywords=list(enrichment["keywords"]),
            tags=list(enrichment["tags"]),
            source_span=(events[0].id, events[-1].id),
            embedding=self._select_embedding_provider().embed(enrichment["context"]),
        )

        candidates = self.store.top_k(note.embedding, self.k_neighbors)
        if candidates:
            raw = provider.chat(
                prompt=build_link_decision_prompt(note, candidates),
                json_schema=link_json_schema([c.id for c in candidates]),
            )
            candidate_ids = {c.id for c in candidates}
            chosen = [i for i in parse_json_response(raw)["links"] if i in candidate_ids]
            note.links.extend(chosen)

        return self.store.add(note)

    def retrieve(self, query: str) -> str:
        """Memories relevant to `query`, rendered ready for a prompt; "" when none."""
        return "\n\n".join(note.render() for note in self._retrieve_notes(query))

    def _retrieve_notes(self, query: str) -> list[MemoryNote]:
        """Top-k notes by embedding similarity, expanded one hop along links."""
        if len(self.store) == 0:
            return []
        # Cap the embedding input to the most-recent tail so it fits the model's context.
        query = query[-self.retrieval_query_chars:]
        try:
            embedding = self._select_embedding_provider().embed(query)
            hits = self.store.top_k(embedding, self.k_retrieve)
        except Exception as exc:
            print(f"Memory retrieval failed; continuing without memories: {exc}")
            return []

        seen = {n.id for n in hits}
        expanded = list(hits)
        for note in hits:
            for link_id in note.links:
                if link_id not in seen:
                    linked = self.store.get(link_id)
                    if linked is not None:
                        expanded.append(linked)
                        seen.add(link_id)
        return expanded
