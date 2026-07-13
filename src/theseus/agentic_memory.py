"""AgenticMemory — A-MEM-inspired long-term memory over the StimulusLog.

Pipeline (per Xu et al., arXiv 2502.12110, minus memory evolution, which is
deferred): a batch of stimulus events is distilled into an enriched MemoryNote
(context/keywords/tags via one LLM call), embedded, then agentically linked to
its nearest existing notes (a second LLM call chooses which neighbors are
genuinely related). Retrieval is embedding similarity plus one hop of link
expansion.

Formation is called post-cycle by CognitiveCore and must never take the loop
down: every failure is caught, reported to stdout, and swallowed.
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
from theseus.stimulus_log import StimulusEvent, new_id


class AgenticMemory:
    def __init__(
        self,
        model_providers: List[ModelProvider],
        store: MemoryStore,
        k_neighbors: int = 5,
        k_retrieve: int = 5,
    ):
        self.model_providers = model_providers
        self.store = store
        self.k_neighbors = k_neighbors
        self.k_retrieve = k_retrieve

    def _select_model_provider(self) -> ModelProvider:
        """Selects the first available provider, in priority order."""
        for provider in self.model_providers:
            if provider.is_available():
                return provider
        raise RuntimeError("No model providers are currently available.")

    def form(self, events: list[StimulusEvent]) -> MemoryNote | None:
        """Distill `events` into one linked, embedded note and persist it.

        Returns the stored note, or None if there was nothing to form or any
        step failed — memory formation must never crash the cognitive loop.
        """
        if not events:
            return None
        try:
            return self._form(events)
        except Exception as exc:
            print(f"Memory formation failed; skipping this batch: {exc}")
            return None

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
            embedding=provider.embed(enrichment["context"]),
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

    def retrieve(self, query: str) -> list[MemoryNote]:
        """Top-k notes by embedding similarity, expanded one hop along links."""
        if len(self.store) == 0:
            return []
        try:
            provider = self._select_model_provider()
            hits = self.store.top_k(provider.embed(query), self.k_retrieve)
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
