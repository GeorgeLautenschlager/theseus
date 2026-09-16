# Assertion event provenance (#64)

Issue #64 requires each new extracted claim to cite eligible stimulus events and
to carry explicit report and action status. The consolidation boundary already
separates recall tool results as context only; event IDs in that block cannot
support new assertions.

The extractor returns `support_event_ids`, `attribution`, optional `reported_by`,
and `action_status` with each assertion. The routing gate checks nonempty,
distinct IDs against evidence events supplied in the extraction prompt, rejects
context-only and out-of-episode IDs, checks enum values, and ties `reported_by`
to a supporting event actor. It dead-letters invalid candidates while accepting
other candidates in the same response. Raw event material stays in the episode
record, including full content when the prompt had to excerpt it.

Knowledge, Memory, and Wisdom records store the same metadata, and recall renders
it. Old JSONL rows read with `support_event_ids=None`, shown as unknown. No event
IDs are inferred from `source_episode_id`. The pending transaction contains the
serialized records, so replay and retries preserve metadata without re-extracting.

Structural checks establish eligible references, not semantic truth. A reported
completion may be false, and a model may misread a plan as a confirmed outcome
even when it cites a real event. Multi-event counterexamples must test these
errors separately from source-ID validity.
