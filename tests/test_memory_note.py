from __future__ import annotations

from datetime import datetime, timezone

from theseus.memory_note import MemoryNote

RAW_EVENTS = (
    '{"id":"01ABC","ts":"2026-07-27T10:00:00+00:00","actor":"george",'
    '"type":"exchange","content":{"message":"I drink tea."}}'
)


def make_note(**overrides) -> MemoryNote:
    fields = dict(
        id="n1",
        ts=datetime.now(timezone.utc),
        content=RAW_EVENTS,
        context="George told me he drinks tea, not coffee.",
        keywords=["George", "tea"],
        tags=["preferences"],
    )
    fields.update(overrides)
    return MemoryNote(**fields)


def test_render_leads_with_the_id_and_context():
    rendered = make_note().render()
    assert rendered.startswith("[n1] George told me he drinks tea, not coffee.")


def test_render_omits_the_raw_source_events():
    """`content` is provenance — the raw JSON the note was distilled from. Rendering it
    puts JSON inside the JSON `output` field of a tool_result, at enormous token cost,
    and duplicates log content the agent may already have in-window. `context` carries
    the meaning; it is also the only field retrieval embeds."""
    rendered = make_note().render()

    assert RAW_EVENTS not in rendered
    assert "01ABC" not in rendered
    assert "content:" not in rendered


def test_render_includes_keywords_and_tags():
    rendered = make_note().render()
    assert "keywords: George, tea" in rendered
    assert "tags: preferences" in rendered


def test_render_omits_empty_keyword_and_tag_lines():
    rendered = make_note(keywords=[], tags=[]).render()
    assert "keywords:" not in rendered
    assert "tags:" not in rendered


def test_content_is_still_stored_as_provenance():
    # Dropping it from render() must not drop it from the note — source_span plus
    # content are what make notes a replayable projection over the stimulus log.
    note = make_note()
    assert note.content == RAW_EVENTS
    assert MemoryNote.from_json(note.to_json()).content == RAW_EVENTS
