from __future__ import annotations

from datetime import datetime, timezone

from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.web.debug_pagination import most_recent_page, older_batch


def _event(id: str) -> StimulusEvent:
    return StimulusEvent(
        id=id,
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        actor="test",
        type="observation",
        content={},
    )


def _events(*ids: str) -> list[StimulusEvent]:
    return [_event(i) for i in ids]


class TestMostRecentPage:
    def test_fewer_than_page_size(self):
        events = _events("a", "b", "c")
        page, has_more = most_recent_page(events, 25)
        assert page == events
        assert has_more is False

    def test_exactly_page_size(self):
        events = _events(*[str(i).zfill(2) for i in range(25)])
        page, has_more = most_recent_page(events, 25)
        assert page == events
        assert has_more is False

    def test_more_than_page_size_takes_tail(self):
        events = _events(*[str(i).zfill(2) for i in range(30)])
        page, has_more = most_recent_page(events, 25)
        assert page == events[-25:]
        assert has_more is True

    def test_empty_log(self):
        page, has_more = most_recent_page([], 25)
        assert page == []
        assert has_more is False


class TestOlderBatch:
    def test_oldest_boundary_has_no_more(self):
        events = _events(*[str(i).zfill(2) for i in range(10)])
        batch, has_more = older_batch(events, before_id="03", limit=25)
        assert batch == events[0:3]
        assert has_more is False

    def test_mid_log_cursor(self):
        events = _events(*[str(i).zfill(2) for i in range(60)])
        batch, has_more = older_batch(events, before_id="40", limit=25)
        assert batch == events[15:40]
        assert has_more is True

    def test_cursor_not_exactly_present(self):
        events = _events("00", "05", "10", "15", "20")
        batch, has_more = older_batch(events, before_id="12", limit=25)
        # bisect_left("12") lands between "10" and "15" -> everything strictly before it
        assert batch == events[0:3]
        assert has_more is False

    def test_cursor_before_start_of_log(self):
        events = _events("05", "10", "15")
        batch, has_more = older_batch(events, before_id="00", limit=25)
        assert batch == []
        assert has_more is False


class TestAgainstRealStimulusLog:
    def test_pagination_matches_appended_order(self, tmp_path):
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        for i in range(40):
            log.append(actor="test", type="observation", content={"i": i})

        events = log.read_all()
        assert len(events) == 40

        page, has_more = most_recent_page(events, 25)
        assert has_more is True
        assert [e.content["i"] for e in page] == list(range(15, 40))

        older, has_more_older = older_batch(events, before_id=page[0].id, limit=25)
        assert has_more_older is False
        assert [e.content["i"] for e in older] == list(range(0, 15))
