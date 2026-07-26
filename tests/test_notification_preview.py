from __future__ import annotations

from theseus.web.preview import notification_preview


class TestShortText:
    def test_short_text_passes_through_unchanged(self):
        assert notification_preview("Hello George.") == "Hello George."

    def test_empty_text_falls_back(self):
        assert notification_preview("") == "New message"

    def test_whitespace_only_text_falls_back(self):
        assert notification_preview("   \n\t  ") == "New message"


class TestWhitespaceCollapse:
    def test_newlines_and_runs_collapse_to_single_spaces(self):
        assert notification_preview("one\n\ntwo   three\tfour") == "one two three four"

    def test_leading_and_trailing_whitespace_is_stripped(self):
        assert notification_preview("  padded  ") == "padded"


class TestMarkdownNoise:
    def test_heading_markers_are_stripped(self):
        assert notification_preview("## Status report\nAll clear.") == "Status report All clear."

    def test_dash_bullets_are_stripped(self):
        assert notification_preview("- first\n- second") == "first second"

    def test_star_bullets_are_stripped(self):
        assert notification_preview("* first\n* second") == "first second"

    def test_backticks_are_stripped(self):
        assert notification_preview("run `make test` now") == "run make test now"

    def test_dashes_inside_a_line_survive(self):
        assert notification_preview("a well-known trade-off") == "a well-known trade-off"


class TestTruncation:
    def test_long_text_is_truncated_at_a_word_boundary(self):
        text = "word " * 60

        preview = notification_preview(text, limit=20)

        assert preview == "word word word word…"
        assert not preview.startswith(" ")

    def test_truncated_preview_respects_the_limit(self):
        preview = notification_preview("alpha beta gamma delta epsilon zeta", limit=16)

        assert len(preview.rstrip("…")) <= 16

    def test_text_exactly_at_the_limit_is_not_truncated(self):
        text = "a" * 10

        assert notification_preview(text, limit=10) == text

    def test_a_single_word_longer_than_the_limit_is_hard_cut(self):
        preview = notification_preview("a" * 50, limit=10)

        assert preview == "a" * 10 + "…"

    def test_default_limit_is_140_characters(self):
        preview = notification_preview("x " * 200)

        assert len(preview.rstrip("…")) <= 140
