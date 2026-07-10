from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.templating import Jinja2Templates

from src.modules.stimulus_log import StimulusEvent

_TEMPLATES_DIR = (
    Path(__file__).parent.parent / "src" / "modules" / "web" / "templates"
)


def _templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["pretty_json"] = lambda content: json.dumps(
        content, indent=2, ensure_ascii=False
    )
    return templates


def _event(content: dict) -> StimulusEvent:
    return StimulusEvent(
        id="01ABCDEFGHJKMNPQRSTVWXYZ0",
        ts=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        actor="tam",
        type="decision",
        content=content,
    )


class TestDebugRowEscaping:
    def test_content_is_html_escaped_not_raw(self):
        templates = _templates()
        macros = templates.env.get_template("_debug_macros.html").module
        html = macros.debug_row(_event({"x": "<script>alert(1)</script>"}))
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    def test_pretty_json_output_is_indented_and_parseable(self):
        templates = _templates()
        content = {"nested": {"a": 1}, "list": [1, 2, 3]}
        macros = templates.env.get_template("_debug_macros.html").module
        html = macros.debug_row(_event(content))
        assert "\n  " in html  # indent=2 produced multi-line JSON
        # Round-trip: the escaped content, unescaped back, must parse as the original dict.
        import html as html_module

        start = html.index("<code>") + len("<code>")
        end = html.index("</code>")
        raw_json = html_module.unescape(html[start:end])
        assert json.loads(raw_json) == content


class TestTemplatesCompile:
    def test_debug_html_renders(self):
        templates = _templates()
        rendered = templates.get_template("debug.html").render(
            events=[_event({"a": 1})], has_more=True, oldest_id="01ABCDEFGHJKMNPQRSTVWXYZ0"
        )
        assert "debug-log" in rendered

    def test_debug_html_renders_empty(self):
        templates = _templates()
        rendered = templates.get_template("debug.html").render(
            events=[], has_more=False, oldest_id=None
        )
        assert "No stimuli recorded yet." in rendered

    def test_older_fragment_renders(self):
        templates = _templates()
        rendered = templates.get_template("_debug_older_fragment.html").render(
            events=[_event({"a": 1})], has_more=False, oldest_id="01ABCDEFGHJKMNPQRSTVWXYZ0"
        )
        assert "debug-row" in rendered
        assert "load-older" not in rendered

    def test_new_rows_fragment_renders(self):
        templates = _templates()
        rendered = templates.get_template("_debug_new_rows_fragment.html").render(
            events=[_event({"a": 1})]
        )
        assert 'hx-swap-oob="beforeend:#debug-log"' in rendered
