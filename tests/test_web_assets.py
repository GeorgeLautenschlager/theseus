"""Tests for the shared web assets module (issue #116, item 1)."""

from __future__ import annotations

from pathlib import Path

import theseus.web
from theseus.web.assets import STATIC_DIR, TEMPLATES_DIR, format_sse_event


def test_asset_dirs_point_at_packaged_directories():
    web_dir = Path(theseus.web.__file__).parent
    assert TEMPLATES_DIR == web_dir / "templates"
    assert STATIC_DIR == web_dir / "static"
    assert TEMPLATES_DIR.is_dir()
    assert STATIC_DIR.is_dir()


def test_format_sse_event_shape():
    assert format_sse_event("x") == "event: message\ndata: x\n\n"
