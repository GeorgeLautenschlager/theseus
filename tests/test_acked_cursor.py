"""AckedCursor — how far the host has acknowledged, and whether it survives a restart."""

from __future__ import annotations

import json
import os
import threading

import pytest

from theseus.surrogates.cursor import AckedCursor

ORIGIN = "surrogate-a"


def _cursor(tmp_path):
    return AckedCursor(tmp_path / "cursor.json", origin=ORIGIN)


def test_fresh_cursor_has_acked_nothing(tmp_path):
    cursor = _cursor(tmp_path)
    # None, not 0: nothing acked is a different claim from "seq 0 was acked".
    assert cursor.acked_seq is None


def test_advanced_cursor_survives_reload(tmp_path):
    path = tmp_path / "cursor.json"
    AckedCursor(path, origin=ORIGIN).advance(7)
    assert AckedCursor(path, origin=ORIGIN).acked_seq == 7


def test_cursor_never_moves_backwards(tmp_path):
    cursor = _cursor(tmp_path)
    cursor.advance(10)
    cursor.advance(4)
    assert cursor.acked_seq == 10


def test_concurrent_advances_land_on_maximum(tmp_path):
    path = tmp_path / "cursor.json"
    cursor = AckedCursor(path, origin=ORIGIN)
    threads = [threading.Thread(target=cursor.advance, args=(seq,)) for seq in range(1, 51)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cursor.acked_seq == 50
    # The file agrees with the in-memory value.
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["acked_seq"] == 50


def test_crash_mid_write_leaves_previous_value(tmp_path, monkeypatch):
    path = tmp_path / "cursor.json"
    cursor = AckedCursor(path, origin=ORIGIN)
    cursor.advance(3)

    def boom(*args, **kwargs):
        raise OSError("simulated crash mid-replace")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        cursor.advance(4)

    # The file still holds the earlier value and is readable.
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["acked_seq"] == 3
    assert AckedCursor(path, origin=ORIGIN).acked_seq == 3


def test_corrupt_cursor_file_loads_as_none(tmp_path):
    (tmp_path / "cursor.json").write_text("not json at all {{{", encoding="utf-8")
    cursor = _cursor(tmp_path)
    assert cursor.acked_seq is None


def test_cursor_naming_different_origin_is_ignored(tmp_path):
    path = tmp_path / "cursor.json"
    path.write_text(json.dumps({"origin": "surrogate-b", "acked_seq": 9}), encoding="utf-8")
    # A cursor is per-origin: applying one stream's position to another skips real events.
    assert AckedCursor(path, origin=ORIGIN).acked_seq is None


def test_seq_below_one_is_rejected(tmp_path):
    cursor = _cursor(tmp_path)
    with pytest.raises(ValueError):
        cursor.advance(0)
    assert cursor.acked_seq is None
