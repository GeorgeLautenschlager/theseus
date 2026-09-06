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
    live = AckedCursor(path, origin=ORIGIN)
    live.advance(7)

    assert live.acked_seq == 7
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
    # And the live object did not run ahead of what actually landed. It is the failed write
    # that makes this observable at all — on a successful one the two always agree, which is
    # why asserting it anywhere else proves nothing.
    assert cursor.acked_seq == 3


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all {{{",
        # Every case below carries the RIGHT origin, so the origin guard cannot be what
        # rejects it — otherwise this test would prove only that guard, over and over.
        '{"origin": "%(o)s", "acked_seq": true}',   # bool is an int in Python; seq 1 it is not
        '{"origin": "%(o)s", "acked_seq": 2.5}',
        '{"origin": "%(o)s", "acked_seq": -3}',
        '{"origin": "%(o)s", "acked_seq": "7"}',
        '{"origin": "%(o)s"}',
        '["%(o)s", 7]',
    ],
)
def test_an_unusable_cursor_file_loads_as_none(tmp_path, payload):
    """Every one of these fails in the safe direction — the surrogate re-sends and the host
    dedupes. `true` is the one that bites without the bool guard: `isinstance(True, int)` is
    True, so it would load as seq 1 and skip the real seq 1 permanently."""
    (tmp_path / "cursor.json").write_text(payload % {"o": ORIGIN}, encoding="utf-8")

    assert _cursor(tmp_path).acked_seq is None


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
