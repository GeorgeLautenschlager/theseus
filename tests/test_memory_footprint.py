"""Loaded memory stores must hold embeddings compactly.

A list of Python floats costs ~32 bytes per element (8-byte pointer + 24-byte
float object); a production store of ~1,400 4096-dim vectors held that way sat
at ~470 MB RSS and OOM-killed a 1 GiB container once a second process loaded
the same store. Vectors are held as packed float64 instead — exact, so the
JSONL on disk round-trips unchanged.
"""

from __future__ import annotations

import gc
import json
import random
import tracemalloc
from datetime import datetime, timezone

from theseus.memory_layer import MemoryRecord
from theseus.memory_module import MemoryModule
from theseus.stimulus_log import StimulusLog
from theseus.wisdom_layer import WisdomRecord

DIM = 1024
ROWS = 40
MODEL = "tests.Embedder:fake"


def _vector(rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(DIM)]


def _write_store(memory_dir) -> None:
    rng = random.Random(7)
    memory_dir.mkdir()
    ts = datetime(2026, 9, 1, tzinfo=timezone.utc)
    with open(memory_dir / "memory.jsonl", "w") as mem, open(memory_dir / "embeddings.jsonl", "w") as emb:
        for i in range(ROWS):
            rid = f"m{i}"
            mem.write(MemoryRecord(rid, ts, "evidence", f"summary {i}", _vector(rng),
                                   embedding_model=MODEL).to_json() + "\n")
            emb.write(json.dumps({"id": rid, "layer": "memory", "model": MODEL, "vector": _vector(rng)}) + "\n")


def test_loaded_store_holds_vectors_compactly(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_store(memory_dir)
    log = StimulusLog(tmp_path / "log.jsonl")

    gc.collect()
    tracemalloc.start()
    try:
        module = MemoryModule(memory_dir, log)
        gc.collect()
        retained, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    elements = 2 * ROWS * DIM  # inline record vectors + embeddings.jsonl index
    assert len(module.memory) == ROWS
    # Packed float64 is 8 bytes/element; lists of floats are ~32.
    assert retained < 12 * elements, f"{retained / elements:.1f} bytes per vector element"


def test_loading_streams_records_instead_of_buffering_the_file(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_store(memory_dir)
    log = StimulusLog(tmp_path / "log.jsonl")
    on_disk = sum(f.stat().st_size for f in memory_dir.iterdir())

    gc.collect()
    tracemalloc.start()
    try:
        module = MemoryModule(memory_dir, log)
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(module.memory) == ROWS
    # Parsing one line at a time costs about one record over what is retained;
    # buffering a whole file (as bytes, then decoded lines) costs its size twice.
    assert peak - retained < on_disk / 4, f"load transient {peak - retained} bytes for {on_disk} on disk"


def test_memory_record_vector_round_trips_exactly():
    vector = [0.011393250897526741, -0.043674129992723465, 1e-300, 3.0]
    line = MemoryRecord("m1", datetime(2026, 9, 1, tzinfo=timezone.utc), "c", "s", vector).to_json()

    loaded = MemoryRecord.from_json(line)

    assert list(loaded.embedding) == vector
    assert loaded.to_json() == line


def test_wisdom_record_vector_round_trips_exactly():
    vector = [0.011393250897526741, -0.043674129992723465, 1e-300, 3.0]
    line = WisdomRecord("w1", datetime(2026, 9, 1, tzinfo=timezone.utc), "statement", vector).to_json()

    loaded = WisdomRecord.from_json(line)

    assert list(loaded.embedding) == vector
    assert loaded.to_json() == line
