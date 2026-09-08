# Buffer Retention and Storage-Pressure Eviction (#33)

> **For agentic workers:** REQUIRED SUB-SKILL: Use steward:steward-local-sdd to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **How this plan is written.** It specifies **contracts and the claim each test must prove** — not
> module source and not test bodies. Blueberry writes both the tests and the implementation; the
> frontier writes the contracts and reviews the diff. A fenced block containing a target module's
> body, or a full test function, means the skill has been inverted. The signature stubs are
> interface, not implementation: they pin names across tasks.

**Goal:** Let a surrogate observe into a full disk. The buffer stays bounded, the oldest events go
first, and every own-origin range that goes is declared on the tape as a `storage_pressure` gap —
so the host learns what it will never receive, rather than inferring a silence.

**Architecture:** One new type, `BufferedStimulusLog`, subclassing `StimulusLog` inside
`surrogates/`. `StimulusLog` itself is not touched and grows no flag: eviction is reachable only by
importing a surrogate type on purpose, which is the issue's "a host `StimulusLog` cannot be put into
the evicting mode by configuration alone". Eviction runs inline after an append that crosses the
threshold — nothing else has to remember to call it, which is what "keeps observing" actually
requires. The surviving events **and** the gap marker are written to one temp file and land with a
single `os.replace`, so truncation and its declaration are one atomic act.

**Tech Stack:** Python 3.12, Poetry, pytest. Run with `env -u VIRTUAL_ENV poetry run ...`.

**Source issue:** GitHub #33. **Spec:** `docs/superpowers/specs/2026-09-04-surrogate-replication-protocol.md`
section Retention (line 204) and the **Resolved: buffer bounding** paragraph (line 273) — read both
first. **Builds on:** #26/#27 (`StimulusEvent` envelope, `replication_events`), #31/#32
(`Replicator`, `AckedCursor`), merged through `0d5649b`.

---

## File Structure

- **Create: `src/theseus/surrogates/buffer.py`** — `BufferPolicy` and `BufferedStimulusLog`.
  Standard library plus `theseus.stimulus_log` and `theseus.replication_events`.
- **Modify: `src/theseus/__init__.py`** — export `BufferedStimulusLog` and `BufferPolicy`
  (a surrogate deployment composes and tunes these).
- **Tests:** create `tests/test_buffered_log.py`; extend `tests/test_replication_round_trip.py`
  and `tests/test_package_exports.py`.

## Decisions locked in for this plan

| Decision | Value |
|---|---|
| What measures pressure | **Bytes of the buffer file**, read off `path.stat().st_size`. No scan, and it measures the thing that actually fills: one 2 MB vision caption counts as 2 MB, not as "one event". Event count is blind to payload size; free disk is pressure the surrogate neither caused nor can relieve, so it would evict because of somebody else's log rotation. |
| Hysteresis | Evict when size exceeds `max_bytes`; evict down to `max_bytes * low_water`. Without the gap, every append past the line rewrites the file — the buffer would spend its life at the threshold copying itself. |
| Where the behaviour lives | A **subclass** in `surrogates/`, not a flag and not a wrapper. A wrapper would need `_append_lock`, `path` and `_write_durably`, so it would force `StimulusLog` to expose exactly the internals the issue says must not be relaxed. |
| When eviction runs | **Inline, at the end of an append that crosses the threshold.** The appending thread pays for the rewrite. No append ever fails or blocks on eviction; observation continues throughout. |
| Atomicity | Survivors **and** the marker are written to one temp file in the same directory, fsynced, then `os.replace`d over the log. Marker-first would leave, after a crash, a marker declaring events that are still present. Marker-last would leave events gone with nothing saying so — the silent forgetting the spec forbids. One replace has neither failure. |
| The seq counter must be recovered **before** the file shrinks | `_recover_next_seq` derives the next seq by scanning the log for the highest own-origin seq. Once the oldest events are gone that scan returns a *lower* number, and a lazy recovery afterwards would reissue seqs the host has already accepted — a duplicate under an identity the protocol assumes is unique. Eviction therefore forces the counter to be populated before it touches the file. **This is the sharpest trap in the task; gate it.** |
| What the marker describes | **This log's own origin only.** `declared_gap` names one origin, and it is the seq space the host dedupes on. Foreign-origin events (host commands, once #34 lands) may be evicted alongside and get no marker — the host authored them and has not lost them. Evicted events with `seq is None` (pre-envelope) get no marker either: they have no range to name, and they were already unreplicable — `Replicator.drain` counts them as `skipped_unsequenced`. |
| No marker when nothing describable went | An eviction that removed only foreign-origin or unsequenced events rewrites the file and emits nothing. A `stimulus.gap` covering no own-origin seq would be a false claim. |
| The buffer never touches the cursor | `BufferedStimulusLog` does not know `AckedCursor` exists. Evicting *ahead of* the cursor is permitted by the spec and needs no cursor move: the events are simply not in the log any more, so `drain` never sees them, and the cursor stays exactly where the host put it. A buffer that could advance the cursor could advance it past events that were never delivered — and behind is recoverable, ahead is not. |
| Never evict everything | Eviction stops when one event remains, even if the buffer is still over budget. A single event larger than the whole budget is kept and the buffer sits over its cap. Dropping the only thing you observed to satisfy a byte target is the failure this issue exists to prevent, and the marker would then be the sole content of the log. |
| Known cost | Eviction rewrites the whole surviving buffer. At the defaults that is a ~205 MB copy roughly every 51 MB appended. Acceptable for a buffer whose alternative is an unbounded disk; **document it in the class docstring**, do not optimise it here. |
| Relationship to #44 | Bounding the buffer caps the *storage* cost of the marker-about-a-marker recursion #44 describes, because the markers become evictable like anything else. It does not stop the recursion. **#44 stays open**; do not close it and do not attempt it here. |

---

### Task 1: the policy, as arithmetic

**Files:** create `src/theseus/surrogates/buffer.py`; test `tests/test_buffered_log.py`

**Contract:**

```python
@dataclass(frozen=True, slots=True)
class BufferPolicy:
    """How large the surrogate's buffer may grow, and how far back eviction cuts."""
    max_bytes: int = 256 * 1024 * 1024
    low_water: float = 0.8

    def __post_init__(self) -> None: ...
```

Behaviour to satisfy:

1. Defaults are the ones above. 256 MB is a buffer a modest edge device can spare and holds days
   of text-rate observation; it is a starting point a deployment overrides, not a law.
2. `max_bytes` must be at least 1. Zero or negative is a buffer that can hold nothing.
3. `low_water` must be strictly between 0 and 1, exclusive at both ends. At `1.0` there is no
   hysteresis and the buffer rewrites itself on every append past the line; at `0.0` a single
   eviction empties the whole buffer.
4. Rejections are `ValueError` at construction, with a message naming the field and the reason —
   this is configuration, and it should fail before a surrogate is running rather than at 3 a.m.
   under pressure.

**Tests to write:**

| Test | Must prove |
|---|---|
| the defaults are the documented ones | `BufferPolicy()` has `max_bytes == 256 * 1024 * 1024` and `low_water == 0.8`. |
| a non-positive `max_bytes` is refused | `0` and `-1` each raise `ValueError`. |
| `low_water` outside the open interval is refused | `0.0`, `1.0`, `-0.5` and `1.5` each raise `ValueError`. Both ends are exclusive — assert on `1.0` specifically, since "less than or equal to 1" is the natural wrong guard and it is the one that reintroduces the thrash. |
| a legal policy at the boundary is accepted | `low_water=0.99` and `low_water=0.01` construct without raising. |
| the policy is frozen | Assigning to a field raises. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect import errors.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full offline suite (720 passing before).
- [ ] **Step 5:** Commit — `git commit -m "Give the surrogate buffer a size policy"`

---

### Task 2: the evicting log, oldest first

**Files:** modify `src/theseus/surrogates/buffer.py`; test `tests/test_buffered_log.py`

This task builds the truncation and its crash-safety. The marker is Task 3 — leave it out here so
the two claims are gated separately.

**Contract:**

```python
class BufferedStimulusLog(StimulusLog):
    """A surrogate's local log: an append-only tape that is allowed to forget its oldest end.

    Not a `StimulusLog` a host could configure into this — a distinct type, imported on purpose.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        origin: str = DEFAULT_ORIGIN,
        *,
        policy: BufferPolicy = BufferPolicy(),
    ) -> None: ...

    def append(self, ...) -> StimulusEvent: ...          # same signature as StimulusLog.append
    def append_many(self, events) -> list[StimulusEvent]: ...

    def _evict_if_needed(self) -> StimulusEvent | None:
        """Evict oldest-first if the file is over budget. Returns the gap marker, if one
        was emitted (Task 3); `None` when nothing was evicted."""
```

Behaviour to satisfy:

1. `append` and `append_many` call `super()` first, then `_evict_if_needed()`, and return what
   `super()` returned. An append is never rejected, delayed or altered by pressure.
2. `_evict_if_needed` starts with a bare `stat().st_size` check and returns immediately when the
   file is within budget. The common case must not read the log.
3. When over budget it takes `_append_lock`, **re-checks the size under the lock** (another thread
   may have just evicted), and then — before reading or writing anything — forces the seq counter
   to be recovered if it has not been already. See the decisions table; this is the trap.
4. Survivors are chosen by walking the events oldest-first, accumulating each one's serialised
   byte length, and dropping from the front until the remainder is at or below
   `max_bytes * low_water`. Byte length is measured the way the file measures it: the event's
   `to_json()` line plus its newline, UTF-8 encoded.
5. Eviction never removes the last remaining event, even if the buffer is still over budget.
6. Survivors are written to a temp file created in the log's **own directory** (so `os.replace`
   is a same-filesystem rename and therefore atomic), flushed, `os.fsync`ed, then replaced over
   the log. A crash leaves either the whole old file or the whole new one.
7. `_append_lock` is held across the whole rewrite, so no append can land in the file between the
   read and the replace and be lost by it.

**Tests to write:**

| Test | Must prove |
|---|---|
| a buffer under budget is never rewritten | Appending well under `max_bytes` leaves every event readable and nothing evicted. |
| crossing the threshold evicts oldest-first | With a small `max_bytes`, append enough events that the cap is crossed; the surviving events are a **contiguous suffix** of what was appended — the newest ones — and the evicted ones are the oldest. Assert on identity (seq), not just on count. |
| eviction lands under the low-water mark, not merely under the cap | After eviction `stat().st_size <= max_bytes * low_water`. A rule that stopped at `max_bytes` would evict on nearly every subsequent append; this is the assertion that makes the hysteresis load-bearing. |
| the log is still readable and well-formed after eviction | `read_all()` after eviction returns the survivors, in order, with no torn or interior-corrupt line. |
| **the seq counter survives eviction** | Append past the cap so the low seqs are evicted, then append again, and the new event's `seq` is **above every seq ever issued** — not restarted from the surviving events. Construct this so a fresh `_recover_next_seq()` on the shrunken file would give a demonstrably lower number: evict enough that the highest surviving seq is well below the highest issued. |
| appends continue during pressure | A loop of appends that crosses the threshold several times completes without raising, and the final event is readable. Observation does not pause for eviction. |
| a lone oversized event is kept | With a `max_bytes` smaller than one event's own line, appending that event leaves it in the log and does not empty the buffer. |
| eviction is safe under concurrent appends | Several threads appending across the threshold leave a log whose every line parses and whose own-origin seqs contain **no duplicate**. |
| a plain `StimulusLog` never evicts | The same append load against `StimulusLog` grows the file past any cap. There is no policy argument to pass it. This is the issue's last acceptance box; assert it as a test, not as a comment. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full offline suite.
- [ ] **Step 5:** Commit — `git commit -m "Let the surrogate's buffer forget its oldest end"`

---

### Task 3: declare what went

**Files:** modify `src/theseus/surrogates/buffer.py`; test `tests/test_buffered_log.py`

**Contract:** no new public names. `_evict_if_needed` now mints a `stimulus.gap` event and returns
it. The marker's content comes from `replication_events.declared_gap` with
`reason="storage_pressure"`, `origin=self.origin`, `from_seq`/`to_seq` the **inclusive** bounds of
the evicted own-origin seqs, and `span_start`/`span_end` the oldest and newest `ts` among those
same events.

Behaviour to satisfy:

1. The marker is built **in memory** with a freshly allocated own seq and an id minted from its
   `appended_ts`, exactly as `StimulusLog.append` builds one — then written into the same temp file
   as the survivors, after them, and landed by the same single `os.replace`. It is **not** appended
   through `append`: a second write would be a second crash window, and the whole point is that the
   truncation and its declaration are indivisible.
2. Because its seq is above every survivor's, the marker sits at the tail of the file. The host
   therefore sees the declaration *after* the surviving events, never before the range it describes.
3. Listeners are notified of the marker after the replace is durable, and **outside**
   `_append_lock` — the same ordering `append` uses, for the same reason: a listener is free to
   append.
4. `from_seq` and `to_seq` cover only own-origin evicted events carrying a seq. Foreign-origin and
   `seq is None` events may be evicted in the same pass and are not described.
5. If the evicted set contains no own-origin sequenced event, the file is still rewritten and
   **no marker is emitted**; `_evict_if_needed` returns `None`.
6. The marker itself is an ordinary event on the buffer and is evictable on a later pass like
   anything else. A surrogate that evicts a gap marker before it replicates has forgotten that it
   forgot — but the eviction that removed it declares a range that covers it, so the tape stays
   honest. Say this in the docstring.

**Tests to write:**

| Test | Must prove |
|---|---|
| eviction declares the range it dropped | After eviction the log's last event is a `stimulus.gap` whose `from_seq` and `to_seq` are **exactly** the lowest and highest evicted own-origin seq — inclusive at both ends — and whose `reason` is `storage_pressure`. Assert the exact numbers against the seqs the test itself appended. |
| the marker's span is the evicted events' own clock | `span_start` and `span_end` match the oldest and newest `ts` among the evicted events, not the eviction's wall clock. Give the appended events distinct explicit `ts` values so a wall-clock implementation cannot pass. |
| the marker is declared, not inferred | Its content carries `declared: True`; the host must be able to tell this from a hole it inferred itself. |
| the marker's seq is above every survivor | So it sorts to the tail and replicates after the events it survives. |
| truncation and marker are one act | After eviction, a single `read_all()` shows both the shortened event list and the marker. There is no observable state in which events are gone and no marker exists. A crash test is out of reach here; assert instead that the marker is present in the *same* file read that shows the survivors, and that eviction performs exactly one `os.replace`. |
| an eviction that dropped nothing describable emits no marker | With a buffer holding only foreign-origin events (append them with an explicit foreign `origin` and `seq`), crossing the cap rewrites the file and adds **no** `stimulus.gap`. |
| unsequenced events are evicted but not described | A hand-written log line with a null `seq` is evicted without appearing in any marker's range. |
| a listener hears the marker | A `subscribe`d listener receives the gap event, after the survivors are durable. |
| successive evictions declare successive ranges | Two evictions produce two markers whose ranges do not overlap and ascend. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full offline suite.
- [ ] **Step 5:** Commit — `git commit -m "Declare the range storage pressure took"`

---

### Task 4: the round trip, and the cursor ahead of the eviction

**Files:** modify `tests/test_replication_round_trip.py`, `tests/test_package_exports.py`,
`src/theseus/__init__.py`

This is the acceptance box the issue words as *"events at or ahead of the acked cursor can be
evicted; the cursor stays consistent"*. It is the scenario the whole issue exists for, and it is
end-to-end: a `BufferedStimulusLog` under pressure, a `Replicator`, and a real host ingress.

Behaviour to satisfy:

1. `BufferedStimulusLog` and `BufferPolicy` are exported from `theseus/__init__.py`, per the
   repo convention that a module joining the public API is added to the curated exports.

**Tests to write:**

| Test | Must prove |
|---|---|
| **unacked events are evicted and the host learns of the hole** | Fill a buffered log past its cap while the cursor is at (or behind) the evicted range — so the events dropped were never delivered. Drain to a real host ingress. The host's log ends with the surviving events and a `stimulus.gap` naming the evicted range with reason `storage_pressure`. The host's `HighWaterMarks` for that origin lands on the highest seq actually sent. Nothing raises, and the drain reports no rejected batch. |
| the cursor is never moved by eviction | The cursor's `acked_seq` immediately after an eviction equals what it was immediately before. Eviction is not an ack. |
| a drain after eviction does not re-send or stall | Following the eviction, `drain()` ships the survivors and the marker, and a second `drain()` ships nothing — the cursor advanced past everything the log still holds. |
| the host does not double-count the evicted range | The host's high-water mark ascends across the hole without a `400`; the declared marker explains the jump, so the ingress records no *inferred* gap for the same range. Assert the host log has exactly one gap event covering it. |
| eviction under an active drain is safe | Appends crossing the cap while a drain is in flight leave the host's accepted seqs strictly ascending with no duplicate. |
| the exports are reachable | `from theseus import BufferedStimulusLog, BufferPolicy` works and the existing export test's list is updated. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full offline suite.
- [ ] **Step 5:** Commit — `git commit -m "Prove eviction ahead of the cursor round trips"`

---

## Out of scope

- **#44** — the marker-about-a-marker recursion against a broken-but-answering host. Retention
  bounds its storage cost; the recursion itself stays open and #44 stays open with it.
- **Command-channel events** (#34). The buffer is written to tolerate foreign-origin events
  because they are coming, but nothing here produces them.
- **Choosing a threshold for a real deployment.** The defaults are a starting point; the issue
  scopes this to "configurable", and it is.
- **Compaction, indexing, or a non-JSONL substrate.** Eviction rewrites the file. If that cost
  ever bites, that is its own issue.
