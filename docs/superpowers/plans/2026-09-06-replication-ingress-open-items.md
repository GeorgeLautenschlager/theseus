# Replication Ingress: Closing the Open Items

> **For agentic workers:** REQUIRED SUB-SKILL: Use steward:steward-local-sdd to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **How this plan is written.** It specifies **contracts and the claim each test must prove** — not
> module source and not test bodies. Blueberry writes both the tests and the implementation; the
> frontier writes the contracts and reviews the diff. A fenced block containing a target module's
> body, or a full test function, means the skill has been inverted.

**Goal:** Close the five items `.steward/runs/30/runlog.md` lists as still open after #30's final
review, so the branch can merge.

**Architecture:** Four small, independent changes to shipped-and-reviewed modules. Two are
correctness (`previous_ts` never reaching the planner; a listener that `ingest`s deadlocking
forever), one is a bounded-input hardening, one is packaging. Nothing here changes the protocol
or the dedupe rules — `replication_batch.py` and `replication_dedupe.py` keep their behaviour.

**Tech Stack:** Python 3.12, Poetry (in-project `.venv`), pytest, FastAPI + Starlette `TestClient`.
Run everything with `env -u VIRTUAL_ENV poetry run ...` — the shell exports a `VIRTUAL_ENV`
pointing at another checkout.

**Branch:** `worktree-replication-ingress`, 19 commits off `main`, `605 passed` before this plan.
**Source issue:** GitHub #30. **Runlog:** `.steward/runs/30/runlog.md` § "Still open after this round".

---

## Decisions locked in for this plan

| Decision | Value |
|---|---|
| Where the ingress remembers an origin's last committed `ts` | An **in-memory dict on the ingress**, written under the existing `_commit_lock` alongside the mark advance. Not derived from the log at construction, and not persisted. `plan_batch`'s own docstring already says `None` is honest when nothing remembers it, and a zero-width span after a restart is a true statement about what the host knows. Deriving it from the log at boot is a real improvement and a **follow-up**, not this round — it would touch `HighWaterMarks`, which is reviewed bedrock. |
| The listener deadlock (I7) | **Detect the re-entry and raise**, do not make it work. A `threading.Lock` held across `append_many` is what makes the commit atomic; swapping it for an `RLock` would let a listener's nested `ingest` run inside a half-finished commit, converting a visible hang into a silent double-append. Raising turns a permanent, undiagnosable deadlock into an immediate error naming the cause. The real fix — an append-without-notify path on the log — stays deferred until a relay actually exists. |
| Oversize check (I6) | Bound the body **while reading it**, so a chunked or lying client is cut off at the limit instead of being buffered first. The declared-`Content-Length` fast path stays; this is the second half of the same check, not a replacement. |
| Exports (I5) | **Export `ReplicationIngress` and `HighWaterMarks`** from `src/theseus/__init__.py`. This overrides the runlog's "decide after #31 and #32": #30 *is* the caller #27 and #29 deferred exporting for, and a mountable ingress a composer can only reach by deep path is an export decision already made badly. `parse_batch`, `BatchRejected`, `CoalescingTrigger`, `plan_batch` and `PendingAppend` stay importable by path — `__all__` is a compatibility commitment for tag-pinned consumers, so it gets the two composition entry points and no more. |
| Not in this round | The append-without-notify path on `StimulusLog`; deriving `previous_ts` from the log at boot; auth (#40); backpressure (#38). The missing Task 4 section in `docs/superpowers/plans/2026-09-04-replication-ingress.md` is historical — the reasoning shipped in the code and the runlog, and plans are not edited after the fact. |

---

### Task 1: give the planner the lower bound it already accepts

**Files:**
- Modify: `src/theseus/replication_ingress.py`
- Test: `tests/test_replication_ingress.py`

**The defect:** `plan_batch` takes `previous_ts` and uses it as an inferred gap's `span_start`
(`replication_dedupe.py:75,123`), and `tests/test_replication_dedupe.py` pins that behaviour. But
`ReplicationIngress.ingest` never passes it, so **every inferred gap the host has ever minted has a
zero-width span** — `span_start == span_end` — regardless of how much the host actually knows.

**Contract:**

- `ReplicationIngress` keeps the last committed `ts` per origin, in memory.
- It is read and written **inside `_commit_lock`**, in the same critical section as the mark
  advance, so a concurrent retry cannot observe a torn pairing of mark and clock.
- It is updated only on a commit that actually appended, to the `ts` of the **last replicated event
  in `plan.to_append`** — not a host-minted marker, whose `ts` is the host's clock and not the
  origin's.
- It is passed as `previous_ts` on the next `plan_batch` call for that origin.
- An origin with nothing remembered passes `None`, and the span stays zero-width. That is the
  honest answer after a restart.

**Tests to write:**

| Test | Must prove |
|---|---|
| a second batch's inferred gap spans from the first batch's last event | Commit seqs 1–2, then a batch starting at seq 5. The minted gap's `span_start` is the `ts` of the seq-2 event and its `span_end` the `ts` of the seq-5 event — **not** equal to each other. |
| the first batch from an origin still gets an honest zero-width span | A fresh ingress with nothing committed for the origin mints `span_start == span_end`. |
| the remembered clock is per origin | Two origins interleaved: a hole in origin B's stream spans from B's own last event, not A's. |
| a duplicate batch does not move the remembered clock | Commit 1–2, re-POST 1–2, then jump to 5: the span still starts at the seq-2 event's `ts`. |
| a host-minted marker is not mistaken for the origin's clock | A batch that mints a gap **and** appends events leaves the remembered clock at the last **replicated** event's `ts`, so the next hole's span does not start from the host's own clock. |

- [ ] **Step 1: Write the failing tests** from the table.
- [ ] **Step 2: Run them** — `env -u VIRTUAL_ENV poetry run pytest tests/test_replication_ingress.py -v`. They must fail because the spans come back zero-width, not because of an import error.
- [ ] **Step 3: Implement the remembered clock.**
- [ ] **Step 4: Run the whole ingress and dedupe suites** — `env -u VIRTUAL_ENV poetry run pytest tests/test_replication_ingress.py tests/test_replication_dedupe.py -v`. `tests/test_replication_dedupe.py` must pass **unmodified**; the planner's behaviour is not changing.
- [ ] **Step 5: Commit** — `git commit -m "Give an inferred span the lower bound the host actually has"`

---

### Task 2: make the listener deadlock loud instead of permanent

**Files:**
- Modify: `src/theseus/replication_ingress.py`
- Test: `tests/test_replication_ingress.py`

**The defect:** `ReplicationIngress` holds `_commit_lock` across `append_many`, and `StimulusLog`
notifies listeners on the appending thread. A listener that calls `ingest` — a relay forwarding what
it receives is the plausible way to write it — re-enters a non-reentrant lock and **hangs that thread
forever; no batch from any origin ever commits again.** The class docstring names this; nothing
enforces it.

**Contract:**

- Track the thread that currently holds `_commit_lock`.
- If `ingest` is entered on a thread that already holds it, raise immediately with a message that
  names the cause — a listener calling `ingest`, re-entrant commit — rather than blocking.
- The exception type is a module-level class so a caller can catch it specifically; a bare
  `RuntimeError` would be indistinguishable from anything else going wrong under the lock.
- **Normal concurrency is unaffected:** two *different* threads still serialise on the lock exactly
  as before. Only same-thread re-entry raises.
- The docstring's "a listener must not call `ingest`" paragraph is updated to say it is now
  enforced, and how.

**Tests to write:**

| Test | Must prove |
|---|---|
| a listener that ingests raises instead of deadlocking | Subscribe a listener that calls `ingest` with a second batch; the outer `ingest` surfaces the re-entry error **within a few seconds** (the test must fail by timing out today, so give it a real bound rather than hanging the suite). |
| the re-entry error names the cause | The message mentions the listener/re-entrancy, not just "lock". |
| the outer batch is still committed | A relay listener firing does not roll back the append that notified it — the first batch is on disk and its mark advanced. *(If the implementation makes this false, say so in the runlog rather than changing the assertion.)* |
| two threads still serialise normally | The existing concurrent-delivery test still passes: different threads block and commit once, they do not raise. |

- [ ] **Step 1: Write the failing tests.** Bound the deadlock test with a timeout so it fails rather than hanging the suite.
- [ ] **Step 2: Run them** — the re-entry test must fail by hitting its bound.
- [ ] **Step 3: Implement re-entry detection.**
- [ ] **Step 4: Run the whole ingress suite** — including the pre-existing concurrency tests, unmodified.
- [ ] **Step 5: Commit** — `git commit -m "Turn the relay deadlock into an error that says so"`

---

### Task 3: bound the body while reading it

**Files:**
- Modify: `src/theseus/replication_ingress.py` (`add_routes`)
- Test: `tests/test_replication_ingress.py`

**The defect:** the declared `Content-Length` is checked before a byte is buffered, which handles an
honest client. A **chunked** request, or one that lies about its length, is buffered in full and only
then measured — so an unbounded stream is still a way to hurt this endpoint.

**Contract:**

- Read the body with `request.stream()`, accumulating chunks and checking the running total against
  `max_bytes` **as it goes**. Stop and answer `413` the moment the total exceeds it, without reading
  the rest.
- The existing `Content-Length` fast path stays — rejecting an honest oversized client before it
  uploads is strictly better than after.
- Behaviour for every within-limit request is unchanged, including the exact `413` payload shape
  the existing tests assert.
- Note in the code that this bounds the body, not the connection: auth (#40) and backpressure (#38)
  are still the answer to a hostile client, and this is not pretending otherwise.

**Tests to write:**

| Test | Must prove |
|---|---|
| a chunked oversized body is rejected | A request sent without a `Content-Length` (a generator body, which `httpx` sends chunked) exceeding `max_bytes` answers `413` and appends nothing. |
| a body that lies about its length is rejected | A declared length under the limit with a real body over it answers `413`, nothing appended. |
| the stream stops early on an oversized body | The endpoint does not consume the whole oversized stream before answering — assert on the bytes the client actually managed to send, or on a generator that records how far it got. |
| an honest oversized body still gets the fast path | The existing declared-`Content-Length` `413` test passes unmodified. |
| a within-limit body is unaffected | The existing success and duplicate tests pass unmodified. |

- [ ] **Step 1: Write the failing tests.**
- [ ] **Step 2: Run them** — the chunked and lying cases must fail today by returning `200`.
- [ ] **Step 3: Implement the streaming bound.**
- [ ] **Step 4: Run the whole ingress suite.**
- [ ] **Step 5: Commit** — `git commit -m "Bound the request body as it arrives, not after"`

---

### Task 4: declare httpx, export the composition entry points

**Files:**
- Modify: `pyproject.toml`, `poetry.lock`, `src/theseus/__init__.py`, `CLAUDE.md`
- Test: the whole offline suite

**The defect:** two tests need `httpx` (`TestClient`) and it is present only transitively through
another package — the suite breaks the day that package drops it. And nothing in the #27/#29/#30
series is exported, so a composer reaches the ingress only by deep path.

**Contract:**

- `httpx` is a declared **dev** dependency.
- `src/theseus/__init__.py` imports and adds to `__all__`: `HighWaterMarks` (from
  `theseus.high_water`) and `ReplicationIngress` (from `theseus.replication_ingress`). **These two
  and no more** — see the decision table.
- `CLAUDE.md`'s Architecture section gains one bullet after Observers: a **Replication ingress**
  (`replication_ingress.py`) entry saying it is the host's door for a surrogate's stimulus batches,
  that a composer mounts it with `ingress.add_routes(observer.app)`, and that it dedupes against
  `HighWaterMarks`, commits with `StimulusLog.append_many`, records unexplained holes as
  `stimulus.gap`, and triggers a coalesced `orient`.

**Tests to write:**

| Test | Must prove |
|---|---|
| the composition entry points import from the package root | `from theseus import HighWaterMarks, ReplicationIngress` works and both are in `theseus.__all__`. |
| the internals stay off `__all__` | `parse_batch`, `BatchRejected`, `CoalescingTrigger`, `plan_batch` and `PendingAppend` are **not** in `theseus.__all__` — the commitment is deliberate and narrow. |

- [ ] **Step 1: Add httpx** — `env -u VIRTUAL_ENV poetry add --group dev httpx`
- [ ] **Step 2: Write the failing export tests.**
- [ ] **Step 3: Run them** — expect `ImportError`.
- [ ] **Step 4: Add the exports and the `CLAUDE.md` bullet.**
- [ ] **Step 5: Run the whole offline suite** — `env -u VIRTUAL_ENV poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py --ignore=tests/e2e`. All pass, with no pre-existing test file modified except where a task above says so.
- [ ] **Step 6: Commit** — `git commit -m "Declare httpx and export what a composer needs"`

---

## Closing the runlog

After all four tasks, append a final section to `.steward/runs/30/runlog.md` recording which of the
five open items are now closed and which remain deferred (the append-without-notify path, deriving
`previous_ts` from the log at boot, and the missing Task 4 plan section), so the PR's review trail
does not still advertise fixed defects as open.
