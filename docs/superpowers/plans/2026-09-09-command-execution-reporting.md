# Command Execution Reporting (#35)

> **For agentic workers:** REQUIRED SUB-SKILL: Use steward:steward-local-sdd to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **How this plan is written.** It specifies **contracts and the claim each test must prove** — not
> module source and not test bodies. Blueberry writes both the tests and the implementation; the
> frontier writes the contracts and reviews the diff. A fenced block containing a target module's
> body, or a full test function, means the skill has been inverted. The signature stubs are
> interface, not implementation: they pin names across tasks.

**Goal:** Make a surrogate honest about what it did with a command. Every command the surrogate
executes produces exactly one stimulus event — `executed`, `partial`, `failed`, or `barged_in` —
on its local log, and that report rides the ordinary upstream replication path home. Without it the
host logs "I said X" while the speaker was muted, and the tape holds a false memory.

**Architecture:** Two pieces. A pure vocabulary module (`command_reports.py`) defines the four
report event types, the `Outcome` values an injected renderer returns, and write-side-validating
content constructors — the same discipline as `commands.py` and `replication_events.py`. A
surrogate-side `CommandExecutor` (`surrogates/command_executor.py`) is the reporting discipline: it
renders one command, converts the outcome (or any raised exception) into exactly one report
appended to the local log, and — driving a `CommandChannel` — advances the cursor only *after* the
report is durable. Reports are ordinary own-origin events, so `Replicator.drain()` ships them with
no special casing. The renderer itself (the reflex layer: speaker, screen, VAD, playback control)
is **injected and out of scope** — this issue defines the event and the reporting discipline, not
the barge-in detector.

**Tech Stack:** Python 3.12, Poetry, pytest, FastAPI, httpx. Run with `env -u VIRTUAL_ENV poetry run ...`
(this shell exports theseus's own venv; unset it so Poetry resolves the project's).

**Source issue:** GitHub #35. **Spec:** `docs/superpowers/specs/2026-09-04-surrogate-replication-protocol.md`
section "Downstream: command channel" → "**Execution reporting**", and Acceptance scenario #11 —
read them first.
**Builds on:** #34 (`commands.py`, `command_feed.py`, `CommandChannel`, `SseCommandChannel`), #31
(`Replicator`, `AckedCursor`, `StimulusTransport`), #30 (`ReplicationIngress`). All merged on the
current branch.

---

## File Structure

- **Create: `src/theseus/command_reports.py`** — what a report *is*: the four event types, the
  `Outcome` value objects a renderer returns, the content constructors (write-side validation), and
  the read-side predicates. Pure; imports only `stimulus_log`, `dataclasses`, and `clean_reason`
  from `replication_events`. Does no I/O and knows nothing about channels or logs.
- **Create: `src/theseus/surrogates/command_executor.py`** — the reporting discipline. `execute_one`
  turns one command + the injected renderer into exactly one appended report; `run` drives a
  `CommandChannel`, reporting each command and then advancing the cursor.
- **Modify: `src/theseus/__init__.py`** — export what a composer needs (`CommandExecutor`, the
  `Outcome` factories, and the report predicates), following how `Replicator`/`CommandFeed` are
  exported.
- **Tests:** create `tests/test_command_reports.py`, `tests/test_command_executor.py`,
  `tests/test_command_report_round_trip.py`. All three are offline (no live LLM endpoint); the
  round trip drives a real socket via the shared `serve` fixture, exactly as `test_command_round_trip.py`
  does.

## Decisions locked in for this plan

| Decision | Value |
|---|---|
| Report namespace is **not** `command.` | Report types live under a separate `command_report.` prefix. This is load-bearing, not cosmetic: `commands.is_command()` recognises anything under `command.` with a verb, so a report typed `command.report` would read back as a command — `CommandFeed` would try to re-serve the surrogate's own report to the surrogate, and `command_target` would parse it. A distinct prefix keeps reports off that gate on both logs. |
| One type per outcome | `command_report.executed` / `.partial` / `.failed` / `.barged_in`. The outcome is the discriminator the host reads; the acceptance criteria name them individually ("produces a `failed` stimulus"). These four strings are **wire protocol** and are pinned by a test, like `GAP`/`BATCH_REJECTED` in `test_replication_events`. |
| A report references its command by cross-node identity | Every report carries `command_seq`, `command_origin`, and `command_id`. Identity across nodes is `(origin, seq)` — never `id` (an event is re-identified when it lands on another log). `command_id` is kept anyway, for a human tracing one command end to end. The referenced `seq`/`origin` are the **host's** — a command is a host-origin event on the host's own log — which is exactly what lets the host correlate a report against the command it issued. |
| The renderer is injected and returns an `Outcome` | `render: Callable[[StimulusEvent], Outcome]`. `Partial`/`BargedIn`/`Failed` are **return values**, not exceptions: barge-in is a normal interruption and muted output is a deliberate refusal the renderer knows about — neither is a crash. An *unexpected* `raise` from the renderer (or from mapping its outcome to content) is caught and becomes a `failed` report. The renderer's body — actual speaking, playback, VAD — is #35's out-of-scope reflex layer and is never implemented here; tests inject fakes. |
| Exactly one report per command, failures included | `execute_one` has exactly **one** reachable `log.append`. `render` + outcome-mapping + content-construction all sit inside one `try`; any exception there is turned into a `failed` report rather than escaping. A command that fails before starting (muted, renderer raises immediately) therefore still lands exactly one event. "A command that vanishes without a report is indistinguishable from one the surrogate never received" — so there is no silent-drop path. |
| The one thing that *may* escape | Only `log.append` itself failing (disk full) propagates — the report did not become durable, so pretending otherwise would be the false memory this issue exists to prevent. Because `run` advances the cursor *after* the append, a failed append leaves the cursor unmoved and the command replays on the next connect. |
| Report-then-advance (at-least-once) | `run` does `execute_one(command)` then `cursor.advance(command.seq)`, never the reverse. This is #34's cursor rule made literal: the cursor means *executed*, a crash between the two replays the command and produces a **second** report, and a duplicate report is visible on the tape where a dropped one is not. Advancing first would convert the protocol to at-most-once and lose the very report this issue adds. |
| Reports replicate through the ordinary path | A report is a plain **own-origin** `StimulusEvent` on the surrogate's local log. `Replicator.drain()` already ships every own-origin, seq'd event above the cursor. There is no report-specific transport, endpoint, or flush — "no bypass" means the drain code is not touched at all. |
| An out-of-contract command is a loud seam bug, not a report | A command handed to `execute_one` with no `seq` (or empty `origin`/`id`) cannot be referenced by `(origin, seq)` and is not something the wire produces — `CommandFeed.pending` serves only seq'd host-origin events. `execute_one` raises `ValueError` up front for such an event rather than inventing a seq. "Every *delivered* command yields exactly one report" is about commands that came through the channel; a malformed event fed in directly is a caller bug, failed at the mistake. |
| Renderer failure reason is bounded | The `failed` report's `reason` reuses `replication_events.clean_reason`: stripped, non-empty, and truncated at `MAX_REASON_CHARS`. A renderer exception can stringify to an arbitrarily long traceback, and this goes onto a permanent, replicated tape. |
| Report `actor` | Appended with `actor="surrogate"` (configurable on the executor). `origin` is the log's own origin, assigned by `append` — which is what makes the drain pick it up. |

---

### Task 1: the report vocabulary

**Files:** create `src/theseus/command_reports.py`; test `tests/test_command_reports.py`

**Contract:**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Union
from theseus.stimulus_log import StimulusEvent

REPORT_PREFIX = "command_report."
EXECUTED = "command_report.executed"
PARTIAL = "command_report.partial"
FAILED = "command_report.failed"
BARGED_IN = "command_report.barged_in"
OUTCOMES = (EXECUTED, PARTIAL, FAILED, BARGED_IN)

# --- Outcome value objects: what an injected renderer returns. Plain data; validation
# --- of their payloads happens in the content constructors below, write-side.
@dataclass(frozen=True, slots=True)
class Executed: ...

@dataclass(frozen=True, slots=True)
class Partial:
    progress: str

@dataclass(frozen=True, slots=True)
class BargedIn:
    playback_position: str | int | float

@dataclass(frozen=True, slots=True)
class Failed:
    reason: str

Outcome = Union[Executed, Partial, BargedIn, Failed]

# --- Content constructors: build one report's `content` dict, validating write-side.
def executed(*, command_seq: int, command_origin: str, command_id: str) -> dict[str, Any]: ...
def partial(*, command_seq: int, command_origin: str, command_id: str, progress: str) -> dict[str, Any]: ...
def failed(*, command_seq: int, command_origin: str, command_id: str, reason: str) -> dict[str, Any]: ...
def barged_in(*, command_seq: int, command_origin: str, command_id: str,
              playback_position: str | int | float) -> dict[str, Any]: ...

# --- Read side: never raises (a report comes off a log that may hold anything).
def is_report(event: StimulusEvent) -> bool: ...
def report_outcome(event: StimulusEvent) -> str | None: ...
```

**Behaviour to satisfy:**

1. The four type constants equal exactly `"command_report.executed"`, `"command_report.partial"`,
   `"command_report.failed"`, `"command_report.barged_in"`. Pin them literally — they are wire
   protocol a non-Python surrogate and a host from another release must agree on.
2. Every constructor validates the shared reference fields: `command_seq` is an `int` ≥ 1 and
   **not** a `bool` (a `bool` is an `int` in Python and would serialise as `true` on a numeric wire
   field); `command_origin` and `command_id` are non-empty strings. Each violation raises
   `ValueError` naming the field.
3. `partial` additionally requires `progress` to be a non-empty string; empty or non-string raises.
4. `failed` additionally requires `reason`; it is passed through `clean_reason` (imported from
   `replication_events`) — stripped, non-empty (empty/whitespace/non-string raises), and truncated
   with a trailing `…` past `MAX_REASON_CHARS`.
5. `barged_in` additionally requires `playback_position` to be **either** a non-empty string **or** a
   real number (`int`/`float`, not `bool`) that is ≥ 0. An empty string, a negative number, `None`,
   or a `bool` raises. (Freedom for the reflex layer — a millisecond offset, `"00:03:12"`, a token
   index — without accepting a meaningless marker.)
6. Each constructor returns a dict carrying the shared fields plus its own; the dict is JSON-native
   (no datetimes, no custom objects) so `StimulusEvent.to_json` can encode it.
7. `is_report` is true iff `event.type` starts with `REPORT_PREFIX` **and** has a non-empty suffix
   (the bare prefix names the namespace, not a report in it), mirroring `is_command`.
8. `report_outcome` returns the event's type when it `is_report` and its `content` is a dict,
   else `None`. It never raises, even on a report whose content is a list or is missing fields —
   a filtering reader must not be taken down by a malformed line.
9. A command event (type `command.say`, built via `commands.command_type`) is **not** a report:
   `is_report` is false and `report_outcome` is `None`. Symmetrically, a report event is not a
   command: `commands.is_command` is false for it. This is the namespace-separation guarantee.

**Tests must prove (claims, not bodies):**

- Each of items 1–9 above, one focused test apiece (or one per constructor for 2–5).
- The cross-namespace test (item 9) drives the **real** `commands.is_command`/`is_report` pair, not
  reimplementations — it is guarding the exact collision the separate prefix exists to prevent.
- `clean_reason` truncation is exercised through `failed` (a reason longer than `MAX_REASON_CHARS`
  comes back ending in `…` and no longer), so the reuse is real, not asserted.

- [ ] Write the failing tests for items 1–9.
- [ ] Run `env -u VIRTUAL_ENV poetry run pytest tests/test_command_reports.py -v`; expected: fail (module/constructors absent).
- [ ] Implement `command_reports.py` to the contract.
- [ ] Run the tests; expected: pass.
- [ ] Commit.

---

### Task 2: the reporting primitive (`execute_one`)

**Files:** create `src/theseus/surrogates/command_executor.py`; test `tests/test_command_executor.py`

**Contract:**

```python
from __future__ import annotations
from collections.abc import Callable
from theseus import command_reports as reports
from theseus.command_reports import Outcome
from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.surrogates.cursor import AckedCursor

Renderer = Callable[[StimulusEvent], Outcome]

class CommandExecutor:
    def __init__(
        self,
        log: StimulusLog,
        render: Renderer,
        cursor: AckedCursor,
        *,
        actor: str = "surrogate",
    ) -> None: ...

    def execute_one(self, command: StimulusEvent) -> StimulusEvent:
        """Render one command and append exactly one report to the local log; return it."""
```

**Behaviour to satisfy:**

1. `execute_one` validates the command up front: `command.seq` is an `int` ≥ 1, `command.origin` is
   non-empty, `command.id` is non-empty. A violation raises `ValueError` **before** any append —
   an out-of-contract event is a seam bug, not a reportable command (see the decisions table).
2. It calls `render(command)` and maps the returned `Outcome` to a report content + type:
   `Executed`→`executed`, `Partial`→`partial` (carrying `progress`), `BargedIn`→`barged_in`
   (carrying `playback_position`), `Failed`→`failed` (carrying `reason`). The report's
   `command_seq`/`command_origin`/`command_id` come from the command event.
3. `render`, the outcome mapping, **and** the content-constructor call all sit inside one `try`.
   Any `Exception` there is converted into a `failed` report whose `reason` is the stringified
   exception (`f"{type(e).__name__}: {e}"`), passed through the `failed` constructor (so it is
   bounded). A renderer that raises, and a renderer that returns e.g. `Partial("")` (which makes the
   `partial` constructor raise), both therefore yield exactly one `failed` report — never an escape,
   never a second event.
4. Exactly **one** `log.append(self._actor, <type>, <content>)` is reachable per call. It is a
   local, own-origin append (no `origin`/`seq` args), so the log allocates the surrogate's next seq
   and the event replicates through the ordinary drain. The appended event is returned.
5. The only thing that may propagate is `log.append` itself raising — the report never became
   durable, and swallowing that would be the false memory this issue prevents.

**Tests must prove (claims, not bodies):**

- A renderer returning `Executed()` yields exactly one `command_report.executed` on the log,
  carrying the command's `seq`/`origin`/`id`. (Assert on `log.read_all()` count and the event.)
- `Partial("halfway")` yields one `command_report.partial` whose content `progress == "halfway"`.
- `BargedIn(1234)` yields one `command_report.barged_in` whose content `playback_position == 1234`;
  a string position round-trips too.
- `Failed("output muted")` yields one `command_report.failed` whose `reason == "output muted"`.
- A renderer that **raises** `RuntimeError("boom")` yields exactly one `command_report.failed` whose
  `reason` contains `boom`, and `execute_one` does **not** re-raise. (This is "fails before starting".)
- A renderer returning `Partial("")` yields exactly one `command_report.failed` (the misbehaving
  renderer is reported, not crashed on) — the log still gains exactly one event.
- Over N mixed commands (executed/partial/failed/barged_in in one sequence), the log gains exactly
  N report events, one per command, in order. (This is the exactly-one invariant at count level.)
- A command with `seq=None` raises `ValueError` from `execute_one` and appends nothing.
- Use an in-memory `StimulusLog` on `tmp_path` and a real `AckedCursor`; the renderer is a fake
  closure/list-driven stub. No network, no LLM.

- [ ] Write the failing tests above.
- [ ] Run `env -u VIRTUAL_ENV poetry run pytest tests/test_command_executor.py -v`; expected: fail (class absent).
- [ ] Implement `CommandExecutor.__init__` and `execute_one` to the contract.
- [ ] Run the tests; expected: pass.
- [ ] Commit.

---

### Task 3: the driver loop (`run`)

**Files:** modify `src/theseus/surrogates/command_executor.py`; test `tests/test_command_executor.py`

**Contract:**

```python
    # added to CommandExecutor
    def run(self, channel: "CommandChannel") -> None:
        """Drain `channel`, reporting each command then advancing the cursor.

        For every command the channel yields: execute_one(command), then
        cursor.advance(command.seq) — report first, advance second (at-least-once).
        Returns when the channel's stream() ends (doorbell drained, or close()).
        """
```

(`CommandChannel` is `theseus.surrogates.command_channel.CommandChannel`; import it.)

**Behaviour to satisfy:**

1. `run` iterates `channel.stream()`. For each command it calls `execute_one` and *then*
   `cursor.advance(command.seq)` — in that order. It never advances before the report is appended.
2. `run` returns when `stream()` ends: a `MemoryCommandChannel(block=False)` doorbell that has
   drained, or any channel whose `close()` was called.
3. `run` adds no report-specific transport or flush — it only consumes the channel and advances the
   cursor. Shipping the reports upstream is `Replicator.drain()`'s job, unchanged (proven in Task 4).

**Tests must prove (claims, not bodies):**

- Offering three commands to a `MemoryCommandChannel(block=False)` and calling `run` leaves exactly
  three reports on the log **and** the cursor advanced to the third command's seq.
- **Report-before-advance ordering:** with a renderer (or a `log.subscribe` listener) that records
  the cursor value at report time, every report is appended while the cursor still sits at the
  *previous* command's position — i.e. the advance for command _n_ happens strictly after command
  _n_'s report. (A test that only checks final counts would pass on the wrong order; this one pins it.)
- A `run` over an empty drained doorbell channel returns immediately and appends nothing.

- [ ] Write the failing tests above.
- [ ] Run `env -u VIRTUAL_ENV poetry run pytest tests/test_command_executor.py -v`; expected: fail (`run` absent).
- [ ] Implement `run` to the contract.
- [ ] Run the tests; expected: pass.
- [ ] Commit.

---

### Task 4: export the composer surface

**Files:** modify `src/theseus/__init__.py`; test folded into `tests/test_command_executor.py`

**Contract:** add to the curated exports, alongside the existing replication/command surface
(`Replicator`, `CommandFeed`, `SseCommandChannel`, etc.):

```python
from theseus.command_reports import (
    Executed, Partial, BargedIn, Failed,
    is_report, report_outcome,
)
from theseus.surrogates.command_executor import CommandExecutor
# ...and add all six names to __all__.
```

**Behaviour to satisfy:**

1. `import theseus; theseus.CommandExecutor`, `theseus.Executed`, `theseus.Partial`,
   `theseus.BargedIn`, `theseus.Failed`, `theseus.is_report`, `theseus.report_outcome` all resolve.
2. Each new name is in `theseus.__all__` (so `from theseus import *` and the API surface test see it).

**Tests must prove:**

- One test importing each of the seven names off the top-level `theseus` package and asserting they
  are the same objects as their defining modules expose. (Follow whatever existing test guards
  `__all__`/public exports, if one exists; otherwise a short new test in `test_command_executor.py`.)

- [ ] Write the failing import/export test.
- [ ] Run it; expected: fail (`AttributeError` on `theseus.CommandExecutor`).
- [ ] Add the exports and `__all__` entries.
- [ ] Run the test; expected: pass.
- [ ] Commit.

---

### Task 5: the round trip — a `failed` report reaches the host log with no bypass

**Files:** create `tests/test_command_report_round_trip.py` (no source changes — this task proves
the whole path composes and closes acceptance scenario #11)

This test reuses the exact harness shape of `tests/test_command_round_trip.py`: a host `StimulusLog`,
a `CommandFeed` served over a real socket via the `serve` fixture, a `ReplicationIngress` mounted on
the same app for the upstream direction, an `SseCommandChannel` on the surrogate resuming from its
`AckedCursor`, an `HttpTransport` + `Replicator` for the surrogate→host drain, and now a
`CommandExecutor`. Read that file first and mirror its fixtures (`HOST`, `SURROGATE`, `TARGET`,
`FAST`, `FAST_BUDGET`, `_issue`, the threaded consumer, deadline-bounded waits).

**Behaviour to prove (claims, not bodies):**

1. **Muted → `failed` on the host log (acceptance #11).** The host issues one `command.say`. The
   surrogate runs a `CommandExecutor` whose injected renderer returns `Failed("output muted")` (it
   models a muted speaker — the reflex layer is faked, per scope). Drive the executor over the
   `SseCommandChannel`; it appends one `command_report.failed` to the **surrogate** log. Then
   `Replicator.drain()` ships it, and the **host's** log — read through the ingress — contains that
   `command_report.failed`, carrying the issued command's `seq`/`origin`, with `reason == "output
   muted"`. Assert it arrived on the host, under the surrogate's origin, via the ingress — i.e. the
   ordinary replication path, with **no** report-specific route touched.
2. **`partial` carries a usable progress marker end to end.** A command whose renderer returns
   `Partial("spoke 3 of 5 words")` produces a `command_report.partial` that reaches the host with
   `progress == "spoke 3 of 5 words"` intact.
3. **`barged_in` carries a playback position end to end.** A renderer returning `BargedIn(1830)`
   produces a `command_report.barged_in` on the host with `playback_position == 1830`.
4. **Exactly one report per delivered command, failures included.** Issue three commands over the
   feed (one that succeeds, one whose renderer raises, one muted). After the executor drains them
   and the replicator drains upstream, the host log holds **exactly three** report events — one per
   command, none missing, none doubled — and the surrogate's `AckedCursor` for commands advanced to
   the third command's seq.

**Test mechanics to respect (so it is not flaky):**

- Run the `CommandExecutor.run` loop on a background thread against the live `SseCommandChannel`;
  close the channel once the expected number of reports is on the surrogate log (poll with a
  deadline, as `test_command_round_trip.py` does — never a bare sleep).
- The surrogate log and host log are **distinct** `StimulusLog`s with distinct origins; the report
  is minted on the surrogate log and re-identified on the host log on append (identity is
  `(origin, seq)`, not `id`), so assert on `(origin, seq)`/type/content, never on `id` equality
  across the two logs.
- Everything is offline. Use `FAST`/`FAST_BUDGET` so idle paths are sub-second and assert against
  deadlines, not intervals.

- [ ] Write the four round-trip tests above (they will fail until the harness is wired, then pass).
- [ ] Run `env -u VIRTUAL_ENV poetry run pytest tests/test_command_report_round_trip.py -v`; expected: pass.
- [ ] Run the full offline suite: `env -u VIRTUAL_ENV poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py`; expected: pass.
- [ ] Commit.

---

## Self-review notes (author)

- **Spec coverage.** Every acceptance checkbox in #35 maps to a task: muted→`failed` on host
  (Task 5.1), `partial` marker (Tasks 1.3/2/5.2), `barged_in` position (Tasks 1.5/2/5.3), ordinary
  path with no bypass (Task 5.1 asserts arrival via the ingress with the drain untouched), exactly
  one report per delivered command incl. failures (Tasks 2, 3, 5.4).
- **Out of scope, honoured.** No VAD, playback control, or barge-in detector is built — the renderer
  is injected everywhere and only ever faked in tests. The `barged_in` *event* is defined; the
  machinery that fires it is not.
- **No source embedded.** Only signature stubs and behaviour/claims appear above, per the
  local-sdd contract; Blueberry writes the module bodies and the test bodies.
