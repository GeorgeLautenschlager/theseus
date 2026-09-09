# CommandChannel Interface and SSE Transport (#34)

> **For agentic workers:** REQUIRED SUB-SKILL: Use steward:steward-local-sdd to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **How this plan is written.** It specifies **contracts and the claim each test must prove** — not
> module source and not test bodies. Blueberry writes both the tests and the implementation; the
> frontier writes the contracts and reviews the diff. A fenced block containing a target module's
> body, or a full test function, means the skill has been inverted. The signature stubs are
> interface, not implementation: they pin names across tasks.

**Goal:** Give the host a way to act with no triggering stimulus. Hours of silence is the normal
case, so this is a real downstream channel, not a reply to the surrogate's next POST. Commands
carry the host's own `seq`, the surrogate holds a cursor, and a reconnect resumes from it — a
command issued while the surrogate was rebooting is queued, not lost.

**Architecture:** One interface and two sides of it. `CommandChannel` is the seam that lets the
mobile push-as-doorbell transport drop in later without a caller noticing: it yields commands from
the cursor forward, blocking between them on a live connection and simply returning when a
doorbell drain runs dry. The host serves `GET /commands/{target}` as SSE over its own log; the
surrogate runs a reconnecting client that resumes from its cursor via `Last-Event-ID`. Only
bytes-moving differs between transports, never what the protocol means.

**Tech Stack:** Python 3.12, Poetry, pytest, FastAPI, httpx. Run with `env -u VIRTUAL_ENV poetry run ...`.

**Source issue:** GitHub #34. **Spec:** `docs/superpowers/specs/2026-09-04-surrogate-replication-protocol.md`
section "Downstream: command channel" — read it first.
**Builds on:** #26 (`StimulusEvent` envelope), #31 (`AckedCursor`), merged through `dce15b2`.

---

## File Structure

- **Create: `src/theseus/commands.py`** — what a command event *is*: type namespace, content
  shape, and the predicates that identify and address one. Pure; imports only `stimulus_log`.
- **Create: `src/theseus/command_feed.py`** — the host side. Reads its own log, filters commands
  for one target above a cursor, and serves them as SSE. Mounts with `add_routes`.
- **Create: `src/theseus/surrogates/command_channel.py`** — the `CommandChannel` protocol and
  `MemoryCommandChannel`, the in-process implementation that proves the seam.
- **Create: `src/theseus/surrogates/sse_command_channel.py`** — the surrogate side: a
  reconnecting SSE client that resumes from its cursor.
- **Modify: `src/theseus/surrogates/cursor.py`** — docstring only; `AckedCursor` is reused here
  for a second purpose and should say so.
- **Modify: `src/theseus/__init__.py`** — export what a composer needs.
- **Tests:** create `tests/test_commands.py`, `tests/test_command_feed.py`,
  `tests/test_command_channel.py`, `tests/test_command_round_trip.py`.

## Decisions locked in for this plan

| Decision | Value |
|---|---|
| Cursor advances **after** hand-off | At-least-once. The caller executes, then advances. A crash between the two re-delivers the command, so a command can run twice. This **inverts** the upstream asymmetry and the inversion is the point: upstream, behind is recoverable because the *host* dedupes — nothing dedupes a spoken sentence. Between a duplicate and a silence, the duplicate is the one #35's execution reporting can show on the tape. Say this in the protocol's docstring; it is the kind of thing that gets "fixed" by someone who remembers the upstream rule. |
| Addressing | **Endpoint per surrogate**: `GET /commands/{target}`. The host filters its own log by the command's target *before* anything reaches the wire, so a surrogate never receives, buffers, or evicts another machine's commands. |
| Interface shape | `CommandChannel.stream()` returns an **iterator that may end**. SSE blocks between commands and never ends on its own; a doorbell drain returns when the queue is empty. The caller's loop is identical, which is the issue's "a second fake implementation works with no change to callers". |
| What a command is | A **host-origin event on the host's own log** whose `type` starts with `command.`, carrying `{"target": <surrogate origin>, "payload": {...}}` in `content`. The verb lives in the type (`command.say`), the arguments in the payload. #34 defines no verbs — it is transport. |
| Identity on the wire | The SSE `id:` field is the **host's own `seq`** for that event, as a decimal string. That is what `Last-Event-ID` carries back, and it is already monotonic per origin. |
| The cursor type | Reuse **`AckedCursor`**, unchanged, with the *host's* origin. It is already "the highest seq of origin X I have durably processed", which is exactly this. Do not fork it or rename it; add one paragraph to its module docstring naming the second use, because a reader who has only seen the replicator will otherwise think the file is misplaced. |
| Replay-then-follow ordering | **Subscribe to the log first, then replay from the file, then drain the subscription discarding anything at or below the last replayed seq.** Reading first and subscribing second drops every command appended in between. This is the sharpest correctness trap in the task; gate it. |
| No `Last-Event-ID` | Stream **from the beginning** of the host's command history for that target, not from now. "Queued, not lost" is the whole point, and a surrogate always sends its own cursor — the case only arises for a genuinely new surrogate. That a brand-new surrogate would then execute months of history is **#37's** problem (command staleness); leave the seam, do not pre-empt the answer. Name it in the docstring. |
| A malformed `Last-Event-ID` | Treated as absent. It arrives from the network; refusing to serve is worse than replaying, and replaying is the direction the protocol already tolerates. |
| Heartbeats | An SSE comment line (`: ` prefix) every 15–30s of idleness, so proxies and NAT tables do not declare a silent connection dead. Configurable, defaulting inside that range. |
| Reuse, not re-invention | `web_chat_ui_observer.py` already runs SSE — `_sse_stream` (`:161`) and `_format_sse_event` (`:299`). Follow its shape: a per-listener `Queue`, `request.is_disconnected()` to end the generator, a keep-alive on queue timeout, and `run_in_threadpool` for the blocking get. **But do not import it**, and note the two things it does not do: it never emits an `id:` field, and it has no replay — a listener there sees only what is published after it connects. Both are load-bearing here. |
| Graceful shutdown | An infinite stream hangs uvicorn's shutdown; `web_chat_ui_observer.serve` works around it with `timeout_graceful_shutdown=3` (`:296`). A composer mounting this endpoint inherits that requirement. Document it at `add_routes`, and make the generator exit promptly on disconnect so the workaround is a backstop rather than the mechanism. |

---

### Task 1: what a command is

**Files:** create `src/theseus/commands.py`; test `tests/test_commands.py`

**Contract:**

```python
COMMAND_PREFIX = "command."

def command_type(verb: str) -> str:
    """`"say"` -> `"command.say"`. Rejects an empty or already-prefixed verb."""

def command_content(*, target: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Content for a command addressed to one surrogate."""

def is_command(event: StimulusEvent) -> bool: ...

def command_target(event: StimulusEvent) -> str | None:
    """The surrogate a command is for, or None if the event is not a well-formed command."""
```

Behaviour to satisfy:

1. `command_type` rejects an empty verb, a verb already carrying the prefix (double-prefixing
   is a caller bug, not something to paper over), and a verb containing whitespace.
2. `command_content` rejects an empty `target` — an unaddressed command is one every surrogate
   or no surrogate executes, and both are wrong. `payload` may be empty; a verb with no
   arguments is legitimate.
3. `is_command` is true only for a `type` that starts with the prefix **and has a verb after
   it** — `"command."` alone is not a command.
4. `command_target` returns `None` rather than raising for anything malformed: these events
   arrive off a log that may hold anything, and a feed that raises while filtering is a feed
   that stops serving.
5. Follow `src/theseus/replication_events.py` for house style — it is the closest sibling.

**Tests to write:**

| Test | Must prove |
|---|---|
| a verb becomes a namespaced type | `command_type("say") == "command.say"`. |
| a bad verb is refused | Empty, whitespace-containing, and already-prefixed verbs each raise `ValueError`. Assert the already-prefixed case specifically — it is the one a caller reaches by being helpful. |
| content carries target and payload | Round-trips both; an empty payload is accepted. |
| an unaddressed command is refused | An empty `target` raises `ValueError`. |
| the bare prefix is not a command | An event typed exactly `"command."` is not `is_command`. |
| a non-command event is not a command | An ordinary `"observation"` event is false, and `command_target` on it is `None`. |
| a malformed command yields None, not an exception | A command-typed event whose content is missing `target`, or whose `target` is not a string, returns `None` from `command_target` without raising. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect import errors.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full offline suite (762 passing before).
- [ ] **Step 5:** Commit — `git commit -m "Say what a command is"`

---

### Task 2: the seam

**Files:** create `src/theseus/surrogates/command_channel.py`; test `tests/test_command_channel.py`

This is the interface the whole issue exists to establish, plus the second implementation its
acceptance criteria demand. Building it *before* the SSE transport is deliberate: an interface
extracted from one implementation fits that implementation and nothing else.

**Contract:**

```python
class CommandChannel(Protocol):
    def stream(self) -> Iterator[StimulusEvent]:
        """Yield commands from the cursor forward, in seq order.

        Blocks between commands on a live transport and ends only when the transport does;
        returns when a doorbell-style transport has drained. A caller loops over it the same
        way either way — that sameness is the point of the interface.
        """


class MemoryCommandChannel:
    """An in-process channel: commands handed to `offer` are yielded by `stream`."""
    def __init__(self, *, block: bool = False) -> None: ...
    def offer(self, event: StimulusEvent) -> None: ...
    def close(self) -> None: ...
    def stream(self) -> Iterator[StimulusEvent]: ...
```

Behaviour to satisfy:

1. `MemoryCommandChannel` is a real, useful implementation — the fake a composer wires up to
   test an agent without a network, and the second implementation the issue's acceptance box
   asks for. It is **not** test-only scaffolding, so it lives in `src/`, not `tests/`.
2. With `block=False` (the default), `stream()` yields whatever has been offered and then
   **returns** — the doorbell shape.
3. With `block=True`, `stream()` waits for more and ends only when `close()` is called — the
   live-connection shape. A `close()` from another thread must end an in-progress `stream()`
   promptly rather than after a timeout.
4. Commands are yielded **in the order offered**, and each exactly once per stream.
5. `stream()` may be called again after it returns, and resumes with whatever has been offered
   since. Two *concurrent* streams on one channel are not supported; say so in the docstring
   rather than half-supporting it.

**Tests to write:**

| Test | Must prove |
|---|---|
| the doorbell shape drains and returns | Offer three, `list(stream())` is those three in order, and the call **returns** rather than blocking. Guard the test with a timeout so a regression fails instead of hanging the suite. |
| the live shape blocks until closed | With `block=True`, a `stream()` running on a worker thread is still alive after offers stop, and ends once `close()` is called. Mark the worker `daemon=True` — a wedged non-daemon worker turns a failing assertion into a hang at interpreter shutdown. |
| a close from another thread ends the stream promptly | Time-bounded: closing wakes the stream in well under the poll interval, not after it. |
| nothing is yielded twice | Across two successive `stream()` calls, no command appears in both. |
| a second stream resumes | Offer, drain, offer again, drain again — the second call yields only the new ones. |
| the protocol is satisfied structurally | `MemoryCommandChannel` is accepted where `CommandChannel` is annotated, and a deliberately incomplete class is not. Use `isinstance` against a `runtime_checkable` protocol, or a static assertion — whichever the codebase already prefers. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect import errors.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full offline suite.
- [ ] **Step 5:** Commit — `git commit -m "Establish the CommandChannel seam"`

---

### Task 3: the host serves its commands

**Files:** create `src/theseus/command_feed.py`; modify `src/theseus/surrogates/cursor.py`
(docstring only); test `tests/test_command_feed.py`

**Contract:**

```python
class CommandFeed:
    """Serves one host log's commands to surrogates over SSE, per target, from a cursor."""

    def __init__(
        self,
        log: StimulusLog,
        *,
        heartbeat_seconds: float = 20.0,
        poll_seconds: float = 1.0,
    ) -> None: ...

    def add_routes(self, app: FastAPI, *, path: str = "/commands") -> None:
        """Mount `GET {path}/{target}` on an existing app — a host already serving a chat
        UI and a replication ingress does not need a third port."""

    def build_app(self) -> FastAPI: ...

    def pending(self, target: str, *, after: int | None) -> list[StimulusEvent]:
        """Commands for `target` with seq above `after`, oldest first. The replay half,
        pulled out so it is testable without a connection."""
```

Behaviour to satisfy:

1. `pending` returns only events that are `is_command`, whose `command_target` matches, whose
   `origin` is the log's own, and whose `seq` is above `after` (`None` meaning from the
   beginning). Ordered by `seq`, not by file order.
2. The endpoint reads `Last-Event-ID` from the request headers. Absent or unparseable is
   treated as `None` — see the decisions table.
3. **Subscribe to the log before reading it for replay.** Then replay, then serve from the
   subscription discarding anything at or below the highest seq already replayed. The reverse
   order silently drops every command appended during the read; on a busy host that is the
   most likely command to be lost, because it is the newest one.
4. Each event is framed with an `id:` line carrying its seq, an `event:` line, and `data:`
   lines carrying the event's `to_json()`. Multi-line data is framed as multiple `data:` lines,
   the way `_format_sse_event` already does it — a raw newline inside one `data:` line ends the
   event early.
5. A heartbeat comment line is emitted after `heartbeat_seconds` of idleness and does not
   disturb the cursor or the client's parse.
6. The generator ends promptly when `request.is_disconnected()`, and **unsubscribes from the
   log in a `finally`**. A listener left behind on every dropped connection is a leak that
   grows with reconnects — which, on a flaky link, is the normal case.
7. A command for a *different* target never appears on this stream.

**Tests to write:**

| Test | Must prove |
|---|---|
| pending filters by target, origin and cursor | With commands for two targets and one plain observation on the log, `pending("a", after=None)` returns only `a`'s commands, in seq order. With `after=` a real seq, only those above it. |
| pending ignores foreign-origin events | A replicated event that happens to be command-typed and addressed to the target is **not** served — the host serves only commands it issued. |
| the endpoint replays from Last-Event-ID | Against a log holding several commands, a request carrying `Last-Event-ID: <seq>` yields only the ones above it, each with an `id:` matching its seq. |
| a missing Last-Event-ID replays everything | Not "nothing", and not "from now" — the queued-not-lost rule. |
| a malformed Last-Event-ID is treated as absent | `Last-Event-ID: banana` yields the full history rather than a 4xx or a crash. |
| **a command appended during replay is not dropped** | Append a command *between* the subscribe and the replay read — for instance by appending from a log listener registered first, or by monkeypatching `read_all` to append on its way through — and assert the stream still delivers it exactly once. This is the ordering trap; a test that cannot distinguish subscribe-first from read-first is not testing it. |
| a command is delivered exactly once | It appears in neither the replay nor the live half twice, when its seq sits exactly at the boundary. |
| another target's command never appears | Two targets, two streams, no crossover. |
| the stream heartbeats when idle | With a short `heartbeat_seconds`, an otherwise-silent connection produces comment lines and no events. |
| a disconnect ends the generator and unsubscribes | After the client disconnects, the log has no listener left. Assert on the log's listener count, not on the absence of an exception. |
| data framing survives a multi-line payload | A command whose payload contains a newline round-trips through the framing to an equal `StimulusEvent`. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect import errors.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full offline suite.
- [ ] **Step 5:** Commit — `git commit -m "Serve the host's commands per surrogate"`

---

### Task 4: the surrogate listens

**Files:** create `src/theseus/surrogates/sse_command_channel.py`; test extends
`tests/test_command_channel.py`

**Contract:**

```python
class SseCommandChannel:
    """A `CommandChannel` over SSE: connects, resumes from the cursor, reconnects on drop."""

    def __init__(
        self,
        url: str,
        cursor: AckedCursor,
        *,
        client: httpx.Client | None = None,
        max_reconnects: int | None = None,
        reconnect_seconds: float = 2.0,
    ) -> None: ...

    def stream(self) -> Iterator[StimulusEvent]: ...
```

Behaviour to satisfy:

1. On connect it sends `Last-Event-ID` from `cursor.acked_seq`, omitting the header entirely
   when the cursor has never advanced. Omitting is not the same as sending `0`; the server
   distinguishes them.
2. It yields a `StimulusEvent` per SSE event, parsed from the `data:` lines with
   `StimulusEvent.from_json`. **It does not advance the cursor** — the caller does, after
   executing. That is the at-least-once decision, and putting the advance in here would quietly
   convert it to at-most-once. Say so in the docstring.
3. Comment lines (heartbeats) are consumed and yield nothing.
4. A dropped connection is retried, resuming from the cursor as it stands *now* — so commands
   the caller has already executed and advanced past are not replayed. `max_reconnects=None`
   means retry forever; a number bounds it so a test can end.
5. A malformed `data:` payload is skipped with a warning, not raised: one bad frame must not
   take the channel down, and the next command is still worth having. Log it — a silently
   skipped command is the failure this whole protocol exists to prevent.
6. `client` is injectable so tests drive an ASGI app in-process, exactly as
   `HttpTransport`/`test_replication_round_trip.py` do with `TestClient`.

**Tests to write:**

| Test | Must prove |
|---|---|
| it resumes from the cursor | With a cursor at seq N, the request carries `Last-Event-ID: N` and only later commands arrive. |
| a never-advanced cursor sends no header | The header is **absent**, not `0` — assert on the request the server actually received. |
| the channel does not advance the cursor | After consuming commands, `cursor.acked_seq` is unchanged. This pins the at-least-once decision; without it a later change to "helpfully" advance here would pass every other test. |
| heartbeats yield nothing | A stream of comment lines produces no events and does not end the iteration. |
| a dropped connection reconnects from the cursor | Serve two commands, drop, and assert the second connection's `Last-Event-ID` reflects what the caller advanced to — not what the channel received. |
| a malformed frame is skipped, not fatal | A `data:` line that is not valid JSON is skipped and the following good command still arrives. |
| it satisfies the protocol | Same structural check as `MemoryCommandChannel`, so the seam is real. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect import errors.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full offline suite.
- [ ] **Step 5:** Commit — `git commit -m "Let a surrogate listen for its commands"`

---

### Task 5: the round trip, and the acceptance scenarios

**Files:** create `tests/test_command_round_trip.py`; modify `src/theseus/__init__.py` and
`tests/test_package_exports.py`

End to end, host log to surrogate, through a real `CommandFeed` mounted on a real app driven
in-process by `TestClient` — the same rig shape as `tests/test_replication_round_trip.py`.

**Behaviour to satisfy:** export `CommandFeed`, `CommandChannel`, `MemoryCommandChannel`,
`SseCommandChannel` and the `commands` helpers a composer needs, per the repo convention.

**Tests to write:**

| Test | Must prove |
|---|---|
| **a command issued during downtime is delivered on reconnect** | The issue's first acceptance box. Append commands while no channel is connected; connect with a cursor from before them; every one arrives, in order. |
| **no replay of already-executed commands** | The second box. Execute and advance through several, reconnect, and assert none of the executed ones arrive again. |
| an interleaved issue-and-consume stays in order | Commands appended while a stream is live arrive in seq order with none missed. |
| a second implementation needs no caller change | The fifth box. Write the consumer loop **once**, run it against both `MemoryCommandChannel` and `SseCommandChannel`, and assert the same commands come out. Parametrise so it is literally one loop, not two that look alike. |
| the two directions are independent | A replicator drain and a command stream against the same host in one test do not interfere: upstream events land, commands arrive, and neither's cursor moves the other's. |
| a disconnected stream leaves no listener | After the round trip, the host log's listener list is empty — the leak in #6 of Task 3, asserted end to end. |
| the exports are reachable | `from theseus import CommandFeed, MemoryCommandChannel, SseCommandChannel` works and the export test's list is updated. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full offline suite.
- [ ] **Step 5:** Commit — `git commit -m "Prove the command channel end to end"`

---

## Out of scope

- **The mobile push transport.** The interface is built so it can drop in; nothing here
  implements it.
- **Execution reporting** (#35). Commands are fire-and-forget on the wire; confirmation
  arrives as experience, later.
- **Command staleness** (#37). A brand-new surrogate replaying old history is real and is
  #37's question. Leave the seam, name it in the docstring, do not answer it here.
- **Command verbs.** #34 is transport. Nothing here decides what `command.say` means.
- **Auth and pairing** (#40). Still a precondition for anything off-LAN, still unanswered.
