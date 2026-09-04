# BRIEF: Surrogate Replication Protocol (Phase 1)

**Project:** Theseus Agent Construction Kit
**Date:** 2026-09-04
**Status:** Specified — not yet implemented
**Supersedes:** the wire-format section of `docs/windows_surrogate_plan.md` (see below)

---

## Context

A Theseus surrogate is remote embodiment: sensory apparatus and effectors that live on a
machine other than the one the host agent runs on (a Windows desktop with mic/speaker/screen,
later an Android phone). The surrogate keeps its own local StimulusLog. This brief specifies
the **replication edge** — how surrogate observations reach the host's log, and how host
commands reach the surrogate.

It specifies the wire contract only. Reflex processing on the surrogate (VAD, STT, vision
captioning, barge-in) and cognition on the host are out of scope.

## Relationship to prior work

`docs/windows_surrogate_plan.md` is an earlier, unimplemented plan for a one-directional
Windows screen surrogate. Its "Wire format" section makes two decisions this brief
reverses, and they must not both be implemented:

| `windows_surrogate_plan.md` | This brief |
|---|---|
| The wire payload is deliberately **not** a `StimulusEvent`; the ingress owns the event shape. | The surrogate keeps its own `StimulusLog` and ships events; the envelope is the contract. |
| Event ids and timestamps are minted by the **receiving** log, because "remote minting imports PC clock skew". | The surrogate stamps `event_ts` and `seq`; the host adds `appended_ts`. Skew is retained and observable, not erased. |

This brief supersedes those decisions. The tiered Windows surrogate becomes *a* surrogate
under this protocol — its escalation pipeline is the reflex layer that produces events,
and this protocol is how those events travel — rather than a parallel mechanism with its
own transport.

## Goals

- Best-effort delivery of surrogate stimuli to the host log: ordered, never duplicated, and
  explicit about what failed to arrive.
- A command path that works when the host wants to act with no triggering stimulus (hours of
  silence is the normal case, not an edge case).
- One contract, multiple transports — the LAN desktop surrogate and a mobile surrogate must
  differ only in how bytes move, not in what the protocol means.
- Design for deletion: both channels sit behind an interface; swapping SSE for something else
  must not touch the log, the Assembler, or the agent.

## Non-goals

- Auth, TLS, pairing/enrollment **for LAN deployments only**. A desktop surrogate on the same
  trusted network may ship without them in Phase 1. They are a **hard precondition** for any
  surrogate reachable off-LAN (edge device on cellular, a machine at a friend's house): the
  upstream endpoint appends directly to the agent's memory, and an unauthenticated one on the
  open internet is a memory-injection surface, not a deferred nicety.
- Multi-surrogate arbitration (two surrogates, one room). Schema supports it; policy doesn't.
- Salience-based flush policy on the surrogate. The protocol allows a batch at any time; *when*
  to send is a surrogate-local decision, specified separately.

## Roles

A node is a **host** or a **surrogate**. Role is configuration, not negotiation — the
relationship is prearranged and asymmetric.

- **Host**: accepts stimulus batches, appends them to its StimulusLog, issues commands.
- **Surrogate**: emits stimulus events, ships them upstream, executes commands on arrival.

The two directions are **independent and asynchronous**. The surrogate sends stimuli on its own
schedule; the host issues commands on its own schedule. Neither is a response to the other.

## Event envelope

Every stimulus event carries:

| Field | Source | Purpose |
|---|---|---|
| `seq` | surrogate | Monotonic per-surrogate. Dedupe and gap detection. Gaps are legal. |
| `origin` | surrogate | *Where* — `kitchen-surrogate`, `android-01`, `webchat`. Routing. |
| `actor` | surrogate | *Who* — `user`, `host`, `timer`. Provenance. |
| `event_ts` | surrogate | When it happened, by the surrogate's clock. |
| `appended_ts` | host | When it landed on the host's log. |
| `payload` | surrogate | Event type and body. |

`origin` and `actor` are separate fields. The same mind reaches the agent through several
channels; collapsing them makes the agent either believe in two users or unable to decide
which mouth to answer from.

Both timestamps are retained. `appended_ts` is authoritative for log order; `event_ts` is
authoritative for meaning. The gap between them is observable clock skew — a surrogate drifting
900ms should show up in the trace, not silently scramble the agent's sense of before-and-after.

## Delivery posture: augmentation, not dependency

A host agent is a complete agent on its own. A surrogate *augments* it with presence somewhere
else; the host functions without one. Nothing in this protocol may be designed as though the
host depends on the surrogate's stream being complete.

Stable network conditions are explicitly **not** assumed. A surrogate may be an edge device
kilometres from the nearest tower, or a machine in another country on someone else's uplink.
The host is enriched by as much as the surrogate manages to deliver, and unbothered by the rest.

Three consequences that run through the rest of this brief:

- **Gaps are normal operation, not errors.** The host accepts a batch that begins above its
  high-water mark, appends it, and carries on.
- **`seq` is monotonic, not contiguous.** Duplicate suppression and ordering are guaranteed;
  completeness is not.
- **Recent data beats complete data.** Anywhere the two conflict, the surrogate ships what it
  has now and abandons what it couldn't send.

### Gap markers

The surrogate always knows what it dropped — `seq` is assigned at local write time, so
eviction or abandonment happens with the range in hand. Whenever it skips forward it emits a
`stimulus.gap` event carrying the abandoned range, a reason (`link_down`, `retry_exhausted`,
`storage_pressure`), and the wall-clock span covered.

This preserves a distinction worth keeping:

- **Declared gap** (marker present): normal operation. "I wasn't observing from 16:02 to 16:40,
  the link was down." The agent can reason about it.
- **Inferred gap** (host sees a `seq` jump with no marker): the surrogate died mid-buffer, or
  something is genuinely broken. Host logs it as such.

Same hole, different diagnosis. Neither is rejected.

## Upstream: stimulus replication

**Transport:** HTTP POST, JSONL body, one event per line, ordered by `seq`.

**Batch:** a contiguous `seq` range from exactly one `origin`. One batch in flight at a time.
Pipelining is prohibited in Phase 1 — it buys nothing on a LAN and costs ordering.

**Application is all-or-nothing.** The whole batch is appended to the host's StimulusLog or
none of it is. There is no partial state for the surrogate to reason about.

**Append, do not interleave.** The host appends in arrival order. It does not sort incoming
events into the existing log by `event_ts` — the log is append-only and arrival-ordered.
Chronological ordering is a *read-time* concern: the Assembler sorts by `event_ts` when it
builds a context window. This is the load-bearing decision in this brief.

**Response:**

| Code | Meaning | Surrogate behaviour |
|---|---|---|
| `2xx` | Batch committed (or already committed — see dedupe) | Advance cursor. |
| `4xx` | Permanently unacceptable. Do not retry. | Advance cursor, emit `replication.batch_rejected`. |
| `5xx` / network failure | Unknown or transient | Retry same `seq` range with backoff. |

**Dedupe:** the host tracks a high-water `seq` per `origin`.

- Batch entirely at or below the high-water mark → duplicate. Discard, return `2xx`.
- Batch straddling the mark → append only the events above it, return `2xx`.
- Batch starting above `high_water + 1` → **gap. Append it and carry on.** If the surrogate
  declared the gap with a `stimulus.gap` marker, that marker is just another event on the tape.
  If it didn't, the host records an inferred-gap event of its own and continues. Never reject.

A lost ack (host commits, response dies in flight) resolves as a duplicate on retry and
returns `2xx`. The surrogate's job on retry is to stop worrying, not to find out it was wrong.

**Batch limits:** max event count and max byte size, both configured. A surrogate returning
from an outage chunks its backlog into limit-sized batches and drains them in `seq` order with
no inter-batch delay. Recovery is therefore serialized — accepted tradeoff, in exchange for a
bounded blast radius per batch and no unbounded POST bodies.

**Abandon rule:** retry is bounded by max attempts *and* max age. On exhaustion the surrogate
skips the batch, emits a `stimulus.gap` marker (`retry_exhausted`), advances its cursor past
the abandoned range, and proceeds to the next batch.

Without this, unbounded buffering plus retry-until-success head-of-line blocks the whole
channel on one undeliverable batch while the buffer grows behind it — the surrogate hoards
forever, stuck on batch 12, delivering nothing. Recent data beats complete data.

**Retention:** the surrogate buffers as much as its storage allows. Truncation ahead of the
acked cursor **is permitted** under storage pressure: oldest-first, and it emits a
`stimulus.gap` marker (`storage_pressure`) for the evicted range. A surrogate observing into a
full disk keeps observing; it does not stop, and it does not silently forget.

**Poison batch:** the `4xx` class exists so one malformed or oversized batch cannot wedge the
channel forever. Rejection emits a local event, so the gap is visible on the tape rather than
being a silence the agent can't account for.

## Downstream: command channel

Commands are the host's own log entries, transmitted wholesale. The surrogate executes on
arrival; it does not evaluate, defer, or negotiate.

**Semantics:** identical to upstream — commands carry a monotonic `seq`, the surrogate holds a
cursor, and reconnection resumes from that cursor rather than dropping what it missed. A
command issued while the surrogate was rebooting is queued, not lost.

**Transports (behind one `CommandChannel` interface):**

- **LAN (desktop):** SSE. Idle-tolerant by design; heartbeat comment lines every 15–30s keep
  proxies and NAT tables from declaring the connection dead. `Last-Event-ID` carries the cursor
  on reconnect.
- **Mobile:** push-as-doorbell. Push wakes the app, the app opens a channel, drains by cursor,
  closes. Doze will kill a long-lived connection regardless of heartbeating; the radio is off.

**Execution reporting:** the surrogate emits a stimulus event for every command it executes —
`executed`, `partial` (with how far it got), `failed`, `barged_in` (with playback position).
These replicate upstream through the normal path. Commands stay fire-and-forget on the wire;
confirmation arrives as experience. Without this the host logs "I said X" while the speaker was
muted, and the tape holds a false memory.

## Surrogates do not reason

A surrogate has **no agency**: no goals, no memory beyond its local log, no policy, and it never
decides *whether* to act — only how to render what it was told.

This is not a ban on inference. A surrogate runs VAD, STT, and a local vision model; those are
transducers turning signal into events. The prohibited thing is a second mind the host must
negotiate with, which would defeat the purpose of presence.

## Acceptance scenarios

1. Normal batch appends in arrival order; Assembler window sorts by `event_ts`.
2. Duplicate batch returns `2xx` and appends nothing.
3. Lost ack → retry → `2xx`, no duplicate events on the host log.
4. Declared gap: surrogate abandons a range, emits `stimulus.gap`, host appends both marker and
   subsequent events without error; high-water mark advances past the hole.
5. Inferred gap: `seq` jump with no marker is appended, and the host records an inferred-gap
   event.
6. Oversized/malformed batch returns `4xx`; surrogate advances and emits `batch_rejected`.
7. Retry exhaustion: undeliverable batch is abandoned after max attempts/age; the channel
   drains subsequent batches rather than blocking on it.
8. Storage pressure: surrogate evicts oldest buffered events, emits `storage_pressure` gap, and
   keeps observing.
9. Surrogate offline 1h, then drains backlog in chunks with correct final high-water mark.
10. Command issued during surrogate downtime is delivered on reconnect via cursor.
11. Command executed with muted output produces a `failed` stimulus on the host log.
12. Surrogate clock skewed 900ms: both timestamps present, skew derivable.

## Open questions

- Command timeout: how long is a queued command still worth executing? A "say hello" from six
  hours ago probably shouldn't fire on reconnect.
- Backpressure: what the host does when a surrogate floods it. Note this now interacts with the
  abandon rule — sustained `429`s should burn retry budget and produce declared gaps rather than
  an ever-growing surrogate buffer.
- Concrete retry budget: max attempts and max age values.
- Auth and pairing scheme for off-LAN surrogates (precondition, not deferred — see Non-goals).

**Resolved:** buffer bounding. The surrogate buffers as much as it can, evicts oldest-first
under pressure, declares what it dropped, and keeps observing throughout.
