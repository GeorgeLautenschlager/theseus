# Memory reliability for a paired experiment

Memory formation is now recoverable and can be scheduled by Auto. Recall remains
an explicit agent tool call. Each partner owns a separate `memory/` directory;
reading a peer log does not import that peer's durable memory.

## Configuration

```python
memory=MemorySpec(
    "module",
    model=ModelSpec("ollama", "gemma4:e4b"),
    embedding=ModelSpec("ollama", "nomic-embed-text"),
    consolidate_every_seconds=300,
    episode_max_events=20,
    episode_max_chars=24000,
    recall_budget_tokens=2000,
)
```

These local model names illustrate configuration, not validated choices for the
experiment. Select the extractor and embedder deliberately. Extraction is a
separate model call from cognition; remote extraction and embeddings consume the
configured provider's balance.

The schedule is opt-in. At the end of an Auto step, a due tick consolidates at most
one bounded batch and repairs up to five missing embeddings. It is synchronous:
there is no background worker, and an idle agent does not tick. The first tick
starts with the beginning of the local log. Set the batch size and interval for
the expected event rate; inspect `core.memory_consolidator.pending_events` and
`last_error` for backlog and failures. Failures are logged without aborting the
completed cognitive turn. A-MEM keeps its existing lifecycle.

## Recovery and compatibility

- A durable `pending.json` contains fixed record IDs and the complete prepared
  transaction before any layer changes. Restart replays missing writes, commits
  the episode ledger, writes its trace once, and removes the preparation.
- `formation/cursor.json` saves episode boundaries before extraction. Failure or
  restart retries the same range, including when new stimuli arrive. A committed
  episode is skipped if the cursor update was interrupted.
- Module operations use thread and process locks and reload their projections.
  A reader cannot observe a partially committed module transaction. Model calls
  hold the module lock, so slow extraction can delay recall. Use separate runtime
  homes for the pair. Direct layer writes bypass the module transaction boundary.
- JSONL append repairs an incomplete final suffix, including broken UTF-8;
  malformed interior records raise instead of being silently dropped.
- Existing records without embedding metadata remain lexically searchable.
  `memory.repair_embeddings(limit=20)` rebuilds derived vectors in a sidecar.
  Provider class and model name identify the embedding space, and vectors must
  have compatible dimensions, finite values, and nonzero magnitude. Keep the
  underlying model fixed under that name; changing its weights or endpoint under
  the same name cannot be detected. Recall never rebuilds old embeddings itself.
- Existing ledgers remain readable. Recovery cannot reconstruct transactions
  interrupted by an older version that did not write a preparation record.
  Back up an existing home while its agent is stopped before upgrading.

The droplet's POSIX locking/fsync path is the deployment target. Windows locking
is implemented but has not received equivalent platform testing. File durability
still depends on the filesystem and host honoring fsync.

## Retrieval and extraction limits

Embedding outages fall back to lexical search over summaries, original episode
evidence, and wisdom. Recent-event search filters for the query and excludes
recall results. Historical episodes remain available and are labeled separately
from current facts.

A newly extracted fact is reconciled against the subject's existing current
knowledge (bounded to `reconciliation_context_k` records, default 6) before it is
written, rather than simply replacing whatever shares its subject+predicate key
and has an equal-or-newer episode timestamp. Reconciliation decides one of:
`new` (nothing existing describes the same attribute), `reinforce` (restates an
existing value — recorded, but not a change), `replace` (a genuine update, in
whatever wording), `coexist` (a different attribute sharing a broad predicate,
e.g. two independent preferences both filed under "preference"), `contradiction`
(conflicting reports with no clear resolution — both stay current, the newer one
names the other in `contradicts`), or `historical` (the claim itself describes a
past or superseded state and must never become current, regardless of when its
episode was processed — report time is not effective time). A subject with no
existing knowledge costs no reconciliation call at all. A failed, unavailable, or
malformed reconciliation response falls back to the pre-reconciliation default
(replace an exact subject+predicate match, otherwise write new) rather than
blocking the episode; failures are recorded to `reconciliation_failures.jsonl`.
`KnowledgeRecord.supersedes` is therefore decided by this step, not inferred by
`KnowledgeLayer` from keys or timestamps — more than one record can legitimately
be current for the same subject+predicate at once (coexistence or an unresolved
contradiction), and a `reconciliation="historical"` record is kept in the
append-only file but never enters the current set.

Lexical and vector rankings combine within a layer. Across layers the result is
a weighted rank interleave, favoring current facts; it does not detect semantic
agreement between different records beyond what reconciliation already resolved
at write time.

Extraction requires a nonempty summary and an assertions list. Malformed responses
try the next provider and leave the episode retryable if none succeeds. Invalid
individual assertions go to `dead_letter.jsonl`. Newly accepted assertions must
name one or more exact `support_event_ids` from the episode's extraction evidence.
Missing, unknown, duplicate, and recall-context IDs are rejected before routing.
Each accepted claim also stores `attribution` (`direct_observation`,
`partner_report`, or `inference`), `reported_by` for partner reports, and
`action_status` (`not_applicable`, `intention`, `attempt`, `failure`, or
`confirmed_outcome`). These distinctions appear in recall text. The original
stimulus events remain in the episode record, even when extraction receives
truncated excerpts. Records written before these fields existed read with
unknown provenance; their source episode ID is not converted into invented
event IDs. The pending transaction replays the metadata unchanged after a crash.

Valid event references establish only that a claim cites eligible input. They do
not prove that the cited events say what the claim says, that a partner report is
true, or that a planned action succeeded. Multi-event offline tests include a
falsely confirmed payment with a valid event reference and mark it semantically
wrong; live extraction still requires accuracy evaluation.

Requests have character and output-token bounds. Complete events are packed into
a shared extraction request using redistributed unused capacity rather than a
fixed equal share per event; an event too large even after redistribution is
pulled out and chunked into its own bounded request(s) that together cover it
completely, so decisive evidence near its end (a success/failure marker, a final
total) still reaches extraction instead of being cut away by a prefix
truncation. Complete source evidence is retained in the episode record
regardless. A bound on characters is an approximate token bound. Huge individual
events still occupy disk and memory. Reads rebuild file-backed projections, so
long-run performance needs measurement at the experiment's actual event volume.

## Evaluation and costs

Run the deterministic, offline diagnostic in a fresh directory:

```sh
poetry run python -m theseus.memory_experiment_eval --workdir /tmp/memory-trial-1
```

It consolidates 14 reference-labeled episodes, restarts memory after moving the
evidence out of the live tail, and compares 12 queries with a raw lexical control.
The initial run found expected text in the first result on **10/12** layered
queries versus **7/12** raw-control queries; top-three results were **11/12** versus
**8/12**. Forbidden text appeared in the first result on **0/12** versus **4/12**.
An ownership-wording query missed at rank one, and the unknown bank-account query
returned related but insufficient evidence. These misses are retained in reports.

This uses reference extractions and exact-text diagnostics, not semantic scoring
or evidence that Fable/Astra can sustain the proposed experiment. The raw control
has no extraction cost. Some forbidden strings occur in negations or historical
reports, so manual review matters.

For an explicitly chosen live model, add `--provider PROVIDER --model MODEL` and
optionally `--embedding-provider PROVIDER --embedding-model MODEL --answers`.
The answer mode runs both retrieval systems through the selected extractor model
as answerer. Run separate trials for each actual experiment model; inspect
answers and evidence IDs for stale facts, unsupported claims, and fabricated
completion. Empty citation lists alone do not establish grounded answers.

`report.json` records results, latency, request counts, token estimates, and
provider-reported usage when available. Consolidation traces live in
`memory/traces/consolidation.jsonl`; failed extraction attempts are recorded in
`extraction_failures.jsonl`. Estimates and reported counts are distinct. Failed
requests, internal provider retries, cognitive turns, and maintenance outside the
trial may not be included. These are diagnostics, not a CAD billing ledger or a
hard spending limit. Reconcile actual provider billing before setting the
30-day operating budget.

The offline tests additionally exercise interrupted writes, failed extraction,
embedding outages/model changes, competing module instances, and real assembled
Auto formation plus recall after restart when the clue is absent from both
partners' context windows. No live inference was used for this validation.

### Multi-event consolidation accuracy eval

`src/theseus/memory_accuracy_eval.py` is a second, parallel evaluation focused on
**knowledge transitions across multi-event episodes** (the older
`memory_experiment_eval.py` keeps the single-event raw-lexical retrieval control).
It consolidates labeled multi-event scenarios — plans followed by failure or
success, alternate-wording corrections (under a genuinely different predicate,
not just a shared key), coexisting preferences under one broad predicate,
historical reports, an unresolved contradiction between two partner reports,
decisive evidence at the end of an oversized event, recall repetition, split
action/result pairs, and repeated or contradictory principles — and reports
correct knowledge updates, supported-claim retention, unsupported claims,
source-ID validity, and agreement with labeled assertion metadata **separately**.
The latter compares attribution, action status, and labeled supporting events;
valid IDs alone do not count as semantic correctness. Reconciliation decisions
for scenarios that need one (replace across differently worded predicates,
coexistence, contradiction) are scripted per episode against the module's live
knowledge state, the same way reference extractions are scripted. It also
verifies recall after a cold restart and after the source events leave the live
context window.

Run it offline (deterministic, no live endpoint — reference extractions, part of
the offline suite):

```bash
env -u VIRTUAL_ENV poetry run python -m theseus.memory_accuracy_eval --workdir /tmp/acc-run
```

Run it live (explicitly selected, never part of CI):

```bash
env -u VIRTUAL_ENV poetry run python -m theseus.memory_accuracy_eval \
  --workdir /tmp/acc-live --provider PROVIDER --model MODEL \
  [--embedding-provider PROVIDER --embedding-model MODEL] [--answers]
```

The offline oversized-tail scenario reports `decisive_tail_in_budget: true` — the
decisive fact at the end of an oversized event reaches extraction because the
event is chunked rather than head-truncated (#65). Substring matches and valid
citation IDs are recorded for review but are **not** treated as proof of
semantic support.
