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
recall results. Current facts use normalized subject/predicate keys and the newest
source episode timestamp; late consolidation of an older episode cannot replace
a newer fact with the same key. Historical episodes remain available and are
labeled separately from current facts.

Lexical and vector rankings combine within a layer. Across layers the result is
a weighted rank interleave, favoring current facts; it does not detect semantic
agreement or contradictions between different records. Different wording for the
same predicate can still create separate facts. Timestamp order is not a substitute
for understanding when a reported fact actually became effective.

Extraction requires a nonempty summary and an assertions list. Malformed responses
try the next provider and leave the episode retryable if none succeeds. Invalid
individual assertions go to `dead_letter.jsonl`. The prompt distinguishes plans,
attempts, failures, and completed actions, but schema validation does not prove
semantic accuracy or verify claims against their source events.

Requests have character and output-token bounds. Oversized events are explicitly
excerpted in the prompt; complete source evidence is retained in the episode
record. A bound on characters is an approximate token bound. Huge individual
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
