# Approved memory reliability work

## Design

Harden the existing layered MemoryModule for a month-long paired-agent trial.
Preserve explicit recall and separate stores. Use a durable preparation record
and idempotent replay for multi-file consolidation, model-aware derived vectors
with lexical fallback, and an opt-in bounded Auto formation policy with a durable
cursor. Treat extraction validity and semantic quality as separate concerns.

## Implementation and validation

1. Repair torn append suffixes; test interruption at each transaction write.
2. Validate extraction envelopes and retain retryable episode boundaries.
3. Filter incompatible embeddings and support bounded sidecar repair.
4. Prefer current timestamped facts and query-relevant recall results.
5. Wire scheduled formation into assembled Auto agents; test restart and recall
   after evidence leaves both local and peer context windows.
6. Add an offline experiment fixture, raw lexical control, optional live model
   evaluation, and usage diagnostics. Preserve observed misses in the report.
7. Run the offline suite and document operations and remaining live validation.

See [operating documentation](../../memory-reliability.md) for configuration,
measured diagnostic results, and limitations. Actual Fable/Astra trials require
explicit model identifiers and are not replaced by the offline reference fixture.
