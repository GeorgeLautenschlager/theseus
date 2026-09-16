# Implementation plan: assertion event provenance (#64)

1. Extend the extraction schema and prompt with evidence IDs, report attribution,
   and action status. Keep recall outputs in a context-only block.
2. Validate each candidate's metadata against the evidence events before write
   routing. Dead-letter invalid candidates without discarding valid siblings.
3. Extend all durable record types and recall rendering. Decode old rows with
   unknown event provenance, and retain full episode evidence.
4. Exercise multi-event source-ID and semantic counterexamples, plus crash
   recovery, restart, and idempotent retry tests. Run the offline suite.
