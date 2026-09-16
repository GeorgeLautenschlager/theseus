# Memory review follow-up

User-approved scope: fix the three reproduced defects before a small live trial.

1. Bound optional prior/recall context separately from extraction evidence.
   Preserve original events and persisted boundaries. Verify two scheduled ticks
   with a large first event, both with and without restart.
2. Never supersede current knowledge when a fact reconciliation decision is
   unavailable or invalid. Persist incoming claims as unresolved, outside current
   knowledge; verify timestamps, alternate predicates, malformed output, and reload.
3. Select bounded reconciliation context by exact attribute, lexical relevance,
   recency, and stable tie-break. Verify corrections beyond six existing facts
   and prioritization of an older exact attribute.

Reference evaluation providers must supply explicit labeled reconciliation
instead of relying on the former destructive production fallback. No live-model
validation or unrelated cost-accounting changes are included in this work.
