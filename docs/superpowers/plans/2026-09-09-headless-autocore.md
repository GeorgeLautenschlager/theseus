# Implementation

1. Add headless assembler validation, composition, and main-thread execution.
2. Extract a public Autocore.step while preserving loop behavior.
3. Check requested Unsloth model availability to honor explicit model fallbacks.
4. Verify headless execution, one-turn behavior, and missing-model fallback; run
   the offline suite. Deploy the resulting wheel to Knope's isolated environment.
