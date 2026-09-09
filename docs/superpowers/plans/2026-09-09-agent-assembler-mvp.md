# Implementation plan

1. Add native Python definitions (`AgentSpec`) and shared composition helpers.
2. Snapshot a definition into a readable launcher with validation and atomic write.
3. Add minimal, variant, and Tam definitions; document ownership and memory choices.
4. Exercise boot, a real offline cognitive turn with a fake provider, reassembly,
   preservation of state, and failed assembly. Run the offline suite.
5. Commit on `codex/agent-assembler-mvp` and open a pull request.
