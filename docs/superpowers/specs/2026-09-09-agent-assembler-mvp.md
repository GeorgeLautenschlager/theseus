# Agent Assembler MVP

Approved scope: a small **Python** authoring tool for repeated assembly of Tam,
isolated E2E agents, and experiments. Python definitions are maintained sources.
Generated agents use the existing composition APIs. No second language or DSL.

There is no manifest loader or capability policy engine in this checkout. Use
ordinary Python composition instead of inventing either. Resolve providers and
tools through existing registries, and validate cadence with `Cadence.parse`.
Support auto and OODA cores, terminal or web chat, explicit tool selection, optional
A-MEM on either core, and MemoryModule on Autocore. Tam uses plain Autocore.

One generated `agent.py` snapshots the definition's values. Validate before
atomically replacing that one file; refuse to replace non-generated files. The
runtime home defaults to `state/` next to the generated file; `--home` overrides it.
On boot, apply the three assembler-owned Markdown settings (constitution, persona,
cadence) before constructing the core. Logs, memory, goals, tasks, schedule, and
credentials are runtime-owned and never emitted or reset by assembly. Restart an
agent after reassembly. Consolidation remains an explicit agent policy, as in the existing Autocore.

No catalog export, packaging system, fleet manager, grant engine, or deployment
service. Tools are explicitly included or omitted; there is no pretend `ask` policy.
