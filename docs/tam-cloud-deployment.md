# Tam cloud deployment

Deployment activated on 2026-09-20 on the existing 2 GB Ubuntu 24.04 droplet
at `159.203.30.50`. See the live status command below for current state.

## Configuration

- Definition: `agents/tam_cloud.py`, using the approved Autocore + Telegram subset
  of `deployment_manifest_wip.toml`.
- Agent and deployment ID: `tam`; Compose project: `tam`.
- Cognition: OpenRouter `z-ai/glm-5.3-flash`, every hour and on Telegram input,
  with `OPENROUTER_REASONING_EFFORT=low` in the container settings.
- Memory embeddings: OpenRouter `qwen/qwen3-embedding-8b`.
- Memory formation: standard Theseus MemoryConsolidator, checked on cognitive
  turns when its 15-minute interval is due. The old standalone TamMemory worker
  and Claude CLI are not run on the droplet.
- Telegram bot: `@Tamlyn_bot`, with the existing user/chat allowlists baked into
  the generated specification. Tokens are separate files under the host's
  `/srv/theseus/tam/secrets`, readable by the non-root agent UID.
- Container: UID/GID 10001, one CPU, 1 GiB memory limit, read-only root filesystem.
- WebChat and Windows surrogate are deferred. No agent web port is published.

## Locations and controls

- Deployment root: `/srv/theseus/tam`.
- Current release: `/srv/theseus/tam/releases/tam-low-reasoning`.
- Original release retained for recovery: `/srv/theseus/tam/releases/tam`.
- Agent home: `/srv/theseus/tam/data/agents/tam/state`.
- Stimulus log: `/srv/theseus/tam/data/agents/tam/logs/stimulus_log.jsonl`.
- Operator: `/usr/local/bin/tamctl`, using `/opt/theseus-operator` and the runtime
  source shipped in the exact release.

```bash
ssh root@159.203.30.50 tamctl status
ssh root@159.203.30.50 'docker logs --tail 30 tam-tam-1'
ssh root@159.203.30.50 'docker stats --no-stream tam-tam-1'
ssh root@159.203.30.50 systemctl stop tam.service
ssh root@159.203.30.50 systemctl start tam.service
ssh root@159.203.30.50 tamctl backup
```

The systemd unit uses the managed operator for startup and shutdown. It is a
oneshot unit: `active (exited)` alone is not evidence of agent health. Check
`tamctl status` for a running container and lifecycle acknowledgement. A failed
stop requires investigation; do not erase control journals to bypass it.

## Migration evidence

The source home remains at `/home/aldric/tam`. Its four legacy user services
(`tam-discord`, `tam-indexer`, `tam-supervisor`, `tam-web`) were stopped and
disabled before the final copy. Do not restart them while cloud Tam is active.

The copy contains 931 stimulus events, 742 knowledge records, 612 memory records,
and Telegram's SQLite delivery state. All 126 committed legacy episodes were
verified against their source-event hashes. The new formation cursor starts
after their 929 events, leaving two subsequent events eligible for processing.
The old Nomic memory vectors are retained in the original records; Qwen search
vectors are rebuilt in the separate `memory/embeddings.jsonl` index.

`state/migration/import.json` records hashes of the 52 original copied files.
`state/migration/legacy-CADENCE.md` retains the source cadence. The local import
and archives under `build/deployments/tam/` are additional recovery evidence.
They contain private state and are Git-ignored; they are not disposable build
output. A future restore must remain inactive until the other instance is stopped.

R2 backups are **not configured**: endpoint, bucket, and S3 credentials are still
missing. `tamctl backup` creates a local stopped snapshot only. OpenRouter spending
limits are managed by the user on the token.

## Verified activation

- Exact image: `sha256:743b8dd5dbd5945563fe0e40a6e075087f8b2d0906624f56b4950cfebf8a7089`.
- Runtime source commit: `6156b9dd295adec95dda5036d13fd36dd546f156`.
- Local baseline snapshot: `3f12ff70434e44beb0c7c75ec6138920`, captured before
  activation after the Qwen index rebuild. This is stored on the droplet, not R2.
- All 612 imported memory records indexed with 4096-dimensional Qwen vectors;
  original memory-file hashes unchanged. A recall check returned 15 entries;
  SQLite delivery integrity passed.
- Managed lifecycle running, systemd unit enabled on boot, actual model decisions
  and tool executions observed, and more than 100 successful Telegram polls.
- Memory formation advanced from 126 to 128 committed episodes after activation.
- Observed container memory around 222 MiB with a 1 GiB limit; this is one
  observation, not a peak-memory guarantee.

The local Obsidian vault at `/home/aldric/vaults/george` was not part of Tam's
home migration and is not mounted on the droplet. A carried-over task attempted
that path and received a file-not-found error. Vault-dependent tasks need separate
access configuration; do not copy the vault or supply Git credentials implicitly.

## Telegram reply correction — 2026-09-20

Telegram intake worked, but the first user message triggered a roughly five-minute
model turn that produced neither text nor tool calls. OpenRouter's catalog lists
GLM Flash's default reasoning effort as `max`. Added an opt-in chat-only reasoning
setting and selected its supported `low` value for Tam. Embedding requests are
unaffected; other deployments retain their defaults unless configured.

- Replacement image: `sha256:6b31112bf907336c7121d3c9c94188780f2bc1556263aebac36da9b88f4b88ca`.
- Runtime includes the local, uncommitted provider fix on top of the base commit
  recorded above. The release bundle includes that exact source.
- Provider/tool/Autocore tests: 50 passed.
- The old worker exceeded its shutdown deadline. That stop remains recorded as
  unclean; its data and operation evidence were retained under `incoming/unclean-*`.
  A disposable-copy validation passed before restart, with 745 knowledge and
  622 memory records and no pending memory transaction.
- After replacement startup at 20:42:07 UTC, Tam called `respond_in_telegram`
  at 20:42:16 UTC. Telegram confirmed delivery on the first attempt with message
  ID `34`. The original user message did not need to be resent.
