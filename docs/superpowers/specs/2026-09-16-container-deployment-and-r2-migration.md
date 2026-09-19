# Container deployment, R2 backups, and migration

Status: design proposal, not implemented.

Implementation roadmap: [GitHub #75](https://github.com/GeorgeLautenschlager/theseus/issues/75),
with implementation issues #76–#84 and their acceptance criteria/dependencies.

## Outcome and proposed first-version boundary

The assembler produces a portable Docker Compose deployment. The same deployment
can be backed up to Cloudflare R2 and moved to a newly provisioned DigitalOcean
Droplet with one operator command. A deployment contains one or more agents, their pairing graph,
and their declared shared workspaces. Project Flywheel's pair moves together.

Confirmed requirements:

- Brief downtime is acceptable during scheduled backups and migration. Use a
  coordinated stop and consistent local snapshot; ordinary backups resume agents
  before uploading to R2. Migration keeps the source stopped through handoff.
- Push-button migration includes creating and preparing a new DigitalOcean Droplet.
  The source stays live throughout provisioning and destination preparation.

Remaining proposed defaults, pending discussion:

- Linux source host with SSH, Docker Engine, and Compose. New destinations receive
  Docker, Compose, and the version-matched operator through automated bootstrap.
- Same CPU architecture at source and destination for v1.
- Full filesystem snapshots; incremental backups can follow after restore works.
- Operator-initiated migration; a missing source host does not trigger automatic
  failover. DNS migration and a web control panel are later.

Existing Python-only assembly remains available and backwards compatible.

## What the current code requires

`assembly.py` currently writes one generated `agent.py`, supports `--home`, and
reapplies constitution, persona, and cadence at boot. Its other files are runtime
state. `AssembledAgent.run()` starts Auto on a daemon thread, and Auto's loop has
no coordinated stop operation. Telegram has a stop event, but no shared lifecycle
joins the observer, core, tools, and memory work. DeliveryJournal uses SQLite WAL.

Consequently, copying a live home is not a consistent backup: different files can
represent different moments, including a half-written memory transaction or
in-flight delivery. Coordinated shutdown is part of this feature, not an optional
cleanup item. Memory's durable pending transaction remains the recovery mechanism
for an interrupted write; do not discard its preparation or cursor files.

## 1. Deployment definition and assembler output

Keep `AgentSpec` about agent behavior. Add a separate trusted Python
`DeploymentSpec` that wraps existing specs with stable deployment and agent IDs,
a pairing map by agent ID, shared workspaces, resource limits, image/build inputs,
and secret names. Host addresses and actual secret values belong to operator
configuration outside the bundle.

Illustrative API (names are proposed):

```python
DEPLOYMENT = DeploymentSpec(
    id="flywheel-trial",
    agents={"fable": FABLE_SPEC, "astra": ASTRA_SPEC},
    peers={"fable": "astra", "astra": "fable"},
    workspaces={"website": ("fable", "astra")},
    platform="linux/amd64",
)
```

The deployment pairing map resolves the log path for each agent. Validate that
peer names/identities agree with the agent definition; reject conflicting pairing
settings rather than silently choosing one. Container path mappings are generated,
not copied from source-machine relative paths. IDs survive renames and migration.

Add `--target compose` to assembly, accepting a `DEPLOYMENT` definition; an
individual `SPEC` can be wrapped with an explicit deployment ID. Preserve today's
`SPEC` -> `agent.py` behavior by default. Assembly performs no network calls,
starts no agent, and never writes runtime state or secrets.

Generated bundle:

```text
build/flywheel/
  compose.yaml
  Dockerfile
  deployment.json           # resolved behavior, IDs, mounts, secret names
  agents/fable/agent.py
  agents/astra/agent.py
  requirements.lock         # resolved runtime dependencies / build provenance
  .dockerignore
  secrets.example           # names/instructions only, no values
  README.md
```

Use the existing generated-file ownership checks for every generated file. Render
and validate a complete bundle in staging before publishing it. Reassembly may
replace generated configuration, never deployment data. Unresolved dependencies,
missing custom code, and unsupported container interfaces fail validation.

A separate build command produces `deployment.lock.json`, recording the exact
resolved spec hash, Theseus commit/version, image content IDs, architecture,
base-image identity, and build inputs. Migration restores those exact images; it
does not reinstall a floating Git branch or resolve dependencies again. Include
custom Python components in the build context explicitly; do not copy the whole
user checkout, `.env`, state, or vault into the image.

For the first release, support Auto with Telegram/headless interfaces through the
full managed lifecycle. Keep unsupported OODA/terminal/web managed deployments
explicitly rejected until their shutdown/readiness behavior is tested. Ordinary
non-container assembly keeps all existing interfaces.

## 2. Persistent storage and pairing

One operator-owned host root per deployment:

```text
/srv/theseus/flywheel-trial/
  releases/<release-id>/     # generated bundle and lock
  data/
    agents/fable/state/
    agents/fable/logs/stimulus_log.jsonl
    agents/astra/state/
    agents/astra/logs/stimulus_log.jsonl
    workspaces/website/
  control/                  # lifecycle state and migration journal, host-owned
  secrets/                  # runtime secret files, host-owned
  snapshots/                # immutable local staging, outside data
```

Each container sees stable paths, regardless of host:

| Mount | Permission | Purpose |
|---|---|---|
| `/data/state` | read/write | Own memory, goals, schedule, delivery journal, private artifacts |
| `/data/logs` | read/write | Own stimulus log |
| `/peers/<peer-id>/logs` | read-only | Only the peer's log directory |
| `/workspaces/website` | read/write | Explicit shared project files |
| `/run/secrets/<name>` | read-only | Required runtime credentials |
| `/run/theseus-control` | read-only | Operator activation/maintenance state |

Mount peer directories, not individual files: a log can be created after the
partner starts. Pre-create directories and ownership on deployment. Agents never
receive the deployment root, peer private state, Docker socket, or R2 credentials.
Stable numeric UID/GID and a shared workspace group keep ownership portable.

Add optional `log_path` to `build_agent` and `--log-path` to the launcher, defaulting
to today's `<home>/stimulus_log.jsonl`. Inject this log into both supported cores
and memory as today. Auto must not create a misleading second home log when an
external path was supplied. Existing homes require an explicit stopped import
that relocates the log; do not silently split an existing log across paths.

Keep tool defaults relative to `/data/state` for compatibility; tell the agent
where its shared workspace is. Runtime packages/tools belong in the image.
Durable output must live under declared data mounts; `/tmp` and image filesystem
changes are ephemeral. Mounts outside the backup set must be declared ephemeral
or rejected for managed migration. Shared workspaces have no automatic file-edit
conflict resolution; the agents still need to coordinate ownership.

## 3. Runtime contract

Run agents as non-root, with an init process and explicit CPU/memory limits.
Keep writable state and workspace mounts explicit. A coding image can include
Node/browser tools when required; the base image alone does not imply website
browser testing is available. No published ports for Telegram-only deployments.

Compose secrets provide mounted files; the entrypoint can read named files into
the environment expected by existing providers. Do not serialize values in
AgentSpec, generated files, image layers, command arguments, or backup manifests.
R2 and SSH credentials belong to the operator, not the agents. Runtime secrets
must be provisioned separately on the target before activation; a state backup
is not a credential-recovery system.

Add a shared lifecycle to `AssembledAgent`:

- Startup validates mounts, ownership, config, and local activation permission.
- A maintenance/preflight mode opens and validates state without polling Telegram,
  starting cognition, sending queued messages, or making model requests.
- `SIGTERM` requests shutdown. Signal handlers set events only.
- Stop accepting new external work; finish/persist an already-returned incoming
  batch, finish the current core step and memory operation, and start no next turn.
- Stop outbox dispatch, join owned worker threads, close resources, and exit cleanly.
- Account for agent-launched child processes: they must not keep writing shared
  files after the runtime reports stopped. Unsupported unmanaged writers make a
  clean snapshot ineligible.
- A clean exit plus stopped container/process tree is the backup boundary.

Use a finite operator shutdown deadline. On timeout, fail the planned backup or
migration; never quietly label a forced-kill snapshot as clean. A separate crash
recovery operation can use existing recovery semantics, but does not claim all
external effects are known. Docker's eventual SIGKILL is not a clean drain.

Readiness means local recovery and initialization succeeded; it must not spend
inference tokens. Sleeping on cadence is healthy. Do not trigger automatic
restarts simply because no model call happened recently.

## 4. Backup protocol

A deployment-wide operation lock serializes backup, restore, migration, and
managed start/stop. Snapshot all agents in a pair and every shared writer together.
Include private state, own logs, all memory files (including pending preparation,
ledgers, sidecars, and cursors), goals/tasks/schedule, delivery SQLite files and WAL
sidecars, and shared workspaces. Never back up only selected memory JSONL files.

Scheduled or manual backup:

1. Record which services were running and enter durable maintenance mode, preventing
   automatic restart during the snapshot even if the host/controller reboots.
2. Drain and stop all deployment writers; verify clean shutdown.
3. Copy the complete `data/` tree into a new local staging directory. Preserve
   file modes and internal links; never follow symlinks outside the data roots.
   Finalize/fsync this local snapshot before permitting live writes again.
4. Resume only the previously running services. Compression and upload now operate
   on the immutable staging copy; downtime does not include the network upload.
5. Validate staged JSONL and SQLite state using an isolated scratch copy where
   recovery/checkpointing may write. Reject corruption; preserve original evidence.
6. Compress, hash, and upload immutable artifacts to R2; publish the snapshot's
   completion manifest only after every required object is uploaded and verified.
7. Mark success only after remote completion. Record snapshot ID, size, archive
   SHA-256, capture time, last successful upload, and any error.

Upload failures leave the previous completed backup valid and a retryable local
snapshot. Never include snapshots inside snapshots. A controller crash leaves a
journaled maintenance state; recovery can resume the previous deployment after
checking that it is not a migration source already handed off. No blind restart
from a generic `finally` block.

Initial scheduling: manual backups plus an optional host systemd timer, disabled
until configured. Retention is explicit; never delete the last verified backup or
a snapshot pinned by an unfinished migration. Automatic pruning can follow v1.

## 5. R2 object format and release portability

Use R2's S3-compatible API, endpoint from operator configuration, region `auto`,
and credentials scoped to the backup bucket. Use a standard SDK behind a small
storage interface so fake/local storage can exercise protocol failures in tests.
R2 is backup storage, not the agents' live filesystem or a runtime database.

```text
v1/deployments/<deployment-id>/
  releases/<release-id>/bundle.tar.gz
  images/<image-artifact-sha256>.tar.gz
  snapshots/<snapshot-id>/data.tar.gz
  snapshots/<snapshot-id>/manifest.json
```

For an implementation without a registry prerequisite, export the built Docker
images once per release, compress/hash them, and store them in R2. Target hosts
verify and `docker load` those images. Record both the archive hash and exact
Docker image content ID/platform. A future registry backend can replace image
transport without changing the snapshot format. Never identify an image by a
mutable tag alone.

The manifest includes format version, deployment/agent IDs, release/spec hash,
image references, schema/runtime version, capture time, consistency level, data
roots, file inventory/checksums, and required secret names (not values). Archive
SHA-256 is separate from an object ETag. Unique snapshot keys provide history;
do not depend on S3 bucket versioning. Listings consider only completed manifests.
A `latest` convenience pointer may be added later; migration always pins an exact
snapshot ID, not a pointer that can change underneath it.

Write the completion manifest last. R2 provides strong object consistency, but
there is no multi-object transaction: the manifest is our completion boundary.
Validate uploaded object size and stored checksum metadata; verify downloaded
bytes against manifest hashes on restore. Periodic restore drills establish that
backup content is actually usable. Partial/orphan uploads are never restorable
snapshots.

Private R2 access and runtime-secret separation are the initial controls. State
itself can contain sensitive user content or text an agent wrote; excluding
configured secret files does not prove the archive contains no secrets.
Client-side encryption can be added with explicit key management if required.

## 6. Restore and one-command migration

A restore stages a snapshot with execution disabled. Verify release identity,
platform, schema, checksums, free space, and required secrets; safely extract into
an empty directory (reject traversal, device files, and escaping links). Run local
state checks without live integrations. Never overwrite an active home or boot
restored agents as a side effect of downloading a backup. Install the exact image
and spec, because current startup reapplies assembler-owned identity files.

Operator CLI example (proposed, not available today):

```sh
theseus deploy build build/flywheel
theseus deploy up build/flywheel --host current-droplet
theseus backup create flywheel-trial --host current-droplet
theseus backup list flywheel-trial
theseus deploy migrate flywheel-trial --from current-droplet --provision digitalocean --profile flywheel
```

Migration orchestrates this sequence:

1. Journal a migration/provisioning ID, create a destination Droplet from the
   operator's DigitalOcean profile, and wait for completed bootstrap and SSH
   readiness. Preflight the destination, transfer/verify release images, and
   provision named secrets through the operator's secure channel. Destination
   stays inactive and the source continues running throughout this step.
2. Acquire the source deployment operation lock and write a migration journal.
3. Mark source activation suspended persistently, drain/stop all source writers,
   disable their restart, and confirm the entire deployment is stopped.
4. Create/upload a final snapshot using the backup protocol **without resuming the
   source**. Persist its exact ID and hash in the migration journal.
5. Restore and preflight that exact snapshot on the target with execution disabled.
6. Retire source activation, persist the handoff record, then authorize and start
   the target. Record activation intent durably before the first target side effect.
7. Verify local readiness, report success, and retain source files for recovery.

Both entrypoints and managed start commands must honor a host-local activation
guard. It lives outside the backed-up data and is read-only inside agents. A
retired source must remain retired after a reboot or `compose up`. This is an
operator-controlled single-active-host protocol, not a distributed lease service.
Root can bypass it; the product must not promise protection from deliberate
operator bypass or unmanaged duplicate deployments.

Failure handling:

| Failure point | Behavior |
|---|---|
| Before source stops | Leave source running; report preflight failure |
| Source refuses to drain | Abort handoff; target remains inactive |
| Snapshot/upload/restore fails | Source remains suspended; resumable operation can restore source activation only after proving target was never activated |
| Target startup has begun | Never automatically restart the old source; target may already have made external changes |
| Controller loses connection | Read durable journals on both hosts before retrying; do not infer completion from a timeout |
| Source unreachable | Refuse automatic promotion; require external fencing such as verified power-off before disaster recovery |

A rollback after target activity is another stopped migration of its latest state,
not an automatic restart of the old snapshot. Preserving Telegram inbox/outbox
state reduces replay errors, but cannot guarantee exactly-once external effects
when a remote send succeeded before its local acknowledgement was recorded.

### DigitalOcean provisioning contract

Provisioning is part of v1. Add an operator-owned host profile specifying region,
size slug, OS image, architecture, SSH public-key IDs, and firewall/network settings.
Validate availability and compatibility before creation; do not silently choose a
larger machine. Preview the selected resources and current estimated infrastructure
cost separately from the inference budget. An existing-host target can remain an
optional path through the same migration flow.

Use the DigitalOcean API behind a small `HostProvisioner` interface. The first
implementation only needs DigitalOcean; do not build a multi-cloud framework.
`user_data`/cloud-init installs Docker, Compose, the pinned Theseus operator, the
runtime user, and deployment directories. No agent is activated by cloud-init.
API state `active` alone is not readiness: wait for SSH, successful cloud-init,
Docker/Compose checks, and available disk space before stopping the source.
Bootstrap timeout or failure leaves the source untouched.

Keep DigitalOcean credentials on the initiating operator machine. Never pass
OpenRouter, Telegram, or R2 secrets in user-data, which is bootstrap configuration,
not secret storage. Transfer runtime secrets after authenticated SSH is available;
keep the expected SSH host identity pinned for subsequent retries. Install R2
credentials only for the operator where needed, never into agent containers.

Record the intended request and unique operation tag before creating the Droplet;
record the returned Droplet ID immediately afterward. Retries reconcile by that ID
or tag rather than blindly creating another server. On an ambiguous API timeout,
query for the tagged resource with bounded retries; if its identity remains
uncertain or multiple matches exist, stop and report the ambiguity instead of
issuing another create. Persist provisioning phases in the migration journal.

Retain the old deployment's files and leave it inactive after handoff. Never
implicitly destroy the source Droplet: it may host other services. Report both
resource IDs and any remaining billable resources. Explicit cleanup may remove an
unused destination created by this operation, but never an activated target or
resources merely sharing a name. Retiring a deployment is not deleting a host.

## 7. Modules and implementation order

Keep this outside the memory module. Suggested small components:

- `deployment.py`: specs, validation, generated Compose bundle and lock format.
- `runtime_lifecycle.py`: activation guard, signal handling, drain/readiness.
- `deployment_store.py`: host layout, locks, operation journals.
- `backup.py`: consistent staging, manifests, hashes, restore validation.
- `backup_store.py`: R2/local storage adapters.
- `deploy.py`: operator CLI and SSH orchestration.
- `host_provisioner.py`: DigitalOcean creation, bootstrap readiness, resource tracking, and explicit cleanup.

The host-side operator controls Docker and R2; no Docker-in-Docker or privileged
agent backup sidecar. SSH transport runs the version-matched operator on prepared
hosts; automated destination bootstrap installs it before preflight. A future UI calls the same operations.

Implement in independently testable slices:

1. Container bundle + stable mounts + runtime secret loading + clean lifecycle.
2. Stopped local snapshot/restore including pair and workspace; prove fidelity.
3. R2 upload/download + completion manifests + exact image transport.
4. DigitalOcean provisioning and bootstrap with resumable resource tracking.
5. Resumable one-command migration with activation guards and fault injection.
6. Optional backup timer and operator status reporting.

The first two slices are useful on their own tonight. Do not advertise push-button
migration until the handoff and recovery tests pass. This feature does not enforce
the trial's shared inference budget; that remains a separate launch prerequisite.
Any future spending ledger/gateway state must be a declared backed-up component,
with external provider usage reconciled so restoring an old snapshot cannot reset
the spending allowance.

## 8. Acceptance tests

- Existing Python assembly stays compatible; generated output contains no secret
  values; repeated assembly never modifies live state or user-owned files.
- Real Compose smoke test with fake providers: own logs writable, peer logs readable
  but unwritable, peer private state absent, shared artifacts persist after recreate.
- Graceful shutdown mid-turn/consolidation and during Telegram polling exits cleanly
  without starting another turn; deadline expiry cannot publish a clean snapshot.
- Snapshot a pair with memory preparation, formation cursor, delivery SQLite/WAL,
  goals, and shared files; restore on a different root and verify continued identity,
  recall, pending recovery, and recorded delivery state using fake transports.
- Broken archive/hash/version, missing image/secret, unsafe link, or nonempty target
  fails before activation and leaves original state intact.
- Interrupted upload never appears as a completed backup; retry is idempotent.
- Fake DigitalOcean API: create timeout after successful creation, repeated command,
  ambiguous/multiple tagged resources, and bootstrap failure cannot create duplicate
  Droplets or interrupt the source. Persist and resume using the resource ID.
- Bootstrap cannot start agents or contain runtime/API secrets. An `active` Droplet
  with failed cloud-init is not eligible for handoff. Cleanup targets only tracked,
  unused resources created by the operation and never the source host.
- Controller interruption at every handoff step; destination stays inactive until
  source is stopped, retired source cannot restart, and post-activation failure does
  not roll back automatically. Include host reboot and SSH disconnect cases.
- Live infrastructure smoke test only with fake cognitive/Telegram backends first;
  then a small explicitly launched real-agent trial.

## Sources checked during design

- [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/): directory mounts and read-only peer access.
- [Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/): named runtime secret files.
- [Compose stop](https://docs.docker.com/reference/cli/docker/compose/stop/): shutdown timeouts; stopping is not by itself proof of application drain.
- [R2 S3 compatibility](https://developers.cloudflare.com/r2/api/s3/api/): endpoint, region, supported operations, lack of S3 bucket versioning.
- [R2 consistency](https://developers.cloudflare.com/r2/reference/consistency/): object visibility guarantees.
- [R2 authentication](https://developers.cloudflare.com/r2/api/tokens/): separate access-key credentials and bucket-scoped access.

- [DigitalOcean Droplet creation API](https://docs.digitalocean.com/reference/pydo/reference/droplets/create/): region, size, image, SSH keys, tags, and user data.
- [DigitalOcean user data](https://docs.digitalocean.com/products/droplets/how-to/provide-user-data/): cloud-init provisioning at Droplet creation.
