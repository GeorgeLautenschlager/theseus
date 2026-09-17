# Paired deployment acceptance and operations

Push-button migration is release-gated by a real Docker acceptance test. It
assembles and builds a two-agent deployment, starts both managed containers with
the deterministic `fixture` provider, verifies the generated mounts, recreates a
container, captures and transports an exact stopped backup, restores it at a
second root, and performs a single-active-host handoff.

The fixture provider never opens a network connection. It remains unavailable
unless `THESEUS_ENABLE_FIXTURE_PROVIDER=1` is present in the process environment.
The harness supplies that flag as a Compose secret and records calls on each
agent's private state mount. The agents poll a local fake Telegram API container,
whose request journal proves that restore and preflight make neither inference
nor Telegram calls. Delivery recovery is also checked in the restored SQLite
state.

Run the gate on Linux with Docker Engine and Compose available:

```bash
make container-acceptance
```

Ordinary `pytest` skips this test. `make release VERSION=X.Y.Z` runs both the
offline suite and this container gate before changing the version. The test uses
unique Compose project names and removes its containers and image tag after
success or failure.

## First deployment

Assemble and build the exact release before creating activation permission:

```bash
poetry run python -m theseus.assemble deployment.py \
  --target compose --output build/flywheel
poetry run theseus-build build/flywheel
```

Use `DeploymentPaths.prepare()` and `apply_ownership()` from a privileged host
setup step to create `/srv/theseus/flywheel`. Write each file named by
`required_secrets` in `build/flywheel/deployment.json` under the deployment's
`secrets/` directory with mode `0600`. Then start through the managed operator:

```bash
poetry run theseus-deployment build/flywheel status \
  --root /srv/theseus/flywheel
poetry run theseus-deployment build/flywheel start \
  --root /srv/theseus/flywheel
```

The first status may show no activation record and no running services. `start`
runs each container's local preflight before writing active permission. Do not use
`docker compose up` as the activation procedure.

## Importing a stopped Python home

Stop the old process and keep its directory as rollback evidence. Import one home
at a time before the first managed start:

```python
from theseus.deployment_store import DeploymentPaths, import_stopped_home

paths = DeploymentPaths(ROOT, DEPLOYMENT)
paths.prepare()
import_stopped_home(OLD_FABLE_HOME, paths, "fable")
paths.apply_ownership()
```

`import_stopped_home` holds the old home's runtime lock, copies private state,
moves `stimulus_log.jsonl` into the separate logs mount, rejects symlinks and an
occupied destination, and leaves the source intact. Repeat for the other agent,
then provision secrets and use the managed `start` command above.

## Backup, list, and stopped restore

Create and upload a clean snapshot. The ordinary form resumes exactly the agents
that were running; add `--leave-stopped` for maintenance or an export:

```bash
poetry run theseus-backup build/flywheel create \
  --root /srv/theseus/flywheel
poetry run theseus-backup build/flywheel list
poetry run theseus-backup build/flywheel download \
  --snapshot SNAPSHOT_ID --output /var/tmp/flywheel-SNAPSHOT_ID
```

Prepare an empty target root with its required secret files, then restore by exact
snapshot ID:

```bash
poetry run theseus-backup build/flywheel restore \
  --snapshot SNAPSHOT_ID --root /srv/theseus/flywheel-restored
poetry run theseus-deployment \
  /var/tmp/flywheel-SNAPSHOT_ID/release/bundle status \
  --root /srv/theseus/flywheel-restored
```

Restore downloads and verifies every object, loads the exact image archive,
checks content ID and platform, and leaves activation suspended. Inspect before a
managed start. Never restore over active or nonempty data.

## Migration resume and incomplete operations

Use the same bundle, state paths, workspace, profile, and source root every time
the controller is restarted:

```bash
poetry run theseus-migrate build/flywheel migrate \
  --from /srv/theseus/flywheel \
  --profile host-profile.json \
  --provision-state ~/.local/state/theseus/flywheel-provision.json \
  --state ~/.local/state/theseus/flywheel-migration.json \
  --workspace ~/.local/state/theseus/flywheel-migration-work \
  --secrets-dir ~/.config/theseus/flywheel-secrets \
  --source-droplet-id SOURCE_DROPLET_ID
```

A repeated `migrate` reconciles durable state and continues. Inspect all three
journals before intervening:

```bash
poetry run theseus-migrate build/flywheel status \
  --from /srv/theseus/flywheel \
  --profile host-profile.json \
  --provision-state ~/.local/state/theseus/flywheel-provision.json \
  --state ~/.local/state/theseus/flywheel-migration.json \
  --workspace ~/.local/state/theseus/flywheel-migration-work
poetry run theseus-host build/flywheel host-profile.json status \
  --state ~/.local/state/theseus/flywheel-provision.json \
  --source-droplet-id SOURCE_DROPLET_ID
poetry run theseus-deployment build/flywheel recovery \
  --root /srv/theseus/flywheel
```

Do not edit journals or promote after a timeout. Source recovery and external
fencing use the guarded commands documented in [migration](migration.md).

## Executable failure evidence

The Docker gate supplies cross-feature evidence. Focused tests retain faster,
precise fault injection:

| Boundary | Executable coverage |
| --- | --- |
| Real mounts, peer isolation, recreation, clean drain, exact image and restored state | `tests/test_container_acceptance.py` |
| Corrupt archive/hash, platform or release mismatch, missing secrets, unsafe extraction, nonempty targets | `tests/test_deployment_snapshot.py`, `tests/test_remote_backup.py` |
| Shutdown deadline and unclean lifecycle evidence | `tests/test_managed_lifecycle.py` |
| Interrupted provisioning and duplicate/ambiguous Droplet reconciliation | `tests/test_host_provisioner.py` |
| Interrupted upload visibility and retry | `tests/test_remote_backup.py` |
| Controller loss at every handoff boundary, reboot, startup failure, unreachable or unknown hosts | `tests/test_migration.py` |

## Optional DigitalOcean and R2 smoke test

Live infrastructure is never used by ordinary pytest. Create a disposable
deployment that uses `ModelSpec("fixture", "quiet")` and Telegram interfaces.
Host a disposable Telegram Bot API fixture reachable from both Droplets and set
the declared `TELEGRAM_API_BASE_URL` secret to its root URL; it must implement
`getUpdates` and send methods without forwarding to Telegram. The repository's
fixture implements that contract and records only method names and token hashes:

```bash
poetry run python scripts/fake_telegram_api.py \
  --host 0.0.0.0 --port 8080 --log /var/tmp/fake-telegram/requests.jsonl
```

Expose that disposable endpoint only to the two test Droplets. Also declare the
two fixture-provider controls used by the Docker harness. Use a dedicated R2
bucket and a new DigitalOcean operation state. Record the source Droplet ID,
destination Droplet ID, operation tag, R2 bucket, snapshot ID, fake Telegram
request journal, and both local state paths before handoff.

Run `theseus-host ... preview`, then the `theseus-migrate ... migrate` command
above. Confirm the result reports `completed`, the exact snapshot, every ready
agent, and both billable Droplets. SSH to the retained source and verify its
activation is retired and its services cannot remain running. On the target,
verify each fixture-provider log contains calls only after target activation.

For cleanup, first save the migration result and backup manifest. A destination
that never activated can be removed with `theseus-host ... cleanup`. The command
intentionally refuses an activated target; delete that disposable Droplet through
the DigitalOcean control plane only after stopping it and recording its ID. Treat
the retained source Droplet independently because it may host other services.
Delete temporary R2 objects or the dedicated bucket, local migration work, and
local operation state only after confirming no rollback or audit evidence is
needed. Check the DigitalOcean account and R2 dashboard for remaining billable
resources.

## Limits

- The consistent final capture causes brief downtime.
- Image restore and migration require the same `linux/amd64` or `linux/arm64`
  architecture recorded by the release.
- Activation is a host-local file guard. External fencing is required when the
  source host cannot prove retirement.
- Telegram or another external service may accept a request before its local
  acknowledgement is durable, so exactly-once external effects are not promised.
- Each deployment enforces its own inference behavior. A shared trial-wide
  inference cap still requires an independent gateway or spending ledger.
