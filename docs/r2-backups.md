# Cloudflare R2 deployment backups

`theseus-backup` stores a stopped, consistent deployment snapshot together with
the generated release bundle and exact Docker image exports. Uploads use
immutable, versioned object keys. The snapshot `manifest.json` is written only
after every referenced object has been uploaded and verified, so listing and
restore ignore interrupted uploads.

Configure the operator environment with an R2 S3 endpoint and bucket. Boto3
reads credentials from its standard credential chain; in a simple host setup,
use `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Do not put credential values
in a deployment definition, generated bundle, agent environment, or command
line. Agents do not receive access to the backup bucket.

```bash
export THESEUS_R2_ENDPOINT=https://ACCOUNT_ID.r2.cloudflarestorage.com
export THESEUS_R2_BUCKET=theseus-backups
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

Create a local clean-stopped snapshot, resume the services that were running,
and then upload it:

```bash
poetry run theseus-backup build/flywheel create \
  --root /srv/theseus/flywheel
```

The JSON result reports the snapshot ID, capture and upload times, total object
size, and last successful snapshot. If the upload fails, the command reports the
local staged path. Retry that same snapshot without another service stop:

```bash
poetry run theseus-backup build/flywheel upload \
  --snapshot /srv/theseus/flywheel/snapshots/SNAPSHOT_ID
```

Uploaded release and image objects that already have the expected size and
SHA-256 metadata are reused. A conflicting immutable key fails rather than
overwriting prior evidence. `remote-status.json` in the local snapshot records
the last attempt, including a failure message and the previous successful
snapshot ID. It never contains credentials.

List only completed backups and download one exact snapshot ID:

```bash
poetry run theseus-backup build/flywheel list
poetry run theseus-backup build/flywheel download \
  --snapshot SNAPSHOT_ID --output /var/tmp/flywheel-backup
```

Restore verifies every downloaded SHA-256, extracts the stored generated bundle,
loads the stored Docker image archive, checks its content ID and platform, and
restores the data into an inactive, empty target. Provision the required secret
files on the target first. Restore leaves activation suspended.

```bash
poetry run theseus-backup build/flywheel restore \
  --snapshot SNAPSHOT_ID --root /srv/theseus/flywheel
```

For offline exercises and tests, replace the R2 configuration with a filesystem
adapter:

```bash
poetry run theseus-backup build/flywheel list --local-store /var/tmp/r2-fake
```

## Recovery and backup status

Combine local operator state with the remote store's own record of what
actually completed. This never trusts a local capture as evidence of a
successful backup; only a manifest the store confirms as complete counts.

```bash
poetry run theseus-backup build/flywheel status --root /srv/theseus/flywheel
```

The result reports current activation, the last *remote-confirmed* backup
(snapshot ID, capture/upload time, age, and size), every local snapshot whose
upload never reached `completed` (failed or never attempted, each with the
`upload` command that resumes it from immutable local staging), any
unfinished deployment operation with an actionable resume command, and the
migration journal's phase when one is in progress. A failed remote upload
never appears as the last completed backup; it only shows up under pending
uploads until a retry actually completes.

## Opt-in scheduled backups

`theseus-backup-schedule` renders and installs a systemd service/timer pair
that invokes the same `create`-then-upload protocol on a cadence, entirely
outside any dashboard or general scheduler. It is opt-in end to end: installing
a timer never enables or starts it, and its generated unit files never embed
credential values, only an optional `EnvironmentFile` path the operator
populates separately (boto3's standard credential chain, as above).

```bash
poetry run theseus-backup-schedule build/flywheel install \
  --root /srv/theseus/flywheel \
  --unit-dir /etc/systemd/system \
  --on-calendar 03:00 \
  --env-file /etc/theseus/flywheel-backup.env \
  --endpoint https://ACCOUNT_ID.r2.cloudflarestorage.com \
  --bucket theseus-backups
```

The install step only writes `theseus-backup-<id>.service` and
`theseus-backup-<id>.timer` and runs `systemctl daemon-reload`; add `--enable`
to also `enable --now` the timer once you are ready for it to run
unattended. `--scope user` targets a user manager instance instead of the
system one.

```bash
poetry run theseus-backup-schedule build/flywheel timer-status \
  --unit-dir /etc/systemd/system
poetry run theseus-backup-schedule build/flywheel uninstall \
  --unit-dir /etc/systemd/system
```

Each scheduled run reuses the deployment's existing operation lock and the
normal backup protocol unchanged; it is only responsible for deciding whether
a run should be attempted at all:

- If a migration journal exists for the deployment and has not reached its
  `completed` phase, the run is skipped. This is what keeps a scheduled
  backup from racing a migration's stop/capture/restore handoff.
- If activation is not `active` (absent, suspended, or retired), the run is
  skipped rather than resuming a retired source or activating a suspended
  deployment. Only eligible, currently active deployments are captured.
- If the operation lock is already held, or an earlier operation was left
  interrupted, the run is skipped so overlapping jobs never race each other;
  inspect and resolve the interrupted operation with `theseus-backup ...
  status` or `theseus-deployment ... recovery`, then let the next tick retry.

A skip is a normal, expected timer outcome and exits successfully. An actual
capture or upload failure still raises and fails the systemd service, so a
real problem is never silently swallowed as a skip:

```bash
poetry run theseus-backup-schedule build/flywheel run \
  --root /srv/theseus/flywheel \
  --endpoint https://ACCOUNT_ID.r2.cloudflarestorage.com \
  --bucket theseus-backups
```

This is the exact command the generated service unit runs; invoke it directly
to test the eligibility policy and the backup protocol together before
enabling the timer.

## Retention and staging cleanup

Automatic pruning is deferred; nothing in this repository deletes an old
snapshot, local staging directory, or remote object on your behalf. Decide
your own retention policy and apply it explicitly with ordinary tools, with
two invariants that must never be violated:

- Never delete the last verified (remote-`completed`) backup for a
  deployment. Check `theseus-backup ... status` first; it reports exactly
  which snapshot ID is the last one the remote store confirms as complete.
- Never delete a local snapshot or its artifacts while they are pinned by an
  unfinished migration. `migration.json` in the deployment's `control/`
  directory names the pinned `snapshot_id`; leave that snapshot directory and
  its uploaded objects alone until the migration reaches `completed` or is
  abandoned with recorded fencing evidence.

Local staging (`<root>/snapshots/<id>/`) and its `.artifacts` release/image
cache are ordinary directories once you have confirmed via `status` that a
snapshot uploaded successfully and is not the last verified backup; remove
them with normal filesystem tools. Remote objects are versioned and
content-addressed, so removing one only affects backups that still reference
it — cross-check every manifest that shares an image or release object before
deleting anything remotely.

## Optional isolated R2 check

Use a temporary bucket that contains no production backups and credentials
scoped only to that bucket. Assemble and build a disposable deployment, create a
backup, list it, and download its exact snapshot ID into a new directory. Compare
the downloaded manifest hashes, then restore into a newly provisioned inactive
root. Delete the temporary bucket only after the restored data and Docker content
ID have been checked. This live check is optional; the automated suite uses the
filesystem adapter and a fake Docker runner, so it needs no cloud credentials.
