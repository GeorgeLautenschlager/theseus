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

## Optional isolated R2 check

Use a temporary bucket that contains no production backups and credentials
scoped only to that bucket. Assemble and build a disposable deployment, create a
backup, list it, and download its exact snapshot ID into a new directory. Compare
the downloaded manifest hashes, then restore into a newly provisioned inactive
root. Delete the temporary bucket only after the restored data and Docker content
ID have been checked. This live check is optional; the automated suite uses the
filesystem adapter and a fake Docker runner, so it needs no cloud credentials.
