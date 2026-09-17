# DigitalOcean migration destination provisioning

`theseus-host` prepares a new DigitalOcean Droplet while the source deployment
continues to run. Provisioning does not stop, retire, restore, or activate an
agent. Migration handoff is a separate operation.

The operator supplies a local host profile. The profile selects an exact region,
size, Ubuntu OS image, architecture, SSH keys, firewall IDs, and VPC settings; the tool
does not substitute a larger size. For example:

```json
{
  "name": "trial-toronto",
  "region": "tor1",
  "size_slug": "s-2vcpu-4gb",
  "image": "ubuntu-24-04-x64",
  "architecture": "amd64",
  "ssh_key_ids": [12345678],
  "firewall_ids": ["00000000-0000-0000-0000-000000000000"],
  "vpc_uuid": "11111111-1111-1111-1111-111111111111",
  "ipv6": false,
  "monitoring": true,
  "ssh_user": "root",
  "ssh_port": 22
}
```

Keep this file on the operator machine and owned by the operator account. The
DigitalOcean token also remains on that machine:

```bash
export DIGITALOCEAN_TOKEN=...
```

The token is read from the environment and is never written to provisioning
state, cloud-init, the generated deployment, or the destination agent runtime.
Use a token with the narrow DigitalOcean scopes needed to inspect regions,
sizes, images, SSH keys, VPCs and firewalls and to create/read/delete Droplets,
create tags, and assign firewalls.

Preview validates that the exact region, size, image architecture, SSH keys,
firewalls, VPC, and minimum disk size are available. It reports CPU, memory,
disk, transfer, and current hourly and monthly DigitalOcean prices:

```bash
poetry run theseus-host build/flywheel host-profile.json preview \
  --state ~/.local/state/theseus/flywheel-provision.json \
  --minimum-disk-gb 20
```

Provision with a durable state path. If the deployment declares runtime
secrets, place files with those exact names in an operator-only directory and
pass that directory:

```bash
poetry run theseus-host build/flywheel host-profile.json provision \
  --state ~/.local/state/theseus/flywheel-provision.json \
  --secrets-dir ~/.config/theseus/flywheel-secrets \
  --minimum-disk-gb 20 \
  --source-droplet-id 123456789
```

Before creating anything, the tool persists the full intended-request identity
and a unique operation tag. It stores the returned Droplet ID immediately. If a
create response is lost, it looks up that tag instead of issuing another create.
An unresolved outcome or multiple matches stops with durable status and requires
operator resolution.

Cloud-init installs Docker, Compose, and the Theseus operator pinned to the full
Git commit recorded in the assembled bundle. It creates the deployment directory
layout with stable runtime ownership but does not create activation permission.
Cloud-init contains no DigitalOcean, OpenRouter, Telegram, or R2 credential
values.

An API status of `active` is only the start of readiness. The tool waits for a
public address and authenticated SSH, retains the SSH host fingerprint for strict
retries, waits for successful cloud-init, verifies Docker and Compose, checks the
machine architecture and free disk space, and confirms that the destination is
inactive. Runtime secret files are copied only after those authenticated checks.

Inspect durable local status without API credentials:

```bash
poetry run theseus-host build/flywheel host-profile.json status \
  --state ~/.local/state/theseus/flywheel-provision.json \
  --minimum-disk-gb 20 \
  --source-droplet-id 123456789
```

The result includes the operation tag, Droplet ID, address, retained SSH host
fingerprint, selected resources and prices, failure details, and remaining
billable Droplets. Changing the bundle, profile, disk requirement, source ID, or
operator pin causes the command to reject the old state rather than resume a
different request.

Explicit cleanup deletes only the uniquely tagged, tracked destination created
by that operation:

```bash
poetry run theseus-host build/flywheel host-profile.json cleanup \
  --state ~/.local/state/theseus/flywheel-provision.json \
  --minimum-disk-gb 20 \
  --source-droplet-id 123456789
```

Cleanup refuses an activated destination, the source Droplet, an untracked
resource, or an ambiguous tag. Marking a target activated is an irreversible
cleanup guard for this operation state:

```bash
poetry run theseus-host build/flywheel host-profile.json mark-activated \
  --state ~/.local/state/theseus/flywheel-provision.json \
  --minimum-disk-gb 20 \
  --source-droplet-id 123456789
```

Retiring agents never deletes the source Droplet. It may contain unrelated
services and remains a separately managed billable resource.

DigitalOcean documents the [Droplet create API](https://docs.digitalocean.com/reference/api/reference/droplets/),
[size and price fields](https://docs.digitalocean.com/reference/api/reference/sizes/),
[cloud-init user-data](https://docs.digitalocean.com/products/droplets/how-to/provide-user-data/),
and [firewall assignment](https://docs.digitalocean.com/reference/api/reference/firewalls/).
