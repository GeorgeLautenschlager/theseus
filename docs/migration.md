# Resumable single-active-host migration

`theseus-migrate` provisions a DigitalOcean destination, preloads the exact
release, captures and uploads a final stopped snapshot, restores it inactive,
and transfers activation permission. The source deployment must begin active
with every declared agent running. The command never deletes the source Droplet
or its files.

Run the command on the source host so `--from` names its managed deployment root
and local Docker can export the built images. Configure DigitalOcean and R2 as
described in [DigitalOcean provisioning](digitalocean-provisioning.md) and
[R2 backups](r2-backups.md), then provide operator-only state and workspace
paths:

```bash
poetry run theseus-migrate build/flywheel migrate \
  --from /srv/theseus/flywheel \
  --provision digitalocean \
  --profile host-profile.json \
  --provision-state ~/.local/state/theseus/flywheel-provision.json \
  --state ~/.local/state/theseus/flywheel-migration.json \
  --workspace ~/.local/state/theseus/flywheel-migration-work \
  --secrets-dir ~/.config/theseus/flywheel-secrets \
  --source-droplet-id 123456789
```

Before downtime, the destination is provisioned and bootstrapped, named runtime
secrets are transferred after authenticated SSH, and the generated bundle and
Docker images are uploaded and verified by release ID, archive SHA-256, image
content ID, and platform. The target has no activation permission in this phase,
so preloading cannot start an agent.

The handoff then follows durable checkpoints:

1. Record source stop intent in the operator and source journals.
2. Suspend source activation, cleanly drain every agent, and stop Compose.
3. Capture a final leave-stopped snapshot and pin its ID, manifest SHA-256, and
   data archive SHA-256.
4. Upload and read-verify that exact snapshot in R2.
5. Download it, transfer it over the retained SSH host identity, and restore it
   on the target while activation remains suspended.
6. Retire source activation and confirm that no source service is running.
7. Persist target activation intent in both journals, mark the provisioned
   destination as no longer cleanable, start it, and require every Compose
   service and lifecycle record to report running.

The source-side migration journal also participates in managed start checks.
`theseus-deployment ... start` refuses to start a source during handoff. A
retired source remains blocked after a host reboot or manual Compose restart by
the activation guard mounted into each agent.

Every external call is retryable. If the controller exits or SSH disconnects,
run the same command with the same paths. It loads the provisioning operation,
Droplet ID, source and target journals, activation records, standard deployment
operation records, and pinned snapshot identity. A timeout never authorizes the
next phase by itself.

Inspect the operator journal without DigitalOcean or R2 credentials:

```bash
poetry run theseus-migrate build/flywheel status \
  --from /srv/theseus/flywheel \
  --profile host-profile.json \
  --provision-state ~/.local/state/theseus/flywheel-provision.json \
  --state ~/.local/state/theseus/flywheel-migration.json \
  --workspace ~/.local/state/theseus/flywheel-migration-work
```

Before target activation intent, source recovery is available only when the
destination answers over its retained SSH identity and proves it is inactive,
has no running services, and has never recorded activation intent:

```bash
poetry run theseus-migrate build/flywheel recover-source \
  --from /srv/theseus/flywheel \
  --profile host-profile.json \
  --provision-state ~/.local/state/theseus/flywheel-provision.json \
  --state ~/.local/state/theseus/flywheel-migration.json \
  --workspace ~/.local/state/theseus/flywheel-migration-work
```

Recovery records source-recovery intent before restoring activation. Once target
activation intent exists, the source is never restarted automatically. A failed
target startup is retried on the target; rollback requires a new stopped transfer
of the target's latest state.

If the source becomes unreachable after the exact target restore, promotion is
blocked. An operator may record external fencing evidence, such as a verified
DigitalOcean power-off, and then rerun migration:

```bash
poetry run theseus-migrate build/flywheel fence-source \
  --from /srv/theseus/flywheel \
  --profile host-profile.json \
  --provision-state ~/.local/state/theseus/flywheel-provision.json \
  --state ~/.local/state/theseus/flywheel-migration.json \
  --workspace ~/.local/state/theseus/flywheel-migration-work \
  --fence-evidence "Droplet 123456789 powered off in DigitalOcean at 2026-09-17T12:00Z"
```

Fencing is recorded evidence, not a remote power operation. It is accepted only
after exact inactive restore and only while the source is unreachable. The
result reports the created Droplet, final snapshot, source and target activation,
target readiness, and remaining billable source and destination resources.

Telegram delivery journals are included in the stopped snapshot. The handoff
reduces replay risk but does not promise exactly-once remote sends when Telegram
accepted a request before its acknowledgement was durably recorded.
