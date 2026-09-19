---
name: deploy-theseus
description: Assemble, build, deploy, observe, stop, and back up Theseus agents as managed Docker containers. Use when asked to spawn or operate a Theseus agent locally, or prepare a DigitalOcean deployment or migration.
---

# Deploy Theseus agents

Use the Theseus assembler and managed lifecycle. Locate the Theseus checkout;
on George's machine it is `/home/aldric/theseus`. Run commands from that checkout.
The skill is compatible with Pi's `/skill:deploy-theseus` invocation.

## Choose the deployment

Use the user's specified identity, goal, model, interface, host, and budget.
Reuse answers already given. Ask only for missing choices that affect operation.
Keep a new agent's state separate from existing homes unless an import was requested.

Inspect these files before adapting a definition:

- `agents/alty_local.py`: working headless Ollama example.
- `src/theseus/deployment.py`: `DeploymentSpec`, resources, IDs, pairing, workspaces.
- `src/theseus/assembly.py`: `AgentSpec`, models, memory, tools, and interfaces.

Current managed containers support `core="auto"` with `interface="none"` or
`"telegram"`. Terminal chat and WebChat exist in ordinary assembly but are not
supported by the managed container lifecycle. Do not promise a browser or TUI endpoint.

For a local secret-free deployment, follow [the local playbook](references/local.md).
For cloud hosts, Telegram secrets, R2, or migration, use the relevant repository
runbooks: `docs/container-acceptance.md`, `docs/r2-backups.md`, and
`docs/migration.md`; inspect the corresponding CLI `--help` before execution.
The local helper intentionally rejects secret-bearing deployments. Cloud and
R2 behavior has separate test coverage; a local smoke test is not live-cloud validation.

## Operational invariants

- Assign a unique deployment ID, host root, and Compose project per deployment.
  Preserve them for subsequent operations. Agent state is under `data/agents/ID`;
  own logs and shared workspaces have separate mounts.
- Assemble and build before activation. Preserve the generated bundle and image
  lock. Do not hand-edit generated release files; keep local network overrides
  separate and record them with the deployment.
- Use managed `start` and `stop`. `docker compose up` bypasses activation setup;
  Docker reporting "running" alone does not prove agent readiness or inference.
- The host operator owns control state. Non-root agents must read activation but
  must not write it. Give the Docker socket only to the temporary operator, never
  to agent containers. Socket access gives the operator control over host Docker.
- A clean stop requires matching lifecycle acknowledgements. After a timeout or
  failed stop, inspect operation and lifecycle records; do not delete journals,
  fabricate clean evidence, or repeatedly retry to bypass a failure.
- Snapshot through managed backup, which stops writers and resumes prior services.
  Do not archive a live state directory. Keep backup evidence and required secrets
  separate. Restore only into an inactive empty target.
- Starting a model-backed agent makes real inference calls. Use the selected local
  model or the user's authorized hosted model/budget. Do not silently replace a
  local model with a paid provider. Inference limits and autonomous cadence are
  distinct from the container's memory/CPU limits.
- Provisioning new paid infrastructure, deleting deployments, and contacting people
  must be within the user's requested scope. Do not request confirmation again for
  deployment steps already authorized.

## Verify and hand over

Check activation, container identity/user/mounts, lifecycle readiness, a completed
model turn, and the expected durable artifact. For the first deployment of a new
configuration, verify a clean stop and restart with state preserved. Exercise a
local snapshot when backup behavior is part of the requested trial.

When validating container infrastructure changes, run `make container-acceptance`.
It uses deterministic inference and fake Telegram. Check the result actually ran;
missing Docker can cause the test module to skip. Do not substitute live paid tests.

Report the deployment ID, root, image identity, model/cadence, interface, current
state, observation and stop commands, validation performed, and remaining limits.
Record useful deployment-specific evidence in the playbook rather than silently
turning one agent's settings into universal defaults.
