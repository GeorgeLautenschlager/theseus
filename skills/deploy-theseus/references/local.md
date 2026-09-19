# Local Linux/Docker playbook

## Prerequisites and decisions

From the Theseus checkout, check `git status --short`, `docker info`,
`docker compose version`, and `poetry run python --version`. Preserve unrelated
edits. If dependencies are missing, install from the existing Poetry lock.
Use Python module entry points: an older virtualenv may lack newly added console
scripts such as `theseus-build` even though imports work.

For Ollama, inspect `ollama list` and the listener. The current provider defaults
to `http://127.0.0.1:11434/v1`. In a bridged container, that address is the container
itself. This playbook uses an explicit Linux host-network override so it reaches
host Ollama without rebinding the server. This gives the agent access to host
network services; it does not mount host files or the Docker socket into the agent.
For network isolation, implement and test a reachable endpoint configuration instead.

The local helper assumes local Docker at `/var/run/docker.sock` and a compatible
Linux Docker CLI/Compose plugin. It is not a remote Docker-context or Docker Desktop
deployment helper. It uses a temporary root operator container and the exact built
agent image; agents themselves retain the configured non-root UID/GID.

## Alty example

Definition: `agents/alty_local.py`. Fresh deployment ID `alty-local`, agent ID
`alty`, `gemma4:e4b`, 8K configured context, 15-minute autonomous cadence, headless
logs, 1 CPU and 768 MiB **agent** memory limit. Ollama runs on the host and its model
memory is outside that limit. Tools are read/write/list. The layered memory module
is attached, without automatic extraction or embedding models in this smoke test.

Alty's task is to write and retain `DEPLOYMENT_CHECK.md` in his private state.
For another agent, create a separate definition with its own IDs and task; do not
overwrite Alty's definition or reuse his root.

```bash
cd /home/aldric/theseus
alty_root="$PWD/build/deployments/alty-local"
alty_bundle="$alty_root/releases/alty-local"

poetry run python -m theseus.assemble agents/alty_local.py \
  --target compose --output "$alty_bundle"
poetry run python -m theseus.build_deployment "$alty_bundle"
python scripts/local_agent.py prepare \
  --root "$alty_root" --bundle "$alty_bundle" --host-network
python scripts/local_agent.py start \
  --root "$alty_root" --bundle "$alty_bundle" --host-network
```

`prepare` is for a new, empty deployment only. If it reports existing data or
activation, inspect `status`; do not erase state to make preparation succeed.
For an existing deployment, use its existing bundle and inspect status before
starting. Reassembly is not required for a restart and is not an upgrade procedure.

The generated bundle stays unchanged. The helper creates `local-compose.json`
outside the bundle and consistently supplies it and the explicit deployment
project name on managed Compose commands. Pass the same `--host-network` option
on every helper invocation. This override is host configuration, not part of the
portable state snapshot; retain it and reapply the appropriate network configuration
when restoring elsewhere.

## Observe

```bash
python scripts/local_agent.py status \
  --root "$alty_root" --bundle "$alty_bundle" --host-network
python scripts/local_agent.py logs \
  --root "$alty_root" --bundle "$alty_bundle" --host-network --lines 10
python scripts/local_agent.py artifacts \
  --root "$alty_root" --bundle "$alty_bundle" --host-network
docker logs --tail 30 alty-local-alty-1
docker stats --no-stream alty-local-alty-1
```

`logs` reads the durable stimulus log; Docker logs show process diagnostics.
`artifacts` reads the Alty-specific `DEPLOYMENT_CHECK.md`. For another task, inspect
its named artifact through a suitable read-only operator command.

Inspect status until activation is `active`, the agent is listed in
`running_services`, and lifecycle state is `running`. Then verify a completed
model response/tool result and artifact. Cold model loading can delay the first
turn; lifecycle readiness is not evidence that inference completed.

## Clean stop, restart, and snapshot

```bash
python scripts/local_agent.py stop \
  --root "$alty_root" --bundle "$alty_bundle" --host-network
python scripts/local_agent.py status \
  --root "$alty_root" --bundle "$alty_bundle" --host-network
python scripts/local_agent.py start \
  --root "$alty_root" --bundle "$alty_bundle" --host-network
python scripts/local_agent.py backup \
  --root "$alty_root" --bundle "$alty_bundle" --host-network
```

Allow an in-flight model turn to finish before testing clean stop. Confirm
`suspended`, no running services, and lifecycle `stopped` with `clean: true`.
After restart, confirm a new run ID and the retained artifact. A failed stop
needs diagnosis; container exit alone does not prove cleanliness.

`backup` creates a local snapshot under `ROOT/snapshots/` and resumes previously
running agents. It does not upload to R2. Preserve the returned snapshot ID and
inspect status afterwards. Root-owned control, log, and snapshot files are
intentionally accessed through the operator; do not chmod them open for convenience.

`build/` is Git-ignored but holds durable state in this example. Do not run
`git clean -fdx`, delete build output, or prune deployment images against a live
root. For a long-lived installation, choose a dedicated persistent root and keep
its release bundles beneath it before first deployment.

## Pi/Blueberry installation

The canonical skill lives at `skills/deploy-theseus` in the Theseus checkout.
Blueberry can discover it through a directory symlink at
`/home/aldric/Blueberry/.pi/skills/deploy-theseus`. This is project-local; it does
not change every Pi agent's configuration. Restart or reload Pi after installation,
then invoke `/skill:deploy-theseus` with the desired agent and deployment request.
From another working directory, use Pi's `--skill` option with the canonical path.

## Recorded Alty trial — 2026-09-17

- Runtime source: `0f1ff0f`; image content ID
  `sha256:57410a9685a0fac4cb134a44f92996bbd5ab43f7d58f88f2a45eb9f48cf8d399`.
- Updated Docker acceptance gate: 2 passed. Focused migration, deployment,
  lifecycle, and host provisioning regressions: 69 passed. Pytest emitted
  cleanup warnings for old root-owned temporary directories.
- Confirmed real Ollama inference and a successful write to
  `DEPLOYMENT_CHECK.md`; observed about 92 MiB agent-container memory.
  This excludes host Ollama memory and is a single observation, not a sizing bound.
- Clean stop acknowledged; local snapshot
  `c6cd3a0d99ae47a09e209bd999f89eb5` captured while stopped.
- No R2 upload or live cloud migration performed in this trial.
- Pi's actual skill loader discovered `deploy-theseus` with no diagnostics.
- Trial cleaned up at the user's request on 2026-09-17: clean shutdown verified;
  container, dedicated image tag, generated release, state, logs, and local
  snapshots removed. The snapshot ID above is historical evidence, not an
  available backup. The definition, operator helper, and installed skill remain.
