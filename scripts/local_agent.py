"""Linux/local-Docker operator for a built, secret-free Theseus deployment.

Run with --help. The operator is temporary and root; agents retain their configured
non-root UID. Only the operator receives the Docker socket. Generated release
files remain unchanged; host networking is a separate local override.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


OPERATOR = r'''
import json, os, subprocess, sys
from dataclasses import asdict
from pathlib import Path
from theseus.deployment_control import DeploymentController
from theseus.deployment_snapshot import DeploymentSnapshots

action, root_arg, bundle_arg, networking, lines = sys.argv[1:]
root, bundle = Path(root_arg), Path(bundle_arg)
manifest = json.loads((bundle / 'deployment.json').read_text())
uid, gid = manifest['uid'], manifest['gid']
agents = tuple(manifest['agents'])
override = root / 'local-compose.json'
expected_override = {'services': {name: {'network_mode': 'host'} for name in agents}} if networking == 'host' else {'services': {}}

if action == 'prepare':
    if manifest['required_secrets']:
        raise SystemExit('This local helper requires a secret-free deployment.')
    if (root / 'control' / 'activation.json').exists():
        raise SystemExit('Already initialized: use status/start, not prepare.')
    if (root / 'data').exists() and any((root / 'data').iterdir()):
        raise SystemExit('Refusing to prepare over existing agent data.')
    for name in ('control', 'secrets', 'snapshots'):
        path = root / name
        path.mkdir(exist_ok=True)
        path.chmod(0o700)
    os.chown(root / 'control', 0, gid)
    (root / 'control').chmod(0o2750)
    for name in agents:
        for suffix in ('', 'state', 'logs'):
            path = root / 'data' / 'agents' / name / suffix
            path.mkdir(parents=True, exist_ok=True)
            os.chown(path, uid, gid)
            path.chmod(0o750)
    workspaces = {w for agent in manifest['agents'].values() for w in agent['workspaces']}
    for name in workspaces:
        path = root / 'data' / 'workspaces' / name
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, uid, manifest['workspace_gid'])
        path.chmod(0o2770)
    override.write_text(json.dumps(expected_override, indent=2) + '\n')
    override.chmod(0o644)
    print(json.dumps({'prepared': str(root), 'networking': networking, 'agents': agents}))
    sys.exit(0)

if not override.is_file() or json.loads(override.read_text()) != expected_override:
    raise SystemExit('Local networking differs from prepared configuration; supply the same --host-network setting.')

def run(command, **kwargs):
    command = list(command)
    if command[:2] == ['docker', 'compose']:
        command[2:2] = ['--project-name', manifest['deployment_id']]
        index = command.index('-f') + 2
        command[index:index] = ['-f', str(override)]
    return subprocess.run(command, **kwargs)

controller = DeploymentController(root=root, deployment_id=manifest['deployment_id'],
    agent_ids=agents, compose_file=bundle / 'compose.yaml', run=run)
if action == 'start':
    print(json.dumps(asdict(controller.start()), indent=2))
elif action == 'stop':
    print(json.dumps(asdict(controller.stop()), indent=2))
elif action == 'status':
    print(json.dumps(controller.status(), indent=2))
elif action == 'backup':
    result = DeploymentSnapshots(controller, bundle).create()
    print(json.dumps({'snapshot_id': result.snapshot_id, 'path': str(result.path)}, indent=2))
elif action == 'logs':
    for name in agents:
        path = root / 'data' / 'agents' / name / 'logs' / 'stimulus_log.jsonl'
        print(f'--- {name} stimulus log ---')
        if path.exists():
            from collections import deque
            with path.open() as stream:
                print(''.join(deque(stream, maxlen=int(lines))), end='')
elif action == 'artifacts':
    for name in agents:
        path = root / 'data' / 'agents' / name / 'state' / 'DEPLOYMENT_CHECK.md'
        print(f'--- {name} DEPLOYMENT_CHECK.md ---')
        print(path.read_text() if path.exists() else '(not written yet)')
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start", "stop", "status", "backup", "logs", "artifacts"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host-network", action="store_true", help="Linux only: let agents reach host loopback (e.g. Ollama)")
    parser.add_argument("--lines", type=int, default=10)
    args = parser.parse_args()
    root, bundle = args.root.resolve(), args.bundle.resolve()
    if not bundle.is_relative_to(root) or bundle == root:
        parser.error("bundle must live below root, e.g. ROOT/releases/alty-local")
    if args.lines < 1:
        parser.error("--lines must be positive")
    lock = json.loads((bundle / "deployment.lock.json").read_text())
    images = lock["images"]
    if len(images) != 1:
        parser.error("local helper currently supports a single shared image")
    image = images[0]["content_id"]
    docker = shutil.which("docker")
    if not docker:
        parser.error("Docker is required")
    info = json.loads(subprocess.check_output([docker, "info", "--format", "{{json .ClientInfo.Plugins}}"], text=True))
    compose = next((p["Path"] for p in info if p["Name"] == "compose"), None)
    if not compose:
        parser.error("Docker Compose plugin is required")
    command = [docker, "run", "--rm", "-i", "--user", "0:0", "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=64m", "--entrypoint", "python",
        "-v", f"{root}:{root}", "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", f"{docker}:/usr/bin/docker:ro",
        "-v", f"{compose}:/usr/libexec/docker/cli-plugins/docker-compose:ro",
        image, "-", args.action, str(root), str(bundle),
        "host" if args.host_network else "bridge", str(args.lines)]
    subprocess.run(command, input=OPERATOR, text=True, check=True)


if __name__ == "__main__":
    main()
