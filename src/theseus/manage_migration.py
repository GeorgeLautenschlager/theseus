"""One-command resumable migration to a provisioned DigitalOcean destination."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

from theseus.backup_store import LocalObjectStore
from theseus.host_provisioner import DigitalOceanAPI, HostProfile, HostProvisioner, OpenSSH
from theseus.manage_backup import _store
from theseus.manage_deployment import _controller
from theseus.migration import (
    LocalMigrationSource,
    MigrationCoordinator,
    SSHMigrationTarget,
)
from theseus.deployment_snapshot import DeploymentSnapshots
from theseus.remote_backup import RemoteBackups


def _secret_files(deployment: dict[str, object], directory: Path | None) -> dict[str, Path]:
    names = deployment.get("required_secrets", [])
    if not names:
        return {}
    if directory is None:
        raise ValueError("--secrets-dir is required for this deployment")
    return {str(name): directory / str(name) for name in names}


def _needs_api(action: str, provision_state: Path) -> bool:
    if action != "migrate":
        return False
    if not provision_state.is_file():
        return True
    try:
        state = json.loads(provision_state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return state.get("phase") not in ("ready", "activated")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate one managed deployment without overlapping agent execution"
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("action", choices=("migrate", "status", "recover-source", "fence-source"))
    parser.add_argument("--from", dest="source_root", type=Path, required=True)
    parser.add_argument("--provision", choices=("digitalocean",), default="digitalocean")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--provision-state", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--secrets-dir", type=Path)
    parser.add_argument("--source-droplet-id", type=int)
    parser.add_argument("--minimum-disk-gb", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--fence-evidence")
    parser.add_argument("--endpoint", help="Cloudflare R2 S3 endpoint URL")
    parser.add_argument("--bucket", help="Cloudflare R2 bucket")
    parser.add_argument("--local-store", type=Path, help="filesystem object store for offline use")
    args = parser.parse_args()
    try:
        deployment = json.loads((args.bundle / "deployment.json").read_text(encoding="utf-8"))
        if deployment.get("_generated_by") != "theseus-compose-assembler":
            raise ValueError("bundle is not an assembler-owned deployment")
        profile = HostProfile.from_file(args.profile)
        ssh = OpenSSH()
        api = DigitalOceanAPI() if _needs_api(args.action, args.provision_state) else None
        provisioner = HostProvisioner(
            api,
            ssh,
            deployment,
            profile,
            args.provision_state,
            minimum_disk_gb=args.minimum_disk_gb,
            source_droplet_id=args.source_droplet_id,
        )
        if (
            args.action != "migrate"
            and args.local_store is None
            and not (args.endpoint or os.environ.get("THESEUS_R2_ENDPOINT"))
        ):
            store = LocalObjectStore(args.workspace / ".offline-status-store")
        else:
            store = _store(args)
        backups = RemoteBackups(store, args.bundle)
        source_controller = _controller(args.bundle, args.source_root)
        source = LocalMigrationSource(
            source_controller,
            DeploymentSnapshots(source_controller, args.bundle),
            backups,
        )

        def target_factory(result):
            if not result.address:
                raise ValueError("provisioned destination has no retained address")
            return SSHMigrationTarget(
                ssh,
                profile,
                provisioner.known_hosts,
                result.address,
                deployment,
                backups.release_id,
            )

        coordinator = MigrationCoordinator(
            deployment_id=deployment["deployment_id"],
            agent_ids=tuple(deployment["agents"]),
            source=source,
            provisioner=provisioner,
            target_factory=target_factory,
            backups=backups,
            state_path=args.state,
            workspace=args.workspace,
            secrets=(
                _secret_files(deployment, args.secrets_dir)
                if args.action == "migrate"
                else {}
            ),
            timeout_seconds=args.timeout,
        )
        if args.action == "migrate":
            value = coordinator.migrate()
        elif args.action == "status":
            value = coordinator.status()
        elif args.action == "recover-source":
            value = coordinator.recover_source()
        else:
            if not args.fence_evidence:
                parser.error("fence-source requires --fence-evidence")
            value = coordinator.fence_source(args.fence_evidence)
        print(json.dumps(asdict(value), indent=2, sort_keys=True))
    except (
        ValueError,
        PermissionError,
        RuntimeError,
        TimeoutError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"theseus-migrate: {exc}\n")


if __name__ == "__main__":
    main()
