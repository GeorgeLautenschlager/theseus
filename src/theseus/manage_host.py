"""Operator CLI for resumable DigitalOcean destination provisioning."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from theseus.host_provisioner import DigitalOceanAPI, HostProfile, HostProvisioner, OpenSSH


def _secrets(deployment: dict, directory: Path | None) -> dict[str, Path]:
    names = deployment.get("required_secrets", [])
    if not names:
        return {}
    if directory is None:
        raise ValueError("--secrets-dir is required for this deployment")
    return {name: directory / name for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision a ready, inactive DigitalOcean migration destination"
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("action", choices=("preview", "provision", "status", "cleanup", "mark-activated"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--secrets-dir", type=Path)
    parser.add_argument("--minimum-disk-gb", type=int, default=10)
    parser.add_argument("--source-droplet-id", type=int)
    args = parser.parse_args()
    try:
        deployment = json.loads((args.bundle / "deployment.json").read_text(encoding="utf-8"))
        if deployment.get("_generated_by") != "theseus-compose-assembler":
            raise ValueError("bundle is not an assembler-owned deployment")
        api = DigitalOceanAPI() if args.action in ("preview", "provision", "cleanup") else None
        provisioner = HostProvisioner(
            api,
            OpenSSH(),
            deployment,
            HostProfile.from_file(args.profile),
            args.state,
            minimum_disk_gb=args.minimum_disk_gb,
            source_droplet_id=args.source_droplet_id,
        )
        if args.action == "preview":
            value = asdict(provisioner.preview())
        elif args.action == "provision":
            value = asdict(
                provisioner.provision(_secrets(deployment, args.secrets_dir))
            )
        elif args.action == "status":
            value = asdict(provisioner.status())
        elif args.action == "cleanup":
            value = asdict(provisioner.cleanup())
        else:
            value = asdict(provisioner.mark_activated())
        print(json.dumps(value, indent=2, sort_keys=True))
    except (
        ValueError,
        PermissionError,
        RuntimeError,
        TimeoutError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"theseus-host: {exc}\n")


if __name__ == "__main__":
    main()
