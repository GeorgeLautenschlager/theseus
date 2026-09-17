"""Operator CLI for Theseus's opt-in scheduled backup timer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from theseus.backup_schedule import DEFAULT_EXECUTABLE, DEFAULT_ON_CALENDAR, install, status, uninstall
from theseus.deployment_snapshot import DeploymentSnapshots
from theseus.manage_backup import _store
from theseus.manage_deployment import _controller
from theseus.remote_backup import RemoteBackups
from theseus.scheduled_backup import run_scheduled_backup


def _deployment_id(bundle: Path) -> str:
    manifest = json.loads((bundle / "deployment.json").read_text(encoding="utf-8"))
    if manifest.get("_generated_by") != "theseus-compose-assembler":
        raise ValueError("bundle is not an assembler-owned deployment")
    return manifest["deployment_id"]


def _store_args(args: argparse.Namespace) -> list[str]:
    if args.local_store:
        return ["--local-store", str(args.local_store)]
    parts = []
    if args.endpoint:
        parts += ["--endpoint", args.endpoint]
    if args.bucket:
        parts += ["--bucket", args.bucket]
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install, inspect, or run Theseus's opt-in scheduled backup timer"
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("action", choices=("install", "uninstall", "timer-status", "run"))
    parser.add_argument("--root", type=Path, help="deployment root")
    parser.add_argument("--unit-dir", type=Path, default=Path("/etc/systemd/system"))
    parser.add_argument("--scope", choices=("system", "user"), default="system")
    parser.add_argument("--on-calendar", default=DEFAULT_ON_CALENDAR)
    parser.add_argument("--executable", default=DEFAULT_EXECUTABLE)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--enable", action="store_true",
        help="also enable and start the timer now; installs are disabled by default",
    )
    parser.add_argument("--endpoint", help="Cloudflare R2 S3 endpoint URL")
    parser.add_argument("--bucket", help="Cloudflare R2 bucket")
    parser.add_argument("--local-store", type=Path, help="filesystem object store for offline use")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    try:
        deployment_id = _deployment_id(args.bundle)
        if args.action == "install":
            if args.root is None:
                parser.error("install requires --root")
            installed = install(
                deployment_id, args.bundle, args.root, args.unit_dir,
                on_calendar=args.on_calendar,
                executable=args.executable,
                env_file=args.env_file,
                store_args=tuple(_store_args(args)),
                scope=args.scope,
                enable=args.enable,
            )
            value = {
                **asdict(installed),
                "unit_dir": str(installed.unit_dir),
                "service_path": str(installed.service_path),
                "timer_path": str(installed.timer_path),
            }
        elif args.action == "uninstall":
            uninstall(deployment_id, args.unit_dir, scope=args.scope)
            value = {"deployment_id": deployment_id, "uninstalled": True}
        elif args.action == "timer-status":
            value = status(deployment_id, args.unit_dir, scope=args.scope)
        else:
            if args.root is None:
                parser.error("run requires --root")
            controller = _controller(args.bundle, args.root)
            snapshots = DeploymentSnapshots(controller, args.bundle)
            backups = RemoteBackups(_store(args), args.bundle)
            value = asdict(
                run_scheduled_backup(controller, snapshots, backups, timeout_seconds=args.timeout)
            )
        print(json.dumps(value, indent=2, sort_keys=True))
    except (
        ValueError,
        PermissionError,
        RuntimeError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"theseus-backup-schedule: {exc}\n")


if __name__ == "__main__":
    main()
