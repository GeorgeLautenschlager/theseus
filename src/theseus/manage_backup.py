"""Command-line management for verified remote deployment backups."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from theseus.backup_store import LocalObjectStore, S3ObjectStore
from theseus.deployment_snapshot import DeploymentSnapshots
from theseus.manage_deployment import _controller
from theseus.remote_backup import RemoteBackups


def _store(args: argparse.Namespace):
    if args.local_store is not None:
        if args.endpoint or args.bucket:
            raise ValueError("--local-store cannot be combined with an R2 endpoint or bucket")
        return LocalObjectStore(args.local_store)
    endpoint = args.endpoint or os.environ.get("THESEUS_R2_ENDPOINT")
    bucket = args.bucket or os.environ.get("THESEUS_R2_BUCKET")
    if not endpoint or not bucket:
        raise ValueError(
            "set --endpoint/--bucket (or THESEUS_R2_ENDPOINT/THESEUS_R2_BUCKET)"
        )
    return S3ObjectStore(endpoint_url=endpoint, bucket=bucket)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload, inspect, and restore exact Theseus deployment backups"
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("action", choices=("create", "upload", "list", "download", "restore"))
    parser.add_argument("--endpoint", help="Cloudflare R2 S3 endpoint URL")
    parser.add_argument("--bucket", help="Cloudflare R2 bucket")
    parser.add_argument("--local-store", type=Path, help="filesystem object store for offline use")
    parser.add_argument("--root", type=Path, help="deployment root")
    parser.add_argument("--snapshot", help="exact snapshot ID, or local path for upload")
    parser.add_argument("--output", type=Path, help="new directory for downloaded objects")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--leave-stopped", action="store_true")
    args = parser.parse_args()
    try:
        service = RemoteBackups(_store(args), args.bundle)
        if args.action == "create":
            controller = _controller(args.bundle, args.root)
            captured = DeploymentSnapshots(controller, args.bundle).create(
                leave_stopped=args.leave_stopped, timeout_seconds=args.timeout
            )
            try:
                value = asdict(service.upload(captured.path))
            except Exception as exc:
                raise RuntimeError(
                    f"snapshot {captured.snapshot_id} is staged at {captured.path}; "
                    f"upload failed: {exc}"
                ) from exc
            value["local_snapshot"] = str(captured.path)
        elif args.action == "upload":
            if not args.snapshot:
                parser.error("upload requires --snapshot with a local snapshot directory")
            value = asdict(service.upload(Path(args.snapshot)))
        elif args.action == "list":
            value = [
                {
                    "snapshot_id": item["snapshot_id"],
                    "captured_at": item["captured_at"],
                    "uploaded_at": item["uploaded_at"],
                    "total_size": item["total_size"],
                    "release_id": item["release_id"],
                }
                for item in service.list()
            ]
        elif args.action == "download":
            if not args.snapshot or args.output is None:
                parser.error("download requires an exact --snapshot ID and --output")
            downloaded = service.download(args.snapshot, args.output)
            value = {
                "snapshot_id": downloaded.snapshot_id,
                "path": str(downloaded.root),
                "bundle": str(downloaded.bundle),
                "snapshot": str(downloaded.snapshot),
            }
        else:
            if not args.snapshot:
                parser.error("restore requires an exact --snapshot ID")
            workspace = args.output
            cleanup_root = None
            if workspace is None:
                cleanup_root = Path(
                    tempfile.mkdtemp(prefix=f"theseus-restore-{args.snapshot}-")
                )
                workspace = cleanup_root / "download"
            try:
                downloaded = service.download(args.snapshot, workspace)
                service.load_images(downloaded)
                target_controller = _controller(downloaded.bundle, args.root)
                restored = DeploymentSnapshots(target_controller, downloaded.bundle).restore(
                    downloaded.snapshot
                )
                value = {
                    "snapshot_id": args.snapshot,
                    "data": str(restored),
                    "activation": "suspended",
                    "release_id": downloaded.manifest["release_id"],
                }
            finally:
                if cleanup_root is not None:
                    shutil.rmtree(cleanup_root, ignore_errors=True)
        print(json.dumps(value, indent=2, sort_keys=True))
    except (
        ValueError,
        PermissionError,
        RuntimeError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"theseus-backup: {exc}\n")


if __name__ == "__main__":
    main()
