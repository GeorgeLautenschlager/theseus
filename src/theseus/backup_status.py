"""Operator status: activation, unfinished operations, and remote backups.

Combines durable on-disk state only -- nothing here is cached in memory, so a
restarted controller reports exactly what a fresh one would. A failed or
still-uploading local snapshot never contributes to ``last_completed_backup``;
that field comes only from the remote store's own completed manifests, so a
failed remote upload can never be mistaken for a fresh successful backup.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from theseus.deployment_control import DeploymentController, MIGRATION_FILE, OperationRecord
from theseus.remote_backup import REMOTE_STATUS_FILE, RemoteBackups


def _resume_command(bundle: Path, root: Path, operation: OperationRecord) -> str:
    base = f"poetry run theseus-deployment {bundle}"
    if operation.kind == "start":
        return f"{base} start --root {root}"
    if operation.kind == "stop":
        return f"{base} stop --root {root}"
    if operation.kind == "backup":
        return f"poetry run theseus-backup {bundle} create --root {root}"
    if operation.kind == "restore":
        return (
            f"{base} recovery --root {root}  # inspect operation detail, "
            "then re-run restore with the recorded --snapshot"
        )
    return (
        f"{base} recovery --root {root}  # migration operation: resume with "
        "the original theseus-migrate command and state paths"
    )


def pending_uploads(root: Path) -> list[dict[str, Any]]:
    """Local snapshots whose remote upload never reached a completed state."""
    snapshots_dir = Path(root) / "snapshots"
    pending: list[dict[str, Any]] = []
    if not snapshots_dir.is_dir():
        return pending
    for entry in sorted(snapshots_dir.iterdir()):
        if entry.is_symlink() or not entry.is_dir() or entry.name.startswith("."):
            continue
        status_path = entry / REMOTE_STATUS_FILE
        if not status_path.is_file():
            pending.append(
                {
                    "snapshot_id": entry.name,
                    "state": "never-uploaded",
                    "failure": None,
                    "path": str(entry),
                    "resume_command": f"poetry run theseus-backup <bundle> upload --snapshot {entry}",
                }
            )
            continue
        try:
            value = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if value.get("state") != "completed":
            pending.append(
                {
                    "snapshot_id": value.get("snapshot_id", entry.name),
                    "state": value.get("state"),
                    "failure": value.get("failure"),
                    "path": str(entry),
                    "resume_command": f"poetry run theseus-backup <bundle> upload --snapshot {entry}",
                }
            )
    return pending


def deployment_recovery_status(
    controller: DeploymentController,
    backups: RemoteBackups,
    bundle: Path,
) -> dict[str, Any]:
    root = controller.root.resolve()
    activation = controller.activation.read()
    operation = controller.interrupted_operation()

    migration_path = controller.control_dir / MIGRATION_FILE
    migration: dict[str, Any] | None = None
    if migration_path.is_file():
        try:
            migration = json.loads(migration_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            migration = {"phase": "unreadable"}

    remote_backups = backups.list()
    last_backup = None
    if remote_backups:
        latest = remote_backups[0]
        age_seconds = (
            datetime.now(timezone.utc) - datetime.fromisoformat(latest["uploaded_at"])
        ).total_seconds()
        last_backup = {
            "snapshot_id": latest["snapshot_id"],
            "captured_at": latest["captured_at"],
            "uploaded_at": latest["uploaded_at"],
            "size": latest["total_size"],
            "age_seconds": age_seconds,
        }

    return {
        "deployment_id": controller.deployment_id,
        "activation": asdict(activation) if activation is not None else None,
        "last_completed_backup": last_backup,
        "pending_or_failed_uploads": pending_uploads(root),
        "unfinished_operation": (
            {**asdict(operation), "resume_command": _resume_command(bundle, root, operation)}
            if operation is not None
            else None
        ),
        "migration": migration,
    }
