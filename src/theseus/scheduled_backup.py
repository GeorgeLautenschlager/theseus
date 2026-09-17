"""One eligibility-checked scheduled backup attempt, safe to run unattended.

Reuses the existing deployment operation lock and the normal backup/upload
protocol unchanged; this module only decides whether a run should be
attempted at all. A skip is a normal, expected outcome for a timer tick (a
migration mid-flight, a suspended/retired deployment, or a concurrent
operation) and is reported, not raised. A genuine failure -- a real backup
attempt that then fails to capture or upload -- still raises, so an
unattended scheduler surfaces it as a failed run rather than a silent skip.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from theseus.deployment_control import DeploymentController, MIGRATION_FILE
from theseus.deployment_snapshot import DeploymentSnapshots
from theseus.remote_backup import RemoteBackups


# The only phase a migration journal reaches once it is done influencing
# activation and running services; anything else means the migration is
# still touching this deployment's activation, services, or operation lock.
TERMINAL_MIGRATION_PHASES = ("completed",)


@dataclass(frozen=True)
class ScheduledBackupOutcome:
    outcome: str  # "completed" or "skipped"
    reason: str | None = None
    snapshot_id: str | None = None
    captured_at: str | None = None
    uploaded_at: str | None = None
    total_size: int | None = None


def eligible(controller: DeploymentController) -> str | None:
    """Return a skip reason, or None if a scheduled backup may run now."""
    migration_path = controller.control_dir / MIGRATION_FILE
    if migration_path.is_file():
        try:
            migration = json.loads(migration_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            migration = {}
        phase = migration.get("phase")
        if phase not in TERMINAL_MIGRATION_PHASES:
            return f"migration is in progress at phase {phase!r}"
    activation = controller.activation.read()
    state = activation.state if activation is not None else "absent"
    if state != "active":
        return f"deployment activation is {state}, not active"
    return None


def run_scheduled_backup(
    controller: DeploymentController,
    snapshots: DeploymentSnapshots,
    backups: RemoteBackups,
    *,
    timeout_seconds: int = 30,
) -> ScheduledBackupOutcome:
    reason = eligible(controller)
    if reason is not None:
        return ScheduledBackupOutcome("skipped", reason=reason)
    try:
        captured = snapshots.create(timeout_seconds=timeout_seconds)
    except RuntimeError as exc:
        # Another operation (an overlapping scheduled run, a manual backup, a
        # migration phase, or a prior interrupted operation) holds the lock or
        # left the journal non-terminal. Recovery review is for an operator;
        # a timer tick just tries again next time.
        return ScheduledBackupOutcome("skipped", reason=str(exc))
    result = backups.upload(captured.path)
    return ScheduledBackupOutcome(
        "completed",
        snapshot_id=result.snapshot_id,
        captured_at=result.captured_at,
        uploaded_at=result.uploaded_at,
        total_size=result.total_size,
    )
