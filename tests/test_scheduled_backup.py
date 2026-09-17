from __future__ import annotations

import json

import pytest

from theseus.backup_store import LocalObjectStore
from theseus.deployment_control import MIGRATION_FILE, operation_lock
from theseus.remote_backup import RemoteBackups
from test_deployment_snapshot import manager
from test_remote_backup import Docker, RecordingStore
from theseus.scheduled_backup import ScheduledBackupOutcome, run_scheduled_backup


def prepared(tmp_path, **manager_kwargs):
    snapshots, compose, root, bundle = manager(tmp_path, **manager_kwargs)
    store = RecordingStore(tmp_path / "objects")
    backups = RemoteBackups(store, bundle, run=Docker())
    return snapshots.controller, snapshots, backups, store, compose, root, bundle


def test_active_deployment_completes_a_normal_backup(tmp_path):
    controller, snapshots, backups, store, compose, _, _ = prepared(tmp_path)
    outcome = run_scheduled_backup(controller, snapshots, backups)
    assert outcome.outcome == "completed"
    assert outcome.snapshot_id is not None
    assert outcome.total_size and outcome.total_size > 0
    assert [item["snapshot_id"] for item in backups.list()] == [outcome.snapshot_id]
    # The normal backup protocol resumed exactly what was running before.
    assert compose.running == set(("fable", "astra"))


def test_suspended_deployment_is_skipped_not_activated(tmp_path):
    controller, snapshots, backups, *_ = prepared(tmp_path)
    controller.activation.set("suspended", reason="maintenance")
    outcome = run_scheduled_backup(controller, snapshots, backups)
    assert outcome.outcome == "skipped"
    assert "suspended" in outcome.reason
    assert backups.list() == []
    assert controller.activation.read().state == "suspended"


def test_retired_deployment_is_skipped_not_resumed(tmp_path):
    controller, snapshots, backups, *_ = prepared(tmp_path, running=())
    controller.retire(reason="decommissioned")
    outcome = run_scheduled_backup(controller, snapshots, backups)
    assert outcome.outcome == "skipped"
    assert "retired" in outcome.reason
    assert backups.list() == []
    assert controller.activation.read().state == "retired"


def test_migration_in_progress_is_skipped_even_when_active(tmp_path):
    controller, snapshots, backups, *_ = prepared(tmp_path)
    (controller.control_dir / MIGRATION_FILE).write_text(
        json.dumps({"phase": "source-stopped", "host_role": "source"})
    )
    outcome = run_scheduled_backup(controller, snapshots, backups)
    assert outcome.outcome == "skipped"
    assert "migration" in outcome.reason
    assert backups.list() == []


def test_completed_migration_no_longer_blocks_scheduled_backups(tmp_path):
    controller, snapshots, backups, *_ = prepared(tmp_path)
    (controller.control_dir / MIGRATION_FILE).write_text(
        json.dumps({"phase": "completed", "host_role": "source"})
    )
    outcome = run_scheduled_backup(controller, snapshots, backups)
    assert outcome.outcome == "completed"


def test_interrupted_operation_is_skipped_not_raised(tmp_path):
    controller, snapshots, backups, *_ = prepared(tmp_path)
    controller.journal.begin("backup", controller.running_services())
    outcome = run_scheduled_backup(controller, snapshots, backups)
    assert outcome.outcome == "skipped"
    assert "interrupted" in outcome.reason


def test_concurrent_operation_lock_is_skipped_not_raised(tmp_path):
    controller, snapshots, backups, *_ = prepared(tmp_path)
    with operation_lock(controller.control_dir):
        outcome = run_scheduled_backup(controller, snapshots, backups)
    assert outcome.outcome == "skipped"
    assert "in progress" in outcome.reason
    assert backups.list() == []


def test_failed_remote_upload_raises_rather_than_reporting_completed(tmp_path):
    controller, snapshots, backups, store, *_ = prepared(tmp_path)
    store.fail_suffix = "/data.tar.gz"
    with pytest.raises(OSError):
        run_scheduled_backup(controller, snapshots, backups)
    assert backups.list() == []
