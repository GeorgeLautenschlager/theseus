from __future__ import annotations

import json

from theseus.backup_status import deployment_recovery_status, pending_uploads
from theseus.deployment_control import DeploymentController, MIGRATION_FILE
from theseus.remote_backup import RemoteBackups
from test_deployment_snapshot import AGENTS, manager
from test_remote_backup import Docker, RecordingStore


def prepared(tmp_path):
    snapshots, compose, root, bundle = manager(tmp_path)
    store = RecordingStore(tmp_path / "objects")
    backups = RemoteBackups(store, bundle, run=Docker())
    return snapshots, compose, root, bundle, store, backups


def fresh_controller(root, bundle, compose):
    return DeploymentController(
        root=root, deployment_id="flywheel", agent_ids=AGENTS,
        compose_file=bundle / "compose.yaml", run=compose,
    )


def test_status_reports_no_backups_yet(tmp_path):
    snapshots, compose, root, bundle, _, backups = prepared(tmp_path)
    value = deployment_recovery_status(snapshots.controller, backups, bundle)
    assert value["last_completed_backup"] is None
    assert value["pending_or_failed_uploads"] == []
    assert value["unfinished_operation"] is None
    assert value["migration"] is None
    assert value["activation"]["state"] == "active"


def test_status_reports_age_size_and_id_of_last_completed_backup(tmp_path):
    snapshots, compose, root, bundle, _, backups = prepared(tmp_path)
    captured = snapshots.create()
    result = backups.upload(captured.path)

    value = deployment_recovery_status(snapshots.controller, backups, bundle)
    last = value["last_completed_backup"]
    assert last["snapshot_id"] == result.snapshot_id
    assert last["size"] == result.total_size
    assert last["age_seconds"] >= 0


def test_failed_upload_is_reported_pending_and_never_becomes_last_completed(tmp_path):
    snapshots, compose, root, bundle, store, backups = prepared(tmp_path)
    captured = snapshots.create()
    store.fail_suffix = "/data.tar.gz"
    try:
        backups.upload(captured.path)
    except OSError:
        pass

    value = deployment_recovery_status(snapshots.controller, backups, bundle)
    assert value["last_completed_backup"] is None
    pending = value["pending_or_failed_uploads"]
    assert len(pending) == 1
    assert pending[0]["snapshot_id"] == captured.snapshot_id
    assert pending[0]["state"] == "failed"
    assert "storage unavailable" in pending[0]["failure"]
    assert str(captured.path) in pending[0]["resume_command"]


def test_locally_captured_but_never_uploaded_snapshot_is_pending(tmp_path):
    snapshots, compose, root, bundle, _, backups = prepared(tmp_path)
    captured = snapshots.create()

    value = deployment_recovery_status(snapshots.controller, backups, bundle)
    pending = value["pending_or_failed_uploads"]
    assert len(pending) == 1
    assert pending[0]["snapshot_id"] == captured.snapshot_id
    assert pending[0]["state"] == "never-uploaded"


def test_unfinished_operation_has_an_actionable_resume_command(tmp_path):
    snapshots, compose, root, bundle, _, backups = prepared(tmp_path)
    controller = snapshots.controller
    controller.journal.begin("start", controller.running_services())

    value = deployment_recovery_status(controller, backups, bundle)
    operation = value["unfinished_operation"]
    assert operation["kind"] == "start"
    assert "theseus-deployment" in operation["resume_command"]
    assert str(bundle) in operation["resume_command"]
    assert str(root) in operation["resume_command"]


def test_migration_in_progress_is_surfaced(tmp_path):
    snapshots, compose, root, bundle, _, backups = prepared(tmp_path)
    controller = snapshots.controller
    (controller.control_dir / MIGRATION_FILE).write_text(
        json.dumps({"phase": "snapshot-uploaded", "host_role": "source"})
    )

    value = deployment_recovery_status(controller, backups, bundle)
    assert value["migration"]["phase"] == "snapshot-uploaded"


def test_status_is_unchanged_across_a_fresh_controller_instance(tmp_path):
    snapshots, compose, root, bundle, _, backups = prepared(tmp_path)
    captured = snapshots.create()
    backups.upload(captured.path)

    before = deployment_recovery_status(snapshots.controller, backups, bundle)
    restarted = fresh_controller(root, bundle, compose)
    after = deployment_recovery_status(restarted, backups, bundle)
    # age_seconds keeps advancing between calls; everything else is durable state.
    before["last_completed_backup"].pop("age_seconds")
    after["last_completed_backup"].pop("age_seconds")
    assert after == before


def test_pending_uploads_ignores_a_missing_snapshots_directory(tmp_path):
    assert pending_uploads(tmp_path / "does-not-exist") == []
