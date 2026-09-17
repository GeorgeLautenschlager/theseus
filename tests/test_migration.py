from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from theseus.deployment_snapshot import SnapshotResult
from theseus.migration import (
    HostInspection,
    HostUnreachable,
    MigrationCoordinator,
    SnapshotEvidence,
    UnknownActivation,
)
from theseus.remote_backup import DownloadedBackup, ReleaseArtifacts
from test_deployment_snapshot import manager


AGENTS = ("astra", "fable")


class Faults:
    def __init__(self, name=None):
        self.name = name
        self.fired = False

    def after(self, name):
        if self.name == name and not self.fired:
            self.fired = True
            raise TimeoutError(f"controller lost after {name}")


class Provision:
    def __init__(self, faults):
        self.faults = faults
        self.ready = False
        self.activated = False
        self.calls = 0

    def _result(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            operation_id="provision-1",
            phase="activated" if self.activated else "ready",
            droplet_id=4242,
            address="203.0.113.42",
            ssh_host_fingerprint="SHA256:target",
            remaining_billable_resources=("droplet:4242", "source-droplet:41"),
        )

    def provision(self, secrets):
        self.calls += 1
        self.ready = True
        self.faults.after("provision")
        return self._result()

    def status(self):
        if not self.ready:
            raise ValueError("not provisioned")
        return self._result()

    def mark_activated(self):
        self.activated = True
        return self._result()


class Backups:
    release_id = "release-1"

    def __init__(self, root, faults):
        self.root = root
        self.faults = faults
        self.evidence = None
        self.uploaded = False

    def stage_release(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        bundle = directory / "bundle.tar.gz"
        image = directory / "image.tar.gz"
        bundle.write_bytes(b"bundle")
        image.write_bytes(b"image")
        return ReleaseArtifacts(
            self.release_id,
            bundle,
            "bundle-sha",
            (image,),
            ({
                "archive_sha256": "image-archive-sha",
                "sha256": "image-object-sha",
                "content_id": "sha256:image",
                "platform": "linux/amd64",
            },),
        )

    def register(self, evidence):
        self.evidence = evidence

    def get_manifest(self, snapshot_id):
        assert self.uploaded and self.evidence.result.snapshot_id == snapshot_id
        return {
            **self.evidence.result.manifest,
            "objects": {"images": []},
        }

    def download(self, snapshot_id, destination):
        assert self.uploaded and self.evidence.result.snapshot_id == snapshot_id
        shutil.copytree(self.evidence.result.path, destination / "snapshot")
        bundle = destination / "release" / "bundle"
        bundle.mkdir(parents=True)
        (bundle / "deployment.json").write_text("{}")
        images = destination / "images"
        images.mkdir()
        return DownloadedBackup(
            snapshot_id,
            destination,
            destination / "snapshot",
            bundle,
            (),
            self.get_manifest(snapshot_id),
        )


class Source:
    agent_ids = AGENTS

    def __init__(self, root, backups, faults):
        self.root = root
        self.identity = f"fake:{root}"
        self.backups = backups
        self.faults = faults
        self.activation = "active"
        self.running = set(AGENTS)
        self.records = []
        self.evidence = None
        self.unreachable = False
        self.retired_files = root / "source-files"
        self.retired_files.write_text("retained")

    def inspect(self):
        if self.unreachable:
            raise HostUnreachable("source unreachable")
        return HostInspection(
            self.activation,
            tuple(sorted(self.running)),
            tuple(sorted(self.running)),
        )

    def record(self, state):
        if self.unreachable:
            raise HostUnreachable("source unreachable")
        self.records.append(json.loads(json.dumps(state)))

    def stop(self, prior, *, timeout_seconds):
        assert set(prior) == set(AGENTS)
        self.activation = "suspended"
        self.running.clear()
        self.faults.after("stop")

    def capture(self, migration_id, started_at):
        if self.evidence is None:
            path = self.root / "snapshot" / "snap-1"
            path.mkdir(parents=True)
            archive = path / "data.tar.gz"
            archive.write_bytes(b"state")
            manifest = {
                "snapshot_id": "snap-1",
                "archive": {
                    "name": "data.tar.gz",
                    "sha256": "archive-sha",
                    "size": 5,
                },
            }
            manifest_path = path / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            import hashlib
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            self.evidence = SnapshotEvidence(
                SnapshotResult("snap-1", path, manifest), digest, "archive-sha"
            )
        self.faults.after("capture")
        return self.evidence

    def upload(self, snapshot):
        self.backups.register(snapshot)
        self.backups.uploaded = True
        self.faults.after("upload")
        return self.backups.get_manifest(snapshot.result.snapshot_id)

    def retire(self, migration_id):
        if self.unreachable:
            raise HostUnreachable("source unreachable")
        self.activation = "retired"
        self.running.clear()
        self.faults.after("retire")

    def recover(self, services, migration_id):
        assert self.activation != "retired"
        self.activation = "active"
        self.running = set(services)

    def reboot(self):
        if self.activation == "active":
            self.running = set(AGENTS)
        else:
            self.running.clear()


class Target:
    agent_ids = AGENTS

    def __init__(self, source, faults):
        self.source = source
        self.faults = faults
        self.activation = "absent"
        self.running = set()
        self.ready = set()
        self.restored = None
        self.intent = False
        self.preloaded = False
        self.records = []
        self.unknown = False
        self.fail_start = False

    def inspect(self):
        if self.unknown:
            raise UnknownActivation("target state unknown")
        return HostInspection(
            self.activation,
            tuple(sorted(self.running)),
            tuple(sorted(self.ready)),
            self.restored,
            self.intent,
        )

    def record(self, state):
        self.records.append(json.loads(json.dumps(state)))
        if state["phase"] in ("target-activation-intent", "target-starting", "completed"):
            self.intent = True
        self.faults.after(f"{state['phase']}-record")

    def preload(self, artifacts):
        assert self.activation in ("absent", "suspended") and not self.running
        self.preloaded = True
        self.faults.after("preload")

    def restore(self, downloaded):
        assert self.preloaded
        self.activation = "suspended"
        self.restored = downloaded.snapshot_id
        self.faults.after("restore")

    def start(self):
        assert self.intent
        assert self.source.activation in ("retired",) or self.source.unreachable
        assert not self.source.running
        if self.fail_start:
            raise RuntimeError("destination startup failed")
        self.activation = "active"
        self.running = set(AGENTS)
        self.ready = set(AGENTS)
        self.faults.after("start")


def coordinator(tmp_path, fault=None):
    faults = Faults(fault)
    backups = Backups(tmp_path / "objects", faults)
    source = Source(tmp_path, backups, faults)
    target = Target(source, faults)
    provision = Provision(faults)
    provision.state_path = tmp_path / "provision.json"
    migration = MigrationCoordinator(
        deployment_id="flywheel",
        agent_ids=AGENTS,
        source=source,
        provisioner=provision,
        target_factory=lambda _: target,
        backups=backups,
        state_path=tmp_path / "migration.json",
        workspace=tmp_path / "work",
        secrets={},
        timeout_seconds=5,
    )
    return migration, source, target, provision, backups, faults


def test_complete_handoff_pins_snapshot_retires_source_and_reports_resources(tmp_path):
    migration, source, target, provision, _, _ = coordinator(tmp_path)
    result = migration.migrate()

    assert result.phase == "completed"
    assert result.droplet_id == 4242
    assert result.snapshot_id == "snap-1"
    assert len(result.snapshot_manifest_sha256) == 64
    assert result.source_activation == "retired"
    assert result.target_activation == "active"
    assert set(result.target_ready_services) == set(AGENTS)
    assert result.remaining_billable_resources == (
        "droplet:4242", "source-droplet:41"
    )
    assert provision.calls == 1
    assert source.retired_files.read_text() == "retained"
    assert source.running == set()
    assert target.running == set(AGENTS)
    phases = [item["phase"] for item in source.records]
    assert "source-stop-intent" in phases
    assert "snapshot-uploaded" in phases
    assert "source-retired" in phases
    assert any(item["phase"] == "target-activation-intent" for item in target.records)


@pytest.mark.parametrize(
    "boundary",
    ["provision", "preload", "stop", "capture", "upload", "restore", "retire", "start"],
)
def test_controller_loss_after_each_handoff_boundary_resumes_without_overlap(
    tmp_path, boundary
):
    migration, source, target, provision, _, faults = coordinator(tmp_path, boundary)
    with pytest.raises(TimeoutError, match=boundary):
        migration.migrate()
    if boundary == "provision":
        failed = migration.status()
        assert failed.phase == "initialized"
        assert failed.droplet_id == 4242
        assert failed.remaining_billable_resources == (
            "droplet:4242", "source-droplet:41"
        )

    result = migration.migrate()
    assert result.phase == "completed"
    assert provision.calls <= 2
    assert source.activation == "retired"
    assert not source.running
    assert target.running == set(AGENTS)
    assert faults.fired


def test_source_reboot_after_stop_cannot_restart_suspended_agents(tmp_path):
    migration, source, target, _, _, _ = coordinator(tmp_path, "upload")
    with pytest.raises(TimeoutError):
        migration.migrate()
    assert source.activation == "suspended"
    source.reboot()
    assert source.running == set()
    assert target.running == set()

    assert migration.migrate().phase == "completed"
    source.reboot()
    assert source.activation == "retired"
    assert source.running == set()


def test_destination_startup_failure_never_recovers_stale_source(tmp_path):
    migration, source, target, _, _, _ = coordinator(tmp_path)
    target.fail_start = True
    with pytest.raises(RuntimeError, match="startup failed"):
        migration.migrate()

    assert migration.status().phase == "target-activation-intent"
    assert source.activation == "retired"
    assert not source.running
    with pytest.raises(RuntimeError, match="recovery is not allowed"):
        migration.recover_source()

    target.fail_start = False
    assert migration.migrate().phase == "completed"


def test_pre_activation_recovery_requires_proof_target_never_activated(tmp_path):
    migration, source, target, _, _, _ = coordinator(tmp_path, "upload")
    with pytest.raises(TimeoutError):
        migration.migrate()
    assert migration.recover_source().phase == "recovered"
    assert source.activation == "active"
    assert source.running == set(AGENTS)
    assert target.running == set()


def test_unknown_target_activation_blocks_source_recovery(tmp_path):
    migration, source, target, _, _, _ = coordinator(tmp_path, "upload")
    with pytest.raises(TimeoutError):
        migration.migrate()
    target.unknown = True
    with pytest.raises(UnknownActivation, match="unknown"):
        migration.recover_source()
    assert source.activation == "suspended"
    assert not source.running


def test_unreachable_source_requires_explicit_fence_before_promotion(tmp_path):
    migration, source, target, _, _, faults = coordinator(
        tmp_path, "target-restored-record"
    )
    with pytest.raises(TimeoutError):
        migration.migrate()
    assert migration.status().phase == "target-restored"
    source.unreachable = True

    with pytest.raises(HostUnreachable):
        migration.migrate()
    assert target.activation == "suspended"
    assert not target.running

    fenced = migration.fence_source("Droplet 41 powered off in control plane")
    assert fenced.phase == "source-retire-intent"
    assert migration.migrate().phase == "completed"
    assert target.activation == "active"


def test_retry_refuses_changed_pinned_snapshot_evidence(tmp_path):
    migration, source, _, _, _, _ = coordinator(tmp_path, "upload")
    with pytest.raises(TimeoutError):
        migration.migrate()
    source.evidence = replace(source.evidence, archive_sha256="different")
    with pytest.raises(ValueError, match="pinned final snapshot evidence changed"):
        migration.migrate()


def test_source_migration_journal_blocks_manual_restart_until_recovery_intent(tmp_path):
    snapshots, compose, root, _ = manager(tmp_path)
    controller = snapshots.controller
    controller.stop()
    journal = root / "control" / "migration.json"
    journal.write_text(json.dumps({"host_role": "source", "phase": "source-stopped"}))

    with pytest.raises(PermissionError, match="blocked by migration"):
        controller.start()
    assert compose.running == set()

    journal.write_text(
        json.dumps({"host_role": "source", "phase": "source-recovery-intent"})
    )
    controller.start(AGENTS)
    assert compose.running == set(AGENTS)


def test_interrupted_target_start_resumes_only_from_durable_activation_intent(tmp_path):
    snapshots, compose, root, _ = manager(tmp_path, running=())
    controller = snapshots.controller
    controller.activation.set("suspended")
    operation = controller.journal.begin("start", ())
    controller.journal.transition(operation.operation_id, "starting")
    journal = root / "control" / "migration.json"
    journal.write_text(
        json.dumps({"host_role": "target", "phase": "target-activation-intent"})
    )

    completed = controller.resume_start()

    assert completed.phase == "completed"
    assert controller.activation.read().state == "active"
    assert compose.running == set(AGENTS)
