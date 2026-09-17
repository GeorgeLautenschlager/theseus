from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from theseus.backup_store import LocalObjectStore, S3ObjectStore
from theseus.deployment_control import DeploymentController
from theseus.deployment_snapshot import DeploymentSnapshots
from theseus.remote_backup import REMOTE_STATUS_FILE, RemoteBackups
from test_deployment_snapshot import AGENTS, Compose, manager


class Docker:
    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            value = [{"Id": command[-1], "Os": "linux", "Architecture": "amd64"}]
            return subprocess.CompletedProcess(command, 0, json.dumps(value), "")
        if command[1:3] == ["image", "save"]:
            Path(command[command.index("--output") + 1]).write_bytes(b"exact docker image\n")
        return subprocess.CompletedProcess(command, 0, "", "")


class RecordingStore(LocalObjectStore):
    def __init__(self, root: Path):
        super().__init__(root)
        self.puts: list[str] = []
        self.fail_suffix: str | None = None

    def put_file(self, key, source, metadata):
        if self.fail_suffix and key.endswith(self.fail_suffix):
            raise OSError("object storage unavailable")
        self.puts.append(key)
        return super().put_file(key, source, metadata)

    def put_bytes(self, key, value, metadata):
        if self.fail_suffix and key.endswith(self.fail_suffix):
            raise OSError("object storage unavailable")
        self.puts.append(key)
        return super().put_bytes(key, value, metadata)


def prepared(tmp_path):
    snapshots, _, root, bundle = manager(tmp_path)
    captured = snapshots.create(leave_stopped=True)
    store = RecordingStore(tmp_path / "objects")
    docker = Docker()
    remote = RemoteBackups(store, bundle, run=docker)
    return remote, store, docker, captured, root, bundle


def test_completion_manifest_is_published_last_and_list_is_manifest_driven(tmp_path):
    remote, store, _, captured, _, _ = prepared(tmp_path)
    result = remote.upload(captured.path)

    assert result.snapshot_id == captured.snapshot_id
    assert result.total_size > 0
    assert store.puts[-1].endswith("/manifest.json")
    assert [item["snapshot_id"] for item in remote.list()] == [captured.snapshot_id]
    status = json.loads((captured.path / REMOTE_STATUS_FILE).read_text())
    assert status["state"] == "completed"
    assert status["last_successful_snapshot_id"] == captured.snapshot_id


def test_partial_upload_is_invisible_and_retry_reuses_staged_snapshot(tmp_path):
    remote, store, docker, captured, _, _ = prepared(tmp_path)
    store.fail_suffix = "/data.tar.gz"
    with pytest.raises(OSError, match="unavailable"):
        remote.upload(captured.path)

    assert remote.list() == []
    failed = json.loads((captured.path / REMOTE_STATUS_FILE).read_text())
    assert failed["state"] == "failed"
    assert "storage unavailable" in failed["failure"]
    first_puts = list(store.puts)
    inspect_count = sum(command[1:3] == ["image", "inspect"] for command in docker.commands)

    store.fail_suffix = None
    result = remote.upload(captured.path)
    assert result.snapshot_id == captured.snapshot_id
    assert store.puts.count(first_puts[0]) == 1
    assert sum(command[1:3] == ["image", "save"] for command in docker.commands) == 1
    assert sum(command[1:3] == ["image", "inspect"] for command in docker.commands) == inspect_count + 1


def test_previous_completed_backup_survives_later_failed_upload(tmp_path):
    remote, store, _, captured, root, _ = prepared(tmp_path)
    remote.upload(captured.path)
    original = remote.list()[0]["snapshot_id"]

    # A copied local snapshot with a fresh valid ID models a later completed capture.
    import uuid
    newer_id = uuid.uuid4().hex
    newer = root / "snapshots" / newer_id
    shutil.copytree(captured.path, newer)
    manifest_path = newer / "manifest.json"
    value = json.loads(manifest_path.read_text())
    value["snapshot_id"] = newer_id
    value["captured_at"] = "2099-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(value))
    store.fail_suffix = "/data.tar.gz"
    with pytest.raises(OSError):
        remote.upload(newer)

    assert [item["snapshot_id"] for item in remote.list()] == [original]
    status = json.loads((newer / REMOTE_STATUS_FILE).read_text())
    assert status["last_successful_snapshot_id"] == original


def test_completed_upload_is_idempotent_without_new_writes(tmp_path):
    remote, store, _, captured, _, _ = prepared(tmp_path)
    first = remote.upload(captured.path)
    writes = list(store.puts)
    second = remote.upload(captured.path)
    assert second == first
    assert store.puts == writes


def test_corrupt_download_and_missing_image_are_rejected(tmp_path):
    remote, store, _, captured, _, _ = prepared(tmp_path)
    remote.upload(captured.path)
    manifest = remote.get_manifest(captured.snapshot_id)
    data_key = manifest["objects"]["data"]["key"]
    store._path(data_key).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="failed verification"):
        remote.download(captured.snapshot_id, tmp_path / "bad-download")

    # Restore data, then remove the exact image object referenced by the manifest.
    shutil.copyfile(captured.path / "data.tar.gz", store._path(data_key))
    image_key = manifest["objects"]["images"][0]["key"]
    store._path(image_key).unlink()
    with pytest.raises(FileNotFoundError, match="object does not exist"):
        remote.download(captured.snapshot_id, tmp_path / "missing-image")


def test_full_fake_store_download_load_and_inactive_restore(tmp_path):
    remote, _, docker, captured, _, _ = prepared(tmp_path)
    remote.upload(captured.path)
    downloaded = remote.download(captured.snapshot_id, tmp_path / "download")
    remote.load_images(downloaded)

    target = tmp_path / "target"
    (target / "control").mkdir(parents=True)
    (target / "secrets").mkdir()
    (target / "secrets" / "TELEGRAM_TOKEN").write_text("target secret")
    compose = Compose(target, ())
    controller = DeploymentController(
        root=target,
        deployment_id="flywheel",
        agent_ids=AGENTS,
        compose_file=downloaded.bundle / "compose.yaml",
        run=compose,
    )
    restored = DeploymentSnapshots(controller, downloaded.bundle).restore(downloaded.snapshot)

    assert (restored / "workspaces" / "website" / "index.html").read_text() == (
        "<h1>shared artifact</h1>\n"
    )
    assert any(command[1:3] == ["image", "load"] for command in docker.commands)
    assert compose.running == set()


def test_s3_adapter_uses_auto_region_and_standard_boto_credentials(monkeypatch):
    called = {}

    def client(*args, **kwargs):
        called.update(args=args, kwargs=kwargs)
        return object()

    monkeypatch.setattr("theseus.backup_store.boto3.client", client)
    S3ObjectStore(endpoint_url="https://account.r2.cloudflarestorage.com", bucket="backups")
    assert called == {
        "args": ("s3",),
        "kwargs": {
            "endpoint_url": "https://account.r2.cloudflarestorage.com",
            "region_name": "auto",
        },
    }
