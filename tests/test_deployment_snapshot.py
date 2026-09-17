from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tarfile

import pytest

from theseus.deployment_control import ActivationStore, DeploymentController, LifecycleStatusStore
from theseus.deployment_snapshot import (
    ARCHIVE_FILE,
    DeploymentSnapshots,
    MANIFEST_FILE,
    SNAPSHOT_FORMAT_VERSION,
)
from theseus.memory_module import MemoryModule
from theseus.manage_deployment import _controller
from theseus.stimulus_log import StimulusLog


AGENTS = ("fable", "astra")


class Compose:
    def __init__(self, root: Path, running=AGENTS):
        self.root = root
        self.running = set(running)
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if "ps" in command:
            output = "".join(f"{name}\n" for name in sorted(self.running))
        elif "stop" in command:
            index = command.index("--timeout") + 2
            for name in command[index:]:
                status = self.root / "data" / "agents" / name / "logs" / "lifecycle-status.json"
                store = LifecycleStatusStore(status, name)
                current = store.read()
                assert current is not None
                store.stopped(current.run_id, clean=True)
                self.running.discard(name)
            output = ""
        elif "up" in command:
            index = command.index("-d") + 1
            self.running.update(command[index:])
            output = ""
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")


def make_bundle(path: Path, *, secret_names=("TELEGRAM_TOKEN",)) -> Path:
    path.mkdir()
    deployment = {
        "_generated_by": "theseus-compose-assembler",
        "format_version": 1,
        "deployment_id": "flywheel",
        "platform": "linux/amd64",
        "runtime": "base",
        "resolved_spec_sha256": "spec-sha",
        "required_secrets": list(secret_names),
        "theseus": {"version": "1.0", "commit": "abc", "source_sha256": "source-sha"},
        "agents": {name: {"spec": {"name": name}} for name in AGENTS},
    }
    release = {
        "_generated_by": "theseus-compose-builder",
        "format_version": 1,
        "deployment_id": "flywheel",
        "resolved_spec_sha256": "spec-sha",
        "platform": "linux/amd64",
        "images": [{"tag": "theseus-flywheel:fixed", "content_id": "sha256:image"}],
        "theseus": deployment["theseus"],
    }
    (path / "deployment.json").write_text(json.dumps(deployment))
    (path / "deployment.lock.json").write_text(json.dumps(release))
    (path / "compose.yaml").write_text("services: {}\n")
    return path


def populate(root: Path) -> None:
    for name in AGENTS:
        state = root / "data" / "agents" / name / "state"
        logs = root / "data" / "agents" / name / "logs"
        memory = state / "memory"
        (memory / "formation").mkdir(parents=True)
        logs.mkdir(parents=True)
        (state / "GOALS.md").write_text(f"Goals for {name}\n")
        (state / "TASKS.md").write_text("- pending\n")
        (state / "SCHEDULE.md").write_text("# schedule\n")
        (logs / "stimulus_log.jsonl").write_text(
            json.dumps({"id": f"{name}-event", "actor": "user"}) + "\n"
        )
        pending = {
            "episode_id": f"{name}-episode",
            "start_id": f"{name}-event",
            "end_id": f"{name}-event",
            "records": {"knowledge": [], "memory": [], "wisdom": []},
            "dead_letters": [],
            "trace": {"episode_id": f"{name}-episode"},
        }
        (memory / "pending.json").write_text(json.dumps(pending))
        (memory / "consolidation_ledger.jsonl").write_text("")
        (memory / "embeddings.jsonl").write_text(
            json.dumps({"model": "fake", "layer": "memory", "id": "sidecar", "vector": [1.0]}) + "\n"
        )
        (memory / "formation" / "cursor.json").write_text(
            json.dumps({"pending": {"episode_id": f"{name}-episode"}})
        )
        database = state / "delivery.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE delivery (id INTEGER PRIMARY KEY, body TEXT)")
            connection.execute("INSERT INTO delivery(body) VALUES ('pending')")
            connection.commit()
        finally:
            connection.close()
        (state / "delivery.sqlite3-wal").touch()
        LifecycleStatusStore(logs / "lifecycle-status.json", name).running()
    workspace = root / "data" / "workspaces" / "website"
    workspace.mkdir(parents=True)
    (workspace / "index.html").write_text("<h1>shared artifact</h1>\n")
    (workspace / "index.html").chmod(0o640)
    (workspace / "current.html").symlink_to("index.html")
    (workspace / "index-hardlink.html").hardlink_to(workspace / "index.html")
    (root / "control").mkdir()
    (root / "secrets").mkdir()
    (root / "snapshots").mkdir()
    (root / "secrets" / "TELEGRAM_TOKEN").write_text("super-secret-value")


def manager(tmp_path, *, running=AGENTS):
    root = tmp_path / "source"
    populate(root)
    bundle = make_bundle(tmp_path / "bundle")
    compose = Compose(root, running)
    controller = DeploymentController(
        root=root,
        deployment_id="flywheel",
        agent_ids=AGENTS,
        compose_file=bundle / "compose.yaml",
        run=compose,
    )
    ActivationStore(root / "control", "flywheel").set("active")
    return DeploymentSnapshots(controller, bundle), compose, root, bundle


def test_backup_captures_whole_pair_then_resumes_only_prior_services(tmp_path):
    snapshots, compose, root, _ = manager(tmp_path)
    original_archive = snapshots._archive

    def archive_after_resume(data, archive):
        assert compose.running == set(AGENTS)
        original_archive(data, archive)

    snapshots._archive = archive_after_resume
    result = snapshots.create()

    assert compose.running == set(AGENTS)
    assert result.path == root / "snapshots" / result.snapshot_id
    assert (result.path / "data" / "agents" / "fable" / "state" / "memory" / "pending.json").exists()
    assert (result.path / "data" / "agents" / "astra" / "state" / "delivery.sqlite3-wal").exists()
    assert (result.path / "data" / "workspaces" / "website" / "index.html").exists()
    assert (result.path / "data" / "workspaces" / "website" / "current.html").is_symlink()
    captured_workspace = result.path / "data" / "workspaces" / "website"
    assert (captured_workspace / "index.html").stat().st_ino == (
        captured_workspace / "index-hardlink.html"
    ).stat().st_ino
    assert not (result.path / "control").exists()
    assert not (result.path / "secrets").exists()
    assert not (result.path / "data" / "snapshots").exists()
    serialized = json.dumps(result.manifest)
    assert "TELEGRAM_TOKEN" in serialized
    assert "super-secret-value" not in serialized
    assert result.manifest["snapshot_format_version"] == SNAPSHOT_FORMAT_VERSION
    assert result.manifest["consistency"] == "clean-stopped"
    assert result.manifest["resolved_spec_sha256"] == "spec-sha"
    assert result.manifest["images"][0]["content_id"] == "sha256:image"
    source_uid = (root / "data" / "workspaces" / "website" / "index.html").stat().st_uid
    captured_entry = next(
        entry for entry in result.manifest["files"]
        if entry["path"] == "workspaces/website/index.html"
    )
    assert captured_entry["uid"] == source_uid
    assert snapshots.controller.journal.read().phase == "completed"


def test_leave_stopped_keeps_maintenance_and_does_not_resume(tmp_path):
    snapshots, compose, root, _ = manager(tmp_path, running=("fable",))
    snapshots.create(leave_stopped=True)
    assert compose.running == set()
    assert not any("up" in command for command in compose.commands)
    assert ActivationStore(root / "control", "flywheel").read().state == "suspended"


def test_ordinary_backup_resumes_only_the_service_that_was_running(tmp_path):
    snapshots, compose, _, _ = manager(tmp_path, running=("fable",))
    snapshots.create()
    assert compose.running == {"fable"}
    up = next(command for command in compose.commands if "up" in command)
    assert up[up.index("-d") + 1:] == ["fable"]


def test_copy_failure_preserves_source_and_never_restarts_from_finally(tmp_path, monkeypatch):
    snapshots, compose, root, _ = manager(tmp_path)
    before = (root / "data" / "workspaces" / "website" / "index.html").read_bytes()

    def fail_copy(*args, **kwargs):
        raise OSError("copy interrupted")

    monkeypatch.setattr("theseus.deployment_snapshot.shutil.copytree", fail_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        snapshots.create()
    assert (root / "data" / "workspaces" / "website" / "index.html").read_bytes() == before
    assert compose.running == set()
    assert not any("up" in command for command in compose.commands)
    operation = snapshots.controller.journal.read()
    assert operation.phase == "failed" and set(operation.prior_running_services) == set(AGENTS)


def test_validation_uses_scratch_and_preserves_original_evidence(tmp_path, monkeypatch):
    snapshots, _, _, _ = manager(tmp_path)

    def recover_in_scratch(data):
        (data / "recovery-write").write_text("scratch only")

    monkeypatch.setattr("theseus.deployment_snapshot.validate_state_copy", recover_in_scratch)
    result = snapshots.create()
    assert not (result.path / "data" / "recovery-write").exists()
    assert not any(entry["path"] == "recovery-write" for entry in result.manifest["files"])


def test_corrupt_staged_state_is_rejected_after_source_resumes(tmp_path):
    snapshots, compose, root, _ = manager(tmp_path)
    (root / "data" / "agents" / "fable" / "logs" / "broken.jsonl").write_text("{bad\n")
    with pytest.raises(ValueError, match="invalid JSONL"):
        snapshots.create()
    assert compose.running == set(AGENTS)
    operation = snapshots.controller.journal.read()
    assert operation.phase == "failed"
    detail = json.loads(operation.detail)
    evidence = root / "snapshots" / detail["snapshot_id"]
    assert evidence.is_dir()
    assert not (evidence / MANIFEST_FILE).exists()
    assert (evidence / "data" / "agents" / "fable" / "logs" / "broken.jsonl").exists()


def test_round_trip_restores_pending_state_to_different_inactive_root(tmp_path):
    snapshots, _, _, bundle = manager(tmp_path)
    result = snapshots.create(leave_stopped=True)

    target = tmp_path / "target"
    (target / "control").mkdir(parents=True)
    (target / "secrets").mkdir()
    (target / "secrets" / "TELEGRAM_TOKEN").write_text("different-target-value")
    target_compose = Compose(target, ())
    target_controller = DeploymentController(
        root=target,
        deployment_id="flywheel",
        agent_ids=AGENTS,
        compose_file=bundle / "compose.yaml",
        run=target_compose,
    )
    restored = DeploymentSnapshots(target_controller, bundle).restore(result.path)

    assert restored == target / "data"
    assert target_compose.running == set()
    assert not any("up" in command for command in target_compose.commands)
    assert ActivationStore(target / "control", "flywheel").read().state == "suspended"
    assert (restored / "workspaces" / "website" / "index.html").read_text() == "<h1>shared artifact</h1>\n"
    assert (restored / "workspaces" / "website" / "current.html").is_symlink()
    assert ((restored / "workspaces" / "website" / "index.html").stat().st_mode & 0o777) == 0o640
    assert (restored / "workspaces" / "website" / "index.html").stat().st_ino == (
        restored / "workspaces" / "website" / "index-hardlink.html"
    ).stat().st_ino

    memory_dir = restored / "agents" / "fable" / "state" / "memory"
    log = StimulusLog(restored / "agents" / "fable" / "logs" / "stimulus_log.jsonl")
    memory = MemoryModule(memory_dir, log)
    assert not (memory_dir / "pending.json").exists()
    assert "fable-episode" in memory._processed_episodes
    with sqlite3.connect(restored / "agents" / "fable" / "state" / "delivery.sqlite3") as db:
        assert db.execute("SELECT body FROM delivery").fetchone()[0] == "pending"


def test_restore_refuses_active_nonempty_or_unprovisioned_target(tmp_path):
    snapshots, _, _, bundle = manager(tmp_path)
    result = snapshots.create(leave_stopped=True)

    target = tmp_path / "target"
    (target / "control").mkdir(parents=True)
    (target / "secrets").mkdir()
    controller = DeploymentController(
        root=target, deployment_id="flywheel", agent_ids=AGENTS,
        compose_file=bundle / "compose.yaml", run=Compose(target, ()),
    )
    restore = DeploymentSnapshots(controller, bundle)
    with pytest.raises(ValueError, match="secrets are missing"):
        restore.restore(result.path)

    (target / "secrets" / "TELEGRAM_TOKEN").write_text("target")
    ActivationStore(target / "control", "flywheel").set("active")
    with pytest.raises(RuntimeError, match="active"):
        restore.restore(result.path)

    ActivationStore(target / "control", "flywheel").set("suspended")
    (target / "data").mkdir()
    (target / "data" / "occupied").write_text("keep")
    with pytest.raises(RuntimeError, match="not empty"):
        restore.restore(result.path)
    assert (target / "data" / "occupied").read_text() == "keep"


def test_interrupted_restore_keeps_target_unpublished_and_snapshot_intact(tmp_path):
    snapshots, _, _, bundle = manager(tmp_path)
    result = snapshots.create(leave_stopped=True)
    archive_hash = result.manifest["archive"]["sha256"]
    target = tmp_path / "target"
    (target / "control").mkdir(parents=True)
    (target / "secrets").mkdir()
    (target / "secrets" / "TELEGRAM_TOKEN").write_text("target")
    controller = DeploymentController(
        root=target, deployment_id="flywheel", agent_ids=AGENTS,
        compose_file=bundle / "compose.yaml", run=Compose(target, ()),
    )
    restore = DeploymentSnapshots(controller, bundle)

    def interrupt(_archive, destination, **_kwargs):
        (destination / "partial").write_text("incomplete")
        raise OSError("extract interrupted")

    restore._extract_safe = interrupt
    with pytest.raises(OSError, match="extract interrupted"):
        restore.restore(result.path)
    assert not (target / "data").exists()
    assert controller.journal.read().phase == "failed"
    assert ActivationStore(target / "control", "flywheel").read().state == "suspended"
    assert result.manifest["archive"]["sha256"] == archive_hash
    assert (result.path / ARCHIVE_FILE).is_file()


def test_restore_checks_release_platform_hashes_and_free_space(tmp_path):
    snapshots, _, _, bundle = manager(tmp_path)
    result = snapshots.create(leave_stopped=True)
    manifest_path = result.path / MANIFEST_FILE
    original = json.loads(manifest_path.read_text())

    changed = dict(original, platform="linux/arm64")
    manifest_path.write_text(json.dumps(changed))
    target = tmp_path / "target"
    (target / "control").mkdir(parents=True)
    (target / "secrets").mkdir()
    (target / "secrets" / "TELEGRAM_TOKEN").write_text("target")
    controller = DeploymentController(
        root=target, deployment_id="flywheel", agent_ids=AGENTS,
        compose_file=bundle / "compose.yaml", run=Compose(target, ()),
    )
    with pytest.raises(ValueError, match="platform"):
        DeploymentSnapshots(controller, bundle).restore(result.path)

    manifest_path.write_text(json.dumps(original))
    (result.path / ARCHIVE_FILE).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        DeploymentSnapshots(controller, bundle).restore(result.path)

    # Restore the archive via a fresh snapshot, then force the disk-space guard.
    second = snapshots.create(leave_stopped=True)
    usage = shutil.disk_usage(target)

    def no_space(_path):
        return usage.__class__(usage.total, usage.used, 0)

    with pytest.raises(OSError, match="free space"):
        DeploymentSnapshots(controller, bundle, disk_usage=no_space).restore(second.path)


def test_safe_extractor_rejects_traversal_devices_and_escaping_links(tmp_path):
    cases = [
        ("../escape", b"bad", None),
        ("data/device", b"", "device"),
        ("data/link", b"", "symlink"),
    ]
    for index, (name, payload, kind) in enumerate(cases):
        archive = tmp_path / f"unsafe-{index}.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            member = tarfile.TarInfo(name)
            if kind == "device":
                member.type = tarfile.CHRTYPE
                stream.addfile(member)
            elif kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside"
                stream.addfile(member)
            else:
                member.size = len(payload)
                stream.addfile(member, io.BytesIO(payload))
        destination = tmp_path / f"extract-{index}"
        destination.mkdir()
        with pytest.raises(ValueError, match="unsafe|device|escapes"):
            DeploymentSnapshots._extract_safe(archive, destination)
        assert not (tmp_path / "escape").exists()


def test_operator_controller_reads_current_manifest_field(tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    controller = _controller(bundle, tmp_path / "root")
    assert controller.deployment_id == "flywheel"
    assert controller.agent_ids == AGENTS
