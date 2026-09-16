from __future__ import annotations

from dataclasses import replace
import fcntl
from pathlib import Path, PurePosixPath

import pytest

from theseus.assembly import AgentSpec, InterfaceSpec, ModelSpec, PairingSpec
from theseus.deployment import DeploymentSpec, ResourceSpec
from theseus.deployment_store import (
    ContainerMount, DeploymentPaths, RUNTIME_LOCK, import_stopped_home,
)


def agent(name: str, pairing: PairingSpec | None = None) -> AgentSpec:
    return AgentSpec(
        name=name,
        constitution=f"I am {name}",
        core="auto",
        models=(ModelSpec("ollama", "test"),),
        interface=InterfaceSpec("none"),
        pairing=pairing,
    )


def deployment(**changes) -> DeploymentSpec:
    base = DeploymentSpec(
        id="flywheel-trial",
        agents={
            "fable": agent("Fable", PairingSpec("../../host/path.jsonl", "Astra")),
            "astra": agent("Astra"),
        },
        peers={"fable": "astra", "astra": "fable"},
        workspaces={"website": ("fable", "astra")},
        resources={"fable": ResourceSpec(cpus=0.5, memory_mb=256)},
        build_inputs=("agents/helpers.py",),
        secrets=("TELEGRAM_TOKEN",),
    )
    return replace(base, **changes)


def test_deployment_resolves_pair_by_id_instead_of_source_host_path():
    spec = deployment()
    spec.validate()
    resolved = spec.resolved_agent("fable")
    assert resolved.pairing == PairingSpec(
        "/peers/astra/logs/stimulus_log.jsonl", "Astra", 0.25
    )
    assert spec.resolved_agent("astra").pairing.peer_log_path == (
        "/peers/fable/logs/stimulus_log.jsonl"
    )


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"id": "Not Portable"}, "stable ID"),
        ({"peers": {"fable": "missing"}}, "unknown agent"),
        ({"peers": {"fable": "fable"}}, "own peer"),
        ({"workspaces": {"website": ("missing",)}}, "unknown agent"),
        ({"resources": {"missing": ResourceSpec()}}, "unknown agent"),
        ({"build_inputs": ("../secret",)}, "build context"),
    ],
)
def test_deployment_validation(changes, message):
    with pytest.raises(ValueError, match=message):
        deployment(**changes).validate()


def test_deployment_rejects_conflicting_agent_pair_identity():
    agents = dict(deployment().agents)
    agents["fable"] = replace(
        agents["fable"], pairing=PairingSpec("somewhere", "Someone else")
    )
    with pytest.raises(ValueError, match="deployment peer"):
        deployment(agents=agents).validate()


def test_portable_layout_and_mount_visibility(tmp_path):
    spec = deployment()
    paths = DeploymentPaths(tmp_path / spec.id, spec)
    paths.prepare()
    assert paths.releases.is_dir()
    assert paths.state("fable").is_dir()
    assert paths.logs("astra").is_dir()
    assert paths.workspace("website").is_dir()
    assert paths.control.is_dir() and paths.secrets.is_dir() and paths.snapshots.is_dir()
    mounts = paths.mounts_for("fable")
    by_container = {str(mount.container_path): mount for mount in mounts}
    assert set(by_container) == {
        "/data/state", "/data/logs", "/peers/astra/logs", "/workspaces/website",
        "/run/theseus-control", "/run/secrets/TELEGRAM_TOKEN",
    }
    assert by_container["/data/state"].host_path == paths.state("fable")
    assert by_container["/peers/astra/logs"].read_only
    assert all("agents/astra/state" not in str(mount.host_path) for mount in mounts)


def test_layout_applies_stable_private_and_shared_ownership(tmp_path, monkeypatch):
    spec = deployment(uid=12001, gid=12002, workspace_gid=12003)
    paths = DeploymentPaths(tmp_path, spec)
    paths.prepare()
    ownership = []
    monkeypatch.setattr("theseus.deployment_store.os.chown", lambda path, uid, gid: ownership.append((Path(path), uid, gid)))
    paths.apply_ownership()
    assert (paths.state("fable"), 12001, 12002) in ownership
    assert (paths.workspace("website"), 12001, 12003) in ownership
    assert paths.workspace("website").stat().st_mode & 0o2000


def test_undeclared_durable_mount_is_rejected(tmp_path):
    paths = DeploymentPaths(tmp_path, deployment())
    mount = ContainerMount(
        tmp_path / "other", PurePosixPath("/durable/other"), False, True
    )
    with pytest.raises(ValueError, match="undeclared durable"):
        paths.validate_managed_mounts("fable", [mount])


def test_stopped_import_splits_log_once_and_preserves_bytes(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    history = b'{"history":"exact"}\n'
    (source / "stimulus_log.jsonl").write_bytes(history)
    (source / "GOALS.md").write_text("Keep me")
    (source / "memory").mkdir()
    (source / "memory" / "knowledge.jsonl").write_text("knowledge\n")
    paths = DeploymentPaths(tmp_path / "portable", deployment())
    import_stopped_home(source, paths, "fable")
    assert paths.log("fable").read_bytes() == history
    assert (paths.state("fable") / "GOALS.md").read_text() == "Keep me"
    assert (paths.state("fable") / "memory" / "knowledge.jsonl").exists()
    assert not (paths.state("fable") / "stimulus_log.jsonl").exists()
    assert (source / "stimulus_log.jsonl").read_bytes() == history
    with pytest.raises(ValueError, match="occupied"):
        import_stopped_home(source, paths, "fable")


def test_stopped_import_refuses_live_writer(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    lock_path = source / RUNTIME_LOCK
    with lock_path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="live writer"):
            import_stopped_home(
                source, DeploymentPaths(tmp_path / "portable", deployment()), "fable"
            )
