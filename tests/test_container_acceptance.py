"""Opt-in release gate for the real paired Compose/backup/handoff path.

Run with ``THESEUS_RUN_CONTAINER_ACCEPTANCE=1``. Ordinary offline pytest skips
this module because it builds an image and requires a local Docker daemon.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from uuid import uuid4

import pytest

from theseus.assembly import AgentSpec, InterfaceSpec, MemorySpec, ModelSpec
from theseus.backup_store import LocalObjectStore
from theseus.deployment import DeploymentSpec
from theseus.deployment_bundle import assemble_compose, build_bundle
from theseus.deployment_control import DeploymentController
from theseus.deployment_snapshot import DeploymentSnapshots
from theseus.deployment_store import DeploymentPaths
from theseus.durable_delivery import DeliveryJournal
from theseus.knowledge_layer import KnowledgeRecord
from theseus.memory_module import MemoryModule
from theseus.remote_backup import RemoteBackups
from theseus.stimulus_log import StimulusLog


pytestmark = pytest.mark.skipif(
    not shutil.which("docker")
    or os.environ.get("THESEUS_RUN_CONTAINER_ACCEPTANCE") != "1",
    reason="set THESEUS_RUN_CONTAINER_ACCEPTANCE=1 to run the Docker release gate",
)

AGENTS = ("astra", "fable")
ROOT = Path(__file__).resolve().parents[1]


def _run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs)


def _project_runner(project: str):
    def run(command, **kwargs):
        command = list(command)
        if command[:2] == ["docker", "compose"]:
            command[2:2] = ["--project-name", project]
        return subprocess.run(command, **kwargs)

    return run


def _wait(message: str, predicate, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {message}")


def _spec(
    *,
    uid: int | None = None,
    gid: int | None = None,
    workspace_gid: int | None = None,
) -> DeploymentSpec:
    def agent(name: str) -> AgentSpec:
        return AgentSpec(
            name=name,
            constitution=f"You are {name}, an acceptance fixture.",
            core="auto",
            models=(ModelSpec("fixture", "quiet", tick=3600),),
            memory=MemorySpec("module"),
            interface=InterfaceSpec(
                "telegram",
                bot_token_env=f"{name.upper()}_TELEGRAM_TOKEN",
                allowed_user_ids=(1,),
                poll_timeout_seconds=1,
            ),
        )

    return DeploymentSpec(
        id="paired-acceptance",
        agents={"astra": agent("Astra"), "fable": agent("Fable")},
        peers={"astra": "fable", "fable": "astra"},
        workspaces={"website": AGENTS},
        secrets=(
            "ASTRA_TELEGRAM_TOKEN",
            "FABLE_TELEGRAM_TOKEN",
            "TELEGRAM_API_BASE_URL",
            "THESEUS_ENABLE_FIXTURE_PROVIDER",
            "THESEUS_FIXTURE_PROVIDER_LOG",
        ),
        uid=os.getuid() if uid is None else uid,
        gid=os.getgid() if gid is None else gid,
        workspace_gid=os.getgid() + 10000 if workspace_gid is None else workspace_gid,
    )


def _secrets(paths: DeploymentPaths, telegram_url: str) -> None:
    values = {
        "ASTRA_TELEGRAM_TOKEN": "astra-fixture-token",
        "FABLE_TELEGRAM_TOKEN": "fable-fixture-token",
        "TELEGRAM_API_BASE_URL": telegram_url,
        "THESEUS_ENABLE_FIXTURE_PROVIDER": "1",
        "THESEUS_FIXTURE_PROVIDER_LOG": "/data/state/fixture-provider-calls.jsonl",
    }
    for name, value in values.items():
        destination = paths.secrets / name
        destination.write_text(value + "\n")
        destination.chmod(0o600)


def _prepare(paths: DeploymentPaths, telegram_url: str) -> None:
    paths.prepare()
    paths.workspace("website").chmod(0o2770)
    _secrets(paths, telegram_url)


def _prepare_empty_target(paths: DeploymentPaths, telegram_url: str) -> None:
    for path in (paths.releases, paths.control, paths.secrets, paths.snapshots):
        path.mkdir(parents=True, exist_ok=True)
    _secrets(paths, telegram_url)


def _controller(root: Path, bundle: Path, project: str) -> DeploymentController:
    return DeploymentController(
        root=root,
        deployment_id="paired-acceptance",
        agent_ids=AGENTS,
        compose_file=bundle / "compose.yaml",
        run=_project_runner(project),
    )


def _calls(root: Path, agent_id: str) -> list[dict[str, str]]:
    path = root / "data" / "agents" / agent_id / "state" / "fixture-provider-calls.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _telegram_calls(root: Path) -> list[dict[str, str]]:
    path = root / "requests.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _ready(controller: DeploymentController, expected=AGENTS) -> bool:
    if set(controller.running_services()) != set(expected):
        return False
    return all(
        (record := controller._lifecycle(agent_id)) is not None
        and record.get("state") == "running"
        and record.get("clean") is None
        for agent_id in expected
    )


def _seed_recovery_state(root: Path, agent_id: str) -> None:
    state = root / "data" / "agents" / agent_id / "state"
    log = StimulusLog(root / "data" / "agents" / agent_id / "logs" / "stimulus_log.jsonl")
    memory = MemoryModule(state / "memory", log)
    memory.knowledge.add(
        KnowledgeRecord(
            id=f"{agent_id}-known",
            ts=datetime.now(timezone.utc),
            subject="Atlas",
            predicate="delivery code",
            value="RAVEN-42",
            source_episode_id=f"{agent_id}-known-episode",
        )
    )
    pending = KnowledgeRecord(
        id=f"{agent_id}-pending",
        ts=datetime.now(timezone.utc),
        subject="Migration",
        predicate="state",
        value="recoverable",
        source_episode_id=f"{agent_id}-pending-episode",
    )
    memory_dir = state / "memory"
    (memory_dir / "pending.json").write_text(json.dumps({
        "episode_id": f"{agent_id}-pending-episode",
        "start_id": f"{agent_id}-start",
        "end_id": f"{agent_id}-end",
        "records": {"knowledge": [pending.to_json()], "memory": [], "wisdom": []},
        "dead_letters": [],
        "trace": {"episode_id": f"{agent_id}-pending-episode"},
    }))
    formation = memory_dir / "formation"
    formation.mkdir(exist_ok=True)
    (formation / "cursor.json").write_text(json.dumps({
        "pending": {
            "episode_id": f"{agent_id}-pending-episode",
            "start_id": f"{agent_id}-start",
            "end_id": f"{agent_id}-end",
        }
    }))
    (state / "GOALS.md").write_text(f"Preserve {agent_id} goals\n")
    database = state / "delivery.sqlite3"
    DeliveryJournal(database).enqueue_outgoing(
        "telegram",
        "1",
        [{"kind": "text", "text": f"recorded-{agent_id}"}],
        now=time.time(),
    )


def _down(bundle: Path, root: Path, project: str) -> None:
    env = {**os.environ, "THESEUS_DEPLOYMENT_ROOT": str(root)}
    subprocess.run(
        [
            "docker", "compose", "--project-name", project,
            "-f", str(bundle / "compose.yaml"), "down", "--remove-orphans",
        ],
        text=True,
        capture_output=True,
        env=env,
    )


def test_paired_container_backup_restore_and_single_active_handoff(tmp_path):
    suffix = uuid4().hex[:8]
    source_project = f"theseus-source-{suffix}"
    target_project = f"theseus-target-{suffix}"
    telegram_name = f"theseus-telegram-{suffix}"
    telegram_url = f"http://{telegram_name}:8080"
    spec = _spec()
    bundle = assemble_compose(spec, tmp_path / "bundle", definition_root=tmp_path)
    lock = json.loads(build_bundle(bundle).read_text())
    image_tag = lock["images"][0]["tag"]
    image_id = lock["images"][0]["content_id"]
    source = DeploymentPaths(tmp_path / "source-host", spec)
    target = DeploymentPaths(tmp_path / "target-host", spec)
    _prepare(source, telegram_url)
    source_controller = _controller(source.root, bundle, source_project)
    target_bundle = bundle
    telegram_calls = tmp_path / "telegram-calls"
    telegram_calls.mkdir()

    try:
        source_controller._compose("create")
        _run([
            "docker", "run", "-d", "--name", telegram_name,
            "--network", f"{source_project}_default",
            "--entrypoint", "python",
            "-v", f"{telegram_calls}:/calls",
            "-v", f"{ROOT / 'scripts' / 'fake_telegram_api.py'}:/fixture.py:ro",
            image_tag, "/fixture.py", "--host", "0.0.0.0",
            "--log", "/calls/requests.jsonl",
        ])
        # A real preflight container validates state but cannot call the provider.
        source_controller._preflight_service("astra")
        assert _calls(source.root, "astra") == []
        assert _telegram_calls(telegram_calls) == []

        source_controller.start()
        _wait(
            "both source agents",
            lambda: _ready(source_controller),
        )
        _wait(
            "fixture inference calls",
            lambda: all(_calls(source.root, agent_id) for agent_id in AGENTS),
        )
        _wait(
            "fake Telegram polls",
            lambda: {
                item["bot_sha256"]
                for item in _telegram_calls(telegram_calls)
            } == {
                hashlib.sha256(token.encode()).hexdigest()
                for token in ("astra-fixture-token", "fable-fixture-token")
            },
        )

        # Exercise the generated real mounts from inside the non-root agent.
        probe = """
from pathlib import Path
Path('/data/state/own-write.txt').write_text('astra private')
peer = Path('/peers/fable/logs/stimulus_log.jsonl')
assert peer.is_file() and peer.read_text()
assert not Path('/peers/fable/state').exists()
try:
    (peer.parent / 'forbidden').write_text('no')
except OSError:
    pass
else:
    raise AssertionError('peer log mount is writable')
Path('/workspaces/website/astra.txt').write_text('shared artifact')
"""
        source_controller._compose("exec", "-T", "astra", "python", "-c", probe)
        assert (source.state("astra") / "own-write.txt").read_text() == "astra private"
        assert (source.workspace("website") / "astra.txt").read_text() == "shared artifact"

        # Remove and recreate one real container; bind-mounted state must survive.
        source_controller.stop(timeout_seconds=10)
        source_controller._compose("rm", "-f", "astra")
        source_controller.start(("astra",))
        _wait("recreated Astra", lambda: _ready(source_controller, ("astra",)))
        source_controller._compose(
            "exec", "-T", "astra", "python", "-c",
            "from pathlib import Path; "
            "assert Path('/data/state/own-write.txt').read_text() == 'astra private'; "
            "assert Path('/workspaces/website/astra.txt').read_text() == 'shared artifact'",
        )
        source_controller.start(("fable",))
        _wait(
            "restarted pair",
            lambda: _ready(source_controller),
        )
        source_controller.stop(timeout_seconds=10)

        for agent_id in AGENTS:
            _seed_recovery_state(source.root, agent_id)
        snapshot = DeploymentSnapshots(source_controller, bundle).create(leave_stopped=True)

        backups = RemoteBackups(LocalObjectStore(tmp_path / "object-store"), bundle)
        uploaded = backups.upload(snapshot.path)
        downloaded = backups.download(uploaded.snapshot_id, tmp_path / "download")
        backups.load_images(downloaded)
        assert downloaded.manifest["images"][0]["content_id"] == image_id

        _prepare_empty_target(target, telegram_url)
        target_bundle = downloaded.bundle
        target_controller = _controller(target.root, target_bundle, target_project)
        restored = DeploymentSnapshots(target_controller, target_bundle).restore(
            downloaded.snapshot
        )
        assert target_controller.activation.read().state == "suspended"
        assert target_controller.running_services() == ()
        target_controller._compose("create")
        _run([
            "docker", "network", "connect", f"{target_project}_default",
            telegram_name,
        ])

        # Preflight is externally quiet. It may only perform deterministic local
        # recovery of the pending memory transaction copied in the snapshot.
        before = {agent_id: len(_calls(target.root, agent_id)) for agent_id in AGENTS}
        telegram_before = len(_telegram_calls(telegram_calls))
        for agent_id in AGENTS:
            target_controller._preflight_service(agent_id)
        assert {
            agent_id: len(_calls(target.root, agent_id)) for agent_id in AGENTS
        } == before
        assert len(_telegram_calls(telegram_calls)) == telegram_before

        for agent_id in AGENTS:
            state = restored / "agents" / agent_id / "state"
            assert (state / "CONSTITUTION.md").read_text().startswith(
                f"You are {agent_id.title()}"
            )
            assert (state / "GOALS.md").read_text() == f"Preserve {agent_id} goals\n"
            memory = MemoryModule(
                state / "memory",
                StimulusLog(restored / "agents" / agent_id / "logs" / "stimulus_log.jsonl"),
            )
            recalled = memory.recall("Atlas delivery code", 500)
            assert any("RAVEN-42" in entry.text for entry in recalled.entries)
            assert not (state / "memory" / "pending.json").exists()
            assert (state / "memory" / "formation" / "cursor.json").is_file()
            with sqlite3.connect(state / "delivery.sqlite3") as connection:
                payload, status = connection.execute(
                    "SELECT payload, status FROM delivery_outbox"
                ).fetchone()
                assert json.loads(payload)["text"] == f"recorded-{agent_id}"
                assert status == "pending"
        assert (restored / "workspaces" / "website" / "astra.txt").read_text() == "shared artifact"

        source_controller.retire(reason="acceptance handoff")
        with pytest.raises(PermissionError, match="retired"):
            source_controller.start()
        target_controller.start()
        _wait(
            "both target agents",
            lambda: _ready(target_controller),
        )

        # Simulate an unconditional restart after source-host reboot. Containers
        # launch, but the retired activation guard makes both agent processes exit.
        source_controller._compose("up", "-d")
        _wait("retired source exit", lambda: source_controller.running_services() == ())
        assert set(target_controller.running_services()) == set(AGENTS)
        target_controller.stop(timeout_seconds=10)
    finally:
        subprocess.run(
            ["docker", "rm", "-f", telegram_name],
            text=True,
            capture_output=True,
        )
        _down(bundle, source.root, source_project)
        _down(target_bundle, target.root, target_project)
        subprocess.run(
            ["docker", "image", "rm", image_tag],
            text=True,
            capture_output=True,
        )


def test_root_owned_activation_is_readable_but_not_writable_by_non_root_agent(tmp_path):
    suffix = uuid4().hex[:8]
    project = f"theseus-root-control-{suffix}"
    agent_uid = 21001
    agent_gid = 21001
    spec = _spec(uid=agent_uid, gid=agent_gid, workspace_gid=21002)
    bundle = assemble_compose(spec, tmp_path / "root-bundle", definition_root=tmp_path)
    image_tag = json.loads(build_bundle(bundle).read_text())["images"][0]["tag"]
    paths = DeploymentPaths(tmp_path / "root-host", spec)
    _prepare(paths, "http://127.0.0.1:9")
    controller = _controller(paths.root, bundle, project)

    def root_activation(state: str) -> None:
        script = f"""
from pathlib import Path
from theseus.deployment_control import ActivationStore
ActivationStore(Path('/host/control'), 'paired-acceptance').set(
    {state!r}, reason='root acceptance operator'
)
"""
        _run([
            "docker", "run", "--rm", "--user", "0:0",
            "-v", f"{paths.root}:/host", "--entrypoint", "python",
            image_tag, "-c", script,
        ])

    root_setup = f"""
import os
from pathlib import Path
root = Path('/host')
for agent_id in {AGENTS!r}:
    for path in (root / 'data' / 'agents' / agent_id, root / 'data' / 'agents' / agent_id / 'state', root / 'data' / 'agents' / agent_id / 'logs'):
        os.chown(path, {agent_uid}, {agent_gid})
        path.chmod(0o750)
workspace = root / 'data' / 'workspaces' / 'website'
os.chown(workspace, {agent_uid}, 21002)
workspace.chmod(0o2770)
for secret in (root / 'secrets').iterdir():
    os.chown(secret, {agent_uid}, {agent_gid})
    secret.chmod(0o600)
control = root / 'control'
os.chown(control, 0, {agent_gid})
control.chmod(0o2750)
    """

    try:
        _run([
            "docker", "run", "--rm", "--user", "0:0",
            "-v", f"{paths.root}:/host", "--entrypoint", "python",
            image_tag, "-c", root_setup,
        ])
        root_activation("active")
        for agent_id in AGENTS:
            controller._preflight_service(agent_id)
        assert paths.control.stat().st_mode & 0o7777 == 0o2750

        controller._compose("up", "-d")
        _wait(
            "non-root agents with root-owned activation",
            lambda: set(controller.running_services()) == set(AGENTS),
        )
        probe = f"""
import stat
from pathlib import Path
path = Path('/run/theseus-control/activation.json')
assert path.read_text()
metadata = path.stat()
assert metadata.st_uid == 0
assert metadata.st_gid == {agent_gid}
assert stat.S_IMODE(metadata.st_mode) == 0o640
try:
    path.write_text('{{}}')
except OSError:
    pass
else:
    raise AssertionError('agent modified host activation state')
replacement = Path('/tmp/replacement-activation.json')
replacement.write_text('{{}}')
try:
    replacement.replace(path)
except OSError:
    pass
else:
    raise AssertionError('agent replaced host activation state')
"""
        for agent_id in AGENTS:
            controller._compose("exec", "-T", agent_id, "python", "-c", probe)

        controller._compose("stop", "--timeout", "10")
        _wait("stopped root-operated agents", lambda: controller.running_services() == ())
        root_activation("suspended")
        root_activation("active")
        controller._compose("up", "-d")
        _wait(
            "agents after root activation replacement",
            lambda: set(controller.running_services()) == set(AGENTS),
        )
        for agent_id in AGENTS:
            controller._compose("exec", "-T", agent_id, "python", "-c", probe)
    finally:
        _down(bundle, paths.root, project)
        cleanup = f"""
import os
from pathlib import Path
root = Path('/host')
for directory, names, files in os.walk(root, topdown=False):
    for name in names + files:
        os.chown(Path(directory) / name, {os.getuid()}, {os.getgid()})
    os.chown(directory, {os.getuid()}, {os.getgid()})
"""
        subprocess.run(
            [
                "docker", "run", "--rm", "--user", "0:0",
                "-v", f"{paths.root}:/host", "--entrypoint", "python",
                image_tag, "-c", cleanup,
            ],
            text=True,
            capture_output=True,
        )
        subprocess.run(
            ["docker", "image", "rm", image_tag],
            text=True,
            capture_output=True,
        )
