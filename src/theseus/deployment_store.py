"""Portable host paths, container mounts, and stopped-home import."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Iterable, Iterator

from theseus.deployment import DeploymentSpec


RUNTIME_LOCK = ".theseus-runtime.lock"


@dataclass(frozen=True)
class ContainerMount:
    host_path: Path
    container_path: PurePosixPath
    read_only: bool
    backed_up: bool


@dataclass(frozen=True)
class DeploymentPaths:
    """The host-side path contract for one deployment root."""

    root: Path
    spec: DeploymentSpec

    @property
    def releases(self) -> Path:
        return self.root / "releases"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def control(self) -> Path:
        return self.root / "control"

    @property
    def secrets(self) -> Path:
        return self.root / "secrets"

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    def agent_root(self, agent_id: str) -> Path:
        self.spec._known_agent(agent_id, "agent")
        return self.data / "agents" / agent_id

    def state(self, agent_id: str) -> Path:
        return self.agent_root(agent_id) / "state"

    def logs(self, agent_id: str) -> Path:
        return self.agent_root(agent_id) / "logs"

    def log(self, agent_id: str) -> Path:
        return self.logs(agent_id) / "stimulus_log.jsonl"

    def workspace(self, workspace_id: str) -> Path:
        if workspace_id not in self.spec.workspaces:
            raise ValueError(f"unknown workspace {workspace_id!r}")
        return self.data / "workspaces" / workspace_id

    def prepare(self) -> None:
        """Create every directory, including peer log directories before log files."""
        self.spec.validate()
        for path in (self.releases, self.control, self.secrets, self.snapshots):
            path.mkdir(parents=True, exist_ok=True)
        for agent_id in self.spec.agents:
            self.state(agent_id).mkdir(parents=True, exist_ok=True)
            self.logs(agent_id).mkdir(parents=True, exist_ok=True)
        for workspace_id in self.spec.workspaces:
            self.workspace(workspace_id).mkdir(parents=True, exist_ok=True)

    def apply_ownership(self) -> None:
        """Apply stable private IDs and setgid shared-workspace ownership.

        This is deliberately separate from ``prepare`` because it requires the
        provisioning process to have permission to change numeric ownership.
        """
        self.spec.validate()
        os.chown(self.control, os.geteuid(), self.spec.gid)
        self.control.chmod(0o2750)
        for agent_id in self.spec.agents:
            for path in (self.agent_root(agent_id), self.state(agent_id), self.logs(agent_id)):
                os.chown(path, self.spec.uid, self.spec.gid)
        for workspace_id in self.spec.workspaces:
            path = self.workspace(workspace_id)
            os.chown(path, self.spec.uid, self.spec.workspace_gid)
            path.chmod(0o2770)

    def mounts_for(self, agent_id: str) -> tuple[ContainerMount, ...]:
        """Expose own private data, peer logs, and explicitly shared workspaces."""
        self.spec.validate()
        self.spec._known_agent(agent_id, "agent")
        mounts = [
            ContainerMount(self.state(agent_id), PurePosixPath("/data/state"), False, True),
            ContainerMount(self.logs(agent_id), PurePosixPath("/data/logs"), False, True),
            ContainerMount(self.control, PurePosixPath("/run/theseus-control"), True, False),
        ]
        peer_id = self.spec.peers.get(agent_id)
        if peer_id is not None:
            mounts.append(ContainerMount(
                self.logs(peer_id), PurePosixPath(f"/peers/{peer_id}/logs"), True, True
            ))
        for workspace_id in self.spec.workspaces_for(agent_id):
            mounts.append(ContainerMount(
                self.workspace(workspace_id),
                PurePosixPath(f"/workspaces/{workspace_id}"), False, True,
            ))
        for secret in self.spec.secrets:
            mounts.append(ContainerMount(
                self.secrets / secret, PurePosixPath(f"/run/secrets/{secret}"), True, False
            ))
        return tuple(mounts)

    def validate_managed_mounts(
        self, agent_id: str, requested: Iterable[ContainerMount]
    ) -> None:
        """Reject undeclared durable paths from a migration-managed runtime."""
        declared = {
            mount.container_path: mount for mount in self.mounts_for(agent_id)
        }
        for mount in requested:
            expected = declared.get(mount.container_path)
            if mount.backed_up and expected != mount:
                raise ValueError(
                    f"undeclared durable mount {mount.container_path}; add a shared workspace"
                )


@contextmanager
def runtime_lock(home: Path, *, blocking: bool = False) -> Iterator[None]:
    """Hold the advisory lock used by launchers and stopped-state imports."""
    import fcntl  # Unix-only; imported lazily so `import theseus` works on Windows

    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    with (home / RUNTIME_LOCK).open("a+") as stream:
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(stream, operation)
        except BlockingIOError as exc:
            raise RuntimeError(f"agent home has a live writer: {home}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def import_stopped_home(source_home: Path, paths: DeploymentPaths, agent_id: str) -> None:
    """Copy a stopped legacy home into the portable state/log split.

    The source data remains as a rollback copy. The imported tree contains the log
    exactly once, under ``logs/``. Publication is an atomic directory rename.
    """
    source_home = Path(source_home).resolve()
    if not source_home.is_dir():
        raise ValueError(f"source home is not a directory: {source_home}")
    paths.spec.validate()
    destination = paths.agent_root(agent_id).resolve()
    if destination == source_home or destination.is_relative_to(source_home):
        raise ValueError("import destination must be outside the source home")
    if source_home.is_relative_to(destination):
        raise ValueError("source home must be outside the import destination")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"import destination is occupied: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    legacy_log = source_home / "stimulus_log.jsonl"

    with runtime_lock(source_home):
        staging = Path(tempfile.mkdtemp(prefix=f".{agent_id}-import-", dir=destination.parent))
        try:
            state = staging / "state"
            logs = staging / "logs"
            state.mkdir()
            logs.mkdir()
            for source in source_home.iterdir():
                if source.name in ("stimulus_log.jsonl", RUNTIME_LOCK):
                    continue
                target = state / source.name
                if source.is_symlink():
                    raise ValueError(f"legacy home contains unsupported symlink: {source}")
                if source.is_dir():
                    shutil.copytree(source, target, symlinks=True)
                    if any(path.is_symlink() for path in target.rglob("*")):
                        raise ValueError(f"legacy home contains unsupported symlink: {source}")
                else:
                    shutil.copy2(source, target)
            if legacy_log.exists():
                if not legacy_log.is_file() or legacy_log.is_symlink():
                    raise ValueError("legacy stimulus log must be a regular file")
                shutil.copy2(legacy_log, logs / legacy_log.name)
            if destination.exists():
                destination.rmdir()
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
