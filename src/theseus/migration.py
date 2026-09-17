"""Durable, single-active-host deployment migration orchestration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import UUID, uuid4

from theseus.deployment_control import DeploymentController, _atomic_json
from theseus.deployment_snapshot import MANIFEST_FILE, DeploymentSnapshots, SnapshotResult
from theseus.host_provisioner import HostProfile, HostProvisioner, OpenSSH, ProvisionResult
from theseus.remote_backup import (
    DownloadedBackup,
    ReleaseArtifacts,
    RemoteBackups,
)


MIGRATION_FORMAT_VERSION = 1
MIGRATION_FILE = "migration.json"
_PRE_ACTIVATION_PHASES = {
    "source-stop-intent",
    "source-stopped",
    "capture-intent",
    "snapshot-captured",
    "snapshot-uploaded",
    "restore-intent",
    "target-restored",
    "source-retire-intent",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class HostInspection:
    activation: str
    running_services: tuple[str, ...]
    ready_services: tuple[str, ...] = ()
    restored_snapshot_id: str | None = None
    activation_intent: bool = False


@dataclass(frozen=True)
class SnapshotEvidence:
    result: SnapshotResult
    manifest_sha256: str
    archive_sha256: str


@dataclass(frozen=True)
class MigrationResult:
    migration_id: str
    phase: str
    deployment_id: str
    droplet_id: int | None
    snapshot_id: str | None
    snapshot_manifest_sha256: str | None
    source_activation: str | None
    target_activation: str | None
    target_ready_services: tuple[str, ...]
    remaining_billable_resources: tuple[str, ...]
    failure: str | None


class HostUnreachable(RuntimeError):
    """A host could not be inspected, so its execution state is unknown."""


class UnknownActivation(RuntimeError):
    """A host answered but did not provide trustworthy activation evidence."""


class MigrationSource(Protocol):
    agent_ids: tuple[str, ...]
    identity: str
    def inspect(self) -> HostInspection: ...
    def record(self, state: Mapping[str, Any]) -> None: ...
    def stop(self, prior: Sequence[str], *, timeout_seconds: int) -> None: ...
    def capture(self, migration_id: str, started_at: str) -> SnapshotEvidence: ...
    def upload(self, snapshot: SnapshotEvidence) -> dict[str, Any]: ...
    def retire(self, migration_id: str) -> None: ...
    def recover(self, services: Sequence[str], migration_id: str) -> None: ...


class MigrationTarget(Protocol):
    agent_ids: tuple[str, ...]
    def inspect(self) -> HostInspection: ...
    def record(self, state: Mapping[str, Any]) -> None: ...
    def preload(self, artifacts: ReleaseArtifacts) -> None: ...
    def restore(self, downloaded: DownloadedBackup) -> None: ...
    def start(self) -> None: ...


class LocalMigrationSource:
    """Source-host operations backed by existing lifecycle and backup components."""

    def __init__(
        self,
        controller: DeploymentController,
        snapshots: DeploymentSnapshots,
        backups: RemoteBackups,
    ) -> None:
        self.controller = controller
        self.snapshots = snapshots
        self.backups = backups
        self.agent_ids = tuple(controller.agent_ids)
        self.identity = f"local:{controller.root.resolve()}"
        self.journal_path = controller.control_dir / MIGRATION_FILE

    def inspect(self) -> HostInspection:
        activation = self.controller.activation.read()
        running = self.controller.running_services()
        ready = tuple(
            agent_id
            for agent_id in self.agent_ids
            if (record := self.controller._lifecycle(agent_id)) is not None
            and record.get("state") == "running"
        )
        return HostInspection(
            activation.state if activation is not None else "absent", running, ready
        )

    def record(self, state: Mapping[str, Any]) -> None:
        _atomic_json(self.journal_path, {**dict(state), "host_role": "source"})

    def _migration_state(self) -> dict[str, Any]:
        state = _read_object(self.journal_path)
        if (
            state.get("host_role") != "source"
            or state.get("deployment_id") != self.controller.deployment_id
            or not isinstance(state.get("migration_id"), str)
        ):
            raise ValueError("source migration journal identity is invalid")
        return state

    def _stop_path(self, migration_id: str) -> Path:
        try:
            valid = isinstance(migration_id, str) and len(migration_id) == 32
            UUID(hex=migration_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("migration ID is invalid") from exc
        if not valid:
            raise ValueError("migration ID is invalid")
        return self.controller.control_dir / f"migration-stop-{migration_id}.json"

    def _capture_path(self, migration_id: str) -> Path:
        self._stop_path(migration_id)  # Apply the same durable-artifact ID validation.
        return self.controller.control_dir / f"migration-capture-{migration_id}.json"

    def _validate_clean_stop(self, evidence: Mapping[str, Any]) -> None:
        if (
            evidence.get("state") != "verified"
            or evidence.get("deployment_id") != self.controller.deployment_id
            or not isinstance(evidence.get("migration_id"), str)
        ):
            raise UnknownActivation("source clean-stop evidence is not verified")
        services = evidence.get("services")
        expected_runs = evidence.get("expected_runs")
        if (
            not isinstance(services, list)
            or set(services) != set(self.agent_ids)
            or not isinstance(expected_runs, dict)
            or set(expected_runs) != set(services)
        ):
            raise UnknownActivation("source clean-stop evidence is incomplete")
        for service in services:
            lifecycle = self.controller._lifecycle(service)
            if (
                lifecycle is None
                or lifecycle.get("run_id") != expected_runs.get(service)
                or lifecycle.get("state") != "stopped"
                or lifecycle.get("clean") is not True
            ):
                raise UnknownActivation(
                    f"source clean stop is unconfirmed for {service}"
                )
        inspection = self.inspect()
        if inspection.activation != "suspended" or inspection.running_services:
            raise UnknownActivation("source suspension is unconfirmed")

    def _finish_stop_evidence(self, evidence: dict[str, Any]) -> None:
        operation = self.controller.journal.read()
        if (
            operation is None
            or operation.kind != "stop"
            or set(operation.prior_running_services) != set(evidence["services"])
        ):
            raise UnknownActivation("source stop operation did not complete cleanly")
        if operation.finished and operation.phase != "completed":
            raise UnknownActivation("source stop operation did not complete cleanly")
        verified = {
            **evidence,
            "state": "verified",
            "stop_operation_id": operation.operation_id,
            "verified_at": _now(),
        }
        self._validate_clean_stop(verified)
        if operation.phase != "completed":
            self.controller.journal.transition(operation.operation_id, "completed")
        _atomic_json(self._stop_path(evidence["migration_id"]), verified)

    def stop(self, prior: Sequence[str], *, timeout_seconds: int) -> None:
        state = self._migration_state()
        migration_id = state["migration_id"]
        if state.get("phase") != "source-stop-intent":
            raise RuntimeError("source stop requires durable migration stop intent")
        if set(prior) != set(self.agent_ids) or set(
            state.get("source_prior_services", ())
        ) != set(prior):
            raise UnknownActivation("migration cutover does not cover every source agent")
        inspection = self.inspect()
        if inspection.activation == "retired":
            raise RuntimeError("source is already retired")
        stop_path = self._stop_path(migration_id)
        if stop_path.is_symlink():
            raise ValueError("source clean-stop evidence cannot be a symlink")
        evidence = _read_object(stop_path) if stop_path.is_file() else None
        if evidence is not None:
            services = evidence.get("services")
            if (
                evidence.get("migration_id") != migration_id
                or evidence.get("deployment_id") != self.controller.deployment_id
                or not isinstance(services, list)
                or set(services) != set(prior)
            ):
                raise ValueError("source clean-stop evidence does not match migration")
            if evidence.get("state") == "verified":
                self._validate_clean_stop(evidence)
                return
            if evidence.get("state") != "stopping":
                raise ValueError("source clean-stop evidence has an invalid state")
        else:
            expected_runs = {}
            for service in prior:
                lifecycle = self.controller._lifecycle(service)
                if (
                    lifecycle is None
                    or lifecycle.get("state") != "running"
                    or lifecycle.get("clean") is not None
                ):
                    raise UnknownActivation(
                        f"source running lifecycle is unconfirmed for {service}"
                    )
                expected_runs[service] = lifecycle["run_id"]
            if set(inspection.running_services) != set(prior):
                raise UnknownActivation("source containers do not match cutover intent")
            evidence = {
                "state": "stopping",
                "migration_id": migration_id,
                "deployment_id": self.controller.deployment_id,
                "services": list(prior),
                "expected_runs": expected_runs,
                "created_at": _now(),
            }
            _atomic_json(stop_path, evidence)
        interrupted = self.controller.interrupted_operation()
        if interrupted is not None:
            if interrupted.kind != "stop":
                raise RuntimeError(
                    f"source has interrupted {interrupted.kind} operation "
                    f"{interrupted.operation_id}"
                )
            running = self.controller.running_services()
            if running:
                self.controller.clean_stop_services(
                    running, timeout_seconds=timeout_seconds
                )
            if self.controller.running_services():
                raise RuntimeError("source still has running services after stop recovery")
        elif not inspection.running_services and inspection.activation == "suspended":
            # Absence of containers is not clean-stop proof.  Only a completed stop
            # operation plus the exact run acknowledgements recorded above can recover.
            pass
        else:
            self.controller.stop(timeout_seconds=timeout_seconds)
        self._finish_stop_evidence(evidence)

    def _stop_evidence(self, migration_id: str) -> tuple[dict[str, Any], str]:
        path = self._stop_path(migration_id)
        if not path.is_file() or path.is_symlink():
            raise UnknownActivation("final capture requires verified clean-stop evidence")
        evidence = _read_object(path)
        if evidence.get("migration_id") != migration_id:
            raise ValueError("source clean-stop evidence belongs to another migration")
        self._validate_clean_stop(evidence)
        return evidence, _sha256_file(path)

    def _snapshot_evidence(
        self, path: Path, migration_id: str, stop_sha256: str
    ) -> SnapshotEvidence:
        snapshots_root = (self.controller.root / "snapshots").resolve()
        if path.is_symlink() or not path.is_dir() or path.parent.resolve() != snapshots_root:
            raise ValueError("final snapshot path is outside the snapshot store")
        manifest = _read_object(path / MANIFEST_FILE)
        context = manifest.get("capture_context")
        expected = {
            "kind": "migration-final",
            "migration_id": migration_id,
            "stop_evidence_sha256": stop_sha256,
        }
        if context != expected:
            raise ValueError("final snapshot evidence does not match migration stop")
        if (
            manifest.get("snapshot_id") != path.name
            or manifest.get("deployment_id") != self.controller.deployment_id
        ):
            raise ValueError("final snapshot identity does not match deployment")
        result = SnapshotResult(manifest["snapshot_id"], path, manifest)
        return SnapshotEvidence(
            result, _sha256_file(path / MANIFEST_FILE), manifest["archive"]["sha256"]
        )

    def _existing_capture(self, migration_id: str, started_at: str) -> SnapshotEvidence | None:
        del started_at  # Timestamps are not final-snapshot identity.
        _, stop_sha256 = self._stop_evidence(migration_id)
        receipt = self._capture_path(migration_id)
        if receipt.is_file() and not receipt.is_symlink():
            value = _read_object(receipt)
            if (
                value.get("migration_id") != migration_id
                or value.get("stop_evidence_sha256") != stop_sha256
                or not isinstance(value.get("snapshot_id"), str)
            ):
                raise ValueError("final snapshot receipt does not match migration stop")
            path = self.controller.root / "snapshots" / value["snapshot_id"]
            return self._snapshot_evidence(path, migration_id, stop_sha256)
        candidates = []
        mismatched = []
        snapshots_root = self.controller.root / "snapshots"
        if snapshots_root.is_dir():
            for path in snapshots_root.iterdir():
                manifest_path = path / MANIFEST_FILE
                if not path.is_dir() or path.is_symlink() or not manifest_path.is_file():
                    continue
                manifest = _read_object(manifest_path)
                context = manifest.get("capture_context")
                if (
                    isinstance(context, dict)
                    and context.get("migration_id") == migration_id
                ):
                    if (
                        context.get("kind") != "migration-final"
                        or context.get("stop_evidence_sha256") != stop_sha256
                    ):
                        mismatched.append(path)
                    elif manifest.get("deployment_id") == self.controller.deployment_id:
                        candidates.append((path, manifest))
                    else:
                        mismatched.append(path)
        if mismatched:
            raise ValueError("final snapshot evidence does not match migration stop")
        if len(candidates) > 1:
            raise RuntimeError("multiple final snapshot candidates require operator resolution")
        if not candidates:
            return None
        path, manifest = candidates[0]
        evidence = self._snapshot_evidence(path, migration_id, stop_sha256)
        _atomic_json(
            receipt,
            {
                "migration_id": migration_id,
                "snapshot_id": evidence.result.snapshot_id,
                "stop_evidence_sha256": stop_sha256,
                "reconciled": True,
            },
        )
        return evidence

    def capture(self, migration_id: str, started_at: str) -> SnapshotEvidence:
        existing = self._existing_capture(migration_id, started_at)
        if existing is not None:
            return existing
        inspection = self.inspect()
        if inspection.activation != "suspended" or inspection.running_services:
            raise RuntimeError("source must be suspended and stopped for final capture")
        _, stop_sha256 = self._stop_evidence(migration_id)
        context = {
            "kind": "migration-final",
            "migration_id": migration_id,
            "stop_evidence_sha256": stop_sha256,
        }
        result = self.snapshots.create(leave_stopped=True, capture_context=context)
        evidence = SnapshotEvidence(
            result,
            _sha256_file(result.path / MANIFEST_FILE),
            result.manifest["archive"]["sha256"],
        )
        _atomic_json(
            self._capture_path(migration_id),
            {
                "migration_id": migration_id,
                "snapshot_id": result.snapshot_id,
                "stop_evidence_sha256": stop_sha256,
            },
        )
        return evidence

    def upload(self, snapshot: SnapshotEvidence) -> dict[str, Any]:
        result = self.backups.upload(snapshot.result.path)
        manifest = self.backups.get_manifest(result.snapshot_id)
        if manifest["archive"]["sha256"] != snapshot.archive_sha256:
            raise ValueError("uploaded final snapshot archive identity differs")
        return manifest

    def retire(self, migration_id: str) -> None:
        inspection = self.inspect()
        if inspection.activation == "retired" and not inspection.running_services:
            return
        self.controller.retire(reason=f"migration {migration_id} handoff")

    def recover(self, services: Sequence[str], migration_id: str) -> None:
        inspection = self.inspect()
        if inspection.activation == "retired":
            raise RuntimeError("retired source cannot be automatically recovered")
        if inspection.running_services:
            raise RuntimeError("source recovery requires a stopped source")
        journal = _read_object(self.journal_path)
        _atomic_json(
            self.journal_path,
            {**journal, "host_role": "source", "phase": "source-recovery-intent"},
        )
        self.controller.start(tuple(services))


class LocalMigrationTarget:
    """Local adapter used for offline integration and existing-host destinations."""

    def __init__(
        self,
        controller: DeploymentController,
        backups: RemoteBackups,
        bundle: Path,
    ) -> None:
        self.controller = controller
        self.backups = backups
        self.bundle = Path(bundle)
        self.agent_ids = tuple(controller.agent_ids)
        self.journal_path = controller.control_dir / MIGRATION_FILE

    def inspect(self) -> HostInspection:
        status = self.controller.status()
        activation = status["activation"]
        journal = _read_object(self.journal_path) if self.journal_path.is_file() else {}
        restored = None
        operation = status["operation"]
        if operation and operation.get("kind") == "restore" and operation.get("phase") == "completed":
            try:
                restored = json.loads(operation.get("detail") or "{}").get("snapshot_id")
            except json.JSONDecodeError:
                restored = None
        ready = tuple(
            agent_id
            for agent_id, value in status["lifecycle"].items()
            if value is not None and value.get("state") == "running"
        )
        return HostInspection(
            activation["state"] if activation else "absent",
            tuple(status["running_services"]),
            ready,
            restored,
            journal.get("phase") in ("target-activation-intent", "target-starting", "completed"),
        )

    def record(self, state: Mapping[str, Any]) -> None:
        _atomic_json(self.journal_path, {**dict(state), "host_role": "target"})

    def preload(self, artifacts: ReleaseArtifacts) -> None:
        # The local target shares the test Docker daemon; loading is verified from the
        # final downloaded backup before restore.
        if not artifacts.bundle_archive.is_file() or not artifacts.image_archives:
            raise ValueError("release preload artifacts are incomplete")

    def restore(self, downloaded: DownloadedBackup) -> None:
        inspection = self.inspect()
        if inspection.restored_snapshot_id == downloaded.snapshot_id:
            return
        if inspection.restored_snapshot_id is not None:
            raise RuntimeError("target already contains a different restored snapshot")
        self.backups.load_images(downloaded)
        DeploymentSnapshots(self.controller, downloaded.bundle).restore(downloaded.snapshot)

    def start(self) -> None:
        inspection = self.inspect()
        if (
            inspection.activation == "active"
            and set(inspection.running_services) == set(self.agent_ids)
        ):
            return
        self.controller.start()


class SSHMigrationTarget:
    """Destination operations over the host identity retained during provisioning."""

    def __init__(
        self,
        ssh: OpenSSH,
        profile: HostProfile,
        known_hosts: Path,
        address: str,
        deployment: Mapping[str, Any],
        release_id: str,
    ) -> None:
        self.ssh = ssh
        self.profile = profile
        self.known_hosts = Path(known_hosts)
        self.address = address
        self.deployment = dict(deployment)
        self.deployment_id = self.deployment["deployment_id"]
        self.agent_ids = tuple(sorted(self.deployment["agents"]))
        self.release_id = release_id
        self.root = f"/srv/theseus/{self.deployment_id}"
        self.bundle = f"{self.root}/releases/{release_id}/bundle"
        self.operator = "/opt/theseus-operator/bin/theseus-deployment"

    def _run(self, command: str) -> subprocess.CompletedProcess[str]:
        try:
            return self.ssh.run(
                self.address, self.profile, self.known_hosts, command
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise HostUnreachable(f"destination command could not be confirmed: {command}") from exc

    def _upload(self, source: Path, destination: str) -> None:
        try:
            self.ssh.upload(
                self.address, self.profile, self.known_hosts, source, destination
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise HostUnreachable("destination upload could not be confirmed") from exc

    def _status(self) -> dict[str, Any]:
        command = (
            f"if test -f {shlex.quote(self.bundle + '/deployment.json')}; then "
            f"sudo {self.operator} {shlex.quote(self.bundle)} status "
            f"--root {shlex.quote(self.root)}; else "
            "printf '%s\\n' '{\"activation\":null,\"operation\":null,"
            "\"running_services\":[],\"lifecycle\":{}}'; fi"
        )
        try:
            value = json.loads(self._run(command).stdout)
        except json.JSONDecodeError as exc:
            raise UnknownActivation("destination returned invalid status") from exc
        if not isinstance(value, dict):
            raise UnknownActivation("destination returned invalid status")
        return value

    def inspect(self) -> HostInspection:
        status = self._status()
        journal_command = (
            f"if test -f {shlex.quote(self.root + '/control/' + MIGRATION_FILE)}; "
            f"then cat {shlex.quote(self.root + '/control/' + MIGRATION_FILE)}; "
            "else printf '%s\\n' '{}'; fi"
        )
        try:
            journal = json.loads(self._run(journal_command).stdout)
        except json.JSONDecodeError as exc:
            raise UnknownActivation("destination migration journal is invalid") from exc
        activation = status.get("activation")
        activation_state = activation.get("state") if isinstance(activation, dict) else "absent"
        if activation_state not in ("absent", "suspended", "active", "retired"):
            raise UnknownActivation("destination activation state is unknown")
        operation = status.get("operation")
        restored = None
        if isinstance(operation, dict) and operation.get("kind") == "restore":
            if operation.get("phase") not in ("completed", "failed"):
                raise UnknownActivation("destination restore operation is interrupted")
            if operation.get("phase") == "completed":
                try:
                    restored = json.loads(operation.get("detail") or "{}").get("snapshot_id")
                except json.JSONDecodeError:
                    pass
        lifecycle = status.get("lifecycle", {})
        ready = tuple(
            name
            for name, value in lifecycle.items()
            if isinstance(value, dict) and value.get("state") == "running"
        ) if isinstance(lifecycle, dict) else ()
        phase = journal.get("phase") if isinstance(journal, dict) else None
        return HostInspection(
            activation_state,
            tuple(status.get("running_services", [])),
            ready,
            restored,
            phase in ("target-activation-intent", "target-starting", "completed"),
        )

    def record(self, state: Mapping[str, Any]) -> None:
        temporary: Path | None = None
        remote = f"/tmp/.theseus-migration-{state['migration_id']}.json"
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(
                    {**dict(state), "host_role": "target"},
                    stream,
                    sort_keys=True,
                    indent=2,
                )
                stream.write("\n")
            self._upload(temporary, remote)
            self._run(
                f"sudo install -d -o root -g {self.deployment['gid']} -m 2750 "
                f"{shlex.quote(self.root + '/control')} && "
                f"sudo install -m 0600 {shlex.quote(remote)} "
                f"{shlex.quote(self.root + '/control/' + MIGRATION_FILE)} && "
                f"rm -f {shlex.quote(remote)}"
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def preload(self, artifacts: ReleaseArtifacts) -> None:
        inspection = self.inspect()
        if inspection.activation not in ("absent", "suspended") or inspection.running_services:
            raise RuntimeError("destination must remain inactive during release preload")
        bundle_remote = f"/tmp/theseus-release-{artifacts.release_id}.tar.gz"
        self._upload(artifacts.bundle_archive, bundle_remote)
        release_root = f"{self.root}/releases/{artifacts.release_id}"
        self._run(
            f"echo {shlex.quote(artifacts.bundle_sha256 + '  ' + bundle_remote)} | sha256sum -c - && "
            f"sudo install -d -m 0750 {shlex.quote(release_root)} && "
            f"sudo tar -xzf {shlex.quote(bundle_remote)} -C {shlex.quote(release_root)} && "
            f"rm -f {shlex.quote(bundle_remote)}"
        )
        for archive, descriptor in zip(
            artifacts.image_archives, artifacts.image_descriptors, strict=True
        ):
            remote = f"/tmp/theseus-image-{descriptor['archive_sha256']}.tar.gz"
            raw = remote.removesuffix(".gz")
            self._upload(archive, remote)
            expected_platform = "/".join(descriptor["platform"].split("/")[:2])
            self._run(
                f"echo {shlex.quote(descriptor['sha256'] + '  ' + remote)} | sha256sum -c - && "
                f"gzip -dc {shlex.quote(remote)} > {shlex.quote(raw)} && "
                f"sudo docker image load --input {shlex.quote(raw)} >/dev/null && "
                f"test \"$(sudo docker image inspect --format '{{{{.Id}}}} {{{{.Os}}}}/{{{{.Architecture}}}}' "
                f"{shlex.quote(descriptor['content_id'])})\" = "
                f"{shlex.quote(descriptor['content_id'] + ' ' + expected_platform)} && "
                f"rm -f {shlex.quote(remote)} {shlex.quote(raw)}"
            )

    def restore(self, downloaded: DownloadedBackup) -> None:
        inspection = self.inspect()
        if inspection.restored_snapshot_id == downloaded.snapshot_id:
            return
        if inspection.restored_snapshot_id is not None:
            raise RuntimeError("destination already restored a different snapshot")
        if (self._status().get("operation") or {}).get("phase") not in (None, "completed", "failed"):
            raise UnknownActivation("destination has an interrupted operation")
        snapshot_root = f"{self.root}/snapshots/{downloaded.snapshot_id}"
        manifest_remote = f"/tmp/{downloaded.snapshot_id}-manifest.json"
        archive_remote = f"/tmp/{downloaded.snapshot_id}-data.tar.gz"
        self._upload(downloaded.snapshot / MANIFEST_FILE, manifest_remote)
        self._upload(downloaded.snapshot / "data.tar.gz", archive_remote)
        self._run(
            f"if test -d {shlex.quote(self.root + '/data')}; then "
            f"test -z \"$(find {shlex.quote(self.root + '/data')} -mindepth 1 ! -type d -print -quit)\" && "
            f"sudo find {shlex.quote(self.root + '/data')} -depth -type d -empty -delete; fi && "
            f"test ! -e {shlex.quote(self.root + '/data')} && "
            f"sudo install -d -m 0750 {shlex.quote(snapshot_root)} && "
            f"sudo install -m 0600 {shlex.quote(manifest_remote)} "
            f"{shlex.quote(snapshot_root + '/' + MANIFEST_FILE)} && "
            f"sudo install -m 0600 {shlex.quote(archive_remote)} "
            f"{shlex.quote(snapshot_root + '/data.tar.gz')} && "
            f"rm -f {shlex.quote(manifest_remote)} {shlex.quote(archive_remote)} && "
            f"sudo {self.operator} {shlex.quote(self.bundle)} restore "
            f"--root {shlex.quote(self.root)} --snapshot {shlex.quote(snapshot_root)}"
        )
        confirmed = self.inspect()
        if confirmed.restored_snapshot_id != downloaded.snapshot_id:
            raise UnknownActivation("destination restore completion is unconfirmed")
        for descriptor in downloaded.manifest["objects"]["images"]:
            expected_platform = "/".join(descriptor["platform"].split("/")[:2])
            result = self._run(
                f"sudo docker image inspect --format '{{{{.Id}}}} {{{{.Os}}}}/{{{{.Architecture}}}}' "
                f"{shlex.quote(descriptor['content_id'])}"
            ).stdout.strip()
            if result != f"{descriptor['content_id']} {expected_platform}":
                raise ValueError("destination exact image identity differs after restore")

    def start(self) -> None:
        inspection = self.inspect()
        operation = self._status().get("operation")
        interrupted_start = (
            isinstance(operation, dict)
            and operation.get("kind") == "start"
            and operation.get("phase") not in ("completed", "failed")
        )
        if (
            inspection.activation == "active"
            and set(inspection.running_services) == set(self.agent_ids)
            and not interrupted_start
        ):
            return
        if not inspection.activation_intent:
            raise RuntimeError("destination activation intent is not durable")
        action = "resume-start" if interrupted_start else "start"
        self._run(
            f"sudo {self.operator} {shlex.quote(self.bundle)} {action} "
            f"--root {shlex.quote(self.root)}"
        )


class MigrationCoordinator:
    """Advance a migration only from durable, inspected handoff evidence."""

    def __init__(
        self,
        *,
        deployment_id: str,
        agent_ids: Sequence[str],
        source: MigrationSource,
        provisioner: HostProvisioner,
        target_factory: Callable[[ProvisionResult], MigrationTarget],
        backups: RemoteBackups,
        state_path: Path,
        workspace: Path,
        secrets: Mapping[str, Path],
        timeout_seconds: int = 30,
    ) -> None:
        self.deployment_id = deployment_id
        self.agent_ids = tuple(sorted(agent_ids))
        self.source = source
        self.provisioner = provisioner
        self.target_factory = target_factory
        self.backups = backups
        self.state_path = Path(state_path)
        self.workspace = Path(workspace)
        self.secrets = dict(secrets)
        self.timeout_seconds = timeout_seconds
        if tuple(sorted(source.agent_ids)) != self.agent_ids:
            raise ValueError("source agent IDs differ from the migration deployment")

    def _initial(self) -> dict[str, Any]:
        return {
            "format_version": MIGRATION_FORMAT_VERSION,
            "migration_id": uuid4().hex,
            "deployment_id": self.deployment_id,
            "source_identity": self.source.identity,
            "provision_state_path": str(self.provisioner.state_path.resolve()),
            "agent_ids": list(self.agent_ids),
            "release_id": self.backups.release_id,
            "phase": "initialized",
            "created_at": _now(),
            "updated_at": _now(),
            "source_prior_services": [],
            "destination": None,
            "snapshot_id": None,
            "snapshot_manifest_sha256": None,
            "snapshot_archive_sha256": None,
            "source_fence": None,
            "source_activation": None,
            "target_activation": None,
            "target_ready_services": [],
            "remaining_billable_resources": [],
            "failure": None,
        }

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self._initial()
            _atomic_json(self.state_path, state)
            return state
        if not self.state_path.is_file() or self.state_path.is_symlink():
            raise ValueError("migration state must be a real file")
        state = _read_object(self.state_path)
        if state.get("format_version") != MIGRATION_FORMAT_VERSION:
            raise ValueError("unsupported migration state format")
        if (
            state.get("deployment_id") != self.deployment_id
            or state.get("source_identity") != self.source.identity
            or state.get("provision_state_path")
            != str(self.provisioner.state_path.resolve())
            or tuple(state.get("agent_ids", ())) != self.agent_ids
            or state.get("release_id") != self.backups.release_id
        ):
            raise ValueError("migration state belongs to another deployment release")
        return state

    def _save(self, state: dict[str, Any], phase: str | None = None, **values: Any) -> None:
        if phase is not None:
            values["phase"] = phase
        state.update(values, updated_at=_now())
        _atomic_json(self.state_path, state)

    def _failure(self, state: dict[str, Any], exc: BaseException) -> None:
        self._save(state, failure=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _destination_dict(result: ProvisionResult) -> dict[str, Any]:
        return {
            "operation_id": result.operation_id,
            "droplet_id": result.droplet_id,
            "address": result.address,
            "ssh_host_fingerprint": result.ssh_host_fingerprint,
            "remaining_billable_resources": list(result.remaining_billable_resources),
        }

    def _target(self, state: dict[str, Any]) -> tuple[ProvisionResult, MigrationTarget]:
        provision = self.provisioner.status()
        destination = state.get("destination") or {}
        if (
            provision.phase not in ("ready", "activated")
            or provision.droplet_id != destination.get("droplet_id")
            or provision.address != destination.get("address")
        ):
            raise RuntimeError("provisioned destination identity is not ready or has changed")
        target = self.target_factory(provision)
        if tuple(sorted(target.agent_ids)) != self.agent_ids:
            raise ValueError("target agent IDs differ from the migration deployment")
        return provision, target

    def _assert_target_inactive(self, target: MigrationTarget) -> HostInspection:
        inspection = target.inspect()
        if inspection.activation not in ("absent", "suspended"):
            raise UnknownActivation(
                f"destination activation is {inspection.activation}; expected inactive"
            )
        if inspection.running_services:
            raise UnknownActivation("destination has running services before handoff")
        return inspection

    def migrate(self) -> MigrationResult:
        state = self._load()
        if state["phase"] == "completed":
            return self._result(state)
        if state["phase"] == "recovered":
            raise RuntimeError("migration was aborted and the source was recovered")
        try:
            if state["phase"] == "initialized":
                try:
                    provision = self.provisioner.status()
                except (OSError, ValueError):
                    provision = self.provisioner.provision(self.secrets)
                else:
                    if provision.phase not in ("ready", "activated"):
                        provision = self.provisioner.provision(self.secrets)
                self._save(
                    state,
                    "destination-ready",
                    destination=self._destination_dict(provision),
                    remaining_billable_resources=list(
                        provision.remaining_billable_resources
                    ),
                    failure=None,
                )
            provision, target = self._target(state)

            if state["phase"] == "destination-ready":
                self._assert_target_inactive(target)
                artifacts = self.backups.stage_release(self.workspace / "preload")
                target.preload(artifacts)
                self._save(state, "release-preloaded", failure=None)

            if state["phase"] == "release-preloaded":
                self._assert_target_inactive(target)
                source = self.source.inspect()
                if source.activation == "retired":
                    raise RuntimeError("source is retired before final capture")
                if source.activation != "active" or set(source.running_services) != set(
                    self.agent_ids
                ):
                    raise RuntimeError(
                        "migration requires every source agent to be active and running"
                    )
                prior = tuple(source.running_services)
                self._save(
                    state,
                    "source-stop-intent",
                    source_prior_services=list(prior),
                    source_activation=source.activation,
                )
                self.source.record(state)

            if state["phase"] == "source-stop-intent":
                self._assert_target_inactive(target)
                self.source.record(state)
                self.source.stop(
                    tuple(state["source_prior_services"]),
                    timeout_seconds=self.timeout_seconds,
                )
                source = self.source.inspect()
                if source.activation != "suspended" or source.running_services:
                    raise UnknownActivation("source stop is not durably confirmed")
                self._save(state, "source-stopped", source_activation="suspended")
                self.source.record(state)

            if state["phase"] == "source-stopped":
                self._save(state, "capture-intent")
                self.source.record(state)

            if state["phase"] == "capture-intent":
                evidence = self.source.capture(
                    state["migration_id"], state["created_at"]
                )
                self._save(
                    state,
                    "snapshot-captured",
                    snapshot_id=evidence.result.snapshot_id,
                    snapshot_manifest_sha256=evidence.manifest_sha256,
                    snapshot_archive_sha256=evidence.archive_sha256,
                )
                self.source.record(state)

            if state["phase"] == "snapshot-captured":
                evidence = self.source.capture(
                    state["migration_id"], state["created_at"]
                )
                if (
                    evidence.result.snapshot_id != state["snapshot_id"]
                    or evidence.manifest_sha256 != state["snapshot_manifest_sha256"]
                    or evidence.archive_sha256 != state["snapshot_archive_sha256"]
                ):
                    raise ValueError("pinned final snapshot evidence changed")
                remote = self.source.upload(evidence)
                if remote["snapshot_id"] != state["snapshot_id"]:
                    raise ValueError("uploaded a different final snapshot")
                self._save(state, "snapshot-uploaded", failure=None)
                self.source.record(state)

            if state["phase"] == "snapshot-uploaded":
                self._assert_target_inactive(target)
                self._save(state, "restore-intent")
                target.record(state)

            if state["phase"] == "restore-intent":
                self._assert_target_inactive(target)
                download = self.workspace / "downloads" / state["snapshot_id"]
                if download.exists():
                    shutil.rmtree(download)
                downloaded = self.backups.download(state["snapshot_id"], download)
                if (
                    _sha256_file(downloaded.snapshot / MANIFEST_FILE)
                    != state["snapshot_manifest_sha256"]
                    or downloaded.manifest["archive"]["sha256"]
                    != state["snapshot_archive_sha256"]
                ):
                    raise ValueError("downloaded final snapshot differs from pinned evidence")
                target.restore(downloaded)
                restored = target.inspect()
                if (
                    restored.restored_snapshot_id != state["snapshot_id"]
                    or restored.activation != "suspended"
                    or restored.running_services
                ):
                    raise UnknownActivation("exact inactive target restore is unconfirmed")
                self._save(state, "target-restored", target_activation="suspended")
                target.record(state)

            if state["phase"] == "target-restored":
                self._save(state, "source-retire-intent")
                self.source.record(state)

            if state["phase"] == "source-retire-intent":
                restored = target.inspect()
                if (
                    restored.restored_snapshot_id != state["snapshot_id"]
                    or restored.activation not in ("absent", "suspended")
                    or restored.running_services
                ):
                    raise UnknownActivation("destination is not proven inactive and restored")
                try:
                    self.source.retire(state["migration_id"])
                    source = self.source.inspect()
                except HostUnreachable:
                    if not state.get("source_fence"):
                        raise
                    self._save(state, "source-fenced", source_activation="fenced")
                else:
                    if source.activation != "retired" or source.running_services:
                        raise UnknownActivation("source retirement is unconfirmed")
                    self._save(state, "source-retired", source_activation="retired")
                    self.source.record(state)

            if state["phase"] in ("source-retired", "source-fenced"):
                self._assert_target_inactive(target)
                self._save(state, "target-activation-intent")
                target.record(state)

            if state["phase"] == "target-activation-intent":
                inspection = target.inspect()
                if not inspection.activation_intent:
                    target.record(state)
                    inspection = target.inspect()
                if not inspection.activation_intent:
                    raise UnknownActivation("target activation intent is not durable")
                self.provisioner.mark_activated()
                if inspection.activation in ("absent", "suspended") and not inspection.running_services:
                    target.start()
                elif inspection.activation != "active":
                    raise UnknownActivation("target activation state is unknown")
                self._save(state, "target-starting", target_activation="active")
                target.record(state)

            if state["phase"] == "target-starting":
                inspection = target.inspect()
                if (
                    inspection.activation in ("absent", "suspended", "active")
                    and set(inspection.running_services) != set(self.agent_ids)
                ):
                    target.start()
                    inspection = target.inspect()
                if inspection.activation != "active":
                    raise UnknownActivation("target activation is not confirmed active")
                if set(inspection.running_services) != set(self.agent_ids):
                    raise RuntimeError("target services are not all running")
                if set(inspection.ready_services) != set(self.agent_ids):
                    raise RuntimeError("target lifecycle readiness is incomplete")
                self._save(
                    state,
                    "completed",
                    target_activation="active",
                    target_ready_services=list(inspection.ready_services),
                    failure=None,
                )
                target.record(state)
            return self._result(state)
        except BaseException as exc:
            if state.get("destination") is None:
                try:
                    provision = self.provisioner.status()
                except (OSError, ValueError):
                    pass
                else:
                    if provision.droplet_id is not None:
                        state["destination"] = self._destination_dict(provision)
                        state["remaining_billable_resources"] = list(
                            provision.remaining_billable_resources
                        )
            self._failure(state, exc)
            raise

    def recover_source(self) -> MigrationResult:
        state = self._load()
        if state["phase"] not in _PRE_ACTIVATION_PHASES:
            raise RuntimeError("source recovery is not allowed at this migration phase")
        _, target = self._target(state)
        target_state = target.inspect()
        if (
            target_state.activation not in ("absent", "suspended")
            or target_state.running_services
            or target_state.activation_intent
        ):
            raise UnknownActivation("target is not proven never-activated")
        source_state = self.source.inspect()
        if source_state.activation == "retired":
            raise RuntimeError("retired source cannot be automatically recovered")
        if source_state.running_services:
            raise RuntimeError("source already has running services")
        self.source.recover(
            tuple(state["source_prior_services"]), state["migration_id"]
        )
        confirmed = self.source.inspect()
        if confirmed.activation != "active" or set(confirmed.running_services) != set(
            state["source_prior_services"]
        ):
            raise UnknownActivation("source recovery is unconfirmed")
        self._save(state, "recovered", source_activation="active", failure=None)
        self.source.record(state)
        return self._result(state)

    def fence_source(self, evidence: str) -> MigrationResult:
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("source fencing evidence must be nonempty text")
        state = self._load()
        if state["phase"] not in ("target-restored", "source-retire-intent"):
            raise RuntimeError("source fencing is only allowed after exact target restore")
        try:
            self.source.inspect()
        except HostUnreachable:
            pass
        else:
            raise RuntimeError("source is reachable; retire it normally instead of fencing")
        self._save(
            state,
            source_fence={"recorded_at": _now(), "evidence": evidence.strip()},
            failure=None,
        )
        return self._result(state)

    def status(self) -> MigrationResult:
        return self._result(self._load())

    @staticmethod
    def _result(state: Mapping[str, Any]) -> MigrationResult:
        destination = state.get("destination") or {}
        return MigrationResult(
            migration_id=state["migration_id"],
            phase=state["phase"],
            deployment_id=state["deployment_id"],
            droplet_id=destination.get("droplet_id"),
            snapshot_id=state.get("snapshot_id"),
            snapshot_manifest_sha256=state.get("snapshot_manifest_sha256"),
            source_activation=state.get("source_activation"),
            target_activation=state.get("target_activation"),
            target_ready_services=tuple(state.get("target_ready_services", ())),
            remaining_billable_resources=tuple(
                state.get("remaining_billable_resources", ())
            ),
            failure=state.get("failure"),
        )
