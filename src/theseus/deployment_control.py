"""Host-owned activation, operation journals, and managed Compose control."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Iterator, Sequence
from uuid import uuid4


ACTIVATION_FILE = "activation.json"
OPERATION_FILE = "operation.json"
OPERATION_LOCK = "operation.lock"
LIFECYCLE_STATUS = "lifecycle-status.json"
MIGRATION_FILE = "migration.json"
ACTIVATION_STATES = ("active", "suspended", "retired")
OPERATION_KINDS = ("start", "stop", "backup", "restore", "migration")
TERMINAL_PHASES = ("completed", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ActivationRecord:
    deployment_id: str
    state: str
    generation: str
    updated_at: str
    reason: str | None = None


class ActivationStore:
    """Durable single-host permission checked by every managed agent boot."""

    def __init__(self, control_dir: Path, deployment_id: str) -> None:
        self.control_dir = Path(control_dir)
        self.deployment_id = deployment_id
        self.path = self.control_dir / ACTIVATION_FILE

    def read(self) -> ActivationRecord | None:
        if not self.path.is_file():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        record = ActivationRecord(**data)
        if record.deployment_id != self.deployment_id:
            raise ValueError(
                f"activation belongs to {record.deployment_id!r}, not {self.deployment_id!r}"
            )
        if record.state not in ACTIVATION_STATES:
            raise ValueError(f"invalid activation state {record.state!r}")
        return record

    def require_active(self) -> ActivationRecord:
        record = self.read()
        if record is None:
            raise PermissionError("managed activation permission is absent")
        if record.state != "active":
            raise PermissionError(f"managed activation is {record.state}")
        return record

    def set(self, state: str, *, reason: str | None = None) -> ActivationRecord:
        if state not in ACTIVATION_STATES:
            raise ValueError(f"activation state must be one of {', '.join(ACTIVATION_STATES)}")
        current = self.read()
        if current is not None and current.state == "retired" and state != "retired":
            raise PermissionError("retired deployments require an explicit recovery procedure")
        record = ActivationRecord(
            deployment_id=self.deployment_id,
            state=state,
            generation=uuid4().hex,
            updated_at=_now(),
            reason=reason,
        )
        _atomic_json(self.path, asdict(record))
        return record


@dataclass(frozen=True)
class OperationRecord:
    operation_id: str
    deployment_id: str
    kind: str
    phase: str
    prior_running_services: tuple[str, ...]
    started_at: str
    updated_at: str
    detail: str | None = None

    @property
    def finished(self) -> bool:
        return self.phase in TERMINAL_PHASES


class OperationJournal:
    """One durable deployment operation whose phase is never inferred on recovery."""

    def __init__(self, control_dir: Path, deployment_id: str) -> None:
        self.path = Path(control_dir) / OPERATION_FILE
        self.deployment_id = deployment_id

    def read(self) -> OperationRecord | None:
        if not self.path.is_file():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["prior_running_services"] = tuple(data.get("prior_running_services", ()))
        record = OperationRecord(**data)
        if record.deployment_id != self.deployment_id:
            raise ValueError(
                f"operation belongs to {record.deployment_id!r}, not {self.deployment_id!r}"
            )
        return record

    def begin(self, kind: str, prior_running_services: Sequence[str]) -> OperationRecord:
        if kind not in OPERATION_KINDS:
            raise ValueError(f"unsupported deployment operation {kind!r}")
        current = self.read()
        if current is not None and not current.finished:
            raise RuntimeError(
                f"operation {current.operation_id} is interrupted at phase {current.phase!r}"
            )
        timestamp = _now()
        record = OperationRecord(
            operation_id=uuid4().hex,
            deployment_id=self.deployment_id,
            kind=kind,
            phase="started",
            prior_running_services=tuple(prior_running_services),
            started_at=timestamp,
            updated_at=timestamp,
        )
        _atomic_json(self.path, asdict(record))
        return record

    def transition(
        self, operation_id: str, phase: str, *, detail: str | None = None
    ) -> OperationRecord:
        current = self.read()
        if current is None or current.operation_id != operation_id:
            raise RuntimeError("stale or unknown operation transition")
        if current.finished:
            raise RuntimeError(f"operation is already {current.phase}")
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("operation phase must be nonempty text")
        record = OperationRecord(
            operation_id=current.operation_id,
            deployment_id=current.deployment_id,
            kind=current.kind,
            phase=phase,
            prior_running_services=current.prior_running_services,
            started_at=current.started_at,
            updated_at=_now(),
            detail=detail,
        )
        _atomic_json(self.path, asdict(record))
        return record


@contextmanager
def operation_lock(control_dir: Path) -> Iterator[None]:
    """Serialize every state-changing host operation for one deployment."""
    control_dir = Path(control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    with (control_dir / OPERATION_LOCK).open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another deployment operation is in progress") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@dataclass(frozen=True)
class LifecycleRecord:
    run_id: str
    agent_id: str
    pid: int
    state: str
    clean: bool | None
    started_at: str
    updated_at: str
    detail: str | None = None


class LifecycleStatusStore:
    """Per-run acknowledgement; run IDs prevent old clean exits authorizing capture."""

    def __init__(self, path: Path, agent_id: str) -> None:
        self.path = Path(path)
        self.agent_id = agent_id

    def read(self) -> LifecycleRecord | None:
        if not self.path.is_file():
            return None
        record = LifecycleRecord(**json.loads(self.path.read_text(encoding="utf-8")))
        if record.agent_id != self.agent_id:
            raise ValueError(
                f"lifecycle status belongs to {record.agent_id!r}, not {self.agent_id!r}"
            )
        return record

    def running(self) -> LifecycleRecord:
        timestamp = _now()
        record = LifecycleRecord(
            run_id=uuid4().hex,
            agent_id=self.agent_id,
            pid=os.getpid(),
            state="running",
            clean=None,
            started_at=timestamp,
            updated_at=timestamp,
        )
        _atomic_json(self.path, asdict(record))
        return record

    def stopped(
        self, run_id: str, *, clean: bool, detail: str | None = None
    ) -> LifecycleRecord:
        current = self.read()
        if current is None or current.run_id != run_id:
            raise RuntimeError("refusing stale lifecycle acknowledgement")
        record = LifecycleRecord(
            run_id=current.run_id,
            agent_id=current.agent_id,
            pid=current.pid,
            state="stopped",
            clean=clean,
            started_at=current.started_at,
            updated_at=_now(),
            detail=detail,
        )
        _atomic_json(self.path, asdict(record))
        return record


class DeploymentController:
    """Host-only Compose start/stop/status with activation and clean-stop checks."""

    def __init__(
        self,
        *,
        root: Path,
        deployment_id: str,
        agent_ids: Sequence[str],
        compose_file: Path,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.root = Path(root)
        self.deployment_id = deployment_id
        self.agent_ids = tuple(agent_ids)
        self.compose_file = Path(compose_file)
        self.control_dir = self.root / "control"
        self.activation = ActivationStore(self.control_dir, deployment_id)
        self.journal = OperationJournal(self.control_dir, deployment_id)
        self._run = run

    def _compose(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return self._run(
            ["docker", "compose", "-f", str(self.compose_file), *arguments],
            check=True,
            text=True,
            capture_output=True,
            env={**os.environ, "THESEUS_DEPLOYMENT_ROOT": str(self.root)},
        )

    def running_services(self) -> tuple[str, ...]:
        result = self._compose("ps", "--services", "--status", "running")
        return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())

    def status(self) -> dict[str, Any]:
        activation = self.activation.read()
        operation = self.journal.read()
        return {
            "deployment_id": self.deployment_id,
            "activation": asdict(activation) if activation is not None else None,
            "operation": asdict(operation) if operation is not None else None,
            "running_services": list(self.running_services()),
            "lifecycle": {
                agent_id: self._lifecycle(agent_id) for agent_id in self.agent_ids
            },
        }

    def interrupted_operation(self) -> OperationRecord | None:
        record = self.journal.read()
        return record if record is not None and not record.finished else None

    @contextmanager
    def operation(self, kind: str) -> Iterator[OperationRecord]:
        """Journal a host backup/restore/migration while holding the shared lock.

        The caller may persist detailed phases through ``self.journal.transition``.
        An interrupted process leaves its last nonterminal phase for recovery review.
        """
        if kind not in ("backup", "restore", "migration"):
            raise ValueError("custom managed operation must be backup, restore, or migration")
        with operation_lock(self.control_dir):
            interrupted = self.interrupted_operation()
            if interrupted is not None:
                raise RuntimeError(
                    f"inspect interrupted operation {interrupted.operation_id} at "
                    f"phase {interrupted.phase!r} before resuming"
                )
            operation = self.journal.begin(kind, self.running_services())
            try:
                yield operation
            except BaseException as exc:
                self.journal.transition(
                    operation.operation_id,
                    "failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                raise
            else:
                self.journal.transition(operation.operation_id, "completed")

    def start(self, services: Sequence[str] = ()) -> OperationRecord:
        with operation_lock(self.control_dir):
            self._require_migration_start_permission()
            interrupted = self.interrupted_operation()
            if interrupted is not None:
                raise RuntimeError(
                    f"inspect interrupted operation {interrupted.operation_id} at "
                    f"phase {interrupted.phase!r} before starting"
                )
            prior = self.running_services()
            operation = self.journal.begin("start", prior)
            try:
                current = self.activation.read()
                if current is not None and current.state == "retired":
                    raise PermissionError("retired deployments require an explicit recovery procedure")
                targets = tuple(services) or tuple(
                    agent_id for agent_id in self.agent_ids if agent_id not in prior
                )
                self.journal.transition(operation.operation_id, "preflight")
                for service in targets:
                    self._preflight_service(service)
                self.activation.set("active", reason=f"start {operation.operation_id}")
                self.journal.transition(operation.operation_id, "starting")
                self._compose("up", "-d", *services)
                return self.journal.transition(operation.operation_id, "completed")
            except Exception as exc:
                try:
                    activation = self.activation.read()
                    if activation is None or activation.state != "retired":
                        self.activation.set("suspended", reason="start failed")
                except Exception:
                    pass
                self.journal.transition(
                    operation.operation_id, "failed", detail=f"{type(exc).__name__}: {exc}"
                )
                raise

    def resume_start(self) -> OperationRecord:
        """Reconcile an interrupted target start after durable migration intent."""
        with operation_lock(self.control_dir):
            migration = self._require_migration_start_permission()
            if migration is None or migration.get("host_role") != "target":
                raise PermissionError("start recovery requires target migration intent")
            operation = self.interrupted_operation()
            if operation is None or operation.kind != "start":
                raise RuntimeError("there is no interrupted start to resume")
            if operation.phase not in ("started", "preflight", "starting"):
                raise RuntimeError(
                    f"interrupted start phase {operation.phase!r} cannot be resumed"
                )
            current = self.activation.read()
            if current is not None and current.state == "retired":
                raise PermissionError("retired deployments require an explicit recovery procedure")
            missing = tuple(
                agent_id
                for agent_id in self.agent_ids
                if agent_id not in self.running_services()
            )
            try:
                for service in missing:
                    self._preflight_service(service)
                self.activation.set("active", reason=f"resume start {operation.operation_id}")
                self.journal.transition(operation.operation_id, "starting")
                self._compose("up", "-d", *missing)
                return self.journal.transition(operation.operation_id, "completed")
            except Exception as exc:
                try:
                    self.activation.set("suspended", reason="resumed start failed")
                except Exception:
                    pass
                self.journal.transition(
                    operation.operation_id, "failed", detail=f"{type(exc).__name__}: {exc}"
                )
                raise

    def _require_migration_start_permission(self) -> dict[str, Any] | None:
        migration_path = self.control_dir / MIGRATION_FILE
        if not migration_path.is_file():
            return None
        migration = json.loads(migration_path.read_text(encoding="utf-8"))
        role = migration.get("host_role")
        phase = migration.get("phase")
        if role == "source" and phase not in ("source-recovery-intent", "recovered"):
            raise PermissionError(f"source start is blocked by migration phase {phase!r}")
        if role == "target" and phase not in (
            "target-activation-intent",
            "target-starting",
            "completed",
        ):
            raise PermissionError(f"target start is blocked by migration phase {phase!r}")
        return migration

    def stop(self, *, timeout_seconds: int = 30) -> OperationRecord:
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        with operation_lock(self.control_dir):
            interrupted = self.interrupted_operation()
            if interrupted is not None:
                raise RuntimeError(
                    f"inspect interrupted operation {interrupted.operation_id} before stopping"
                )
            prior = self.running_services()
            operation = self.journal.begin("stop", prior)
            try:
                activation = self.activation.read()
                if activation is None or activation.state != "retired":
                    self.activation.set("suspended", reason=f"stop {operation.operation_id}")
                self.journal.transition(operation.operation_id, "stopping")
                self.clean_stop_services(prior, timeout_seconds=timeout_seconds)
                return self.journal.transition(operation.operation_id, "completed")
            except Exception as exc:
                self.journal.transition(
                    operation.operation_id, "failed", detail=f"{type(exc).__name__}: {exc}"
                )
                raise

    def clean_stop_services(
        self, services: Sequence[str], *, timeout_seconds: int = 30
    ) -> None:
        """Drain selected services and require fresh clean acknowledgements.

        The caller owns the deployment operation lock and activation transition.
        This split lets backup use one journal entry for its whole stop/copy/resume
        transaction rather than nesting a separate stop operation.
        """
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        selected = tuple(services)
        unknown = sorted(set(selected) - set(self.agent_ids))
        if unknown:
            raise ValueError(f"unknown deployment services: {', '.join(unknown)}")
        if not selected:
            return
        expected_runs = {service: self._lifecycle(service) for service in selected}
        self._compose("stop", "--timeout", str(timeout_seconds), *selected)
        problems = []
        for service in selected:
            before = expected_runs[service]
            after = self._lifecycle(service)
            if (
                before is None
                or before.get("state") != "running"
                or before.get("clean") is not None
                or after is None
                or before.get("run_id") != after.get("run_id")
            ):
                problems.append(f"{service}: stale or missing lifecycle acknowledgement")
            elif after.get("state") != "stopped" or after.get("clean") is not True:
                problems.append(f"{service}: shutdown was not a clean drain")
        if problems:
            raise RuntimeError("; ".join(problems))

    def resume_services(self, services: Sequence[str]) -> None:
        """Resume exactly the selected services after an immutable local capture."""
        selected = tuple(services)
        unknown = sorted(set(selected) - set(self.agent_ids))
        if unknown:
            raise ValueError(f"unknown deployment services: {', '.join(unknown)}")
        if selected:
            self._compose("up", "-d", *selected)

    def retire(self, *, reason: str) -> ActivationRecord:
        with operation_lock(self.control_dir):
            if self.running_services():
                raise RuntimeError("stop all services cleanly before retirement")
            return self.activation.set("retired", reason=reason)

    def _lifecycle(self, agent_id: str) -> dict[str, Any] | None:
        path = self.root / "data" / "agents" / agent_id / "logs" / LIFECYCLE_STATUS
        record = LifecycleStatusStore(path, agent_id).read()
        return asdict(record) if record is not None else None

    def _preflight_service(self, agent_id: str) -> None:
        if agent_id not in self.agent_ids:
            raise ValueError(f"unknown deployment service {agent_id!r}")
        self._compose(
            "run", "--rm", "--no-deps", agent_id,
            "python", f"/app/agents/{agent_id}/agent.py",
            "--home", "/data/state",
            "--log-path", "/data/logs/stimulus_log.jsonl",
            "--preflight",
        )
