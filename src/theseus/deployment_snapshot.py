"""Consistent stopped deployment snapshots and inactive local restore."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import tarfile
import tempfile
from typing import Any, Iterator
from uuid import UUID, uuid4

from theseus.deployment_bundle import LOCK_FILE
from theseus.deployment_control import DeploymentController, _atomic_json, operation_lock


SNAPSHOT_FORMAT_VERSION = 1
STATE_SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
ARCHIVE_FILE = "data.tar.gz"
CAPTURE_FILE = "capture.json"
CONSISTENCY = "clean-stopped"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in reversed(directories):
        _fsync_directory(directory)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _walk(root: Path) -> Iterator[Path]:
    """Walk without following links and reject unsupported filesystem objects."""
    root = root.resolve()
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir(), key=lambda item: item.name, reverse=True):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                link = os.readlink(path)
                if os.path.isabs(link):
                    raise ValueError(f"absolute data symlink is not portable: {path}")
                target = (path.parent / link).resolve(strict=False)
                if not _inside(target, root):
                    raise ValueError(f"data symlink escapes snapshot root: {path}")
            elif stat.S_ISDIR(mode):
                pending.append(path)
            elif not stat.S_ISREG(mode):
                raise ValueError(f"unsupported data file type: {path}")
            yield path


def inventory(root: Path) -> list[dict[str, Any]]:
    """Stable inventory of every directory, file, and internal symlink."""
    root = Path(root).resolve()
    entries: list[dict[str, Any]] = []
    inodes: dict[tuple[int, int], str] = {}
    paths = sorted(_walk(root), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        metadata = path.lstat()
        entry: dict[str, Any] = {
            "path": path.relative_to(root).as_posix(),
            "mode": stat.S_IMODE(metadata.st_mode),
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
        }
        if stat.S_ISDIR(metadata.st_mode):
            entry["type"] = "directory"
        elif stat.S_ISLNK(metadata.st_mode):
            entry.update(type="symlink", target=os.readlink(path))
        else:
            entry.update(type="file", size=metadata.st_size, sha256=_sha256_file(path))
            inode = (metadata.st_dev, metadata.st_ino)
            if metadata.st_nlink > 1 and inode in inodes:
                entry["hardlink_to"] = inodes[inode]
            else:
                inodes[inode] = entry["path"]
        entries.append(entry)
    return sorted(entries, key=lambda item: item["path"])


def _verify_inventory(root: Path, expected: list[dict[str, Any]]) -> None:
    actual = inventory(root)
    if actual != expected:
        expected_by_path = {entry.get("path"): entry for entry in expected}
        actual_by_path = {entry.get("path"): entry for entry in actual}
        paths = sorted(set(expected_by_path) | set(actual_by_path))
        changed = next(
            (path for path in paths if expected_by_path.get(path) != actual_by_path.get(path)),
            "unknown",
        )
        raise ValueError(f"snapshot inventory mismatch at {changed}")


def validate_state_copy(data: Path) -> None:
    """Validate JSONL and SQLite in a disposable copy that may be checkpointed."""
    data = Path(data)
    for path in data.rglob("*.jsonl"):
        if path.is_symlink() or not path.is_file():
            continue
        number = 1
        try:
            with path.open("r", encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    if line.strip():
                        json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
    for path in data.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.name.endswith(("-wal", "-shm")):
            continue
        with path.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                continue
        try:
            with sqlite3.connect(path) as connection:
                result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"invalid SQLite database {path}: {exc}") from exc
        if result != "ok":
            raise ValueError(f"SQLite integrity check failed for {path}: {result}")


def _copy_tree(source: Path, destination: Path) -> None:
    """Copy ownership, modes, symlinks, and hard links without following links."""
    source = Path(source).resolve()
    destination = Path(destination)
    linked: dict[tuple[int, int], Path] = {}

    def copy_file(source_name: str, destination_name: str) -> str:
        source_path = Path(source_name)
        destination_path = Path(destination_name)
        metadata = source_path.stat(follow_symlinks=False)
        inode = (metadata.st_dev, metadata.st_ino)
        existing = linked.get(inode) if metadata.st_nlink > 1 else None
        if existing is not None:
            os.link(existing, destination_path)
        else:
            shutil.copy2(source_path, destination_path, follow_symlinks=False)
            if metadata.st_nlink > 1:
                linked[inode] = destination_path
        return str(destination_path)

    shutil.copytree(source, destination, symlinks=True, copy_function=copy_file)
    source_paths = [source, *list(_walk(source))]
    for source_path in source_paths:
        target = destination / source_path.relative_to(source)
        metadata = source_path.lstat()
        try:
            os.chown(target, metadata.st_uid, metadata.st_gid, follow_symlinks=False)
        except PermissionError as exc:
            copied = target.lstat()
            if (copied.st_uid, copied.st_gid) != (metadata.st_uid, metadata.st_gid):
                raise PermissionError(
                    f"cannot preserve snapshot ownership for {source_path}"
                ) from exc


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_id: str
    path: Path
    manifest: dict[str, Any]


class DeploymentSnapshots:
    """Host-only full snapshots for one assembled deployment bundle."""

    def __init__(
        self,
        controller: DeploymentController,
        bundle: Path,
        *,
        disk_usage=shutil.disk_usage,
    ) -> None:
        self.controller = controller
        self.bundle = Path(bundle).resolve()
        self.disk_usage = disk_usage
        self.deployment = self._read_json(self.bundle / "deployment.json")
        self.release = self._read_json(self.bundle / LOCK_FILE)
        if self.deployment.get("_generated_by") != "theseus-compose-assembler":
            raise ValueError("bundle is not an assembler-owned deployment")
        if self.release.get("_generated_by") != "theseus-compose-builder":
            raise ValueError("bundle has no completed immutable build identity")
        deployment_id = self.deployment.get("deployment_id")
        if deployment_id != self.controller.deployment_id:
            raise ValueError("bundle and controller deployment IDs differ")
        if self.release.get("deployment_id") != deployment_id:
            raise ValueError("deployment lock belongs to another deployment")
        if self.release.get("resolved_spec_sha256") != self.deployment.get(
            "resolved_spec_sha256"
        ):
            raise ValueError("deployment lock and resolved spec identity differ")
        if self.release.get("platform") != self.deployment.get("platform"):
            raise ValueError("deployment lock and platform identity differ")
        if sorted(self.controller.agent_ids) != sorted(self.deployment.get("agents", {})):
            raise ValueError("bundle and controller agent IDs differ")
        if not isinstance(self.release.get("images"), list) or not self.release["images"]:
            raise ValueError("deployment lock has no immutable image identity")
        self.release_id = _sha256_json(self.release)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object: {path}")
        return value

    def create(
        self, *, leave_stopped: bool = False, timeout_seconds: int = 30
    ) -> SnapshotResult:
        root = self.controller.root.resolve()
        data = root / "data"
        snapshots = root / "snapshots"
        if not data.is_dir() or data.is_symlink():
            raise ValueError(f"deployment data tree is missing or unsafe: {data}")
        snapshots.mkdir(parents=True, exist_ok=True)
        if _inside(snapshots.resolve(), data.resolve()):
            raise ValueError("snapshot staging must be outside live data")
        snapshot_id = uuid4().hex
        staging = snapshots / f".{snapshot_id}.staging"
        destination = snapshots / snapshot_id
        if (
            staging.exists()
            or staging.is_symlink()
            or destination.exists()
            or destination.is_symlink()
        ):
            raise RuntimeError("new snapshot paths are unexpectedly occupied")
        captured_at = ""
        resumed = False

        with operation_lock(self.controller.control_dir):
            interrupted = self.controller.interrupted_operation()
            if interrupted is not None:
                raise RuntimeError(
                    f"inspect interrupted operation {interrupted.operation_id} before backup"
                )
            prior = self.controller.running_services()
            operation = self.controller.journal.begin("backup", prior)
            try:
                activation = self.controller.activation.read()
                if activation is None or activation.state != "retired":
                    self.controller.activation.set(
                        "suspended", reason=f"backup {operation.operation_id}"
                    )
                self.controller.journal.transition(operation.operation_id, "draining")
                self.controller.clean_stop_services(prior, timeout_seconds=timeout_seconds)
                self.controller.journal.transition(
                    operation.operation_id,
                    "copying",
                    detail=json.dumps({"snapshot_id": snapshot_id}, sort_keys=True),
                )
                _walk_all(data)
                staging.mkdir()
                _copy_tree(data, staging / "data")
                _fsync_tree(staging)
                captured_at = _now()
                capture = {
                    "snapshot_id": snapshot_id,
                    "deployment_id": self.controller.deployment_id,
                    "captured_at": captured_at,
                    "consistency": CONSISTENCY,
                    "prior_running_services": list(prior),
                }
                _atomic_json(staging / CAPTURE_FILE, capture)
                _fsync_tree(staging)
                os.replace(staging, destination)
                _fsync_directory(snapshots)
                self.controller.journal.transition(
                    operation.operation_id,
                    "captured",
                    detail=json.dumps(
                        {"snapshot_id": snapshot_id, "path": str(destination)},
                        sort_keys=True,
                    ),
                )

                if prior and not leave_stopped:
                    if activation is not None and activation.state == "retired":
                        raise RuntimeError("refusing to resume services for a retired deployment")
                    self.controller.journal.transition(operation.operation_id, "resuming")
                    self.controller.activation.set(
                        "active", reason=f"backup captured {snapshot_id}"
                    )
                    self.controller.resume_services(prior)
                    resumed = True

                self.controller.journal.transition(operation.operation_id, "validating")
                self._validate_evidence(destination / "data", snapshots)
                files = inventory(destination / "data")
                self.controller.journal.transition(operation.operation_id, "archiving")
                archive = destination / ARCHIVE_FILE
                self._archive(destination / "data", archive)
                manifest = self._manifest(
                    snapshot_id=snapshot_id,
                    captured_at=captured_at,
                    prior=prior,
                    files=files,
                    archive=archive,
                )
                _atomic_json(destination / MANIFEST_FILE, manifest)
                _fsync_tree(destination)
                self.controller.journal.transition(
                    operation.operation_id,
                    "completed",
                    detail=json.dumps(
                        {"snapshot_id": snapshot_id, "path": str(destination)},
                        sort_keys=True,
                    ),
                )
                return SnapshotResult(snapshot_id, destination, manifest)
            except BaseException as exc:
                current = self.controller.journal.read()
                if current is not None and not current.finished:
                    self.controller.journal.transition(
                        operation.operation_id,
                        "failed",
                        detail=json.dumps(
                            {
                                "snapshot_id": snapshot_id,
                                "error": f"{type(exc).__name__}: {exc}",
                                "services_resumed": resumed,
                            },
                            sort_keys=True,
                        ),
                    )
                raise

    def restore(self, snapshot: Path) -> Path:
        snapshot = Path(snapshot)
        if snapshot.is_symlink() or not snapshot.is_dir():
            raise ValueError("snapshot must be a real directory")
        snapshot = snapshot.resolve()
        if (snapshot / MANIFEST_FILE).is_symlink():
            raise ValueError("snapshot manifest cannot be a symlink")
        manifest = self._read_json(snapshot / MANIFEST_FILE)
        self._validate_manifest_identity(manifest)
        root = self.controller.root.resolve()
        data = root / "data"
        archive = snapshot / ARCHIVE_FILE

        with operation_lock(self.controller.control_dir):
            interrupted = self.controller.interrupted_operation()
            if interrupted is not None:
                raise RuntimeError(
                    f"inspect interrupted operation {interrupted.operation_id} before restore"
                )
            if self.controller.running_services():
                raise RuntimeError("restore target has running services")
            activation = self.controller.activation.read()
            if activation is not None and activation.state == "active":
                raise RuntimeError("restore target is active")
            if data.is_symlink() or (
                data.exists() and (not data.is_dir() or any(data.iterdir()))
            ):
                raise RuntimeError("restore target data is not empty")
            operation = self.controller.journal.begin("restore", ())
            stage = root / f".restore-{operation.operation_id}"
            try:
                self.controller.activation.set(
                    "suspended", reason=f"restore {operation.operation_id}"
                )
                self.controller.journal.transition(operation.operation_id, "verifying")
                self._verify_archive(manifest, archive)
                required = sum(
                    entry.get("size", 0)
                    for entry in manifest["files"]
                    if entry.get("type") == "file"
                )
                available = self.disk_usage(root).free
                if available < required + max(required // 10, 1024 * 1024):
                    raise OSError(
                        f"insufficient free space: need {required} bytes plus safety margin"
                    )
                self._require_secrets(manifest)
                if stage.exists() or stage.is_symlink():
                    raise RuntimeError(f"restore staging already exists: {stage}")
                stage.mkdir()
                self.controller.journal.transition(operation.operation_id, "extracting")
                self._extract_safe(archive, stage, max_uncompressed_bytes=required)
                restored = stage / "data"
                _verify_inventory(restored, manifest["files"])
                self.controller.journal.transition(operation.operation_id, "validating")
                validate_state_copy(restored)
                _fsync_tree(stage)
                self.controller.journal.transition(operation.operation_id, "publishing")
                if data.exists():
                    data.rmdir()
                os.replace(restored, data)
                _fsync_directory(root)
                stage.rmdir()
                self.controller.journal.transition(
                    operation.operation_id,
                    "completed",
                    detail=json.dumps(
                        {"snapshot_id": manifest["snapshot_id"], "data": str(data)},
                        sort_keys=True,
                    ),
                )
                return data
            except BaseException as exc:
                current = self.controller.journal.read()
                if current is not None and not current.finished:
                    self.controller.journal.transition(
                        operation.operation_id,
                        "failed",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                raise

    def _validate_evidence(self, evidence: Path, snapshots: Path) -> None:
        scratch = Path(tempfile.mkdtemp(prefix=".snapshot-validate-", dir=snapshots))
        try:
            _copy_tree(evidence, scratch / "data")
            validate_state_copy(scratch / "data")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    @staticmethod
    def _archive(data: Path, archive: Path) -> None:
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        try:
            with tarfile.open(temporary, "w:gz", dereference=False) as stream:
                stream.add(data, arcname="data", recursive=True)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)

    def _manifest(
        self,
        *,
        snapshot_id: str,
        captured_at: str,
        prior: tuple[str, ...],
        files: list[dict[str, Any]],
        archive: Path,
    ) -> dict[str, Any]:
        return {
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "deployment_id": self.controller.deployment_id,
            "agent_ids": sorted(self.deployment["agents"]),
            "release_id": self.release_id,
            "resolved_spec_sha256": self.deployment["resolved_spec_sha256"],
            "platform": self.deployment["platform"],
            "runtime": self.deployment["runtime"],
            "theseus": self.deployment["theseus"],
            "images": self.release["images"],
            "captured_at": captured_at,
            "consistency": CONSISTENCY,
            "data_root": "data",
            "prior_running_services": list(prior),
            "required_secrets": sorted(self.deployment.get("required_secrets", [])),
            "files": files,
            "archive": {
                "name": ARCHIVE_FILE,
                "size": archive.stat().st_size,
                "sha256": _sha256_file(archive),
            },
        }

    def _validate_manifest_identity(self, manifest: dict[str, Any]) -> None:
        required = {
            "snapshot_format_version", "state_schema_version", "snapshot_id",
            "deployment_id", "agent_ids", "release_id", "resolved_spec_sha256",
            "platform", "runtime", "theseus", "images", "captured_at",
            "consistency", "data_root", "prior_running_services",
            "required_secrets", "files", "archive",
        }
        missing = sorted(required - set(manifest))
        if missing:
            raise ValueError(f"snapshot manifest is missing: {', '.join(missing)}")
        if manifest["snapshot_format_version"] != SNAPSHOT_FORMAT_VERSION:
            raise ValueError("unsupported snapshot format version")
        if manifest["state_schema_version"] != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported state schema version")
        try:
            UUID(hex=manifest["snapshot_id"])
            datetime.fromisoformat(manifest["captured_at"])
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("snapshot ID or capture time is invalid") from exc
        comparisons = {
            "deployment_id": self.controller.deployment_id,
            "release_id": self.release_id,
            "resolved_spec_sha256": self.deployment["resolved_spec_sha256"],
            "platform": self.deployment["platform"],
            "runtime": self.deployment["runtime"],
            "theseus": self.deployment["theseus"],
            "images": self.release["images"],
            "consistency": CONSISTENCY,
            "data_root": "data",
        }
        for name, expected in comparisons.items():
            if manifest.get(name) != expected:
                raise ValueError(f"snapshot {name} does not match target deployment")
        if sorted(manifest["agent_ids"]) != sorted(self.deployment["agents"]):
            raise ValueError("snapshot agent IDs do not match target deployment")
        if sorted(manifest["required_secrets"]) != sorted(
            self.deployment.get("required_secrets", [])
        ):
            raise ValueError("snapshot required secret names do not match target deployment")
        if not isinstance(manifest["files"], list):
            raise ValueError("snapshot file inventory must be a list")
        seen: set[str] = set()
        for entry in manifest["files"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("snapshot file inventory contains an invalid entry")
            path = PurePosixPath(entry["path"])
            if path.is_absolute() or ".." in path.parts or not path.parts or entry["path"] in seen:
                raise ValueError("snapshot file inventory contains an unsafe or duplicate path")
            seen.add(entry["path"])
            if entry.get("type") not in ("file", "directory", "symlink"):
                raise ValueError(f"snapshot inventory type is invalid at {entry['path']}")
            if entry["type"] == "file" and (
                type(entry.get("size")) is not int
                or entry["size"] < 0
                or not isinstance(entry.get("sha256"), str)
                or _SHA256.fullmatch(entry["sha256"]) is None
            ):
                raise ValueError(f"snapshot file checksum is invalid at {entry['path']}")
            if entry["type"] == "symlink" and not isinstance(entry.get("target"), str):
                raise ValueError(f"snapshot symlink target is invalid at {entry['path']}")
        archive = manifest["archive"]
        if (
            not isinstance(archive, dict)
            or type(archive.get("size")) is not int
            or archive["size"] < 0
            or not isinstance(archive.get("sha256"), str)
            or _SHA256.fullmatch(archive["sha256"]) is None
        ):
            raise ValueError("snapshot archive identity is invalid")

    def _require_secrets(self, manifest: dict[str, Any]) -> None:
        missing = []
        secrets = self.controller.root / "secrets"
        for name in manifest["required_secrets"]:
            path = secrets / name
            if not path.is_file() or path.is_symlink():
                missing.append(name)
        if missing:
            raise ValueError(f"required target secrets are missing: {', '.join(missing)}")

    @staticmethod
    def _verify_archive(manifest: dict[str, Any], archive: Path) -> None:
        expected = manifest["archive"]
        if expected.get("name") != ARCHIVE_FILE:
            raise ValueError("snapshot archive name is unsupported")
        if not archive.is_file() or archive.is_symlink():
            raise ValueError("snapshot archive is missing or unsafe")
        if archive.stat().st_size != expected.get("size"):
            raise ValueError("snapshot archive size mismatch")
        if _sha256_file(archive) != expected.get("sha256"):
            raise ValueError("snapshot archive checksum mismatch")

    @staticmethod
    def _extract_safe(
        archive: Path,
        destination: Path,
        *,
        max_uncompressed_bytes: int | None = None,
    ) -> None:
        seen: set[PurePosixPath] = set()
        uncompressed = 0
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream.getmembers():
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or not name.parts:
                    raise ValueError(f"unsafe archive path: {member.name}")
                if name.parts[0] != "data" or name in seen:
                    raise ValueError(f"archive member is outside data or duplicated: {member.name}")
                seen.add(name)
                if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
                    raise ValueError(f"archive contains a device or special file: {member.name}")
                if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise ValueError(f"unsupported archive member: {member.name}")
                if member.isfile():
                    uncompressed += member.size
                    if (
                        max_uncompressed_bytes is not None
                        and uncompressed > max_uncompressed_bytes
                    ):
                        raise ValueError("archive expands beyond its declared file inventory")
                if member.issym():
                    target = name.parent.joinpath(PurePosixPath(member.linkname))
                    normalized = _normalize_posix(target)
                    if not normalized.parts or normalized.parts[0] != "data":
                        raise ValueError(f"archive symlink escapes data: {member.name}")
                if member.islnk():
                    target = _normalize_posix(PurePosixPath(member.linkname))
                    if not target.parts or target.parts[0] != "data":
                        raise ValueError(f"archive hard link escapes data: {member.name}")
            stream.extractall(destination, numeric_owner=True, filter="fully_trusted")


def _normalize_posix(path: PurePosixPath) -> PurePosixPath:
    parts: list[str] = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return PurePosixPath("..")
            parts.pop()
        else:
            parts.append(part)
    return PurePosixPath(*parts)


def _walk_all(root: Path) -> None:
    for _ in _walk(root):
        pass
