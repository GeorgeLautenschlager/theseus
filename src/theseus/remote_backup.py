"""Verified, resumable deployment backups in an S3-compatible object store."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Callable
from uuid import UUID

from theseus.backup_store import ObjectInfo, ObjectStore
from theseus.deployment_bundle import LOCK_FILE
from theseus.deployment_control import DeploymentController, _atomic_json
from theseus.deployment_snapshot import ARCHIVE_FILE, MANIFEST_FILE, DeploymentSnapshots


REMOTE_FORMAT_VERSION = 1
REMOTE_STATUS_FILE = "remote-status.json"
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
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _clean_identity(value: str) -> str:
    return value.removeprefix("sha256:")


def _safe_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive path: {member.name}")
    if member.ischr() or member.isblk() or member.isfifo():
        raise ValueError(f"unsupported archive member: {member.name}")
    if member.issym() or member.islnk():
        link = PurePosixPath(member.linkname)
        if link.is_absolute() or ".." in link.parts:
            raise ValueError(f"unsafe archive link: {member.name}")


def _extract_safe(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            _safe_member(member)
        stream.extractall(destination, members=members, filter="data")


def _deterministic_archive(source: Path, destination: Path, arcname: str) -> None:
    """Archive generated release files without timestamps or host ownership."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        return info

    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as stream:
                    stream.add(source, arcname=arcname, filter=normalize)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class BackupResult:
    snapshot_id: str
    captured_at: str
    uploaded_at: str | None
    total_size: int
    last_successful_snapshot_id: str | None
    failure: str | None = None


@dataclass(frozen=True)
class DownloadedBackup:
    snapshot_id: str
    root: Path
    snapshot: Path
    bundle: Path
    image_archives: tuple[Path, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ReleaseArtifacts:
    release_id: str
    bundle_archive: Path
    bundle_sha256: str
    image_archives: tuple[Path, ...]
    image_descriptors: tuple[dict[str, Any], ...]


class RemoteBackups:
    """Publish completion manifests last and restore only verified backups."""

    def __init__(
        self,
        store: ObjectStore,
        bundle: Path,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.store = store
        self.bundle = Path(bundle).resolve()
        self.deployment = _read_object(self.bundle / "deployment.json")
        self.release = _read_object(self.bundle / LOCK_FILE)
        if self.deployment.get("_generated_by") != "theseus-compose-assembler":
            raise ValueError("bundle is not an assembler-owned deployment")
        if self.release.get("_generated_by") != "theseus-compose-builder":
            raise ValueError("bundle has no completed immutable build identity")
        self.deployment_id = self.deployment.get("deployment_id")
        if not isinstance(self.deployment_id, str) or not self.deployment_id:
            raise ValueError("deployment ID is missing")
        if self.release.get("deployment_id") != self.deployment_id:
            raise ValueError("deployment lock belongs to another deployment")
        if self.release.get("resolved_spec_sha256") != self.deployment.get(
            "resolved_spec_sha256"
        ):
            raise ValueError("deployment lock and resolved spec identity differ")
        self.release_id = _sha256_json(self.release)
        self.run = run

    @property
    def prefix(self) -> str:
        return f"v1/deployments/{self.deployment_id}"

    def _snapshot_prefix(self, snapshot_id: str) -> str:
        self._validate_snapshot_id(snapshot_id)
        return f"{self.prefix}/snapshots/{snapshot_id}"

    @staticmethod
    def _validate_snapshot_id(snapshot_id: str) -> None:
        try:
            UUID(hex=snapshot_id)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("snapshot ID is invalid") from exc

    @staticmethod
    def _descriptor(key: str, path: Path, **extra: Any) -> dict[str, Any]:
        return {
            "key": key,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
            **extra,
        }

    @staticmethod
    def _verify_info(info: ObjectInfo | None, descriptor: dict[str, Any]) -> bool:
        return bool(
            info is not None
            and info.size == descriptor["size"]
            and info.metadata.get("sha256") == descriptor["sha256"]
        )

    def _ensure_file(self, descriptor: dict[str, Any], source: Path) -> None:
        current = self.store.head(descriptor["key"])
        if current is not None:
            if not self._verify_info(current, descriptor):
                raise ValueError(f"immutable object identity conflict: {descriptor['key']}")
            self._verify_stored_file(descriptor)
            return
        uploaded = self.store.put_file(
            descriptor["key"], source, {"sha256": descriptor["sha256"]}
        )
        if not self._verify_info(uploaded, descriptor):
            raise IOError(f"object verification failed after upload: {descriptor['key']}")
        self._verify_stored_file(descriptor)

    def _ensure_bytes(self, descriptor: dict[str, Any], value: bytes) -> None:
        current = self.store.head(descriptor["key"])
        if current is not None:
            if not self._verify_info(current, descriptor):
                raise ValueError(f"immutable object identity conflict: {descriptor['key']}")
            stored, _ = self.store.get_bytes(descriptor["key"])
            if len(stored) != descriptor["size"] or hashlib.sha256(stored).hexdigest() != descriptor["sha256"]:
                raise ValueError(f"stored object checksum mismatch: {descriptor['key']}")
            return
        uploaded = self.store.put_bytes(
            descriptor["key"], value, {"sha256": descriptor["sha256"]}
        )
        if not self._verify_info(uploaded, descriptor):
            raise IOError(f"object verification failed after upload: {descriptor['key']}")
        stored, _ = self.store.get_bytes(descriptor["key"])
        if len(stored) != descriptor["size"] or hashlib.sha256(stored).hexdigest() != descriptor["sha256"]:
            raise IOError(f"object checksum failed after upload: {descriptor['key']}")

    def _verify_stored_file(self, descriptor: dict[str, Any]) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as stream:
                temporary = Path(stream.name)
            self.store.get_file(descriptor["key"], temporary)
            if (
                temporary.stat().st_size != descriptor["size"]
                or _sha256_file(temporary) != descriptor["sha256"]
            ):
                raise ValueError(f"stored object checksum mismatch: {descriptor['key']}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _artifact_root(self, snapshot: Path) -> Path:
        root = snapshot.parent / ".artifacts"
        root.mkdir(exist_ok=True)
        return root

    def _release_archive(self, snapshot: Path) -> Path:
        path = self._artifact_root(snapshot) / f"release-{self.release_id}.tar.gz"
        _deterministic_archive(self.bundle, path, "bundle")
        return path

    def _inspect_image(self, reference: str) -> dict[str, Any]:
        result = self.run(
            ["docker", "image", "inspect", reference],
            check=True,
            text=True,
            capture_output=True,
        )
        value = json.loads(result.stdout)
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
            raise ValueError(f"Docker returned no unique identity for {reference}")
        return value[0]

    def _image_archives(self, snapshot: Path) -> list[tuple[Path, dict[str, Any]]]:
        artifacts = self._artifact_root(snapshot)
        results: list[tuple[Path, dict[str, Any]]] = []
        images = self.release.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError("deployment lock has no immutable image identity")
        for image in images:
            content_id = image.get("content_id")
            if not isinstance(content_id, str) or not content_id.startswith("sha256:"):
                raise ValueError("deployment image has no immutable content ID")
            inspected = self._inspect_image(content_id)
            if inspected.get("Id") != content_id:
                raise ValueError(f"Docker image identity changed for {content_id}")
            for name in ("os", "architecture"):
                expected = image.get(name)
                actual = inspected.get(name.title())
                if expected is not None and actual != expected:
                    raise ValueError(f"Docker image {name} differs for {content_id}")
            platform = image.get("platform", self.release.get("platform"))
            actual_platform = f"{inspected.get('Os')}/{inspected.get('Architecture')}"
            if not isinstance(platform, str) or "/".join(platform.split("/")[:2]) != actual_platform:
                raise ValueError(f"Docker image platform differs for {content_id}")
            raw = artifacts / f"image-{_clean_identity(content_id)}.tar"
            archive = raw.with_suffix(".tar.gz")
            identity_path = archive.with_suffix(archive.suffix + ".json")
            cached = None
            if archive.is_file() and identity_path.is_file():
                try:
                    cached = _read_object(identity_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    cached = None
            if not (
                cached is not None
                and cached.get("content_id") == content_id
                and cached.get("sha256") == _sha256_file(archive)
            ):
                archive.unlink(missing_ok=True)
                identity_path.unlink(missing_ok=True)
                raw.unlink(missing_ok=True)
                self.run(
                    ["docker", "image", "save", "--output", str(raw), content_id],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                if not raw.is_file() or raw.is_symlink():
                    raise ValueError(f"Docker did not export image {content_id}")
                temporary = archive.with_name(f".{archive.name}.tmp")
                try:
                    with raw.open("rb") as source, temporary.open("wb") as output:
                        with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as gz:
                            shutil.copyfileobj(source, gz)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, archive)
                    _atomic_json(
                        identity_path,
                        {"content_id": content_id, "sha256": _sha256_file(archive)},
                    )
                finally:
                    raw.unlink(missing_ok=True)
                    temporary.unlink(missing_ok=True)
            archive_sha = _sha256_file(archive)
            descriptor = self._descriptor(
                f"{self.prefix}/images/{archive_sha}.tar.gz",
                archive,
                archive_sha256=archive_sha,
                content_id=content_id,
                platform=platform,
                os=image.get("os"),
                architecture=image.get("architecture"),
            )
            results.append((archive, descriptor))
        return results

    def _status_path(self, snapshot: Path) -> Path:
        return snapshot / REMOTE_STATUS_FILE

    def stage_release(self, directory: Path) -> ReleaseArtifacts:
        """Export the exact bundle and images before migration downtime begins."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "preload"
        bundle = self._release_archive(marker)
        images = self._image_archives(marker)
        return ReleaseArtifacts(
            release_id=self.release_id,
            bundle_archive=bundle,
            bundle_sha256=_sha256_file(bundle),
            image_archives=tuple(path for path, _ in images),
            image_descriptors=tuple(descriptor for _, descriptor in images),
        )

    def _write_status(self, snapshot: Path, result: BackupResult, state: str) -> None:
        _atomic_json(self._status_path(snapshot), {"state": state, **asdict(result)})

    def list(self) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        for key in self.store.list_keys(f"{self.prefix}/snapshots/"):
            if not key.endswith(f"/{MANIFEST_FILE}"):
                continue
            try:
                value, _ = self.store.get_bytes(key)
                manifest = json.loads(value)
                self._validate_remote_manifest(manifest)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            manifests.append(manifest)
        return sorted(manifests, key=lambda value: value["captured_at"], reverse=True)

    def upload(self, snapshot: Path) -> BackupResult:
        snapshot = Path(snapshot).resolve()
        local = _read_object(snapshot / MANIFEST_FILE)
        snapshot_id = local.get("snapshot_id")
        if snapshot.name != snapshot_id:
            raise ValueError("snapshot directory and manifest IDs differ")
        self._validate_snapshot_id(snapshot_id)
        if local.get("deployment_id") != self.deployment_id:
            raise ValueError("snapshot belongs to another deployment")
        if local.get("release_id") != self.release_id:
            raise ValueError("snapshot belongs to another release")
        data = snapshot / ARCHIVE_FILE
        expected_data = local.get("archive", {})
        if (
            not data.is_file()
            or data.stat().st_size != expected_data.get("size")
            or _sha256_file(data) != expected_data.get("sha256")
        ):
            raise ValueError("local snapshot archive failed verification")
        previous_id = None
        initial = BackupResult(
            snapshot_id, local["captured_at"], None, 0, previous_id, None
        )
        self._write_status(snapshot, initial, "uploading")
        try:
            previous = self.list()
            previous_id = previous[0]["snapshot_id"] if previous else None
            if previous_id is not None:
                self._write_status(
                    snapshot,
                    BackupResult(
                        snapshot_id, local["captured_at"], None, 0, previous_id, None
                    ),
                    "uploading",
                )
            manifest_key = f"{self._snapshot_prefix(snapshot_id)}/{MANIFEST_FILE}"
            if self.store.head(manifest_key) is not None:
                remote = self.get_manifest(snapshot_id)
                if remote["release_id"] != self.release_id:
                    raise ValueError("completed backup release identity differs")
                self._verify_remote_objects(remote)
                result = BackupResult(
                    snapshot_id,
                    remote["captured_at"],
                    remote["uploaded_at"],
                    remote["total_size"],
                    snapshot_id,
                )
                self._write_status(snapshot, result, "completed")
                return result

            release_archive = self._release_archive(snapshot)
            release_descriptor = self._descriptor(
                f"{self.prefix}/releases/{self.release_id}/bundle.tar.gz",
                release_archive,
                release_id=self.release_id,
            )
            data_descriptor = self._descriptor(
                f"{self._snapshot_prefix(snapshot_id)}/{ARCHIVE_FILE}", data
            )
            image_pairs = self._image_archives(snapshot)
            self._ensure_file(release_descriptor, release_archive)
            for archive, descriptor in image_pairs:
                self._ensure_file(descriptor, archive)
            self._ensure_file(data_descriptor, data)

            uploaded_at = _now()
            remote = {
                **local,
                "remote_format_version": REMOTE_FORMAT_VERSION,
                "uploaded_at": uploaded_at,
                "objects": {
                    "release": release_descriptor,
                    "images": [descriptor for _, descriptor in image_pairs],
                    "data": data_descriptor,
                },
            }
            remote["total_size"] = sum(
                item["size"]
                for item in [release_descriptor, data_descriptor, *remote["objects"]["images"]]
            )
            payload = (json.dumps(remote, indent=2, sort_keys=True) + "\n").encode()
            manifest_descriptor = {
                "key": manifest_key,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            self._ensure_bytes(manifest_descriptor, payload)
            result = BackupResult(
                snapshot_id,
                local["captured_at"],
                uploaded_at,
                remote["total_size"],
                snapshot_id,
            )
            self._write_status(snapshot, result, "completed")
            return result
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            failed = BackupResult(
                snapshot_id, local["captured_at"], None, 0, previous_id, failure
            )
            self._write_status(snapshot, failed, "failed")
            raise

    def _validate_descriptor(self, descriptor: Any) -> None:
        if (
            not isinstance(descriptor, dict)
            or not isinstance(descriptor.get("key"), str)
            or type(descriptor.get("size")) is not int
            or descriptor["size"] < 0
            or not isinstance(descriptor.get("sha256"), str)
            or _SHA256.fullmatch(descriptor["sha256"]) is None
        ):
            raise ValueError("remote manifest has an invalid object descriptor")
        if not descriptor["key"].startswith(f"{self.prefix}/"):
            raise ValueError("remote manifest object belongs to another deployment")

    def _validate_remote_manifest(self, manifest: Any) -> None:
        if not isinstance(manifest, dict):
            raise ValueError("remote manifest must be an object")
        if manifest.get("remote_format_version") != REMOTE_FORMAT_VERSION:
            raise ValueError("unsupported remote backup format")
        if manifest.get("deployment_id") != self.deployment_id:
            raise ValueError("remote backup belongs to another deployment")
        snapshot_id = manifest.get("snapshot_id")
        self._validate_snapshot_id(snapshot_id)
        objects = manifest.get("objects")
        if not isinstance(objects, dict) or not isinstance(objects.get("images"), list):
            raise ValueError("remote manifest object inventory is invalid")
        descriptors = [objects.get("release"), objects.get("data"), *objects["images"]]
        for descriptor in descriptors:
            self._validate_descriptor(descriptor)
        for descriptor in objects["images"]:
            if not all(
                isinstance(descriptor.get(name), str)
                for name in ("archive_sha256", "content_id", "platform")
            ):
                raise ValueError("remote image identity is incomplete")
        expected_data_key = f"{self._snapshot_prefix(snapshot_id)}/{ARCHIVE_FILE}"
        if objects["data"]["key"] != expected_data_key:
            raise ValueError("remote data key does not match the snapshot")
        if manifest.get("total_size") != sum(item["size"] for item in descriptors):
            raise ValueError("remote backup total size is invalid")
        datetime.fromisoformat(manifest["captured_at"])
        datetime.fromisoformat(manifest["uploaded_at"])

    def get_manifest(self, snapshot_id: str) -> dict[str, Any]:
        key = f"{self._snapshot_prefix(snapshot_id)}/{MANIFEST_FILE}"
        try:
            value, _ = self.store.get_bytes(key)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"completed backup does not exist: {snapshot_id}") from exc
        manifest = json.loads(value)
        self._validate_remote_manifest(manifest)
        return manifest

    def _verify_remote_objects(self, manifest: dict[str, Any]) -> None:
        objects = manifest["objects"]
        for descriptor in [objects["release"], objects["data"], *objects["images"]]:
            if not self._verify_info(self.store.head(descriptor["key"]), descriptor):
                raise ValueError(f"remote backup object is missing or corrupt: {descriptor['key']}")
            self._verify_stored_file(descriptor)

    def _download_object(self, descriptor: dict[str, Any], destination: Path) -> None:
        info = self.store.get_file(descriptor["key"], destination)
        if (
            info.size != descriptor["size"]
            or destination.stat().st_size != descriptor["size"]
            or _sha256_file(destination) != descriptor["sha256"]
        ):
            destination.unlink(missing_ok=True)
            raise ValueError(f"downloaded object failed verification: {descriptor['key']}")

    def download(self, snapshot_id: str, destination: Path) -> DownloadedBackup:
        manifest = self.get_manifest(snapshot_id)
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"download destination already exists: {destination}")
        staging = destination.with_name(f".{destination.name}.staging")
        if staging.exists() or staging.is_symlink():
            raise ValueError(f"download staging already exists: {staging}")
        staging.mkdir(parents=True)
        try:
            objects = manifest["objects"]
            release_archive = staging / "bundle.tar.gz"
            self._download_object(objects["release"], release_archive)
            _extract_safe(release_archive, staging / "release")
            bundle = staging / "release" / "bundle"
            if _sha256_json(_read_object(bundle / LOCK_FILE)) != manifest["release_id"]:
                raise ValueError("downloaded release identity does not match the backup")
            snapshot = staging / "snapshot"
            snapshot.mkdir()
            self._download_object(objects["data"], snapshot / ARCHIVE_FILE)
            local_manifest = {
                key: value for key, value in manifest.items()
                if key not in ("remote_format_version", "uploaded_at", "objects", "total_size")
            }
            _atomic_json(snapshot / MANIFEST_FILE, local_manifest)
            images_dir = staging / "images"
            images_dir.mkdir()
            image_archives: list[Path] = []
            for index, descriptor in enumerate(objects["images"]):
                path = images_dir / f"{index}-{descriptor['archive_sha256']}.tar.gz"
                self._download_object(descriptor, path)
                image_archives.append(path)
            os.replace(staging, destination)
            return DownloadedBackup(
                snapshot_id,
                destination,
                destination / "snapshot",
                destination / "release" / "bundle",
                tuple(destination / "images" / path.name for path in image_archives),
                manifest,
            )
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def load_images(self, downloaded: DownloadedBackup) -> None:
        descriptors = downloaded.manifest["objects"]["images"]
        for archive, descriptor in zip(downloaded.image_archives, descriptors, strict=True):
            raw = archive.with_suffix("")
            with gzip.open(archive, "rb") as source, raw.open("wb") as output:
                shutil.copyfileobj(source, output)
            try:
                self.run(
                    ["docker", "image", "load", "--input", str(raw)],
                    check=True,
                    text=True,
                    capture_output=True,
                )
            finally:
                raw.unlink(missing_ok=True)
            inspected = self._inspect_image(descriptor["content_id"])
            if inspected.get("Id") != descriptor["content_id"]:
                raise ValueError("Docker loaded an image with the wrong content ID")
            if descriptor.get("os") is not None and inspected.get("Os") != descriptor["os"]:
                raise ValueError("Docker loaded an image for the wrong OS")
            if (
                descriptor.get("architecture") is not None
                and inspected.get("Architecture") != descriptor["architecture"]
            ):
                raise ValueError("Docker loaded an image for the wrong architecture")
            actual_platform = f"{inspected.get('Os')}/{inspected.get('Architecture')}"
            if "/".join(descriptor["platform"].split("/")[:2]) != actual_platform:
                raise ValueError("Docker loaded an image for the wrong platform")

    def restore(
        self,
        snapshot_id: str,
        destination: Path,
        controller: DeploymentController,
    ) -> Path:
        downloaded = self.download(snapshot_id, destination)
        self.load_images(downloaded)
        return DeploymentSnapshots(controller, downloaded.bundle).restore(downloaded.snapshot)
