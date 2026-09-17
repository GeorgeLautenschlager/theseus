"""Object storage boundary for remote deployment backups."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Iterator, Mapping, Protocol

import boto3
from botocore.exceptions import ClientError


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    metadata: dict[str, str]
    last_modified: str | None = None


class ObjectStore(Protocol):
    def put_file(self, key: str, source: Path, metadata: Mapping[str, str]) -> ObjectInfo: ...
    def put_bytes(self, key: str, value: bytes, metadata: Mapping[str, str]) -> ObjectInfo: ...
    def head(self, key: str) -> ObjectInfo | None: ...
    def get_file(self, key: str, destination: Path) -> ObjectInfo: ...
    def get_bytes(self, key: str) -> tuple[bytes, ObjectInfo]: ...
    def list_keys(self, prefix: str) -> Iterator[str]: ...


def _safe_key(key: str) -> PurePosixPath:
    path = PurePosixPath(key)
    if not key or path.is_absolute() or ".." in path.parts or any(not part for part in path.parts):
        raise ValueError(f"unsafe object key: {key!r}")
    return path


class LocalObjectStore:
    """Filesystem object adapter with S3-like immutable-key test semantics."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.metadata_root = self.root / ".object-metadata"
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_root.mkdir(exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root.joinpath(*_safe_key(key).parts)

    def _metadata_path(self, key: str) -> Path:
        return self.metadata_root.joinpath(*_safe_key(key).parts).with_suffix(
            PurePosixPath(key).suffix + ".metadata.json"
        )

    def put_file(self, key: str, source: Path, metadata: Mapping[str, str]) -> ObjectInfo:
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"object source is missing or unsafe: {source}")
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                with source.open("rb") as incoming:
                    shutil.copyfileobj(incoming, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            self._write_metadata(key, metadata)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return self.head(key)  # type: ignore[return-value]

    def put_bytes(self, key: str, value: bytes, metadata: Mapping[str, str]) -> ObjectInfo:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            self._write_metadata(key, metadata)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return self.head(key)  # type: ignore[return-value]

    def _write_metadata(self, key: str, metadata: Mapping[str, str]) -> None:
        path = self._metadata_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        value = {
            "metadata": {str(name): str(item) for name, item in metadata.items()},
            "last_modified": datetime.now(timezone.utc).isoformat(),
        }
        temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    def head(self, key: str) -> ObjectInfo | None:
        path = self._path(key)
        if not path.is_file() or path.is_symlink():
            return None
        metadata_path = self._metadata_path(key)
        value = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {"metadata": {}, "last_modified": None}
        )
        return ObjectInfo(
            key=key,
            size=path.stat().st_size,
            metadata=dict(value.get("metadata", {})),
            last_modified=value.get("last_modified"),
        )

    def get_file(self, key: str, destination: Path) -> ObjectInfo:
        source = self._path(key)
        info = self.head(key)
        if info is None:
            raise FileNotFoundError(f"object does not exist: {key}")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return info

    def get_bytes(self, key: str) -> tuple[bytes, ObjectInfo]:
        info = self.head(key)
        if info is None:
            raise FileNotFoundError(f"object does not exist: {key}")
        return self._path(key).read_bytes(), info

    def list_keys(self, prefix: str) -> Iterator[str]:
        _safe_key(prefix.rstrip("/") or "root")
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.is_relative_to(self.metadata_root):
                continue
            key = path.relative_to(self.root).as_posix()
            if key.startswith(prefix):
                yield key


class S3ObjectStore:
    """Cloudflare R2 adapter using boto3 and its standard credential chain."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        bucket: str,
        client: Any | None = None,
    ) -> None:
        if not isinstance(endpoint_url, str) or not endpoint_url.strip():
            raise ValueError("R2 endpoint URL must be nonempty")
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError("R2 bucket must be nonempty")
        self.endpoint_url = endpoint_url.strip()
        self.bucket = bucket.strip()
        self._client = client or boto3.client(
            "s3", endpoint_url=self.endpoint_url, region_name="auto"
        )

    def put_file(self, key: str, source: Path, metadata: Mapping[str, str]) -> ObjectInfo:
        _safe_key(key)
        self._client.upload_file(
            str(source), self.bucket, key,
            ExtraArgs={"Metadata": {str(name): str(value) for name, value in metadata.items()}},
        )
        return self.head(key)  # type: ignore[return-value]

    def put_bytes(self, key: str, value: bytes, metadata: Mapping[str, str]) -> ObjectInfo:
        _safe_key(key)
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=value,
            Metadata={str(name): str(item) for name, item in metadata.items()},
        )
        return self.head(key)  # type: ignore[return-value]

    def head(self, key: str) -> ObjectInfo | None:
        _safe_key(key)
        try:
            value = self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        modified = value.get("LastModified")
        return ObjectInfo(
            key=key,
            size=int(value["ContentLength"]),
            metadata={str(name): str(item) for name, item in value.get("Metadata", {}).items()},
            last_modified=modified.isoformat() if hasattr(modified, "isoformat") else None,
        )

    def get_file(self, key: str, destination: Path) -> ObjectInfo:
        _safe_key(key)
        info = self.head(key)
        if info is None:
            raise FileNotFoundError(f"object does not exist: {key}")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(destination))
        return info

    def get_bytes(self, key: str) -> tuple[bytes, ObjectInfo]:
        _safe_key(key)
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        value = response["Body"].read()
        info = self.head(key)
        if info is None:
            raise FileNotFoundError(f"object does not exist: {key}")
        return value, info

    def list_keys(self, prefix: str) -> Iterator[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                yield item["Key"]
