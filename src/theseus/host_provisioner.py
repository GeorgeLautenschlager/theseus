"""Resumable DigitalOcean destination provisioning for managed deployments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from time import sleep
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from theseus.deployment_control import _atomic_json


PROVISION_FORMAT_VERSION = 1
_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_SECRET = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SSH_USER = re.compile(r"[a-z_][a-z0-9_-]*\Z")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class HostProfile:
    """Operator-owned DigitalOcean placement and network policy."""

    name: str
    region: str
    size_slug: str
    image: str
    architecture: str
    ssh_key_ids: tuple[int, ...]
    firewall_ids: tuple[str, ...] = ()
    vpc_uuid: str | None = None
    ipv6: bool = False
    monitoring: bool = True
    ssh_user: str = "root"
    ssh_port: int = 22

    def validate(self) -> None:
        for label, value in (
            ("profile name", self.name),
            ("region", self.region),
            ("size slug", self.size_slug),
            ("image", self.image),
            ("SSH user", self.ssh_user),
        ):
            if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
                raise ValueError(f"{label} must be a nonempty token")
        if _SSH_USER.fullmatch(self.ssh_user) is None:
            raise ValueError("SSH user is invalid")
        if self.architecture not in ("amd64", "arm64"):
            raise ValueError("architecture must be 'amd64' or 'arm64'")
        if not isinstance(self.ssh_key_ids, tuple) or not self.ssh_key_ids:
            raise ValueError("at least one SSH key ID is required")
        if any(type(item) is not int or item <= 0 for item in self.ssh_key_ids):
            raise ValueError("SSH key IDs must be positive integers")
        if len(set(self.ssh_key_ids)) != len(self.ssh_key_ids):
            raise ValueError("SSH key IDs contain duplicates")
        if not isinstance(self.firewall_ids, tuple) or any(
            not isinstance(item, str) or _TOKEN.fullmatch(item) is None
            for item in self.firewall_ids
        ):
            raise ValueError("firewall IDs must be nonempty strings")
        if len(set(self.firewall_ids)) != len(self.firewall_ids):
            raise ValueError("firewall IDs contain duplicates")
        if self.vpc_uuid is not None and (
            not isinstance(self.vpc_uuid, str) or _TOKEN.fullmatch(self.vpc_uuid) is None
        ):
            raise ValueError("VPC UUID must be nonempty text")
        if type(self.ipv6) is not bool or type(self.monitoring) is not bool:
            raise ValueError("ipv6 and monitoring must be booleans")
        if type(self.ssh_port) is not int or not 1 <= self.ssh_port <= 65535:
            raise ValueError("SSH port must be between 1 and 65535")

    @classmethod
    def from_file(cls, path: Path) -> HostProfile:
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError("host profile must be a real operator-owned file")
        if path.stat().st_uid != os.getuid():
            raise PermissionError("host profile is not owned by the current operator")
        value = _read_object(path)
        for name in ("ssh_key_ids", "firewall_ids"):
            if name in value:
                value[name] = tuple(value[name])
        profile = cls(**value)
        profile.validate()
        return profile


@dataclass(frozen=True)
class ProvisionPreview:
    profile: str
    region: str
    size_slug: str
    image: str
    architecture: str
    vcpus: int
    memory_mb: int
    disk_gb: int
    transfer_tb: float
    price_hourly_usd: float
    price_monthly_usd: float
    ssh_key_ids: tuple[int, ...]
    firewall_ids: tuple[str, ...]
    vpc_uuid: str | None


@dataclass(frozen=True)
class ProvisionResult:
    operation_id: str
    operation_tag: str
    phase: str
    droplet_id: int | None
    address: str | None
    ssh_host_fingerprint: str | None
    preview: ProvisionPreview
    remaining_billable_resources: tuple[str, ...]
    failure: str | None = None


class DigitalOcean(Protocol):
    def list_regions(self) -> list[dict[str, Any]]: ...
    def list_sizes(self) -> list[dict[str, Any]]: ...
    def get_image(self, image: str) -> dict[str, Any]: ...
    def get_ssh_key(self, key_id: int) -> dict[str, Any]: ...
    def get_firewall(self, firewall_id: str) -> dict[str, Any]: ...
    def get_vpc(self, vpc_uuid: str) -> dict[str, Any]: ...
    def find_droplets_by_tag(self, tag: str) -> list[dict[str, Any]]: ...
    def create_droplet(self, request: dict[str, Any]) -> dict[str, Any]: ...
    def get_droplet(self, droplet_id: int) -> dict[str, Any]: ...
    def assign_firewall(self, firewall_id: str, droplet_id: int) -> None: ...
    def delete_droplet(self, droplet_id: int) -> None: ...


class DigitalOceanAPI:
    """Small HTTP adapter that keeps the operator token out of persisted state."""

    def __init__(
        self,
        token: str | None = None,
        *,
        client: httpx.Client | None = None,
        base_url: str = "https://api.digitalocean.com/v2",
    ) -> None:
        token = token or os.environ.get("DIGITALOCEAN_TOKEN")
        if not token:
            raise ValueError("DIGITALOCEAN_TOKEN is required")
        self._client = client or httpx.Client(timeout=30.0, follow_redirects=False)
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(
                method,
                f"{self._base_url}/{path.lstrip('/')}",
                headers=self._headers,
                **kwargs,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(f"DigitalOcean API {method} {path} request failed") from exc
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"DigitalOcean API {method} {path} failed with HTTP {response.status_code}"
            ) from exc
        if response.status_code == 204 or not response.content:
            return {}
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("DigitalOcean API returned a non-object response")
        return value

    def list_regions(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "regions", params={"per_page": 200})["regions"])

    def list_sizes(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "sizes", params={"per_page": 200})["sizes"])

    def get_image(self, image: str) -> dict[str, Any]:
        return dict(self._request("GET", f"images/{image}")["image"])

    def get_ssh_key(self, key_id: int) -> dict[str, Any]:
        return dict(self._request("GET", f"account/keys/{key_id}")["ssh_key"])

    def get_firewall(self, firewall_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"firewalls/{firewall_id}")["firewall"])

    def get_vpc(self, vpc_uuid: str) -> dict[str, Any]:
        return dict(self._request("GET", f"vpcs/{vpc_uuid}")["vpc"])

    def find_droplets_by_tag(self, tag: str) -> list[dict[str, Any]]:
        value = self._request("GET", "droplets", params={"tag_name": tag, "per_page": 200})
        return list(value.get("droplets", []))

    def create_droplet(self, request: dict[str, Any]) -> dict[str, Any]:
        return dict(self._request("POST", "droplets", json=request)["droplet"])

    def get_droplet(self, droplet_id: int) -> dict[str, Any]:
        return dict(self._request("GET", f"droplets/{droplet_id}")["droplet"])

    def assign_firewall(self, firewall_id: str, droplet_id: int) -> None:
        self._request(
            "POST", f"firewalls/{firewall_id}/droplets", json={"droplet_ids": [droplet_id]}
        )

    def delete_droplet(self, droplet_id: int) -> None:
        self._request("DELETE", f"droplets/{droplet_id}")


class SSH(Protocol):
    def connect(
        self, host: str, profile: HostProfile, known_hosts: Path, expected_fingerprint: str | None
    ) -> str: ...
    def run(
        self, host: str, profile: HostProfile, known_hosts: Path, command: str
    ) -> subprocess.CompletedProcess[str]: ...
    def upload(
        self, host: str, profile: HostProfile, known_hosts: Path, source: Path, destination: str
    ) -> None: ...


class OpenSSH:
    """OpenSSH transport with trust-on-first-authentication and strict retries."""

    @staticmethod
    def _target(host: str, profile: HostProfile) -> str:
        return f"{profile.ssh_user}@{host}"

    @staticmethod
    def _options(profile: HostProfile, known_hosts: Path, policy: str = "yes") -> list[str]:
        return [
            "-p", str(profile.ssh_port),
            "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", f"StrictHostKeyChecking={policy}",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
        ]

    @staticmethod
    def _fingerprint(known_hosts: Path) -> str:
        result = subprocess.run(
            ["ssh-keygen", "-lf", str(known_hosts), "-E", "sha256"],
            check=True, text=True, capture_output=True,
        )
        values = sorted(
            line.split()[1] for line in result.stdout.splitlines() if len(line.split()) >= 2
        )
        if not values:
            raise RuntimeError("SSH did not retain a host identity")
        return ",".join(values)

    def connect(
        self, host: str, profile: HostProfile, known_hosts: Path, expected_fingerprint: str | None
    ) -> str:
        known_hosts.parent.mkdir(parents=True, exist_ok=True)
        policy = "yes" if expected_fingerprint else "accept-new"
        subprocess.run(
            ["ssh", *self._options(profile, known_hosts, policy), self._target(host, profile), "true"],
            check=True, text=True, capture_output=True,
        )
        fingerprint = self._fingerprint(known_hosts)
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise RuntimeError("SSH host identity differs from the retained destination identity")
        return fingerprint

    def run(
        self, host: str, profile: HostProfile, known_hosts: Path, command: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["ssh", *self._options(profile, known_hosts), self._target(host, profile), command],
            check=True, text=True, capture_output=True,
        )

    def upload(
        self, host: str, profile: HostProfile, known_hosts: Path, source: Path, destination: str
    ) -> None:
        options = self._options(profile, known_hosts)
        # scp spells the port option with an uppercase P.
        options[0] = "-P"
        subprocess.run(
            ["scp", *options, str(source), f"{self._target(host, profile)}:{destination}"],
            check=True, text=True, capture_output=True,
        )


def _image_architecture(image: Mapping[str, Any]) -> str | None:
    value = image.get("architecture")
    if value in ("amd64", "x86_64", "x64"):
        return "amd64"
    if value in ("arm64", "aarch64"):
        return "arm64"
    identity = " ".join(str(image.get(name, "")) for name in ("slug", "name")).lower()
    if any(marker in identity for marker in ("arm64", "aarch64")):
        return "arm64"
    if any(marker in identity for marker in ("x64", "amd64", "x86_64")):
        return "amd64"
    return None


def cloud_init(
    deployment: Mapping[str, Any], *, operator_repository: str, operator_ref: str
) -> str:
    """Render non-secret bootstrap configuration; it cannot activate a deployment."""
    deployment_id = deployment["deployment_id"]
    if _ID.fullmatch(deployment_id) is None:
        raise ValueError("deployment ID is unsafe for provisioning")
    if _SHA.fullmatch(operator_ref) is None:
        raise ValueError("the Theseus operator must be pinned to a full Git commit")
    repository = urlsplit(operator_repository)
    if (
        repository.scheme != "https"
        or not repository.hostname
        or repository.username is not None
        or repository.password is not None
        or repository.query
        or repository.fragment
        or any(c.isspace() for c in operator_repository)
    ):
        raise ValueError("operator repository must be a non-secret HTTPS URL")
    uid = deployment.get("uid")
    gid = deployment.get("gid")
    workspace_gid = deployment.get("workspace_gid")
    if any(type(value) is not int or value <= 0 for value in (uid, gid, workspace_gid)):
        raise ValueError("deployment runtime ownership is invalid")
    agents = deployment.get("agents")
    if not isinstance(agents, dict) or not agents:
        raise ValueError("deployment has no agents")
    workspaces = sorted({
        workspace
        for agent in agents.values()
        for workspace in agent.get("workspaces", [])
    })
    root = f"/srv/theseus/{deployment_id}"
    private_dirs = [
        f"{root}/data/agents/{agent_id}/{kind}"
        for agent_id in sorted(agents)
        for kind in ("state", "logs")
    ]
    shared_dirs = [f"{root}/data/workspaces/{name}" for name in workspaces]
    base_dirs = [f"{root}/{name}" for name in ("releases", "secrets", "snapshots")]
    script = [
        "#!/bin/bash", "set -euo pipefail",
        "systemctl enable --now docker",
        f"getent group {gid} >/dev/null || groupadd --gid {gid} theseus-runtime",
        f"getent group {workspace_gid} >/dev/null || groupadd --gid {workspace_gid} theseus-workspace",
        f"id -u {uid} >/dev/null 2>&1 || useradd --uid {uid} --gid {gid} --groups {workspace_gid} --no-create-home --shell /usr/sbin/nologin theseus-runtime",
        "python3 -m venv /opt/theseus-operator",
        f"/opt/theseus-operator/bin/pip install --no-cache-dir {shlex.quote(f'git+{operator_repository}@{operator_ref}')}",
        *[f"install -d -m 0750 {shlex.quote(path)}" for path in base_dirs],
        # setgid keeps atomically replaced activation files in the runtime group.
        f"install -d -o root -g {gid} -m 2750 {shlex.quote(root + '/control')}",
        *[f"install -d -o {uid} -g {gid} -m 0750 {shlex.quote(path)}" for path in private_dirs],
        *[f"install -d -o {uid} -g {workspace_gid} -m 2770 {shlex.quote(path)}" for path in shared_dirs],
        "touch /var/lib/theseus-bootstrap-complete",
    ]
    indented = "\n".join(f"      {line}" for line in script)
    return (
        "#cloud-config\n"
        "package_update: true\n"
        "packages:\n"
        "  - docker.io\n"
        "  - docker-compose-v2\n"
        "  - git\n"
        "  - python3-venv\n"
        "write_files:\n"
        "  - path: /usr/local/sbin/theseus-bootstrap\n"
        "    permissions: '0700'\n"
        "    owner: root:root\n"
        "    content: |\n"
        f"{indented}\n"
        "runcmd:\n"
        "  - [bash, /usr/local/sbin/theseus-bootstrap]\n"
    )


class HostProvisioner:
    """Persist first, reconcile create ambiguity, and require full host readiness."""

    def __init__(
        self,
        api: DigitalOcean | None,
        ssh: SSH,
        deployment: Mapping[str, Any],
        profile: HostProfile,
        state_path: Path,
        *,
        operator_repository: str = "https://github.com/GeorgeLautenschlager/theseus.git",
        minimum_disk_gb: int = 10,
        source_droplet_id: int | None = None,
        poll_attempts: int = 30,
        lookup_attempts: int = 3,
        wait: Callable[[float], None] = sleep,
    ) -> None:
        profile.validate()
        if type(minimum_disk_gb) is not int or minimum_disk_gb <= 0:
            raise ValueError("minimum disk space must be a positive integer")
        if type(poll_attempts) is not int or poll_attempts <= 0:
            raise ValueError("poll attempts must be positive")
        if type(lookup_attempts) is not int or lookup_attempts <= 0:
            raise ValueError("lookup attempts must be positive")
        if source_droplet_id is not None and (
            type(source_droplet_id) is not int or source_droplet_id <= 0
        ):
            raise ValueError("source Droplet ID must be a positive integer")
        self.api = api
        self.ssh = ssh
        self.deployment = dict(deployment)
        self.profile = profile
        self.state_path = Path(state_path)
        self.operator_repository = operator_repository
        self.minimum_disk_gb = minimum_disk_gb
        self.source_droplet_id = source_droplet_id
        self.poll_attempts = poll_attempts
        self.lookup_attempts = lookup_attempts
        self.wait = wait
        deployment_id = self.deployment.get("deployment_id")
        if not isinstance(deployment_id, str) or _ID.fullmatch(deployment_id) is None:
            raise ValueError("deployment identity is invalid")
        self.deployment_id = deployment_id
        theseus = self.deployment.get("theseus")
        self.operator_ref = theseus.get("commit") if isinstance(theseus, dict) else None
        if not isinstance(self.operator_ref, str) or _SHA.fullmatch(self.operator_ref) is None:
            raise ValueError("deployment has no full pinned Theseus commit")

    @property
    def known_hosts(self) -> Path:
        return self.state_path.with_suffix(self.state_path.suffix + ".known_hosts")

    def _require_api(self) -> DigitalOcean:
        if self.api is None:
            raise ValueError("DigitalOcean API credentials are required for this action")
        return self.api

    def preview(self) -> ProvisionPreview:
        api = self._require_api()
        regions = {item.get("slug"): item for item in api.list_regions()}
        region = regions.get(self.profile.region)
        if region is None or region.get("available") is not True:
            raise ValueError(f"DigitalOcean region is unavailable: {self.profile.region}")
        sizes = {item.get("slug"): item for item in api.list_sizes()}
        size = sizes.get(self.profile.size_slug)
        if size is None or size.get("available") is not True:
            raise ValueError(f"DigitalOcean size is unavailable: {self.profile.size_slug}")
        if self.profile.region not in size.get("regions", []):
            raise ValueError(
                f"size {self.profile.size_slug} is unavailable in {self.profile.region}"
            )
        if self.profile.size_slug not in region.get("sizes", []):
            raise ValueError("region does not advertise the exact requested size")
        if int(size.get("disk", 0)) < self.minimum_disk_gb:
            raise ValueError(
                f"size {self.profile.size_slug} has {size.get('disk')} GB disk; "
                f"{self.minimum_disk_gb} GB is required"
            )
        image = api.get_image(self.profile.image)
        if image.get("status", "available") != "available":
            raise ValueError(f"DigitalOcean image is unavailable: {self.profile.image}")
        if image.get("distribution") != "Ubuntu":
            raise ValueError("v1 cloud-init bootstrap requires an Ubuntu image")
        if image.get("regions") and self.profile.region not in image["regions"]:
            raise ValueError("image is unavailable in the requested region")
        architecture = _image_architecture(image)
        if architecture != self.profile.architecture:
            raise ValueError(
                f"image architecture is {architecture or 'unknown'}, "
                f"not {self.profile.architecture}"
            )
        if size.get("architecture") is not None:
            size_architecture = _image_architecture(size)
            if size_architecture != self.profile.architecture:
                raise ValueError("size and requested architecture are incompatible")
        for key_id in self.profile.ssh_key_ids:
            key = api.get_ssh_key(key_id)
            if key.get("id") != key_id:
                raise ValueError(f"SSH key is unavailable: {key_id}")
        for firewall_id in self.profile.firewall_ids:
            firewall = api.get_firewall(firewall_id)
            if firewall.get("id") != firewall_id:
                raise ValueError(f"firewall is unavailable: {firewall_id}")
        if self.profile.vpc_uuid is not None:
            vpc = api.get_vpc(self.profile.vpc_uuid)
            if vpc.get("id") != self.profile.vpc_uuid or vpc.get("region") != self.profile.region:
                raise ValueError("VPC is unavailable or belongs to another region")
        return ProvisionPreview(
            profile=self.profile.name,
            region=self.profile.region,
            size_slug=self.profile.size_slug,
            image=self.profile.image,
            architecture=self.profile.architecture,
            vcpus=int(size["vcpus"]),
            memory_mb=int(size["memory"]),
            disk_gb=int(size["disk"]),
            transfer_tb=float(size["transfer"]),
            price_hourly_usd=float(size["price_hourly"]),
            price_monthly_usd=float(size["price_monthly"]),
            ssh_key_ids=self.profile.ssh_key_ids,
            firewall_ids=self.profile.firewall_ids,
            vpc_uuid=self.profile.vpc_uuid,
        )

    def _new_state(self, preview: ProvisionPreview) -> dict[str, Any]:
        operation_id = uuid4().hex
        request = self._request_identity()
        return {
            "format_version": PROVISION_FORMAT_VERSION,
            "operation_id": operation_id,
            "operation_tag": f"theseus-provision-{operation_id}",
            "request": request,
            "request_sha256": _canonical_sha(request),
            "preview": asdict(preview),
            "phase": "intended",
            "created_at": _now(),
            "updated_at": _now(),
            "droplet_id": None,
            "address": None,
            "ssh_host_fingerprint": None,
            "created_by_operation": False,
            "activated": False,
            "failure": None,
        }

    def _request_identity(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "deployment_sha256": _canonical_sha(self.deployment),
            "profile": asdict(self.profile),
            "operator_repository": self.operator_repository,
            "operator_ref": self.operator_ref,
            "minimum_disk_gb": self.minimum_disk_gb,
            "source_droplet_id": self.source_droplet_id,
        }

    def _load_or_create_state(self, preview: ProvisionPreview) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self._new_state(preview)
            _atomic_json(self.state_path, state)
            return state
        if not self.state_path.is_file() or self.state_path.is_symlink():
            raise ValueError("provisioning state must be a real file")
        state = _read_object(self.state_path)
        expected = self._request_identity()
        if state.get("format_version") != PROVISION_FORMAT_VERSION:
            raise ValueError("unsupported provisioning state format")
        if state.get("request_sha256") != _canonical_sha(expected):
            raise ValueError("provisioning state belongs to a different intended request")
        return state

    def _load_existing_state(self) -> tuple[dict[str, Any], ProvisionPreview]:
        if not self.state_path.is_file() or self.state_path.is_symlink():
            raise ValueError("provisioning state does not exist")
        state = _read_object(self.state_path)
        if state.get("format_version") != PROVISION_FORMAT_VERSION:
            raise ValueError("unsupported provisioning state format")
        if state.get("request_sha256") != _canonical_sha(self._request_identity()):
            raise ValueError("provisioning state belongs to a different intended request")
        value = dict(state.get("preview", {}))
        for name in ("ssh_key_ids", "firewall_ids"):
            if name in value:
                value[name] = tuple(value[name])
        try:
            preview = ProvisionPreview(**value)
        except TypeError as exc:
            raise ValueError("persisted provisioning preview is invalid") from exc
        return state, preview

    def _save(self, state: dict[str, Any], phase: str, **values: Any) -> None:
        state.update(values, phase=phase, updated_at=_now())
        _atomic_json(self.state_path, state)

    def _lookup(self, tag: str) -> list[dict[str, Any]]:
        api = self._require_api()
        matches: list[dict[str, Any]] = []
        for attempt in range(self.lookup_attempts):
            matches = api.find_droplets_by_tag(tag)
            if matches or attempt + 1 == self.lookup_attempts:
                break
            self.wait(1.0)
        if len(matches) > 1:
            ids = ", ".join(str(item.get("id")) for item in matches)
            raise RuntimeError(f"multiple Droplets match operation tag {tag}: {ids}")
        return matches

    def _create_request(self, state: dict[str, Any]) -> dict[str, Any]:
        profile = self.profile
        user_data = cloud_init(
            self.deployment,
            operator_repository=self.operator_repository,
            operator_ref=self.operator_ref,
        )
        if len(user_data.encode("utf-8")) > 65536:
            raise ValueError("cloud-init user-data exceeds DigitalOcean's 64 KiB limit")
        return {
            "name": f"theseus-{self.deployment_id}-{state['operation_id'][:8]}",
            "region": profile.region,
            "size": profile.size_slug,
            "image": profile.image,
            "ssh_keys": list(profile.ssh_key_ids),
            "backups": False,
            "ipv6": profile.ipv6,
            "monitoring": profile.monitoring,
            "tags": [state["operation_tag"], "theseus-destination"],
            "user_data": user_data,
            **({"vpc_uuid": profile.vpc_uuid} if profile.vpc_uuid else {}),
        }

    def _record_match(self, state: dict[str, Any], droplet: Mapping[str, Any]) -> None:
        droplet_id = droplet.get("id")
        if type(droplet_id) is not int or droplet_id <= 0:
            raise ValueError("DigitalOcean returned an invalid Droplet identity")
        if self.source_droplet_id is not None and droplet_id == self.source_droplet_id:
            raise RuntimeError("destination lookup resolved to the source Droplet")
        self._save(
            state, "created", droplet_id=droplet_id, created_by_operation=True, failure=None
        )

    @staticmethod
    def _public_address(droplet: Mapping[str, Any]) -> str | None:
        networks = droplet.get("networks", {})
        for network in networks.get("v4", []) if isinstance(networks, dict) else []:
            if network.get("type") == "public" and network.get("ip_address"):
                address = str(network["ip_address"])
                try:
                    parsed = ipaddress.ip_address(address)
                except ValueError:
                    continue
                if parsed.version == 4:
                    return address
        return None

    def _wait_active(self, droplet_id: int) -> tuple[dict[str, Any], str]:
        api = self._require_api()
        last = None
        for _ in range(self.poll_attempts):
            last = api.get_droplet(droplet_id)
            address = self._public_address(last)
            if last.get("status") == "active" and address:
                return last, address
            self.wait(2.0)
        raise TimeoutError(
            f"Droplet {droplet_id} did not become active with a public address; "
            f"last status was {None if last is None else last.get('status')}"
        )

    def _ready(self, state: dict[str, Any], address: str) -> str:
        expected = state.get("ssh_host_fingerprint")
        failure: BaseException | None = None
        for attempt in range(self.poll_attempts):
            try:
                fingerprint = self.ssh.connect(
                    address, self.profile, self.known_hosts, expected
                )
                break
            except (OSError, subprocess.CalledProcessError) as exc:
                failure = exc
                if attempt + 1 < self.poll_attempts:
                    self.wait(2.0)
        else:
            raise TimeoutError("destination SSH did not become ready") from failure
        if expected is None:
            self._save(state, "ssh-authenticated", ssh_host_fingerprint=fingerprint)
        cloud = self.ssh.run(
            address, self.profile, self.known_hosts, "sudo cloud-init status --wait --long"
        )
        if "status: done" not in cloud.stdout.lower():
            raise RuntimeError("cloud-init did not report successful completion")
        self.ssh.run(address, self.profile, self.known_hosts, "sudo docker --version")
        self.ssh.run(address, self.profile, self.known_hosts, "sudo docker compose version")
        machine = self.ssh.run(
            address, self.profile, self.known_hosts, "uname -m"
        ).stdout.strip()
        expected_machine = "x86_64" if self.profile.architecture == "amd64" else "aarch64"
        if machine != expected_machine:
            raise RuntimeError(f"destination architecture is {machine!r}, expected {expected_machine}")
        root = f"/srv/theseus/{self.deployment_id}"
        disk = self.ssh.run(
            address,
            self.profile,
            self.known_hosts,
            f"df -k --output=avail {shlex.quote(root)} | tail -1",
        ).stdout.strip()
        try:
            available_kb = int(disk)
        except ValueError as exc:
            raise RuntimeError("destination disk availability could not be read") from exc
        if available_kb < self.minimum_disk_gb * 1024 * 1024:
            raise RuntimeError(
                f"destination has {available_kb // (1024 * 1024)} GB free; "
                f"{self.minimum_disk_gb} GB is required"
            )
        self.ssh.run(
            address,
            self.profile,
            self.known_hosts,
            f"test -f /var/lib/theseus-bootstrap-complete && test ! -e {shlex.quote(root + '/control/activation.json')}",
        )
        return fingerprint

    def _transfer_secrets(
        self, state: dict[str, Any], address: str, secrets: Mapping[str, Path]
    ) -> None:
        required = self.deployment.get("required_secrets", [])
        if not isinstance(required, list) or any(
            not isinstance(name, str) or _SECRET.fullmatch(name) is None for name in required
        ):
            raise ValueError("deployment required secret names are invalid")
        if set(secrets) != set(required):
            missing = sorted(set(required) - set(secrets))
            extra = sorted(set(secrets) - set(required))
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unexpected " + ", ".join(extra))
            raise ValueError("runtime secret set differs: " + "; ".join(detail))
        root = f"/srv/theseus/{self.deployment_id}/secrets"
        for name in sorted(required):
            source = Path(secrets[name])
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"runtime secret file is missing or unsafe: {name}")
            temporary = f"/tmp/.theseus-secret-{state['operation_id']}-{name}"
            self.ssh.upload(address, self.profile, self.known_hosts, source, temporary)
            destination = f"{root}/{name}"
            self.ssh.run(
                address,
                self.profile,
                self.known_hosts,
                "sudo install "
                f"-o {self.deployment['uid']} -g {self.deployment['gid']} -m 0600 "
                f"{shlex.quote(temporary)} {shlex.quote(destination)} && rm -f {shlex.quote(temporary)}",
            )

    def provision(self, secrets: Mapping[str, Path]) -> ProvisionResult:
        api = self._require_api()
        if self.state_path.exists():
            state, preview = self._load_existing_state()
        else:
            preview = self.preview()
            state = self._load_or_create_state(preview)
        try:
            droplet_id = state.get("droplet_id")
            if droplet_id is None:
                matches = self._lookup(state["operation_tag"])
                if state.get("phase") == "reconcile-ambiguous":
                    raise RuntimeError(
                        "operation tag was previously ambiguous; resolve tracked resources "
                        "before provisioning can continue"
                    )
                if matches:
                    self._record_match(state, matches[0])
                elif state.get("phase") in ("creating", "create-uncertain"):
                    raise RuntimeError(
                        "the create outcome remains unresolved; no tagged Droplet was found"
                    )
                else:
                    self._save(state, "creating", failure=None)
                    try:
                        droplet = api.create_droplet(self._create_request(state))
                    except BaseException as exc:
                        self._save(
                            state,
                            "create-uncertain",
                            failure=f"{type(exc).__name__}: create outcome is uncertain",
                        )
                        matches = self._lookup(state["operation_tag"])
                        if not matches:
                            raise RuntimeError(
                                "Droplet create outcome is uncertain; retry will reconcile by tag"
                            ) from exc
                        droplet = matches[0]
                    self._record_match(state, droplet)
                droplet_id = state["droplet_id"]
            if type(droplet_id) is not int:
                raise ValueError("persisted Droplet identity is invalid")
            if self.source_droplet_id is not None and droplet_id == self.source_droplet_id:
                raise RuntimeError("destination Droplet is the source host")
            for firewall_id in self.profile.firewall_ids:
                api.assign_firewall(firewall_id, droplet_id)
            _, address = self._wait_active(droplet_id)
            self._save(state, "api-active", address=address)
            fingerprint = self._ready(state, address)
            self._save(state, "preflight-complete", ssh_host_fingerprint=fingerprint)
            self._transfer_secrets(state, address, secrets)
            self._save(state, "ready", failure=None)
            return self._result(state, preview)
        except BaseException as exc:
            if "multiple Droplets match operation tag" in str(exc):
                self._save(
                    state, "reconcile-ambiguous", failure=f"{type(exc).__name__}: {exc}"
                )
            elif state.get("phase") == "creating":
                self._save(
                    state,
                    "create-uncertain",
                    failure=f"{type(exc).__name__}: create outcome is uncertain",
                )
            elif state.get("phase") not in ("create-uncertain", "reconcile-ambiguous"):
                self._save(state, "failed", failure=f"{type(exc).__name__}: {exc}")
            raise

    def _result(self, state: dict[str, Any], preview: ProvisionPreview) -> ProvisionResult:
        resources = []
        if state.get("droplet_id") is not None and state.get("phase") != "cleaned":
            resources.append(f"droplet:{state['droplet_id']}")
        if self.source_droplet_id is not None:
            resources.append(f"source-droplet:{self.source_droplet_id}")
        return ProvisionResult(
            operation_id=state["operation_id"],
            operation_tag=state["operation_tag"],
            phase=state["phase"],
            droplet_id=state.get("droplet_id"),
            address=state.get("address"),
            ssh_host_fingerprint=state.get("ssh_host_fingerprint"),
            preview=preview,
            remaining_billable_resources=tuple(resources),
            failure=state.get("failure"),
        )

    def status(self) -> ProvisionResult:
        state, preview = self._load_existing_state()
        return self._result(state, preview)

    def mark_activated(self) -> ProvisionResult:
        state, preview = self._load_existing_state()
        if state.get("phase") == "activated" and state.get("activated") is True:
            return self._result(state, preview)
        if state.get("phase") != "ready":
            raise RuntimeError("only a ready destination can be marked activated")
        self._save(state, "activated", activated=True)
        return self._result(state, preview)

    def cleanup(self) -> ProvisionResult:
        api = self._require_api()
        state, preview = self._load_existing_state()
        droplet_id = state.get("droplet_id")
        if type(droplet_id) is not int or not state.get("created_by_operation"):
            raise RuntimeError("no destination created by this operation is tracked")
        if state.get("phase") == "cleaned":
            raise RuntimeError("destination has already been cleaned up")
        if state.get("activated") or state.get("phase") == "activated":
            raise RuntimeError("refusing to delete an activated destination")
        if self.source_droplet_id is not None and droplet_id == self.source_droplet_id:
            raise RuntimeError("refusing to delete the source Droplet")
        matches = api.find_droplets_by_tag(state["operation_tag"])
        if len(matches) != 1 or matches[0].get("id") != droplet_id:
            raise RuntimeError("tracked destination tag no longer resolves uniquely to its Droplet")
        api.delete_droplet(droplet_id)
        self._save(state, "cleaned", failure=None)
        return self._result(state, preview)
