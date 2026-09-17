from __future__ import annotations

import json
from pathlib import Path
import subprocess

import httpx
import pytest

from theseus.host_provisioner import DigitalOceanAPI, HostProfile, HostProvisioner, cloud_init


COMMIT = "a" * 40
SECRET_VALUE = "telegram-secret-value"


def deployment():
    return {
        "_generated_by": "theseus-compose-assembler",
        "deployment_id": "flywheel",
        "platform": "linux/amd64",
        "uid": 10001,
        "gid": 10001,
        "workspace_gid": 10002,
        "required_secrets": ["TELEGRAM_TOKEN"],
        "theseus": {"commit": COMMIT},
        "agents": {
            "fable": {"workspaces": ["website"]},
            "astra": {"workspaces": ["website"]},
        },
    }


def profile():
    return HostProfile(
        name="trial-toronto",
        region="tor1",
        size_slug="s-2vcpu-4gb",
        image="ubuntu-24-04-x64",
        architecture="amd64",
        ssh_key_ids=(101, 202),
        firewall_ids=("firewall-1",),
        vpc_uuid="vpc-1",
        ipv6=True,
    )


class API:
    def __init__(self):
        self.droplets = []
        self.create_calls = 0
        self.deleted = []
        self.firewall_assignments = []
        self.timeout_after_create = False
        self.timeout_without_create = False

    def list_regions(self):
        return [{"slug": "tor1", "available": True, "sizes": ["s-2vcpu-4gb"]}]

    def list_sizes(self):
        return [{
            "slug": "s-2vcpu-4gb", "available": True, "regions": ["tor1"],
            "vcpus": 2, "memory": 4096, "disk": 80, "transfer": 4,
            "price_hourly": 0.03571, "price_monthly": 24,
        }]

    def get_image(self, image):
        return {
            "slug": image, "status": "available", "regions": ["tor1"],
            "architecture": "amd64", "type": "distribution", "distribution": "Ubuntu",
        }

    def get_ssh_key(self, key_id):
        return {"id": key_id}

    def get_firewall(self, firewall_id):
        return {"id": firewall_id}

    def get_vpc(self, vpc_uuid):
        return {"id": vpc_uuid, "region": "tor1"}

    def find_droplets_by_tag(self, tag):
        return [item for item in self.droplets if tag in item["tags"]]

    def create_droplet(self, request):
        self.create_calls += 1
        if self.timeout_without_create:
            raise TimeoutError("request timed out")
        droplet = {
            "id": 9000 + self.create_calls,
            "status": "active",
            "tags": list(request["tags"]),
            "networks": {"v4": [{"type": "public", "ip_address": "203.0.113.40"}]},
            "request": request,
        }
        self.droplets.append(droplet)
        if self.timeout_after_create:
            raise TimeoutError("response was lost")
        return droplet

    def get_droplet(self, droplet_id):
        return next(item for item in self.droplets if item["id"] == droplet_id)

    def assign_firewall(self, firewall_id, droplet_id):
        self.firewall_assignments.append((firewall_id, droplet_id))

    def delete_droplet(self, droplet_id):
        self.deleted.append(droplet_id)


class SSH:
    def __init__(self):
        self.events = []
        self.fail_connect = False
        self.cloud_status = "status: done\n"

    def connect(self, host, profile, known_hosts, expected_fingerprint):
        self.events.append(("connect", host, expected_fingerprint))
        if self.fail_connect:
            raise OSError("SSH refused")
        return "SHA256:destination-host"

    def run(self, host, profile, known_hosts, command):
        self.events.append(("run", command))
        if "cloud-init status" in command:
            stdout = self.cloud_status
        elif command == "uname -m":
            stdout = "x86_64\n"
        elif command.startswith("df "):
            stdout = str(50 * 1024 * 1024) + "\n"
        else:
            stdout = "ok\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    def upload(self, host, profile, known_hosts, source, destination):
        self.events.append(("upload", destination, Path(source).read_text()))


def setup(tmp_path, *, api=None, ssh=None):
    api = api or API()
    ssh = ssh or SSH()
    state = tmp_path / "provision.json"
    provisioner = HostProvisioner(
        api,
        ssh,
        deployment(),
        profile(),
        state,
        source_droplet_id=77,
        poll_attempts=2,
        lookup_attempts=2,
        wait=lambda _: None,
    )
    secret = tmp_path / "TELEGRAM_TOKEN"
    secret.write_text(SECRET_VALUE)
    return provisioner, api, ssh, state, {"TELEGRAM_TOKEN": secret}


def test_successful_bootstrap_reports_exact_resources_and_cost(tmp_path):
    provisioner, api, ssh, state_path, secrets = setup(tmp_path)
    result = provisioner.provision(secrets)

    assert result.phase == "ready"
    assert result.droplet_id == 9001
    assert result.address == "203.0.113.40"
    assert result.ssh_host_fingerprint == "SHA256:destination-host"
    assert result.preview.size_slug == "s-2vcpu-4gb"
    assert result.preview.price_monthly_usd == 24
    assert result.remaining_billable_resources == ("droplet:9001", "source-droplet:77")
    assert api.create_calls == 1
    assert api.firewall_assignments == [("firewall-1", 9001)]
    assert ssh.events[0][0] == "connect"
    assert next(index for index, event in enumerate(ssh.events) if event[0] == "upload") > next(
        index for index, event in enumerate(ssh.events)
        if event[:2] == ("run", "sudo cloud-init status --wait --long")
    )
    state = json.loads(state_path.read_text())
    assert state["droplet_id"] == 9001
    assert state["activated"] is False

    offline = HostProvisioner(
        None,
        SSH(),
        deployment(),
        profile(),
        state_path,
        source_droplet_id=77,
        poll_attempts=2,
        lookup_attempts=2,
        wait=lambda _: None,
    )
    assert offline.status().droplet_id == 9001


def test_user_data_bootstraps_without_tokens_or_activation(tmp_path):
    provisioner, api, _, _, secrets = setup(tmp_path)
    provisioner.provision(secrets)
    request = api.droplets[0]["request"]
    user_data = request["user_data"]

    assert request["size"] == "s-2vcpu-4gb"
    assert request["image"] == "ubuntu-24-04-x64"
    assert request["vpc_uuid"] == "vpc-1"
    assert "docker-compose-v2" in user_data
    assert f"@{COMMIT}" in user_data
    assert "/srv/theseus/flywheel/data/agents/fable/state" in user_data
    assert "install -d -o root -g 10001 -m 2750 /srv/theseus/flywheel/control" in user_data
    assert "activation.json" not in user_data
    assert SECRET_VALUE not in user_data
    assert "DIGITALOCEAN_TOKEN" not in user_data
    assert "OPENROUTER" not in user_data
    assert "R2" not in user_data


def test_create_timeout_reconciles_tag_and_never_creates_twice(tmp_path):
    api = API()
    api.timeout_after_create = True
    provisioner, _, ssh, state_path, secrets = setup(tmp_path, api=api)

    first = provisioner.provision(secrets)
    second = provisioner.provision(secrets)

    assert first.droplet_id == second.droplet_id == 9001
    assert api.create_calls == 1
    assert len(api.droplets) == 1
    assert json.loads(state_path.read_text())["droplet_id"] == 9001
    connects = [event for event in ssh.events if event[0] == "connect"]
    assert connects[-1][2] == "SHA256:destination-host"


def test_unresolved_create_timeout_stays_uncertain_and_retry_does_not_create(tmp_path):
    api = API()
    api.timeout_without_create = True
    provisioner, _, _, state_path, secrets = setup(tmp_path, api=api)

    with pytest.raises(RuntimeError, match="outcome is uncertain"):
        provisioner.provision(secrets)
    assert json.loads(state_path.read_text())["phase"] == "create-uncertain"
    with pytest.raises(RuntimeError, match="outcome remains unresolved"):
        provisioner.provision(secrets)
    assert api.create_calls == 1


def test_retry_from_persisted_creating_phase_never_blindly_creates(tmp_path):
    provisioner, api, _, state_path, secrets = setup(tmp_path)
    state = provisioner._load_or_create_state(provisioner.preview())
    state["phase"] = "creating"
    state_path.write_text(json.dumps(state))

    with pytest.raises(RuntimeError, match="outcome remains unresolved"):
        provisioner.provision(secrets)
    assert api.create_calls == 0


def test_ambiguous_tagged_matches_stop_without_creation(tmp_path):
    api = API()
    provisioner, _, _, state_path, secrets = setup(tmp_path, api=api)
    preview = provisioner.preview()
    state = provisioner._load_or_create_state(preview)
    tag = state["operation_tag"]
    api.droplets = [{"id": 1, "tags": [tag]}, {"id": 2, "tags": [tag]}]

    with pytest.raises(RuntimeError, match="multiple Droplets.*1, 2"):
        provisioner.provision(secrets)
    assert api.create_calls == 0
    assert json.loads(state_path.read_text())["phase"] == "reconcile-ambiguous"
    api.droplets = []
    with pytest.raises(RuntimeError, match="previously ambiguous"):
        provisioner.provision(secrets)
    assert api.create_calls == 0


@pytest.mark.parametrize("failure", ["ssh", "cloud-init"])
def test_readiness_failure_retains_destination_and_does_not_touch_source(tmp_path, failure):
    source = tmp_path / "source-agent"
    source.write_text("still running")
    ssh = SSH()
    if failure == "ssh":
        ssh.fail_connect = True
    else:
        ssh.cloud_status = "status: error\n"
    provisioner, api, _, state_path, secrets = setup(tmp_path, ssh=ssh)

    with pytest.raises((OSError, RuntimeError)):
        provisioner.provision(secrets)

    state = json.loads(state_path.read_text())
    assert state["phase"] == "failed"
    assert state["droplet_id"] == 9001
    assert api.deleted == []
    assert source.read_text() == "still running"


def test_runtime_secret_value_is_only_uploaded_after_authenticated_ssh(tmp_path):
    provisioner, api, ssh, state_path, secrets = setup(tmp_path)
    provisioner.provision(secrets)

    state_text = state_path.read_text()
    request_text = json.dumps(api.droplets[0]["request"])
    assert SECRET_VALUE not in state_text
    assert SECRET_VALUE not in request_text
    upload = next(event for event in ssh.events if event[0] == "upload")
    assert upload[2] == SECRET_VALUE
    assert ssh.events.index(upload) > next(
        index for index, event in enumerate(ssh.events) if event[0] == "connect"
    )


def test_cleanup_refuses_activated_target_and_never_deletes_source(tmp_path):
    provisioner, api, _, _, secrets = setup(tmp_path)
    provisioner.provision(secrets)
    provisioner.mark_activated()

    with pytest.raises(RuntimeError, match="activated destination"):
        provisioner.cleanup()
    assert api.deleted == []
    assert 77 not in api.deleted


def test_cleanup_deletes_only_uniquely_tagged_unused_destination(tmp_path):
    provisioner, api, _, _, secrets = setup(tmp_path)
    result = provisioner.provision(secrets)
    cleaned = provisioner.cleanup()

    assert api.deleted == [result.droplet_id]
    assert cleaned.phase == "cleaned"
    assert "source-droplet:77" in cleaned.remaining_billable_resources


def test_preview_rejects_unavailable_exact_size_without_substitution(tmp_path):
    api = API()
    api.list_sizes = lambda: [{
        "slug": "s-4vcpu-8gb", "available": True, "regions": ["tor1"],
        "vcpus": 4, "memory": 8192, "disk": 160, "transfer": 5,
        "price_hourly": 0.07, "price_monthly": 48,
    }]
    provisioner, _, _, _, _ = setup(tmp_path, api=api)
    with pytest.raises(ValueError, match="size is unavailable: s-2vcpu-4gb"):
        provisioner.preview()


def test_cloud_init_requires_full_operator_pin():
    with pytest.raises(ValueError, match="full Git commit"):
        cloud_init(deployment(), operator_repository="https://example.test/repo.git", operator_ref="main")


def test_digitalocean_adapter_keeps_token_in_operator_authorization_header():
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(200, json={"sizes": [{"slug": "exact-size"}]})

    client = httpx.Client(transport=httpx.MockTransport(respond))
    api = DigitalOceanAPI("operator-token", client=client)

    assert api.list_sizes() == [{"slug": "exact-size"}]
    assert seen[0].headers["authorization"] == "Bearer operator-token"
    assert seen[0].url.params["per_page"] == "200"
