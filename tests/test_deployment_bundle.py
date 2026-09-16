from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from theseus.assembly import AgentSpec, InterfaceSpec, MemorySpec, ModelSpec
from theseus.deployment import DeploymentSpec, ResourceSpec
from theseus.deployment_bundle import assemble_compose, build_bundle


def agent(*, interface: InterfaceSpec | None = None, provider: str = "ollama") -> AgentSpec:
    return AgentSpec(
        name="Managed agent",
        constitution="Run without spending inference during assembly.",
        core="auto",
        models=(ModelSpec(provider, "fake-model"),),
        interface=interface or InterfaceSpec("none"),
        memory=MemorySpec(),
    )


def deployment(**changes) -> DeploymentSpec:
    value = DeploymentSpec(
        id="test-deployment",
        agents={"alpha": agent()},
        workspaces={"shared": ("alpha",)},
        resources={"alpha": ResourceSpec(cpus=0.5, memory_mb=192)},
    )
    return replace(value, **changes)


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def test_compose_bundle_is_complete_deterministic_and_contains_no_state(tmp_path):
    helper = tmp_path / "helpers" / "custom.py"
    helper.parent.mkdir()
    helper.write_text("VALUE = 7\n")
    spec = deployment(build_inputs=("helpers/custom.py",), secrets=("SERVICE_TOKEN",))
    output = assemble_compose(spec, tmp_path / "bundle", definition_root=tmp_path)
    before = snapshot(output)
    assemble_compose(spec, output, definition_root=tmp_path)
    assert snapshot(output) == before
    assert {
        "compose.yaml", "Dockerfile", "deployment.json", "requirements.lock",
        ".dockerignore", "secrets.example", "README.md", "entrypoint.py",
        ".theseus-generated.json",
    } <= {path.name for path in output.iterdir()}
    assert (output / "agents" / "alpha" / "agent.py").exists()
    assert (output / "build-inputs" / "helpers" / "custom.py").read_text() == "VALUE = 7\n"
    assert not (output / ".env").exists()
    assert not (output / "state").exists()
    compose = (output / "compose.yaml").read_text()
    assert "init: true" in compose
    assert "read_only: true" in compose
    assert "user: \"10001:10001\"" in compose
    assert "cpus: \"0.5\"" in compose
    assert "mem_limit: \"192m\"" in compose
    assert "/var/run/docker.sock" not in compose
    assert "ports:" not in compose
    assert "SERVICE_TOKEN" in (output / "secrets.example").read_text()
    assert "SERVICE_TOKEN=" not in (output / "secrets.example").read_text()


def test_reassembly_refuses_modified_generated_file_and_preserves_user_data(tmp_path):
    output = assemble_compose(deployment(), tmp_path / "bundle", definition_root=tmp_path)
    state = output / "state" / "keep.txt"
    secret = output / "secrets" / "token"
    state.parent.mkdir()
    secret.parent.mkdir()
    state.write_text("state")
    secret.write_text("credential")
    assemble_compose(deployment(), output, definition_root=tmp_path)
    assert state.read_text() == "state"
    assert secret.read_text() == "credential"
    (output / "compose.yaml").write_text("user edit")
    with pytest.raises(ValueError, match="modified generated"):
        assemble_compose(deployment(), output, definition_root=tmp_path)
    assert (output / "compose.yaml").read_text() == "user edit"


def test_compose_assembly_refuses_unowned_output(tmp_path):
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "compose.yaml").write_text("mine")
    with pytest.raises(ValueError, match="unowned bundle"):
        assemble_compose(deployment(), output, definition_root=tmp_path)
    assert (output / "compose.yaml").read_text() == "mine"


def test_compose_assembly_refuses_symlinked_output(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "bundle"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked bundle"):
        assemble_compose(deployment(), output, definition_root=tmp_path)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    "spec,message",
    [
        (deployment(agents={"alpha": agent(interface=InterfaceSpec("terminal"))}), "supports only"),
        (deployment(agents={"alpha": replace(
            agent(interface=InterfaceSpec("terminal")), core="ooda"
        )}), "supports only"),
        (deployment(agents={"alpha": agent(provider="openrouter")}), "OPENROUTER_API_KEY"),
        (deployment(runtime="coding-browser", image="example/image:tag"), "installed_tools"),
    ],
)
def test_managed_validation_rejects_unsupported_or_unresolved_specs(tmp_path, spec, message):
    with pytest.raises(ValueError, match=message):
        assemble_compose(spec, tmp_path / "bundle", definition_root=tmp_path)


@pytest.mark.parametrize("relative", ["missing.py", ".env", "nested/.env.local"])
def test_compose_assembly_rejects_missing_or_environment_build_inputs(tmp_path, relative):
    path = tmp_path / relative
    if path.name.startswith(".env"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SECRET=value")
    with pytest.raises(ValueError, match="build input|environment"):
        assemble_compose(
            deployment(build_inputs=(relative,)), tmp_path / "bundle",
            definition_root=tmp_path,
        )


def test_telegram_requires_declared_token_and_publishes_no_port(tmp_path):
    telegram = InterfaceSpec(
        "telegram", bot_token_env="BOT_SECRET", allowed_user_ids=(1,)
    )
    missing = deployment(agents={"alpha": agent(interface=telegram)})
    with pytest.raises(ValueError, match="BOT_SECRET"):
        assemble_compose(missing, tmp_path / "missing", definition_root=tmp_path)
    output = assemble_compose(
        replace(missing, secrets=("BOT_SECRET",)),
        tmp_path / "bundle", definition_root=tmp_path,
    )
    compose = (output / "compose.yaml").read_text()
    assert "ports:" not in compose
    assert "/run/secrets" not in compose  # Compose secrets use the canonical mount implicitly.
    assert "BOT_SECRET" in compose


def test_build_records_immutable_image_and_export_identities(tmp_path):
    output = assemble_compose(deployment(), tmp_path / "bundle", definition_root=tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            reference = command[3]
            identity = "base" if reference.startswith("python:") else "runtime"
            return subprocess.CompletedProcess(
                command, 0,
                stdout=json.dumps([{
                    "Id": f"sha256:{identity}",
                    "RepoDigests": [f"example/{identity}@sha256:{identity}"],
                    "Os": "linux",
                    "Architecture": "amd64",
                    "Config": {"User": "10001:10001"},
                }]),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    lock_path = build_bundle(output, run=fake_run)
    lock = json.loads(lock_path.read_text())
    deployment_data = json.loads((output / "deployment.json").read_text())
    assert calls[0][:2] == ["docker", "pull"]
    assert calls[1][:2] == ["docker", "build"]
    assert lock["resolved_spec_sha256"] == deployment_data["resolved_spec_sha256"]
    assert lock["images"][0]["content_id"] == "sha256:runtime"
    assert lock["images"][0]["user"] == "10001:10001"
    assert lock["base_image"]["content_id"] == "sha256:base"
    assert lock["theseus"]["version"]
    assert lock["theseus"]["commit"]
    assert len(lock["theseus"]["source_sha256"]) == 64
    assert lock["image_exports"][0]["reference"] == "sha256:runtime"
    assemble_compose(deployment(), output, definition_root=tmp_path)
    assert not lock_path.exists()  # A new assembly invalidates the old image identity.


def test_compose_cli_loads_deployment_while_default_cli_stays_compatible(tmp_path):
    definition = tmp_path / "deployment.py"
    definition.write_text(
        "from theseus import AgentSpec, DeploymentSpec, InterfaceSpec, ModelSpec\n"
        "AGENT = AgentSpec(name='A', constitution='C', core='auto', "
        "models=(ModelSpec('ollama', 'fake'),), interface=InterfaceSpec('none'))\n"
        "DEPLOYMENT = DeploymentSpec(id='cli-test', agents={'alpha': AGENT})\n"
    )
    output = tmp_path / "bundle"
    result = subprocess.run(
        [sys.executable, "-m", "theseus.assemble", str(definition), "--target", "compose",
         "--output", str(output)],
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "compose.yaml").exists()


@pytest.mark.skipif(
    not shutil.which("docker") or not bool(__import__("os").environ.get("THESEUS_RUN_DOCKER_TESTS")),
    reason="set THESEUS_RUN_DOCKER_TESTS=1 to run the real image build",
)
def test_real_docker_build_with_fake_provider_configuration(tmp_path):
    output = assemble_compose(deployment(), tmp_path / "bundle", definition_root=tmp_path)
    lock = json.loads(build_bundle(output).read_text())
    assert lock["images"][0]["content_id"].startswith("sha256:")
    assert lock["images"][0]["user"] == "10001:10001"
