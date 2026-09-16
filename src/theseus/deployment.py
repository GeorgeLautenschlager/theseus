"""Trusted deployment definitions independent of any container runtime."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import PurePosixPath
import re
from typing import Mapping


_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_SECRET = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _stable_id(value: object, label: str) -> None:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(
            f"{label} must be a lowercase stable ID of 1-63 letters, digits, or hyphens"
        )


@dataclass(frozen=True)
class ResourceSpec:
    """Portable limits for one managed agent."""

    cpus: float = 1.0
    memory_mb: int = 512

    def validate(self) -> None:
        if (
            isinstance(self.cpus, bool)
            or not isinstance(self.cpus, (int, float))
            or not math.isfinite(self.cpus)
            or self.cpus <= 0
        ):
            raise ValueError("resource cpus must be a positive finite number")
        if type(self.memory_mb) is not int or self.memory_mb <= 0:
            raise ValueError("resource memory_mb must be a positive integer")


@dataclass(frozen=True)
class DeploymentSpec:
    """Stable deployment identity and topology around existing ``AgentSpec`` values.

    Paths in an AgentSpec remain useful for ordinary Python assembly. Managed
    deployment pairing is resolved from IDs, so source-host paths never leak into
    a portable runtime definition.
    """

    id: str
    agents: Mapping[str, object]
    peers: Mapping[str, str] = field(default_factory=dict)
    workspaces: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    platform: str = "linux/amd64"
    resources: Mapping[str, ResourceSpec] = field(default_factory=dict)
    image: str | None = None
    build_inputs: tuple[str, ...] = ()
    secrets: tuple[str, ...] = ()
    uid: int = 10001
    gid: int = 10001
    workspace_gid: int = 10002

    def validate(self) -> None:
        from theseus.assembly import AgentSpec

        _stable_id(self.id, "deployment id")
        if not isinstance(self.agents, Mapping) or not self.agents:
            raise ValueError("agents must be a nonempty mapping of stable ID to AgentSpec")
        for agent_id, agent in self.agents.items():
            _stable_id(agent_id, "agent id")
            if not isinstance(agent, AgentSpec):
                raise ValueError(f"agent {agent_id!r} must be an AgentSpec")
            agent.validate()

        if not isinstance(self.peers, Mapping):
            raise ValueError("peers must be a mapping of agent IDs")
        for agent_id, peer_id in self.peers.items():
            self._known_agent(agent_id, "peer owner")
            self._known_agent(peer_id, "peer")
            if agent_id == peer_id:
                raise ValueError(f"agent {agent_id!r} cannot be its own peer")

        for agent_id, agent in self.agents.items():
            pairing = agent.pairing
            peer_id = self.peers.get(agent_id)
            if pairing is not None and peer_id is None:
                raise ValueError(
                    f"agent {agent_id!r} has AgentSpec pairing but no deployment peer"
                )
            if peer_id is not None and pairing is not None:
                peer_name = self.agents[peer_id].name
                if pairing.peer_name != peer_name:
                    raise ValueError(
                        f"agent {agent_id!r} pairing names {pairing.peer_name!r}, "
                        f"but deployment peer {peer_id!r} is named {peer_name!r}"
                    )

        if not isinstance(self.workspaces, Mapping):
            raise ValueError("workspaces must be a mapping of workspace IDs to agent IDs")
        for workspace_id, members in self.workspaces.items():
            _stable_id(workspace_id, "workspace id")
            if not isinstance(members, tuple) or not members:
                raise ValueError(f"workspace {workspace_id!r} must have a nonempty member tuple")
            if len(set(members)) != len(members):
                raise ValueError(f"workspace {workspace_id!r} contains duplicate agent IDs")
            for agent_id in members:
                self._known_agent(agent_id, f"workspace {workspace_id!r} member")

        if not isinstance(self.platform, str) or not re.fullmatch(
            r"linux/(?:amd64|arm64)", self.platform
        ):
            raise ValueError("platform must be 'linux/amd64' or 'linux/arm64'")
        if not isinstance(self.resources, Mapping):
            raise ValueError("resources must be a mapping of agent IDs to ResourceSpec")
        for agent_id, resource in self.resources.items():
            self._known_agent(agent_id, "resource owner")
            if not isinstance(resource, ResourceSpec):
                raise ValueError(f"resources[{agent_id!r}] must be a ResourceSpec")
            resource.validate()
        if self.image is not None and (
            not isinstance(self.image, str) or not self.image.strip()
        ):
            raise ValueError("image must be nonempty text when provided")
        if not isinstance(self.build_inputs, tuple):
            raise ValueError("build_inputs must be a tuple of relative paths")
        for value in self.build_inputs:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("build_inputs must contain nonempty relative paths")
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("build_inputs must stay within the build context")
        if not isinstance(self.secrets, tuple) or any(
            not isinstance(secret, str) or _SECRET.fullmatch(secret) is None
            for secret in self.secrets
        ):
            raise ValueError("secrets must be a tuple of environment-style names")
        if len(set(self.secrets)) != len(self.secrets):
            raise ValueError("secrets contains duplicate names")
        for label, value in (
            ("uid", self.uid), ("gid", self.gid), ("workspace_gid", self.workspace_gid)
        ):
            if type(value) is not int or not 1 <= value <= 2**31 - 1:
                raise ValueError(f"{label} must be a positive numeric identity")
        if self.workspace_gid == self.gid:
            raise ValueError("workspace_gid must differ from the private agent gid")

    def _known_agent(self, agent_id: object, label: str) -> None:
        if not isinstance(agent_id, str) or agent_id not in self.agents:
            raise ValueError(f"{label} references unknown agent {agent_id!r}")

    def resolved_agent(self, agent_id: str):
        """Return the AgentSpec with its peer log mapped to the container contract."""
        from theseus.assembly import PairingSpec

        self.validate()
        self._known_agent(agent_id, "agent")
        agent = self.agents[agent_id]
        peer_id = self.peers.get(agent_id)
        if peer_id is None:
            return agent
        previous = agent.pairing
        return replace(
            agent,
            pairing=PairingSpec(
                peer_log_path=f"/peers/{peer_id}/logs/stimulus_log.jsonl",
                peer_name=self.agents[peer_id].name,
                peer_context_fraction=(
                    previous.peer_context_fraction if previous is not None else 0.25
                ),
            ),
        )

    def workspaces_for(self, agent_id: str) -> tuple[str, ...]:
        self._known_agent(agent_id, "agent")
        return tuple(
            workspace_id
            for workspace_id, members in self.workspaces.items()
            if agent_id in members
        )

