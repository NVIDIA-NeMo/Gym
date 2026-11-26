# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attempt sandbox: the sandbox group an Environment Server starts, lends and stops for one episode attempt.

The Environment Server owns it; the agent and the Resources Server only borrow what it lends. Where the agent runs is
derived: in the group when the task has tools or sidecars or the agent cannot run its loop without a sandbox. Where
the verifier runs is its ``VerifierMode``: the Environment Server's ``verifier.mode`` (``AttemptVerifierConfig``), or
else the mode the Resources Server declares at seed. The Resources Server also declares what the group holds
(``SandboxGroupSpec``). These are the types both sides share; the lifecycle itself is ``BaseEnvironmentServer``'s.
"""

from collections.abc import Mapping
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.config import SandboxConfig
from nemo_gym.sandbox.providers.base import NetworkRule, SandboxResources, SandboxSidecarSpec


# The /prepare_sandbox route a Resources Server serves when it prepares the task in the attempt's sandbox group.
PREPARE_SANDBOX_PATH = "/prepare_sandbox"


class AttemptSandboxConfig(SandboxConfig):
    """The sandbox settings an Environment Server starts each attempt's sandbox group with.

    ``network`` must name a rule, so no attempt starts with the provider's default egress by accident: ``model_only``,
    with ``model_server`` set to the agent's model server, for an agent that only calls the model; ``model_and_tools``
    with ``tool_servers`` naming every server whose tools the episode grants. A rule allows a server's whole host.
    ``ttl_s`` must be set, so the provider reclaims a sandbox the server could not stop; it must exceed the episode
    timeout plus the cleanup timeout.

    Attributes:
        image: The main container's image, unless the Resources Server declares one at seed.
        workdir: The working directory the agent and the verifier run in, unless the Resources Server declares one.
    """

    image: str | None = None
    workdir: str | None = None

    @model_validator(mode="after")
    def _explicit_network_and_lifetime(self) -> "AttemptSandboxConfig":
        if self.network is None:
            raise ValueError("an attempt sandbox needs a network rule, such as network: model_only")
        if self.ttl_s is None:
            raise ValueError("an attempt sandbox needs ttl_s, so the provider reclaims one the server cannot stop")
        return self


# Where /verify runs; the words are Harbor's verifier modes. "none": in the Resources Server, lent no sandbox.
# "shared": in the group the agent borrowed, after the agent closes. "separate": in a fresh group of its own, started
# right before /verify, with no_network unless verifier.sandbox sets a rule, after any agent group stops.
VerifierMode = Literal["none", "shared", "separate"]
# The verifier modes in a fixed order, for messages and the seed reply.
VERIFIER_MODES: tuple[VerifierMode, ...] = get_args(VerifierMode)
# The fields of a sandbox's network rule. A separate verifier's group takes none of them from sandbox_config: its rule
# is no_network unless verifier.sandbox sets one.
_NETWORK_FIELDS = frozenset({"network", "model_server", "tool_servers", "allowed_hosts"})


class VerifierSandboxConfig(SandboxConfig):
    """``verifier.sandbox``: the settings of a separate verifier's group that differ from the task sandbox's.

    Each field it sets replaces ``sandbox_config``'s, and the rest are copied, except the network rule: unset, it is
    ``no_network``, whatever ``sandbox_config`` allows. Without a ``sandbox_config`` it is the whole group's settings,
    so it sets ``image`` and ``ttl_s`` at least. The group never has sidecars.

    Attributes:
        image: The verifier's image, over the one the Resources Server declares at seed and ``sandbox_config.image``.
        workdir: The verifier's working directory, over the declared one and ``sandbox_config.workdir``.
    """

    image: str | None = None
    workdir: str | None = None


class AttemptVerifierConfig(BaseModel):
    """The ``verifier`` block of an Environment Server's config: where each attempt's ``/verify`` runs.

    Attributes:
        mode: The ``VerifierMode``; unset, the mode the Resources Server declares at seed. Either way, the Resources
            Server must say its ``/verify`` can run in it (``supported_verifier_modes``).
        sandbox: For ``mode: separate`` only, the verifier's group settings that differ from ``sandbox_config``'s.
    """

    model_config = ConfigDict(extra="forbid")

    mode: VerifierMode | None = None
    sandbox: VerifierSandboxConfig | None = None

    @model_validator(mode="after")
    def _sandbox_only_when_separate(self) -> "AttemptVerifierConfig":
        if self.sandbox is not None and self.mode != "separate":
            raise ValueError("verifier.sandbox configures a separate verifier's group; set verifier.mode: separate")
        return self

    def separate_sandbox_config(self, task_sandbox: AttemptSandboxConfig | None) -> AttemptSandboxConfig | None:
        """The settings a separate verifier's group starts with: ``sandbox`` over ``task_sandbox`` without its network
        rule, and ``no_network`` unless ``sandbox`` sets a rule. ``None`` when neither is set.

        Raises:
            pydantic.ValidationError: If the merged settings are no complete ``AttemptSandboxConfig``.
        """
        if task_sandbox is None and self.sandbox is None:
            return None
        settings: dict[str, Any] = {"network": "no_network"}
        if task_sandbox is not None:
            settings |= task_sandbox.model_dump(exclude=set(_NETWORK_FIELDS))
        if self.sandbox is not None:
            settings |= self.sandbox.model_dump(include=set(self.sandbox.model_fields_set))
        model = type(task_sandbox) if task_sandbox is not None else AttemptSandboxConfig
        return model.model_validate(settings)


class SandboxGroupSidecar(BaseModel):
    """A sidecar container of the attempt's sandbox group, started from its own image beside the main container.

    It serves an environment tool reached over HTTP, an API or MCP. Starting a group with sidecars needs a provider
    that implements ``SupportsSandboxSidecars`` and applies ``no_network`` and ``hosts_only``.

    Attributes:
        egress_hosts: The only hosts the sidecar reaches, such as the API its tool calls (``hosts_only``); empty, it
            reaches none (``no_network``). The main container's network rule never applies to a sidecar.
        tools_port: The port on which the sidecar serves the Resources Server's HTTP tools, if it does. Only an
            Environment Server that grants them to the agent (``grants_tool_sidecars``) starts such a sidecar.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    image: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)
    egress_hosts: list[str] = Field(default_factory=list)
    tools_port: int | None = Field(default=None, ge=1, le=65535)

    @field_validator("resources")
    @classmethod
    def _known_resources(cls, resources: dict[str, Any]) -> dict[str, Any]:
        SandboxResources.from_mapping(resources)
        return resources

    @property
    def network(self) -> NetworkRule:
        """The sidecar's own network rule: ``hosts_only`` for its ``egress_hosts``, else ``no_network``."""
        return "hosts_only" if self.egress_hosts else "no_network"

    def spec(self, *, provider_options: Mapping[str, Any] | None = None) -> SandboxSidecarSpec:
        """The provider-neutral sidecar spec this declaration starts, with its own ``provider_options``.

        A ``tools_port`` is the one port of the sidecar the main container may reach; the tool host is started on it
        later, at ``/prepare_sandbox``, so the spec names no entrypoint and the create does not wait for it.
        """
        return SandboxSidecarSpec(
            name=self.name,
            image=self.image,
            env=dict(self.env),
            resources=dict(self.resources),
            provider_options=dict(provider_options or {}),
            ports=(self.tools_port,) if self.tools_port is not None else (),
        )


class SandboxGroupSpec(BaseModel):
    """What an attempt's sandbox group holds, as the Resources Server declares it at seed.

    The Environment Server starts the group from its own ``sandbox_config`` (provider, network rule, ``ttl_s``,
    resources) with these declared fields over it, lends it, and stops it. The Resources Server never starts it.

    Attributes:
        image: The main container's image; unset, the Environment Server's ``sandbox_config.image``.
        workdir: The main container's working directory; unset, ``sandbox_config.workdir``.
        env: Environment variables of the main container, over ``sandbox_config.env``.
        files: Text files delivered into the main container before anyone borrows it, keyed by path.
        sidecars: Containers that serve the task's environment tools.
        prepare: The Resources Server prepares the task in the first group the attempt starts through
            ``/prepare_sandbox`` (see ``SimpleResourcesServer.prepare_sandbox``), before the agent, or the verifier,
            borrows it.
    """

    model_config = ConfigDict(extra="forbid")

    image: str | None = None
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    sidecars: list[SandboxGroupSidecar] = Field(default_factory=list)
    prepare: bool = False

    @field_validator("sidecars")
    @classmethod
    def _unique_sidecar_names(cls, sidecars: list[SandboxGroupSidecar]) -> list[SandboxGroupSidecar]:
        names = [sidecar.name for sidecar in sidecars]
        if len(set(names)) != len(names):
            raise ValueError(f"sandbox group sidecar names must be unique: {names}")
        return sidecars


class AttemptSandbox(BaseModel):
    """One attempt's sandbox groups as their Environment Server tracks them.

    An episode that holds one is a restart (see ``BaseEnvironmentServer.seed_attempt_sandbox``), so no checkpoint
    exports it, nor the declared files and environment variables in ``group``.

    Attributes:
        key: The agent's group's key in the server's ``OwnedSandboxes``, unique to the attempt; a separate verifier's
            group is ``verifier_key``.
        lend_to_agent: The group starts at seed and the agent borrows it for its turn: the task has tools or sidecars,
            or the agent cannot run its loop without a sandbox.
        verifier_mode: Where ``/verify`` runs: the configured ``verifier.mode``, or else the declared one.
        group: What the Resources Server declared at seed, if anything.
        image: The agent's group's image, set when ``lend_to_agent``: the declared one, or else
            ``sandbox_config.image``.
        workdir: The agent's group's working directory, set when ``lend_to_agent``: the declared one, or else
            ``sandbox_config.workdir``.
        verifier_image: A separate verifier's image: ``verifier.sandbox.image``, the declared one, or else
            ``sandbox_config.image``.
        verifier_workdir: A separate verifier's working directory, in the same order.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    lend_to_agent: bool = False
    verifier_mode: VerifierMode = "none"
    group: SandboxGroupSpec | None = None
    image: str | None = None
    workdir: str | None = None
    verifier_image: str | None = None
    verifier_workdir: str | None = None

    @model_validator(mode="after")
    def _groups_have_image_and_workdir(self) -> "AttemptSandbox":
        if self.lend_to_agent and (self.image is None or self.workdir is None):
            raise ValueError("the agent's group needs an image and a working directory")
        if self.verifier_mode == "separate" and (self.verifier_image is None or self.verifier_workdir is None):
            raise ValueError("a separate verifier's group needs an image and a working directory")
        return self

    @property
    def verifier_key(self) -> str:
        """The key of the group a separate verifier runs in."""
        return verifier_group_key(self.key)


def verifier_group_key(key: str) -> str:
    """The key of a separate verifier's group, beside the agent's group under ``key``."""
    return f"{key}/verifier"


class ResourcesPrepareSandboxRequest(BaseModel):
    """Ask the Resources Server to prepare the task in the attempt's started sandbox group.

    It carries the seed's identity and ``task_data``, so a server without session state knows what to prepare. The
    server borrows the group with ``borrow(sandbox_access)`` and releases it before it replies; it never stops it.
    """

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    episode_id: EpisodeId
    task_id: TaskId
    task_data: dict[str, Any]
    sandbox_access: SandboxAccess


class ResourcesPrepareSandboxResponse(BaseModel):
    """Confirm the task is prepared in the attempt's sandbox group."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
