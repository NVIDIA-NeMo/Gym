# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attempt sandbox: the sandbox group an Environment Server starts, lends and stops for one episode attempt.

The Environment Server owns it. The Resources Server declares at seed what the group holds (``SandboxGroupSpec``) and
whether its grader runs code; the agent borrows the group for its turn and the Resources Server borrows it for
``/verify``. A pure QA task, or a grader that executes nothing, may opt out (``SandboxOptOut``). These are the types
both sides share; the lifecycle itself is ``BaseEnvironmentServer``'s.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.config import SandboxConfig
from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSidecarSpec


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
        workdir: The working directory the agent and the grader run in, unless the Resources Server declares one.
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


class SandboxOptOut(BaseModel):
    """What may run outside the attempt's sandbox group, to save sandbox capacity on large runs.

    Everything runs in a sandbox unless it opts out, and only what executes nothing may. The Environment Server
    checks both at seed and fails the attempt, as a terminal configuration error, when the opt-out is refused.

    Attributes:
        task: A pure QA task, whose model calls no tools: the agent loop runs in the Agent Server's process and is
            lent no sandbox. Refused for a task that declares tools, for an episode that grants tools, and for a
            sandbox group with sidecars.
        grader: A grader that executes nothing and calls no tools, such as rules, math-verify or a judge call:
            ``/verify`` runs in the Resources Server and is lent no sandbox. Refused when the Resources Server
            declares that its grader runs code (``SandboxGroupSpec.grader_runs_code``).
    """

    model_config = ConfigDict(extra="forbid")

    task: bool = False
    grader: bool = False


class SandboxGroupSidecar(BaseModel):
    """A sidecar container of the attempt's sandbox group, started from its own image beside the main container.

    It serves an environment tool reached over HTTP, an API or MCP. Starting a group with sidecars needs a provider
    that implements ``SupportsSandboxSidecars``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    image: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)

    @field_validator("resources")
    @classmethod
    def _known_resources(cls, resources: dict[str, Any]) -> dict[str, Any]:
        SandboxResources.from_mapping(resources)
        return resources

    def spec(self) -> SandboxSidecarSpec:
        """The provider-neutral sidecar spec this declaration starts."""
        return SandboxSidecarSpec(name=self.name, image=self.image, env=dict(self.env), resources=dict(self.resources))


class SandboxGroupSpec(BaseModel):
    """What an attempt's sandbox group holds and who needs it, as the Resources Server declares it at seed.

    The Environment Server starts the group from its own ``sandbox_config`` (provider, network rule, ``ttl_s``,
    resources) with these declared fields over it, lends it, and stops it. The Resources Server never starts it.

    Attributes:
        image: The main container's image; unset, the Environment Server's ``sandbox_config.image``.
        workdir: The main container's working directory; unset, ``sandbox_config.workdir``.
        env: Environment variables of the main container, over ``sandbox_config.env``.
        files: Text files delivered into the main container before anyone borrows it, keyed by path.
        sidecars: Containers that serve the task's environment tools.
        grader_runs_code: The grader executes model output or calls tools, so it never opts out: it grades in the
            attempt's group, or in a sandbox of its own when the task opted out.
        prepare: The Resources Server prepares the task in the started group through ``/prepare_sandbox`` (see
            ``SimpleResourcesServer.prepare_sandbox``) before the agent, or the grader, borrows it.
    """

    model_config = ConfigDict(extra="forbid")

    image: str | None = None
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    sidecars: list[SandboxGroupSidecar] = Field(default_factory=list)
    grader_runs_code: bool = False
    prepare: bool = False

    @field_validator("sidecars")
    @classmethod
    def _unique_sidecar_names(cls, sidecars: list[SandboxGroupSidecar]) -> list[SandboxGroupSidecar]:
        names = [sidecar.name for sidecar in sidecars]
        if len(set(names)) != len(names):
            raise ValueError(f"sandbox group sidecar names must be unique: {names}")
        return sidecars


class AttemptSandbox(BaseModel):
    """One attempt's sandbox group as its Environment Server tracks it; checkpoint boundaries record it.

    Attributes:
        key: The group's key in the server's ``OwnedSandboxes``, unique to the attempt.
        lend_to_agent: The group starts at seed and the agent borrows it for its turn.
        lend_to_grader: ``/verify`` borrows the group after the agent closes; when the agent borrowed none, the
            group starts just before ``/verify``, for the grader alone.
        group: What the Resources Server declared at seed, if anything.
        workdir: The working directory the group started with, set once it starts.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    lend_to_agent: bool = False
    lend_to_grader: bool = False
    group: SandboxGroupSpec | None = None
    workdir: str | None = None


class ResourcesPrepareSandboxRequest(BaseModel):
    """Ask the Resources Server to prepare the task in the attempt's started sandbox group.

    It borrows the group with ``borrow(sandbox_access)`` and releases it before it replies; it never stops it.
    """

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    sandbox_access: SandboxAccess


class ResourcesPrepareSandboxResponse(BaseModel):
    """Confirm the task is prepared in the attempt's sandbox group."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
