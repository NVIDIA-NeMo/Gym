# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Resources Server's HTTP tools served from a sidecar of the attempt's sandbox group.

A Resources Server declares a ``tool_sidecar`` in its config. The Environment Server that creates the attempt's group
adds the sidecar to the group (:meth:`ToolSidecarConfig.sidecar_spec`), with an egress of its own. Once the group
runs, it asks the Resources Server to start its tool host there through a borrowed handle (``/tool_sidecar/prepare``),
and grants the agent ``http://<name>:<port>`` with the cookies the reply names. After the agent closes and before
``/verify``, it asks the Resources Server to copy the session's tool state out of the sidecar
(``/tool_sidecar/collect``), then stops the group. No tool call of the agent reaches the Resources host.
"""

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PositiveFloat

from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.providers.base import NetworkRule, SandboxSidecarSpec
from nemo_gym.server_runtime.owned_sandboxes import check_network_rule


TOOL_SIDECAR_PREPARE_PATH = "/tool_sidecar/prepare"
TOOL_SIDECAR_COLLECT_PATH = "/tool_sidecar/collect"


class ToolSidecarConfig(BaseModel):
    """Where a Resources Server's tools run: a sidecar of the attempt's sandbox group, never the Resources host.

    The sidecar runs the Resources Server's own class, with this server's config, and serves its tool routes, so a
    tool answers as it does in process. It reaches only ``egress_hosts``, none by default. The agent container reaches
    it as ``http://<name>:<port>``; on OpenSandbox that needs ``networking.enabled``. Code in the agent container can
    also reach the sidecar's other ports, such as the provider's exec daemon, so the sidecar must hold no secret: a
    config that would put one there is refused.

    Attributes:
        image: The sidecar's image. One without the tool host's runtime gets it installed at the first prepare, which
            needs ``sh`` and ``bash`` in the image; an image whose runtime is baked in at the same path skips that.
        name: The sidecar's name, which is also its host name in the group.
        port: The port the tool host listens on.
        resources: The sidecar's CPU and memory, as ``SandboxSidecarSpec.resources``.
        egress_hosts: The hosts the sidecar may reach, such as a tool's own API.
        install_timeout_seconds: How long installing the runtime and starting the tool host may take.
        ready_timeout_seconds: How long the started tool host may take to answer.
    """

    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1)
    name: str = Field(default="tools", pattern=r"^[a-z][a-z0-9-]{0,62}$")
    port: int = Field(default=8000, ge=1024, le=65535)
    resources: dict[str, JsonValue] = Field(default_factory=lambda: {"cpu": 1, "memory_mib": 2048})
    egress_hosts: list[str] = Field(default_factory=list)
    install_timeout_seconds: PositiveFloat = 900
    ready_timeout_seconds: PositiveFloat = 120

    @property
    def url(self) -> str:
        """The tool host's address as the agent container reaches it."""
        return f"http://{self.name}:{self.port}"

    @property
    def network_rule(self) -> NetworkRule:
        return "hosts_only" if self.egress_hosts else "no_network"

    def check_provider(self, provider: object) -> None:
        """Refuse a provider that cannot run sidecars or cannot limit this sidecar's egress.

        Raises:
            ValueError: Naming what the provider lacks.
        """
        if not hasattr(provider, "sidecar_exec") or not hasattr(provider, "upload_sidecar_file"):
            raise ValueError(
                f"sandbox provider {getattr(provider, 'name', type(provider).__name__)!r} cannot run the tool sidecar "
                f"{self.name!r}: it runs no sidecars"
            )
        check_network_rule(provider, self.network_rule)

    def sidecar_spec(self, provider: object) -> SandboxSidecarSpec:
        """The sidecar to add to the group's ``SandboxSpec.sidecars``, with the provider's own egress options.

        Raises:
            ValueError: If the provider cannot run it (:meth:`check_provider`).
        """
        self.check_provider(provider)
        options = provider.network_options(self.network_rule, self.egress_hosts, {})  # type: ignore[attr-defined]
        return SandboxSidecarSpec(
            name=self.name, image=self.image, resources=dict(self.resources), provider_options=options
        )


class ToolSidecarPrepareRequest(BaseModel):
    """Start the tool host in the group's sidecar and seed the episode's tool session there.

    Attributes:
        seed: The ``/seed_session`` body the Resources Server was seeded with; the tool host is seeded with the same.
        sandbox_access: Operate-scope access to the attempt's group, which the Resources Server borrows.
    """

    model_config = ConfigDict(extra="forbid")

    seed: dict[str, JsonValue]
    sandbox_access: SandboxAccess


class ToolSidecarPrepareResponse(BaseModel):
    """What the agent's grant needs besides the sidecar's address.

    Attributes:
        cookies: The tool host's session cookies; every tool call of the episode sends them.
    """

    cookies: dict[str, str] = Field(default_factory=dict)


class ToolSidecarCollectRequest(BaseModel):
    """Copy the episode's tool state out of the sidecar into the Resources Server's session, before ``/verify``.

    Attributes:
        sandbox_access: Operate-scope access to the attempt's group.
        cookies: The tool host's session cookies, from the prepare reply.
    """

    model_config = ConfigDict(extra="forbid")

    sandbox_access: SandboxAccess
    cookies: dict[str, str] = Field(default_factory=dict)


class ToolSidecarCollectResponse(BaseModel):
    """Whether the sidecar's session held state the Resources Server imported."""

    imported: bool


def tool_sidecar_config_of(resources_config: Mapping[str, Any]) -> ToolSidecarConfig | None:
    """The ``tool_sidecar`` a Resources Server's config block declares, for the Environment Server that binds it."""
    raw = resources_config.get("tool_sidecar")
    if raw is None:
        return None
    return ToolSidecarConfig.model_validate(
        OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else raw
    )
