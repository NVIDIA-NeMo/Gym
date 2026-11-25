# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Resources Server's HTTP tools served from a sidecar of the attempt's sandbox group.

A Resources Server declares a ``tool_sidecar`` in its config, and the sidecar in its seed's ``sandbox_group``. The
Environment Server's base class starts the group, and the Resources Server starts its tool host in the sidecar at
``/prepare_sandbox``, through a borrowed handle. The Environment Server grants the agent ``http://<name>:<port>`` with
the Resources session's cookies, which the tool host reads as its own, and checks that the agent container reaches it.
After the agent closes and before ``/verify``, it asks the Resources Server to copy the session's tool state out of
the sidecar (``/tool_sidecar/import_state``), then stops the group. No tool call of the agent reaches the Resources host.
"""

import shlex
from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PositiveFloat

from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SupportsSandboxSidecars
from nemo_gym.server_runtime.attempt_sandbox import SandboxGroupSidecar, SandboxGroupSpec
from nemo_gym.server_runtime.owned_sandboxes import check_network_rule


TOOL_SIDECAR_IMPORT_STATE_PATH = "/tool_sidecar/import_state"
# The tool host's readiness route, which anyone may call.
TOOL_HOST_READY_PATH = "/_tool_sidecar/ready"


class ToolSidecarConfig(BaseModel):
    """Where a Resources Server's tools run: a sidecar of the attempt's sandbox group, never the Resources host.

    The sidecar runs the Resources Server's own class, with this server's config, and serves its tool routes, so a
    tool answers as it does in process. It reaches only ``egress_hosts``, none by default. The agent
    container reaches it as ``http://<name>:<port>``; on OpenSandbox that needs ``networking.enabled``. Code in the
    agent container can also reach the sidecar's other ports, such as the provider's exec daemon, so the sidecar must
    hold no secret: a config that would put one there is refused.

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

    def check_provider(self, provider: object) -> None:
        """Refuse at startup a provider that cannot run this sidecar or apply its own egress, as the attempt's group
        would at every seed.

        Raises:
            ValueError: Naming what the provider lacks.
        """
        if not isinstance(provider, SupportsSandboxSidecars):
            raise ValueError(
                f"sandbox provider {getattr(provider, 'name', type(provider).__name__)!r} cannot run the tool sidecar "
                f"{self.name!r}: it runs no sidecars"
            )
        check_network_rule(provider, self.group_sidecar().network)
        # The agent container reaches the sidecar by its name only where the provider connects a group's members.
        validate_networking = getattr(provider, "validate_networking", None)
        if validate_networking is not None:
            try:
                validate_networking()
            except NotImplementedError as error:
                raise ValueError(
                    f"the agent container cannot reach the tool sidecar {self.name!r} by its name: {error}"
                ) from error

    def group_sidecar(self) -> SandboxGroupSidecar:
        """The sidecar the Resources Server's seed declares in its ``sandbox_group``, with its own egress and the port
        its tools are served on."""
        return SandboxGroupSidecar(
            name=self.name,
            image=self.image,
            resources=dict(self.resources),
            egress_hosts=list(self.egress_hosts),
            tools_port=self.port,
        )


class ToolSidecarImportStateRequest(BaseModel):
    """Copy the episode's tool state out of the sidecar into the Resources Server's session, before ``/verify``.

    The request carries the Resources session's cookies, which name the session in both places.

    Attributes:
        sandbox_access: Operate-scope access to the attempt's group.
    """

    model_config = ConfigDict(extra="forbid")

    sandbox_access: SandboxAccess


class ToolSidecarImportStateResponse(BaseModel):
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


def tool_sidecar_to_grant(
    group: SandboxGroupSpec | None,
    *,
    resources_server: str,
    configured: bool,
    lent_to_agent: bool,
    transports: Sequence[str],
) -> SandboxGroupSidecar | None:
    """The sidecar of the seed's group whose tools the Environment Server grants the agent; None when the Resources
    Server serves its tools in its own process.

    Args:
        group: The group the Resources Server's seed declared.
        resources_server: The Resources Server's name, for the refusal.
        configured: Whether the Resources Server's config sets ``tool_sidecar``.
        lent_to_agent: Whether the agent borrows the attempt's group.
        transports: The Environment Server's ``resources_tool_transports``.

    Raises:
        ValueError: Naming why the tools cannot be granted, which no retry changes: a configured tool sidecar the seed
            does not declare, more than one, a group the agent does not borrow, or a grant other than direct_http.
    """
    serving = [sidecar for sidecar in (group.sidecars if group is not None else []) if sidecar.tools_port]
    name = repr(resources_server)
    if not serving:
        if configured:
            raise ValueError(
                f"{name} serves its tools from a sidecar, but its seed declared no sidecar that serves them"
            )
        return None
    if len(serving) > 1:
        raise ValueError(f"{name} declared {len(serving)} tool sidecars; an agent is granted one direct_http address")
    if not lent_to_agent:
        raise ValueError(
            f"{name} serves its tools from a sidecar of an attempt sandbox group the agent does not borrow"
        )
    if "direct_http" not in transports:
        raise ValueError(f"{name} serves its tools from a sidecar, which this server grants over direct_http only")
    return serving[0]


async def tool_sidecar_reachable(sandbox: AsyncSandbox, host: str, port: int) -> bool:
    """Probe the tool host from the agent container, as the agent's tool calls will reach it.

    With ``python3`` the probe asks the tool host whether it is ready, without any proxy the container's environment
    names; with only ``bash`` it opens a connection to its port. Returns False when the image has neither, so the reach
    was not checked.

    Raises:
        RuntimeError: If the agent container cannot reach the tool host.
    """
    ready = shlex.quote(f"http://{host}:{port}{TOOL_HOST_READY_PATH}")
    fetch = (
        "import sys, urllib.request; "
        "urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1], timeout=10)"
    )
    probe = (
        "if command -v python3 >/dev/null; then "
        f"python3 -c {shlex.quote(fetch)} {ready}; "
        "elif command -v bash >/dev/null; then "
        f"bash -c 'exec 3<>\"/dev/tcp/$0/$1\"' {shlex.quote(host)} {int(port)}; "
        "else exit 127; fi"
    )
    result = await sandbox.exec(probe, cwd="/", timeout_s=60)
    if result.return_code == 127 and result.error_type is None:
        return False
    if result.return_code != 0 or result.error_type is not None:
        detail = result.error_type or (result.stderr or result.stdout or "").strip()[-1000:]
        raise RuntimeError(f"The agent container cannot reach the tool sidecar at {host}:{port}: {detail}")
    return True
