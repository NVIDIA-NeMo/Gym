# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tool sidecar a Resources Server declares, as the Environment Server adds it to the attempt's group."""

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.sandbox.providers.base import SandboxCapabilities
from nemo_gym.server_runtime.tool_sidecar import ToolSidecarConfig, tool_sidecar_config_of


class _SidecarProvider:
    """Runs sidecars and applies the rules OpenSandbox applies on a deployment that enforces egress."""

    name = "fake"
    capabilities = SandboxCapabilities(network_rules=frozenset({"inherit", "no_network", "hosts_only", "model_only"}))

    def network_options(self, rule: str, hosts: Sequence[str], provider_options: Mapping[str, Any]) -> dict[str, Any]:
        return {**provider_options, "network_policy": {"rule": rule, "allow": list(hosts)}}

    async def sidecar_exec(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not called")

    async def upload_sidecar_file(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("not called")


class _NoSidecars:
    name = "plain"
    capabilities = SandboxCapabilities(network_rules=frozenset({"inherit", "no_network"}))

    def network_options(self, rule: str, hosts: Sequence[str], provider_options: Mapping[str, Any]) -> dict[str, Any]:
        return dict(provider_options)


class _OpenEgressOnly(_SidecarProvider):
    capabilities = SandboxCapabilities(network_rules=frozenset({"inherit"}))


def test_the_seed_declares_the_tool_sidecar_and_the_agent_reaches_it_by_its_name() -> None:
    closed = ToolSidecarConfig(image="python:3.13-slim")
    sidecar = closed.group_sidecar()
    assert (sidecar.name, sidecar.image, sidecar.resources, sidecar.egress_hosts, sidecar.tools_port) == (
        "tools",
        "python:3.13-slim",
        {"cpu": 1, "memory_mib": 2048},
        [],
        8000,
    )
    assert sidecar.network == closed.network_rule == "no_network"
    assert (closed.url, closed.network_rule) == ("http://tools:8000", "no_network")

    search = ToolSidecarConfig(image="search:1", name="search", port=9000, egress_hosts=["api.example.com"])
    assert (search.url, search.network_rule) == ("http://search:9000", "hosts_only")


def test_a_provider_that_runs_no_sidecar_or_cannot_close_its_egress_is_refused() -> None:
    config = ToolSidecarConfig(image="python:3.13-slim")
    config.check_provider(_SidecarProvider())
    with pytest.raises(ValueError, match="runs no sidecars"):
        config.check_provider(_NoSidecars())
    with pytest.raises(ValueError, match="cannot apply network 'no_network'"):
        config.check_provider(_OpenEgressOnly())


def test_a_sidecar_name_must_be_a_host_name() -> None:
    with pytest.raises(ValidationError):
        ToolSidecarConfig(image="python:3.13-slim", name="Tools_1")
    with pytest.raises(ValidationError):
        ToolSidecarConfig(image="")


def test_the_environment_server_reads_the_declaration_from_the_resources_config_block() -> None:
    block = OmegaConf.create({"entrypoint": "app.py", "tool_sidecar": {"image": "python:3.13-slim", "port": 9000}})
    assert tool_sidecar_config_of(block) == ToolSidecarConfig(image="python:3.13-slim", port=9000)
    assert tool_sidecar_config_of(OmegaConf.create({"entrypoint": "app.py"})) is None
    with pytest.raises(ValidationError):
        tool_sidecar_config_of({"tool_sidecar": {"image": "python:3.13-slim", "portt": 9000}})
