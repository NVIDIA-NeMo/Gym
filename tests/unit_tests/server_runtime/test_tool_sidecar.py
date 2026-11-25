# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tool sidecar a Resources Server declares, as the Environment Server adds it to the attempt's group."""

import asyncio
import http.server
import shutil
import socket
import subprocess
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.sandbox.providers.base import SandboxCapabilities, SandboxExecResult
from nemo_gym.server_runtime.tool_sidecar import (
    TOOL_HOST_READY_PATH,
    ToolSidecarConfig,
    tool_sidecar_config_of,
    tool_sidecar_reachable,
)


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

    async def download_sidecar_file(self, *args: Any, **kwargs: Any) -> None:
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
    assert (closed.url, sidecar.network) == ("http://tools:8000", "no_network")

    search = ToolSidecarConfig(image="search:1", name="search", port=9000, egress_hosts=["api.example.com"])
    assert (search.url, search.group_sidecar().network) == ("http://search:9000", "hosts_only")
    assert search.group_sidecar().egress_hosts == ["api.example.com"]


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


class _LocalShell:
    """The agent container as a local shell: the probe's script runs for real."""

    def __init__(self, *, exit_code: int | None = None, error_type: str | None = None) -> None:
        self.commands: list[str] = []
        self.exit_code = exit_code
        self.error_type = error_type

    async def exec(self, command: str, *, cwd: str, timeout_s: float) -> SandboxExecResult:
        self.commands.append(command)
        if self.exit_code is not None or self.error_type is not None:
            return SandboxExecResult(stdout="", stderr=None, return_code=self.exit_code, error_type=self.error_type)
        process = await asyncio.create_subprocess_exec(
            "sh", "-c", command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(stdout=stdout.decode(), stderr=stderr.decode(), return_code=process.returncode)


class _Ready(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        self.send_response(200 if self.path == TOOL_HOST_READY_PATH else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def tool_host_port() -> Any:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Ready)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


@pytest.mark.skipif(shutil.which("python3") is None, reason="the probe runs python3 from a shell")
async def test_the_probe_passes_once_the_tool_host_answers_from_the_agent_container(tool_host_port: int) -> None:
    shell = _LocalShell()

    assert await tool_sidecar_reachable(shell, "127.0.0.1", tool_host_port) is True  # type: ignore[arg-type]
    assert f"http://127.0.0.1:{tool_host_port}/_tool_sidecar/ready" in shell.commands[0]


@pytest.mark.skipif(shutil.which("python3") is None, reason="the probe runs python3 from a shell")
async def test_a_tool_host_the_agent_container_cannot_reach_fails_the_probe() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = probe.getsockname()[1]

    with pytest.raises(RuntimeError, match=f"cannot reach the tool sidecar at 127.0.0.1:{closed}"):
        await tool_sidecar_reachable(_LocalShell(), "127.0.0.1", closed)  # type: ignore[arg-type]


async def test_an_agent_image_without_python3_or_bash_is_not_checked_and_an_exec_error_fails() -> None:
    assert await tool_sidecar_reachable(_LocalShell(exit_code=127), "tools", 8000) is False  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="tools:8000: timeout"):
        await tool_sidecar_reachable(_LocalShell(error_type="timeout"), "tools", 8000)  # type: ignore[arg-type]
