# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tool sidecar a Resources Server declares, as the Environment Server adds it to the attempt's group."""

import asyncio
import http.server
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.sandbox.providers.base import SandboxCapabilities, SandboxExecResult
from nemo_gym.server_runtime.attempt_sandbox import SandboxGroupSidecar, SandboxGroupSpec
from nemo_gym.server_runtime.tool_sidecar import (
    TOOL_HOST_READY_PATH,
    ToolSidecarConfig,
    tool_sidecar_config_of,
    tool_sidecar_reachable,
    tool_sidecar_to_grant,
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
    with pytest.raises(ValueError, match="cannot reach the tool sidecar 'tools' by its name: .*networking.enabled"):
        config.check_provider(_UnconnectedGroups())


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


class _UnconnectedGroups(_SidecarProvider):
    """Runs sidecars on a deployment that does not connect a group's members, as OpenSandbox without networking."""

    def validate_networking(self) -> None:
        raise NotImplementedError("OpenSandbox networking requires networking.enabled=true")


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _AgentContainer:
    """An agent container whose commands run in a local ``sh`` with only ``programs`` on its PATH.

    Its environment names a proxy that answers nothing, as an agent image with an egress proxy would.
    """

    def __init__(self, bin_dir: Path, programs: Mapping[str, str]) -> None:
        bin_dir.mkdir()
        for name, target in programs.items():
            (bin_dir / name).symlink_to(target)
        self.env = {"PATH": str(bin_dir), "HTTP_PROXY": f"http://127.0.0.1:{_closed_port()}"}
        self.env["http_proxy"] = self.env["HTTP_PROXY"]

    async def exec(self, command: str, *, cwd: str, timeout_s: float) -> SandboxExecResult:
        process = await asyncio.create_subprocess_exec(
            "/bin/sh", "-c", command, cwd=cwd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )


needs_bash = pytest.mark.skipif(not os.path.exists("/bin/bash"), reason="the probe falls back to bash's /dev/tcp")


async def test_the_agent_container_reaches_the_tool_host_with_python_and_ignores_its_proxy(
    tmp_path: Path, tool_host_port: int
) -> None:
    container = _AgentContainer(tmp_path / "bin", {"python3": sys.executable})

    assert await tool_sidecar_reachable(container, "127.0.0.1", tool_host_port) is True  # type: ignore[arg-type]
    closed = _closed_port()
    with pytest.raises(
        RuntimeError, match=f"(?s)cannot reach the tool sidecar at 127.0.0.1:{closed}: .*Connection refused"
    ):
        await tool_sidecar_reachable(container, "127.0.0.1", closed)  # type: ignore[arg-type]


@needs_bash
async def test_an_agent_container_with_only_bash_checks_that_the_tool_hosts_port_answers(
    tmp_path: Path, tool_host_port: int
) -> None:
    container = _AgentContainer(tmp_path / "bin", {"bash": "/bin/bash"})

    assert await tool_sidecar_reachable(container, "127.0.0.1", tool_host_port) is True  # type: ignore[arg-type]
    closed = _closed_port()
    with pytest.raises(RuntimeError, match=f"cannot reach the tool sidecar at 127.0.0.1:{closed}"):
        await tool_sidecar_reachable(container, "127.0.0.1", closed)  # type: ignore[arg-type]


def _group(*sidecars: SandboxGroupSidecar) -> SandboxGroupSpec:
    return SandboxGroupSpec(sidecars=list(sidecars), prepare=True)


_TOOLS = ToolSidecarConfig(image="python:3.13-slim").group_sidecar()


_GRANT = {"resources_server": "ledger", "configured": True, "lent_to_agent": True, "transports": ["direct_http"]}


def test_the_agent_is_granted_the_one_sidecar_that_serves_tools() -> None:
    other = SandboxGroupSidecar(name="db", image="postgres:17")
    assert tool_sidecar_to_grant(_group(other, _TOOLS), **_GRANT) == _TOOLS
    # A Resources Server whose tools run in process declares no tool sidecar, and nothing is granted.
    assert tool_sidecar_to_grant(_group(other), **_GRANT | {"configured": False}) is None
    assert tool_sidecar_to_grant(None, **_GRANT | {"configured": False}) is None


@pytest.mark.parametrize(
    ("group", "overrides", "reason"),
    [
        (None, {}, "'ledger' serves its tools from a sidecar, but its seed declared no sidecar that serves them"),
        (
            _group(_TOOLS, _TOOLS.model_copy(update={"name": "more"})),
            {},
            "'ledger' declared 2 tool sidecars; an agent is granted one direct_http address",
        ),
        (_group(_TOOLS), {"lent_to_agent": False}, "of an attempt sandbox group the agent does not borrow"),
        (_group(_TOOLS), {"transports": ["mcp"]}, "which this server grants over direct_http only"),
    ],
)
def test_a_tool_sidecar_that_cannot_be_granted_is_refused_with_why(
    group: SandboxGroupSpec | None, overrides: dict[str, Any], reason: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(reason)):
        tool_sidecar_to_grant(group, **_GRANT | overrides)
