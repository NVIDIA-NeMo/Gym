# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An Environment Server that grants a Resources Server's tool sidecar: single_agent_turn checks the pairing at
startup, probes the sidecar from the agent container, grants it, and imports the tool state before /verify.

The agent container's reach probe runs for real, in a local shell, against a stand-in tool host on localhost: the
seed declares the sidecar as ``localhost`` on the stand-in's port, so the address the agent is granted is the one the
probe reached.
"""

import asyncio
import http.server
import os
import shutil
import socket
import subprocess
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from aiohttp import ClientConnectionError
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.agents.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.providers.base import (
    SandboxCapabilities,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.server_runtime.base_resources_server import NG_SANDBOX_ACCESS_KEY
from nemo_gym.server_runtime.owned_sandboxes import OwnedSandboxes
from nemo_gym.server_runtime.tool_access import DirectHTTPToolAccess
from nemo_gym.server_runtime.tool_sidecar import TOOL_HOST_READY_PATH
from tests.helpers.single_agent_turn import _environment_server, _request, _Response


MODEL_HOST = "10.0.0.5"
pytestmark = pytest.mark.skipif(shutil.which("python3") is None, reason="the agent container's probe runs python3")


class _ToolHostStandIn(http.server.ThreadingHTTPServer):
    """Answers the tool host's readiness route, as a started tool host does, and records each request's path."""

    paths: list[str]


class _Ready(http.server.BaseHTTPRequestHandler):
    server: _ToolHostStandIn

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        self.server.paths.append(self.path)
        self.send_response(200 if self.path == TOOL_HOST_READY_PATH else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def tool_host() -> Iterator[_ToolHostStandIn]:
    server = _ToolHostStandIn(("127.0.0.1", 0), _Ready)
    server.paths = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _port(tool_host: _ToolHostStandIn) -> int:
    return tool_host.server_address[1]


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _SidecarProvider:
    """An attempt provider with sidecars: the group's members are records, and a command in the agent container that
    probes the tool host runs in a local shell with ``path`` as its PATH. Any other command only succeeds."""

    name = "fake"
    capabilities = SandboxCapabilities(network_rules=frozenset({"inherit", "model_only", "no_network", "hosts_only"}))

    def __init__(self, *, path: str | None = None) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
        self.probes: list[int] = []
        self.path = path if path is not None else os.environ.get("PATH", "")

    def network_options(self, rule: str, hosts: Sequence[str], provider_options: Mapping[str, Any]) -> dict[str, Any]:
        return {**provider_options, "network_policy": {"rule": rule, "allow": list(hosts)}}

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        sandbox_id = f"sb-{len(self.boxes) + 1}"
        self.boxes[sandbox_id] = {"spec": spec, "state": "running"}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        if TOOL_HOST_READY_PATH not in command:
            return SandboxExecResult(stdout="", stderr=None, return_code=0)
        process = await asyncio.create_subprocess_exec(
            "/bin/sh", "-c", command, cwd="/", env={"PATH": self.path}, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 60)
        self.probes.append(process.returncode)
        return SandboxExecResult(stdout=stdout.decode(), stderr=stderr.decode(), return_code=process.returncode)

    async def upload_file(self, handle: SandboxHandle, source_path: Any, target_path: str) -> None:
        pass

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Any) -> None:
        pass

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING if self.boxes[handle.sandbox_id]["state"] == "running" else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        self.boxes[handle.sandbox_id]["state"] = "stopped"

    async def aclose(self) -> None:
        pass

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id, "scope": scope}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        return SandboxHandle(sandbox_id=str(descriptor["sandbox_id"]), provider_name=self.name, raw=None)

    async def sidecar_exec(self, *args: Any, **kwargs: Any) -> SandboxExecResult:
        raise AssertionError("the Environment Server never runs commands in the tool sidecar")

    async def upload_sidecar_file(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("the Environment Server never writes to the tool sidecar")

    async def download_sidecar_file(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("the Environment Server never reads the tool sidecar")

    def running(self) -> list[str]:
        return [sandbox_id for sandbox_id, box in self.boxes.items() if box["state"] == "running"]


def _seed_body(port: int) -> dict[str, Any]:
    """The Resources seed's reply: the tool sidecar in the attempt's group, on ``port`` of ``localhost``, prepared."""
    sidecars = [{"name": "localhost", "image": "tools:1", "tools_port": port}]
    return {"resources_session_id": "resources-session", "sandbox_group": {"sidecars": sidecars, "prepare": True}}


def _sidecar_server(
    provider: _SidecarProvider, port: int, **config: Any
) -> tuple[SingleAgentTurnEnvironmentServer, Any]:
    """single_agent_turn lending a model-only group from ``provider``, bound to a Resources Server whose config sets
    ``tool_sidecar`` and whose seed declares it on ``port``."""
    base, client = _environment_server()
    settings = {
        "resources_tool_transports": ["direct_http"],
        "sandbox_provider": "attempt_sandboxes",
        "sandbox_config": {
            "image": "python:3.13-slim",
            "workdir": "/workspace",
            "ttl_s": 25200,
            "network": "model_only",
            "model_server": "policy_model",
        },
    }
    server_config = SingleAgentTurnEnvironmentServerConfig.model_validate(base.config.model_dump() | settings | config)
    client.global_config_dict = OmegaConf.merge(
        client.global_config_dict,
        {
            "policy_model": {"responses_api_models": {"vllm_model": {"host": MODEL_HOST, "port": 8000}}},
            "resources": {"resources_servers": {"test": {"tool_sidecar": {"image": "tools:1"}}}},
        },
    )
    server = SingleAgentTurnEnvironmentServer(config=server_config, server_client=client)
    server._owned_sandboxes["sandbox_provider"] = OwnedSandboxes(
        provider,
        state_dir=None,
        server_name="environment",
        provider_config_ref="attempt_sandboxes",
        global_config=client.global_config_dict,
    )
    client.responses[0] = _Response(_seed_body(port))
    # The prepare reply after the group starts, and the state import's reply before /verify.
    client.responses.insert(1, _Response({"resources_session_id": "resources-session"}))
    client.responses.insert(5, _Response({"imported": True}))
    return server, client


def _recording(client: Any, provider: _SidecarProvider) -> list[tuple[str, list[str]]]:
    """Record, at each call the server makes, the sandboxes then running."""
    seen: list[tuple[str, list[str]]] = []
    post = type(client).post

    async def recording_post(self: Any, server_name: str, url_path: str, **kwargs: Any) -> Any:
        seen.append((url_path, provider.running()))
        return await post(self, server_name, url_path, **kwargs)

    object.__setattr__(client, "post", recording_post.__get__(client))
    return seen


def _paths(client: Any) -> list[str]:
    return [path for _, path, _ in client.calls]


async def test_the_agent_is_granted_the_tool_sidecar_it_was_probed_to_reach_and_grading_reads_its_state_first(
    tool_host: _ToolHostStandIn,
) -> None:
    provider = _SidecarProvider()
    server, client = _sidecar_server(provider, _port(tool_host))
    recorded = _recording(client, provider)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert result.result.reward == 1.0
    running_at = dict(recorded)
    # One group: the agent container, model only, and the tool sidecar the seed declared, which reaches no host.
    assert list(provider.boxes) == ["sb-1"]
    spec = provider.boxes["sb-1"]["spec"]
    assert spec.provider_options["network_policy"] == {"rule": "model_only", "allow": [MODEL_HOST]}
    [sidecar] = spec.sidecars
    assert (sidecar.name, sidecar.image) == ("localhost", "tools:1")
    assert sidecar.provider_options == {"network_policy": {"rule": "no_network", "allow": []}}
    assert _paths(client) == [
        "/seed_session",
        "/prepare_sandbox",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/tool_sidecar/import_state",
        "/verify",
        "/close_session",
    ]
    calls = {path: kwargs for _, path, kwargs in client.calls}
    prepare = SandboxAccess.model_validate(calls["/prepare_sandbox"]["json"]["sandbox_access"])
    assert prepare.connection.descriptor["sandbox_id"] == "sb-1"
    # The agent container reached the tool host's readiness route before the agent was seeded.
    assert provider.probes == [0]
    assert tool_host.paths == [TOOL_HOST_READY_PATH]
    # The agent's only tool grant is the address it was probed to reach, with the Resources session's cookies.
    seed = AgentSeedSessionRequest.model_validate(calls["/v1/agent_sessions"]["json"])
    [grant] = seed.tool_accesses
    assert isinstance(grant, DirectHTTPToolAccess)
    assert str(grant.base_url).rstrip("/") == f"http://localhost:{_port(tool_host)}"
    assert dict(grant.cookies) == {"session": "cookie-value"}
    # The tool state is copied out of the group while it runs. The verifier (verifier_mode "none") grades it in the
    # Resources Server, lent nothing, after the group has stopped.
    imported = SandboxAccess.model_validate(calls["/tool_sidecar/import_state"]["json"]["sandbox_access"])
    assert imported.connection.descriptor["sandbox_id"] == "sb-1"
    assert running_at["/tool_sidecar/import_state"] == ["sb-1"]
    assert NG_SANDBOX_ACCESS_KEY not in calls["/verify"]["json"]
    assert running_at["/verify"] == []
    assert provider.running() == []


async def test_a_shared_verifier_reads_the_tool_state_and_borrows_the_same_group(
    tool_host: _ToolHostStandIn,
) -> None:
    provider = _SidecarProvider()
    server, client = _sidecar_server(provider, _port(tool_host))
    client.responses[0] = _Response(_seed_body(_port(tool_host)) | {"verifier_mode": "shared"})
    recorded = _recording(client, provider)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    running_at = dict(recorded)
    calls = {path: kwargs for _, path, kwargs in client.calls}
    imported = calls["/tool_sidecar/import_state"]["json"]["sandbox_access"]
    assert imported == calls["/verify"]["json"][NG_SANDBOX_ACCESS_KEY]
    assert running_at["/verify"] == ["sb-1"]
    assert provider.running() == []


async def test_a_tool_sidecar_the_agent_container_cannot_reach_fails_the_attempt_for_a_retry() -> None:
    provider = _SidecarProvider()
    closed = _closed_port()
    server, client = _sidecar_server(provider, closed)

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("seed", False)
    assert f"cannot reach the tool sidecar at localhost:{closed}" in result.failure.failure_reason
    assert provider.probes and provider.probes[0] != 0
    assert "/v1/agent_sessions" not in _paths(client)
    assert "/verify" not in _paths(client)
    assert provider.running() == []


async def test_a_resources_server_that_cannot_prepare_the_sidecar_fails_the_attempt_before_the_probe(
    tool_host: _ToolHostStandIn,
) -> None:
    provider = _SidecarProvider()
    server, client = _sidecar_server(provider, _port(tool_host))
    client.responses[1] = ClientConnectionError("resources went away")

    result = await server.run_request(_request())

    assert (result.failure.stage, result.failure.terminal) == ("seed", False)
    assert "resources went away" in result.failure.failure_reason
    assert provider.probes == []
    assert "/v1/agent_sessions" not in _paths(client)
    assert provider.running() == []


async def test_an_agent_image_that_cannot_check_the_tool_sidecar_is_granted_nothing(
    tool_host: _ToolHostStandIn, tmp_path: Path
) -> None:
    # An agent image without python3 or bash: the probe's shell finds neither on its PATH.
    provider = _SidecarProvider(path=str(tmp_path))
    server, client = _sidecar_server(provider, _port(tool_host))

    result = await server.run_request(_request())

    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "neither python3 nor bash" in result.failure.failure_reason
    assert provider.probes == [127]
    assert tool_host.paths == []
    assert "/v1/agent_sessions" not in _paths(client)
    assert provider.running() == []


async def test_tool_state_that_cannot_be_imported_is_never_graded(tool_host: _ToolHostStandIn) -> None:
    provider = _SidecarProvider()
    server, client = _sidecar_server(provider, _port(tool_host))
    client.responses[5] = ClientConnectionError("sidecar went away")

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("verification", False)
    assert "/verify" not in _paths(client)
    assert provider.running() == []


@pytest.mark.parametrize(
    ("sidecars", "refusal"),
    [
        ([{"name": "tools", "image": "tools:1"}], "declared no sidecar that serves them"),
        (
            [
                {"name": "tools", "image": "tools:1", "tools_port": 8000},
                {"name": "more", "image": "t:2", "tools_port": 9},
            ],
            "declared 2 tool sidecars",
        ),
    ],
)
async def test_a_seed_that_does_not_declare_one_tool_sidecar_fails_for_good(
    tool_host: _ToolHostStandIn, sidecars: list[dict[str, Any]], refusal: str
) -> None:
    provider = _SidecarProvider()
    server, client = _sidecar_server(provider, _port(tool_host))
    client.responses[0] = _Response(
        {"resources_session_id": "resources-session", "sandbox_group": {"sidecars": sidecars, "prepare": True}}
    )

    result = await server.run_request(_request())

    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert refusal in result.failure.failure_reason
    assert provider.probes == []
    assert "/v1/agent_sessions" not in _paths(client)
    assert provider.running() == []


def test_a_pairing_that_cannot_start_or_grant_the_tool_sidecar_is_refused_at_startup() -> None:
    server, _ = _sidecar_server(_SidecarProvider(), 8000, resources_tool_transports=["mcp"])
    with pytest.raises(ValueError, match="over direct_http only"):
        server._check_tool_sidecar()

    server, _ = _sidecar_server(_SidecarProvider(), 8000, sandbox_provider=None, sandbox_config=None)
    with pytest.raises(ValueError, match="only with sandbox_provider and sandbox_config set"):
        server._check_tool_sidecar()


def test_a_model_only_group_may_grant_tools_that_run_in_its_sidecar() -> None:
    # Without the sidecar, model_only with direct_http tools is refused: the sandbox could not reach the Resources host.
    networking = {"opensandbox": {"networking": {"enabled": True, "enforces_network_policy": True}}}
    server, client = _sidecar_server(_SidecarProvider(), 8000)
    client.global_config_dict = OmegaConf.merge(client.global_config_dict, {"attempt_sandboxes": networking})

    server.check_sandbox_provider()

    class _OpenEgressOnly(_SidecarProvider):
        capabilities = SandboxCapabilities(network_rules=frozenset({"inherit", "model_only"}))

    server, client = _sidecar_server(_OpenEgressOnly(), 8000)
    client.global_config_dict = OmegaConf.merge(client.global_config_dict, {"attempt_sandboxes": networking})
    with pytest.raises(ValueError, match="cannot apply network 'no_network'"):
        server.check_sandbox_provider()


def test_a_config_without_the_resources_servers_block_serves_the_tools_in_process() -> None:
    server, client = _sidecar_server(_SidecarProvider(), 8000)
    client.global_config_dict = OmegaConf.create(
        {key: value for key, value in client.global_config_dict.items() if key != "resources"}
    )

    assert server.tool_sidecar is None
    # Without a tool sidecar there is nothing to refuse, even without an attempt sandbox to run one in.
    server, client = _sidecar_server(_SidecarProvider(), 8000, sandbox_provider=None, sandbox_config=None)
    client.global_config_dict = OmegaConf.create(
        {key: value for key, value in client.global_config_dict.items() if key != "resources"}
    )
    server._check_tool_sidecar()
