# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Resources Server whose tools run in a sidecar of the attempt's sandbox group, granted by this server."""

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServer
from environment_servers.single_agent_turn.tests.test_attempt_sandbox import _lending_server, _Provider
from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.sandbox import SandboxAccess, SandboxCapabilities, SandboxExecResult, SandboxHandle
from nemo_gym.server_runtime import NG_SANDBOX_ACCESS_KEY, DirectHTTPToolAccess
from tests.helpers.single_agent_turn import _request, _Response


TOOL_COOKIES = {"session": "tool-host-cookie"}


class _SidecarProvider(_Provider):
    """The fake attempt provider, with sidecars and the rule that closes a sidecar's egress."""

    capabilities = SandboxCapabilities(network_rules=frozenset({"inherit", "model_only", "no_network", "hosts_only"}))

    def __init__(self, *, probe_exit: int = 0) -> None:
        super().__init__()
        self.commands: list[str] = []
        self.probe_exit = probe_exit

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        self.commands.append(command)
        failed = self.probe_exit if "_tool_sidecar/ready" in command else 0
        return SandboxExecResult(stdout="", stderr="connection refused" if failed else None, return_code=failed)

    async def sidecar_exec(self, *args: Any, **kwargs: Any) -> SandboxExecResult:
        raise AssertionError("the Environment Server never runs commands in the tool sidecar")

    async def upload_sidecar_file(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("the Environment Server never writes to the tool sidecar")

    async def download_sidecar_file(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("the Environment Server never reads the tool sidecar")

    def network_options(self, rule: str, hosts: Sequence[str], provider_options: Mapping[str, Any]) -> dict[str, Any]:
        self.rules.append((rule, list(hosts)))
        return {**provider_options, "network_policy": {"rule": rule, "allow": list(hosts)}}


def _sidecar_server(
    provider: _SidecarProvider, *, tool_sidecar: dict[str, Any] | None = None, **config: Any
) -> tuple[SingleAgentTurnEnvironmentServer, Any]:
    server, client = _lending_server(provider, **({"resources_tool_transports": ["direct_http"]} | config))
    client.global_config_dict = OmegaConf.merge(
        client.global_config_dict,
        {"resources": {"resources_servers": {"test": {"tool_sidecar": tool_sidecar or {"image": "tools:1"}}}}},
    )
    # The prepare reply after the Resources seed, and the collect reply before /verify.
    client.responses.insert(1, _Response({"cookies": TOOL_COOKIES}))
    client.responses.insert(5, _Response({"imported": True}))
    return server, client


def _paths(client: Any) -> list[str]:
    return [path for _, path, _ in client.calls]


async def test_the_agent_calls_the_tools_in_the_groups_sidecar_and_grading_reads_their_state_first() -> None:
    provider = _SidecarProvider()
    server, client = _sidecar_server(provider)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert result.result.reward == 1.0
    # One group: the agent container, model only, and the tool sidecar, which reaches no host.
    assert list(provider.boxes) == ["sb-1"]
    spec = provider.boxes["sb-1"]["spec"]
    assert spec.provider_options["network_policy"] == {"rule": "model_only", "allow": ["10.0.0.5"]}
    [sidecar] = spec.sidecars
    assert (sidecar.name, sidecar.image) == ("tools", "tools:1")
    assert sidecar.provider_options == {"network_policy": {"rule": "no_network", "allow": []}}
    assert _paths(client) == [
        "/seed_session",
        "/tool_sidecar/prepare",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/tool_sidecar/collect",
        "/verify",
        "/close_session",
    ]
    calls = {path: kwargs for _, path, kwargs in client.calls}
    # The Resources Server prepares the sidecar with the seed it got and access to the group.
    prepare = calls["/tool_sidecar/prepare"]["json"]
    assert prepare["seed"] == calls["/seed_session"]["json"]
    assert SandboxAccess.model_validate(prepare["sandbox_access"]).connection.descriptor["sandbox_id"] == "sb-1"
    # The agent container reaches the tool host before the agent is seeded.
    assert provider.commands and "http://tools:8000/_tool_sidecar/ready" in provider.commands[0]
    # The agent's only tool grant is the sidecar, with the tool host's cookies, not the Resources host.
    seed = AgentSeedSessionRequest.model_validate(calls["/v1/agent_sessions"]["json"])
    [grant] = seed.tool_accesses
    assert isinstance(grant, DirectHTTPToolAccess)
    assert (str(grant.base_url).rstrip("/"), dict(grant.cookies)) == ("http://tools:8000", TOOL_COOKIES)
    # The state the tool calls left is copied out of the sidecar before /verify grades, with the same cookies.
    collect = calls["/tool_sidecar/collect"]["json"]
    assert collect["cookies"] == TOOL_COOKIES
    assert collect["sandbox_access"] == calls["/verify"]["json"][NG_SANDBOX_ACCESS_KEY]
    assert provider.running() == []


@pytest.mark.parametrize(
    ("failure", "probe_exit", "message"),
    [
        (ConnectionResetError("resources went away"), 0, "resources went away"),
        (None, 7, "cannot reach the tool sidecar"),
    ],
)
async def test_a_tool_sidecar_that_is_not_ready_fails_the_attempt_for_a_retry_before_the_agent_runs(
    failure: Exception | None, probe_exit: int, message: str
) -> None:
    provider = _SidecarProvider(probe_exit=probe_exit)
    server, client = _sidecar_server(provider)
    if failure is not None:
        client.responses[1] = failure

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("seed", False)
    assert message in result.failure.failure_reason
    assert "/v1/agent_sessions" not in _paths(client)
    assert "/verify" not in _paths(client)
    assert provider.running() == []


async def test_tool_state_that_cannot_be_collected_is_never_graded() -> None:
    provider = _SidecarProvider()
    server, client = _sidecar_server(provider)
    client.responses[5] = ConnectionResetError("sidecar went away")

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("verification", False)
    assert "/verify" not in _paths(client)
    assert provider.running() == []


async def test_a_sandbox_the_resources_seed_lends_has_no_tool_sidecar() -> None:
    provider = _SidecarProvider()
    server, client = _sidecar_server(provider)
    lent = {"connection": {"kind": "direct", "provider_config_ref": "task", "descriptor": {"id": "t"}}, "workdir": "/"}
    client.responses[0] = _Response({"resources_session_id": "resources-session", "sandbox_access": lent})

    result = await server.run_request(_request())

    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "has no tool sidecar" in result.failure.failure_reason
    assert provider.boxes == {}


def test_a_pairing_that_cannot_start_or_grant_the_tool_sidecar_is_refused_at_startup() -> None:
    server, _ = _sidecar_server(_SidecarProvider(), resources_tool_transports=["mcp"])
    with pytest.raises(ValueError, match="over direct_http only"):
        server._check_tool_sidecar()

    server, _ = _sidecar_server(_SidecarProvider(), sandbox_provider=None, sandbox_config=None)
    with pytest.raises(ValueError, match="only with sandbox_provider and sandbox_config set"):
        server._check_tool_sidecar()


def test_a_model_only_group_may_grant_tools_that_run_in_its_sidecar() -> None:
    # Without the sidecar, model_only with direct_http tools is refused: the sandbox could not reach the Resources host.
    server, client = _sidecar_server(_SidecarProvider())
    provider = {"opensandbox": {"networking": {"enabled": True, "enforces_network_policy": True}}}
    client.global_config_dict = OmegaConf.merge(client.global_config_dict, {"attempt_sandboxes": provider})

    server.check_sandbox_provider()

    class _OpenEgressOnly(_SidecarProvider):
        capabilities = SandboxCapabilities(network_rules=frozenset({"inherit", "model_only"}))

    server, client = _sidecar_server(_OpenEgressOnly())
    client.global_config_dict = OmegaConf.merge(client.global_config_dict, {"attempt_sandboxes": provider})
    with pytest.raises(ValueError, match="cannot apply network 'no_network'"):
        server.check_sandbox_provider()
