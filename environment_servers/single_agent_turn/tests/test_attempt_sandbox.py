# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandbox the Environment Server starts for an attempt when the Resources Server lends none."""

import asyncio
import json
import typing
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel, ValidationError

from environment_servers.single_agent_turn.app import (
    AttemptSandboxConfig,
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
    SingleAgentTurnResult,
)
from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.sandbox import (
    SandboxAccess,
    SandboxCapabilities,
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.server_runtime import NG_SANDBOX_ACCESS_KEY, OwnedSandboxes, SingleAgentTurnResponse
from tests.helpers.single_agent_turn import _environment_server, _request, _Response


MODEL_HOST = "10.0.0.5"
# The model_server a rule names: only the rules that allow the model server name one.
MODEL_SERVER = {"model_only": "policy_model", "inherit": None}


class _Provider:
    """A provider that applies the model-only rule, as OpenSandbox does, and lends sandboxes to other processes."""

    name = "fake"
    capabilities = SandboxCapabilities(network_rules=frozenset({"inherit", "model_only"}))

    def __init__(self) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
        self.rules: list[tuple[str, list[str]]] = []
        self.fail_create: Exception | None = None
        # The serialize call, counting from 1, that fails: the second describes the sandbox to the verifier.
        self.fail_serialize_at: int | None = None
        self.serializations = 0

    def network_options(self, rule: str, hosts: Sequence[str], provider_options: Mapping[str, Any]) -> dict[str, Any]:
        self.rules.append((rule, list(hosts)))
        return {**provider_options, "network_policy": {"allow": list(hosts)}}

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if self.fail_create is not None:
            raise self.fail_create
        sandbox_id = f"sb-{len(self.boxes) + 1}"
        self.boxes[sandbox_id] = {"spec": spec, "state": "running"}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

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
        self.serializations += 1
        if self.serializations == self.fail_serialize_at:
            raise ConnectionResetError("provider went away")
        return {"sandbox_id": handle.sandbox_id, "scope": scope}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        return SandboxHandle(sandbox_id=str(descriptor["sandbox_id"]), provider_name=self.name, raw=None)

    def running(self) -> list[str]:
        return [sandbox_id for sandbox_id, box in self.boxes.items() if box["state"] == "running"]


def _attempt_sandbox_config(**overrides: Any) -> dict[str, Any]:
    return {
        "image": "python:3.13-slim",
        "workdir": "/workspace",
        "ttl_s": 25200,
        "network": "model_only",
        "model_server": "policy_model",
    } | overrides


def _lending_server(
    provider: _Provider | None, *, agent_delay_s: float = 0.0, **config: Any
) -> tuple[SingleAgentTurnEnvironmentServer, Any]:
    """A lending server; with no ``provider``, none is built, so a test sees whether the server builds one."""
    base, client = _environment_server()
    config = SingleAgentTurnEnvironmentServerConfig.model_validate(
        base.config.model_dump()
        | {"sandbox_provider": "attempt_sandboxes", "sandbox_config": _attempt_sandbox_config()}
        | config
    )
    client.global_config_dict = OmegaConf.merge(
        client.global_config_dict,
        {"policy_model": {"responses_api_models": {"vllm_model": {"host": MODEL_HOST, "port": 8000}}}},
    )
    if agent_delay_s:
        client.slow_path = "/ng-rollout/rollout-a2/v1/responses"
    server = SingleAgentTurnEnvironmentServer(config=config, server_client=client)
    if provider is not None:
        server._owned_sandboxes["sandbox_provider"] = OwnedSandboxes(
            provider,
            state_dir=None,
            server_name="environment",
            provider_config_ref="attempt_sandboxes",
            global_config=client.global_config_dict,
        )
    return server, client


async def test_an_attempt_borrows_one_model_only_sandbox_for_its_turn_and_its_verification() -> None:
    provider = _Provider()
    server, client = _lending_server(provider)
    # A verifier that echoes its whole request body back, as some do.
    echoing = dict(json.loads(client.responses[4].body))
    client.responses[4] = _Response(echoing | {NG_SANDBOX_ACCESS_KEY: {"connection": "echoed"}})

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert result.result.reward == 1.0
    # One sandbox, started with the rule that allows only the model server's host.
    assert list(provider.boxes) == ["sb-1"]
    assert provider.rules == [("model_only", [MODEL_HOST])]
    spec = provider.boxes["sb-1"]["spec"]
    assert (spec.image, spec.workdir, spec.ttl_s) == ("python:3.13-slim", "/workspace", 25200)
    assert spec.provider_options["network_policy"] == {"allow": [MODEL_HOST]}
    assert spec.metadata["task_id"] == "task"
    # The agent borrows it with operate scope, through the provider block that a borrower resolves.
    seed = AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"])
    assert seed.sandbox_access is not None
    assert seed.sandbox_access.connection.provider_config_ref == "attempt_sandboxes"
    assert seed.sandbox_access.connection.descriptor | {"workdir": None} == {
        "sandbox_id": "sb-1",
        "scope": "operate",
        "workdir": None,
    }
    assert seed.sandbox_access.workdir == "/workspace"
    # After the agent closed, the verifier is lent the same sandbox.
    paths = [path for _, path, _ in client.calls]
    assert paths.index("/v1/agent_sessions/close") < paths.index("/verify")
    verify_body = client.calls[paths.index("/verify")][2]["json"]
    lent = SandboxAccess.model_validate(verify_body[NG_SANDBOX_ACCESS_KEY])
    assert lent.connection.descriptor == seed.sandbox_access.connection.descriptor
    # The lent sandbox's access never reaches the stored result, even from a verifier that echoes it.
    assert NG_SANDBOX_ACCESS_KEY not in result.result.model_dump()
    # It is stopped once the episode ends.
    assert provider.running() == []
    assert server.sandboxes.keys == []


async def test_a_resources_server_that_lends_a_sandbox_of_its_own_fails_the_attempt_and_none_is_started() -> None:
    server, client = _lending_server(None)
    resources_access = {
        "connection": {"kind": "direct", "provider_config_ref": "task_sandboxes", "descriptor": {"id": "task"}},
        "workdir": "/repo",
    }
    client.responses[0] = _Response({"resources_session_id": "resources-session", "sandbox_access": resources_access})

    result = await server.run_request(_request())

    # Two owners of one attempt's sandbox is a configuration error, so retrying the attempt would not help.
    assert result.result is None
    assert result.failure.stage == "seed"
    assert result.failure.terminal is True
    assert "only borrows" in result.failure.failure_reason
    # No agent ran, and the Resources session, which stops its own sandbox, is still closed.
    assert [path for _, path, _ in client.calls] == ["/seed_session", "/close_session"]
    # No provider was built, not even by the cleanup that stops an attempt sandbox.
    assert server._owned_sandboxes == {}


async def test_a_sandbox_that_cannot_start_fails_the_attempt_for_a_retry_before_the_agent_runs() -> None:
    provider = _Provider()
    provider.fail_create = SandboxCreateError("provider unreachable")
    server, client = _lending_server(provider)

    result = await server.run_request(_request())

    assert result.result is None
    assert result.failure is not None
    # An infrastructure failure: retried on a fresh attempt, never graded.
    assert result.failure.terminal is False
    assert result.failure.stage == "seed"
    assert "provider unreachable" in result.failure.failure_reason
    paths = [path for _, path, _ in client.calls]
    assert "/v1/agent_sessions" not in paths
    assert "/verify" not in paths
    # The Resources session it seeded is still closed.
    assert paths == ["/seed_session", "/close_session"]


async def test_a_failed_agent_turn_still_stops_the_sandbox() -> None:
    provider = _Provider()
    server, client = _lending_server(provider)
    client.responses[2] = ConnectionResetError("agent went away")

    result = await server.run_request(_request())

    assert result.failure is not None
    assert list(provider.boxes) == ["sb-1"]
    assert provider.running() == []


async def test_a_cancelled_attempt_stops_its_sandbox_after_closing_the_agent() -> None:
    provider = _Provider()
    server, client = _lending_server(provider, agent_delay_s=0.2)
    stops: list[str] = []
    close = provider.close

    async def recording_close(handle: SandboxHandle) -> None:
        stops.append([path for _, path, _ in client.calls][-1])
        await close(handle)

    provider.close = recording_close

    episode = asyncio.create_task(server.run_request(_request()))
    while "/ng-rollout/rollout-a2/v1/responses" not in [path for _, path, _ in client.calls]:
        await asyncio.sleep(0.01)
    episode.cancel()
    with pytest.raises(asyncio.CancelledError):
        await episode

    assert provider.running() == []
    # The agent session was closed before its sandbox stopped.
    assert stops == ["/v1/agent_sessions/close"]


def test_the_attempt_sandbox_needs_a_network_rule_and_a_lifetime() -> None:
    with pytest.raises(ValidationError, match="network rule"):
        AttemptSandboxConfig.model_validate(_attempt_sandbox_config(network=None, model_server=None))
    with pytest.raises(ValidationError, match="ttl_s"):
        AttemptSandboxConfig.model_validate(_attempt_sandbox_config(ttl_s=None))


def test_the_provider_and_the_sandbox_settings_come_together() -> None:
    base, _ = _environment_server()
    with pytest.raises(ValidationError, match="together"):
        SingleAgentTurnEnvironmentServerConfig.model_validate(
            base.config.model_dump() | {"sandbox_provider": "attempt_sandboxes"}
        )
    with pytest.raises(ValidationError, match="together"):
        SingleAgentTurnEnvironmentServerConfig.model_validate(
            base.config.model_dump() | {"sandbox_config": _attempt_sandbox_config()}
        )


@pytest.mark.parametrize(
    ("provider_block", "network", "refusal"),
    [
        # Docker cannot restrict a container's egress to a list of hosts.
        ({"docker": {}}, "model_only", "cannot apply network 'model_only'"),
        # The local provider runs commands on this host, and no other process can connect to its sandboxes.
        ({"local": {}}, "inherit", "cannot lend a sandbox"),
    ],
)
def test_startup_refuses_a_provider_that_cannot_lend_the_attempt_sandbox(
    provider_block: dict[str, Any], network: str, refusal: str
) -> None:
    server, client = _lending_server(_Provider())
    config = SingleAgentTurnEnvironmentServerConfig.model_validate(
        server.config.model_dump()
        | {"sandbox_config": _attempt_sandbox_config(network=network, model_server=MODEL_SERVER[network])}
    )
    server = SingleAgentTurnEnvironmentServer(config=config, server_client=client)
    client.global_config_dict = OmegaConf.merge(client.global_config_dict, {"attempt_sandboxes": provider_block})

    with pytest.raises(ValueError, match=refusal):
        server.check_sandbox_provider()


def test_a_server_without_a_sandbox_provider_checks_nothing_at_startup() -> None:
    server, _ = _environment_server()

    server.check_sandbox_provider()

    assert server._owned_sandboxes == {}


async def test_a_sandbox_that_cannot_be_lent_to_the_verifier_fails_the_attempt_for_a_retry() -> None:
    provider = _Provider()
    provider.fail_serialize_at = 2
    server, client = _lending_server(provider)

    result = await server.run_request(_request())

    assert result.result is None
    assert result.failure.terminal is False
    assert result.failure.stage == "verification"
    assert "/verify" not in [path for _, path, _ in client.calls]
    assert provider.running() == []


def test_no_episode_result_type_declares_a_sandbox_field() -> None:
    """The lent sandbox reaches /verify only on the wire: no verify response or episode result type declares it, so
    no stored result can carry a sandbox descriptor."""
    carried: list[str] = []

    def walk(model: type[BaseModel], prefix: str, seen: set[type]) -> None:
        if model in seen:
            return
        seen.add(model)
        for name, field in model.model_fields.items():
            if name in {"sandbox_access", "descriptor", "connection", "provider_config_ref"} or field.alias == (
                NG_SANDBOX_ACCESS_KEY
            ):
                carried.append(f"{prefix}.{name}")
            stack = [field.annotation]
            while stack:
                current = stack.pop()
                if isinstance(current, type) and issubclass(current, BaseModel):
                    walk(current, f"{prefix}.{name}", seen)
                stack.extend(typing.get_args(current))

    walk(SingleAgentTurnResponse, "SingleAgentTurnResponse", set())
    walk(SingleAgentTurnResult, "SingleAgentTurnResult", set())

    assert carried == []


@pytest.mark.parametrize(
    ("grants", "network", "tool_servers", "refused"),
    [
        ({"resources_tool_transports": ["direct_http"]}, "model_only", [], True),
        ({"resources_tool_transports": ["mcp"]}, "model_and_tools", ["resources"], False),
        ({}, "model_only", [], False),
    ],
)
def test_startup_refuses_a_rule_that_blocks_the_tools_an_episode_grants(
    grants: dict[str, Any], network: str, tool_servers: list[str], refused: bool
) -> None:
    sandbox_config = _attempt_sandbox_config(network=network, tool_servers=tool_servers)
    server, client = _lending_server(_Provider(), sandbox_config=sandbox_config, **grants)
    # An OpenSandbox deployment that enforces egress allowlists, so it can apply every restricted rule.
    opensandbox = {"opensandbox": {"networking": {"enforces_network_policy": True}}}
    client.global_config_dict = OmegaConf.merge(client.global_config_dict, {"attempt_sandboxes": opensandbox})

    if refused:
        with pytest.raises(ValueError, match=r"does not allow \['resources'\]"):
            server.check_sandbox_provider()
    else:
        server.check_sandbox_provider()


def test_the_attempt_sandbox_must_outlive_the_episode_and_its_cleanup() -> None:
    base, _ = _environment_server()
    with pytest.raises(ValidationError, match="must exceed"):
        SingleAgentTurnEnvironmentServerConfig.model_validate(
            base.config.model_dump()
            | {
                "sandbox_provider": "attempt_sandboxes",
                "sandbox_config": _attempt_sandbox_config(ttl_s=3600),
                "default_episode_timeout_seconds": 3600,
            }
        )
