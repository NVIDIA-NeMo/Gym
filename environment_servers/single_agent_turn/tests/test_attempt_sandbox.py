# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandbox the Environment Server starts for an attempt when the Resources Server lends none."""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from environment_servers.single_agent_turn.app import (
    VERIFY_SANDBOX_ACCESS_FIELD,
    AttemptSandboxConfig,
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
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
from nemo_gym.server_runtime import OwnedSandboxes
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
        return {"sandbox_id": handle.sandbox_id, "scope": scope}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        return SandboxHandle(sandbox_id=str(descriptor["sandbox_id"]), provider_name=self.name, raw=None)

    def running(self) -> list[str]:
        return [sandbox_id for sandbox_id, box in self.boxes.items() if box["state"] == "running"]


def _attempt_sandbox_config(**overrides: Any) -> dict[str, Any]:
    return {
        "image": "python:3.13-slim",
        "workdir": "/workspace",
        "ttl_s": 3600,
        "network": "model_only",
        "model_server": "policy_model",
    } | overrides


def _lending_server(
    provider: _Provider, *, agent_delay_s: float = 0.0
) -> tuple[SingleAgentTurnEnvironmentServer, Any]:
    base, client = _environment_server()
    config = SingleAgentTurnEnvironmentServerConfig.model_validate(
        base.config.model_dump()
        | {"sandbox_provider": "attempt_sandboxes", "sandbox_config": _attempt_sandbox_config()}
    )
    client.global_config_dict = OmegaConf.merge(
        client.global_config_dict,
        {"policy_model": {"responses_api_models": {"vllm_model": {"host": MODEL_HOST, "port": 8000}}}},
    )
    if agent_delay_s:
        client.slow_path = "/ng-rollout/rollout-a2/v1/responses"
    server = SingleAgentTurnEnvironmentServer(config=config, server_client=client)
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

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert result.result.reward == 1.0
    # One sandbox, started with the rule that allows only the model server's host.
    assert list(provider.boxes) == ["sb-1"]
    assert provider.rules == [("model_only", [MODEL_HOST])]
    spec = provider.boxes["sb-1"]["spec"]
    assert (spec.image, spec.workdir, spec.ttl_s) == ("python:3.13-slim", "/workspace", 3600)
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
    lent = SandboxAccess.model_validate(verify_body[VERIFY_SANDBOX_ACCESS_FIELD])
    assert lent.connection.descriptor == seed.sandbox_access.connection.descriptor
    # The lent sandbox does not reach the stored result.
    assert VERIFY_SANDBOX_ACCESS_FIELD not in result.result.model_dump()
    # It is stopped once the episode ends.
    assert provider.running() == []
    assert server.sandboxes.keys == []


async def test_a_sandbox_the_resources_seed_lends_is_lent_on_and_none_is_started() -> None:
    provider = _Provider()
    server, client = _lending_server(provider)
    resources_access = {
        "connection": {"kind": "direct", "provider_config_ref": "task_sandboxes", "descriptor": {"id": "task"}},
        "workdir": "/repo",
    }
    client.responses[0] = _Response({"resources_session_id": "resources-session", "sandbox_access": resources_access})

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert provider.boxes == {}
    seed = AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"])
    assert seed.sandbox_access.connection.provider_config_ref == "task_sandboxes"
    verify_body = next(kwargs["json"] for _, path, kwargs in client.calls if path == "/verify")
    assert VERIFY_SANDBOX_ACCESS_FIELD not in verify_body


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
