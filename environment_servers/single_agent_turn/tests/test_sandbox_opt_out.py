# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When the attempt's sandbox group starts and who borrows it: the derived start, the opt-out, and what Resources
declares."""

from typing import Any

import pytest

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServer
from environment_servers.single_agent_turn.tests.test_attempt_sandbox import MODEL_HOST, _lending_server
from environment_servers.single_agent_turn.tests.test_attempt_sandbox import _Provider as _ModelOnlyProvider
from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxAccess, SandboxCapabilities, SandboxCreateError
from nemo_gym.server_runtime import NG_SANDBOX_ACCESS_KEY, PREPARE_SANDBOX_PATH, ResourcesPrepareSandboxRequest
from tests.helpers.single_agent_turn import _environment_server, _request, _Response


AGENT_PATH = "/ng-rollout/rollout-a2/v1/responses"
# What Simple Agent declares on GET /v1/agent_sessions/capabilities: its loop runs in its own process when lent
# nothing.
IN_PROCESS_AGENT = {"runs_loop_without_sandbox": True}


class _Provider(_ModelOnlyProvider):
    """The attempt sandbox provider, which can also cut all egress of a group that runs the verifier alone."""

    capabilities = SandboxCapabilities(network_rules=frozenset({"inherit", "model_only", "no_network"}))


def _server(provider: _Provider | None, *, agent: dict[str, Any] | None = None, **config: Any) -> tuple[Any, Any]:
    """A lending server whose bound agent declares ``agent`` as its capabilities."""
    server, client = _lending_server(provider, **config)
    client.capabilities = agent or {}
    return server, client


def _recording(client: Any, provider: _Provider) -> list[tuple[str, list[str]]]:
    """Record, at each call the server makes, the sandboxes then running."""
    seen: list[tuple[str, list[str]]] = []
    post = type(client).post

    async def recording_post(self: Any, server_name: str, url_path: str, **kwargs: Any) -> Any:
        seen.append((url_path, provider.running()))
        return await post(self, server_name, url_path, **kwargs)

    object.__setattr__(client, "post", recording_post.__get__(client))
    return seen


def _seed_reply(**fields: Any) -> _Response:
    return _Response({"resources_session_id": "resources-session"} | fields)


def _verify_body(client: Any) -> dict[str, Any]:
    return next(kwargs["json"] for _, path, kwargs in client.calls if path == "/verify")


def _paths(client: Any) -> list[str]:
    return [path for _, path, _ in client.calls]


def _with_tools(request: Any) -> Any:
    request.task.task_input.responses_create_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": "task", "tools": [{"type": "function", "name": "run_python", "parameters": {}, "strict": True}]}
    )
    return request


async def test_a_task_without_tools_and_an_in_process_agent_starts_the_group_late_for_the_verifier_alone() -> None:
    provider = _Provider()
    server, client = _server(provider, agent=IN_PROCESS_AGENT)
    seen = _recording(client, provider)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert result.result.reward == 1.0
    # The agent was lent nothing, so its loop ran in the Agent Server process, and no group was held through its turn.
    assert AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"]).sandbox_access is None
    assert dict(seen)[AGENT_PATH] == []
    assert dict(seen)["/v1/agent_sessions/close"] == []
    # The verifier's group started after the agent closed, was lent to /verify, and stopped once it replied.
    assert dict(seen)["/verify"] == ["sb-1"]
    lent = SandboxAccess.model_validate(_verify_body(client)[NG_SANDBOX_ACCESS_KEY])
    assert (lent.connection.descriptor["sandbox_id"], lent.workdir) == ("sb-1", "/workspace")
    assert seen[-1] == ("/close_session", [])
    # The verifier runs the model's output there, so its group reaches nothing, not even the model server.
    assert provider.rules == [("no_network", [])]
    # The agent's capabilities are asked once per process.
    assert client.capability_requests == 1
    await server._agent_capabilities()
    assert client.capability_requests == 1


@pytest.mark.parametrize("agent", [IN_PROCESS_AGENT, {}], ids=["in-process agent", "agent that needs a container"])
async def test_a_task_with_tools_starts_the_group_at_seed_and_the_verifier_borrows_it_too(
    agent: dict[str, Any],
) -> None:
    provider = _Provider()
    server, client = _server(provider, agent=agent)
    seen = _recording(client, provider)

    result = await server.run_request(_with_tools(_request()))

    assert result.failure is None, result.failure
    seed = AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"])
    assert seed.sandbox_access.connection.descriptor["sandbox_id"] == "sb-1"
    assert dict(seen)[AGENT_PATH] == ["sb-1"]
    # The same group, with the agent's model-only rule, is lent to the verifier.
    lent = SandboxAccess.model_validate(_verify_body(client)[NG_SANDBOX_ACCESS_KEY])
    assert lent.connection.descriptor == seed.sandbox_access.connection.descriptor
    assert provider.rules == [("model_only", [MODEL_HOST])]
    assert list(provider.boxes) == ["sb-1"]
    assert provider.running() == []
    # A task with tools never asks the agent.
    assert client.capability_requests == 0


async def test_a_task_without_tools_and_an_agent_that_needs_a_container_starts_the_group_at_seed() -> None:
    provider = _Provider()
    server, client = _server(provider, agent={"runs_loop_without_sandbox": False})

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    seed = AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"])
    assert seed.sandbox_access.connection.descriptor["sandbox_id"] == "sb-1"
    assert SandboxAccess.model_validate(_verify_body(client)[NG_SANDBOX_ACCESS_KEY]).connection.descriptor == (
        seed.sandbox_access.connection.descriptor
    )
    assert provider.rules == [("model_only", [MODEL_HOST])]


async def test_an_opted_out_pure_qa_task_starts_no_group_at_all() -> None:
    server, client = _server(None, agent=IN_PROCESS_AGENT, sandbox_opt_out=True)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"]).sandbox_access is None
    assert NG_SANDBOX_ACCESS_KEY not in _verify_body(client)
    # No provider was built, not even by the cleanup that stops an attempt sandbox, and the agent was not asked.
    assert server._owned_sandboxes == {}
    assert client.capability_requests == 0


async def test_an_opt_out_for_a_task_that_declares_tools_is_refused_and_no_agent_runs() -> None:
    server, client = _server(_Provider(), agent=IN_PROCESS_AGENT, sandbox_opt_out=True)

    result = await server.run_request(_with_tools(_request()))

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "pure QA task" in result.failure.failure_reason
    assert "run_python" in result.failure.failure_reason
    assert _paths(client) == ["/seed_session", "/close_session"]


async def test_an_opt_out_for_a_group_with_sidecars_is_refused() -> None:
    server, client = _server(_Provider(), agent=IN_PROCESS_AGENT, sandbox_opt_out=True)
    client.responses[0] = _seed_reply(sandbox_group={"sidecars": [{"name": "db", "image": "db:1"}]})

    result = await server.run_request(_request())

    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "sidecar db" in result.failure.failure_reason
    assert _paths(client) == ["/seed_session", "/close_session"]


async def test_the_verifier_of_a_task_without_tools_verifies_in_the_group_its_resources_server_declared() -> None:
    provider = _Provider()
    server, client = _server(provider, agent=IN_PROCESS_AGENT)
    client.responses[0] = _seed_reply(
        sandbox_group={"image": "verifier:1", "files": {"tests/test_task.py": "assert True"}}
    )

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    spec = provider.boxes["sb-1"]["spec"]
    assert (spec.image, spec.files) == ("verifier:1", {"tests/test_task.py": "assert True"})
    assert NG_SANDBOX_ACCESS_KEY in _verify_body(client)
    assert provider.running() == []


async def test_a_late_group_that_cannot_start_fails_the_attempt_for_a_retry_with_the_agents_evidence() -> None:
    provider = _Provider()
    provider.fail_create = SandboxCreateError("provider unreachable")
    server, client = _server(provider, agent=IN_PROCESS_AGENT)

    result = await server.run_request(_request())

    # An infrastructure failure: masked and retried, never a reward of 0, and never verified on this host.
    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("verification", False)
    assert "provider unreachable" in result.failure.failure_reason
    assert "/verify" not in _paths(client)
    assert result.failure.partial_response is not None
    assert provider.running() == []


async def test_a_group_no_attempt_can_start_is_refused_at_seed_before_the_agents_turn() -> None:
    provider = _Provider()
    server, client = _server(provider, agent=IN_PROCESS_AGENT)
    client.responses[0] = _seed_reply(sandbox_group={"image": "verifier:1"})
    server.config.sandbox_config.workdir = None

    result = await server.run_request(_request())

    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "has no workdir" in result.failure.failure_reason
    assert _paths(client) == ["/seed_session", "/close_session"]
    assert provider.boxes == {}


@pytest.mark.parametrize("agent", [{}, IN_PROCESS_AGENT], ids=["group at seed", "group for the verifier alone"])
async def test_the_resources_server_prepares_the_group_before_anyone_borrows_it(agent: dict[str, Any]) -> None:
    provider = _Provider()
    server, client = _server(provider, agent=agent)
    client.responses[0] = _seed_reply(sandbox_group={"prepare": True, "env": {"TASK": "1"}})
    late = bool(agent)
    borrower = "/verify" if late else "/v1/agent_sessions"
    client.responses.insert(4 if late else 1, _Response({"resources_session_id": "resources-session"}))
    seen = _recording(client, provider)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    paths = [path for path, _ in seen]
    assert paths.index(PREPARE_SANDBOX_PATH) < paths.index(borrower)
    assert dict(seen)[PREPARE_SANDBOX_PATH] == ["sb-1"]
    _, _, kwargs = client.calls[paths.index(PREPARE_SANDBOX_PATH)]
    prepare = ResourcesPrepareSandboxRequest.model_validate(kwargs["json"])
    assert prepare.resources_session_id.startswith("resources-session-")
    assert prepare.task_data == {"instance_id": "task"}
    assert prepare.sandbox_access.connection.descriptor["sandbox_id"] == "sb-1"
    # The Resources session's cookie names the episode, as on every other Resources call: after the agent's close,
    # the one the close returned.
    assert kwargs["cookies"] == {"session": "updated-cookie" if late else "cookie-value"}
    assert provider.boxes["sb-1"]["spec"].env["TASK"] == "1"
    assert provider.running() == []


async def test_a_resources_server_that_declares_a_group_with_no_sandbox_provider_fails_the_attempt_for_good() -> None:
    server, client = _environment_server()
    client.responses[0] = _seed_reply(sandbox_group={"image": "verifier:1"})

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "sets no sandbox_provider" in result.failure.failure_reason
    assert _paths(client) == ["/seed_session", "/close_session"]


def test_a_server_that_opts_out_needs_no_provider_block_at_startup() -> None:
    # attempt_sandboxes is not defined: a large QA run that opts out of a config shipping a provider still starts.
    server, _ = _server(None, sandbox_opt_out=True)

    server.check_sandbox_provider()

    assert server._owned_sandboxes == {}


def test_startup_refuses_an_opt_out_for_an_episode_that_grants_tools() -> None:
    server, client = _environment_server(resources_tool_transports=["mcp"])
    server = SingleAgentTurnEnvironmentServer(
        config=server.config.model_copy(update={"sandbox_opt_out": True}), server_client=client
    )

    with pytest.raises(ValueError, match=r"grants the tools of \['resources'\]; drop resources_tool_transports"):
        server.check_sandbox_provider()


@pytest.mark.parametrize("opt_out", [False, True])
async def test_without_a_sandbox_provider_nothing_changes(opt_out: bool) -> None:
    server, client = _environment_server()
    server = SingleAgentTurnEnvironmentServer(
        config=server.config.model_copy(update={"sandbox_opt_out": opt_out}), server_client=client
    )

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert result.result.reward == 1.0
    assert AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"]).sandbox_access is None
    assert NG_SANDBOX_ACCESS_KEY not in _verify_body(client)
    assert server._owned_sandboxes == {}
    assert client.capability_requests == 0
