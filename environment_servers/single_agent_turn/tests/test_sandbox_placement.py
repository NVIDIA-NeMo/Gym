# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the agent and the verifier of an attempt run: the agent's side derived from the task's tools and the agent's
capabilities, the verifier's side declared by the Resources Server (``verifier_mode``)."""

from typing import Any

import pytest
from aiohttp import ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

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
    """The attempt sandbox provider, which can also cut all egress of a separate verifier's group."""

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


def _agent_seed(client: Any) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"])


def _paths(client: Any) -> list[str]:
    return [path for _, path, _ in client.calls]


def _http_error(status: int) -> ClientResponseError:
    url = URL("http://agent:8001/v1/agent_sessions/capabilities")
    info = RequestInfo(url=url, method="GET", headers=CIMultiDictProxy(CIMultiDict()), real_url=url)
    return ClientResponseError(info, (), status=status)


def _with_tools(request: Any) -> Any:
    request.task.task_input.responses_create_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": "task", "tools": [{"type": "function", "name": "run_python", "parameters": {}, "strict": True}]}
    )
    return request


async def test_a_separate_verifier_of_a_task_an_in_process_agent_runs_gets_a_late_group_with_no_egress() -> None:
    provider = _Provider()
    server, client = _server(provider, agent=IN_PROCESS_AGENT)
    client.responses[0] = _seed_reply(verifier_mode="separate")
    seen = _recording(client, provider)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert result.result.reward == 1.0
    # The agent was lent nothing, so its loop ran in the Agent Server process, and no group was held through its turn.
    assert _agent_seed(client).sandbox_access is None
    assert dict(seen)[AGENT_PATH] == []
    # The verifier's group started after the agent closed, was lent to /verify, and stopped once it replied.
    assert dict(seen)["/verify"] == ["sb-1"]
    lent = SandboxAccess.model_validate(_verify_body(client)[NG_SANDBOX_ACCESS_KEY])
    assert (lent.connection.descriptor["sandbox_id"], lent.workdir) == ("sb-1", "/workspace")
    assert seen[-1] == ("/close_session", [])
    # The verifier runs the model's output there, so its group reaches nothing, not even the model server.
    assert provider.rules == [("no_network", [])]


async def test_a_separate_verifier_after_an_agent_group_gets_a_fresh_group_once_the_agents_has_stopped() -> None:
    provider = _Provider()
    server, client = _server(provider)
    client.responses[0] = _seed_reply(verifier_mode="separate")
    seen = _recording(client, provider)

    result = await server.run_request(_with_tools(_request()))

    assert result.failure is None, result.failure
    assert _agent_seed(client).sandbox_access.connection.descriptor["sandbox_id"] == "sb-1"
    assert dict(seen)[AGENT_PATH] == ["sb-1"]
    # The agent's group is gone by /verify, which borrows a second group of its own with no egress.
    assert dict(seen)["/verify"] == ["sb-2"]
    assert SandboxAccess.model_validate(_verify_body(client)[NG_SANDBOX_ACCESS_KEY]).connection.descriptor[
        "sandbox_id"
    ] == ("sb-2")
    assert provider.rules == [("model_only", [MODEL_HOST]), ("no_network", [])]
    assert provider.running() == []


async def test_a_shared_verifier_borrows_the_group_the_agent_borrowed() -> None:
    provider = _Provider()
    server, client = _server(provider)
    client.responses[0] = _seed_reply(verifier_mode="shared")

    result = await server.run_request(_with_tools(_request()))

    assert result.failure is None, result.failure
    seed = _agent_seed(client)
    lent = SandboxAccess.model_validate(_verify_body(client)[NG_SANDBOX_ACCESS_KEY])
    assert lent.connection.descriptor == seed.sandbox_access.connection.descriptor
    assert list(provider.boxes) == ["sb-1"]
    assert provider.running() == []


async def test_a_shared_verifier_is_refused_when_the_agent_borrows_no_group_and_no_agent_runs() -> None:
    provider = _Provider()
    server, client = _server(provider, agent=IN_PROCESS_AGENT)
    client.responses[0] = _seed_reply(verifier_mode="shared")

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "declares verifier_mode 'shared', but the agent borrows no group to share" in result.failure.failure_reason
    assert _paths(client) == ["/seed_session", "/close_session"]
    assert provider.boxes == {}


async def test_a_verifier_that_runs_in_the_resources_server_is_lent_nothing_and_the_agents_group_stops_first() -> None:
    provider = _Provider()
    server, client = _server(provider)
    seen = _recording(client, provider)

    result = await server.run_request(_with_tools(_request()))

    assert result.failure is None, result.failure
    assert dict(seen)[AGENT_PATH] == ["sb-1"]
    # Undeclared means none: /verify runs in the Resources Server, after the agent's group has stopped.
    assert dict(seen)["/verify"] == []
    assert NG_SANDBOX_ACCESS_KEY not in _verify_body(client)


async def test_a_task_without_tools_an_in_process_agent_and_a_verifier_in_resources_start_no_group() -> None:
    server, client = _server(_Provider(), agent=IN_PROCESS_AGENT)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert _agent_seed(client).sandbox_access is None
    assert NG_SANDBOX_ACCESS_KEY not in _verify_body(client)
    assert server.sandboxes.keys == []


@pytest.mark.parametrize(
    "agent, error",
    [
        ({"runs_loop_without_sandbox": False}, None),
        ({}, None),
        ({}, _http_error(404)),
    ],
    ids=["declares it needs a container", "undeclared", "agent server without the route"],
)
async def test_an_agent_that_does_not_say_it_runs_without_a_sandbox_borrows_a_group_at_seed(
    agent: dict[str, Any], error: Exception | None
) -> None:
    provider = _Provider()
    server, client = _server(provider, agent=agent)
    client.capabilities_error = error

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert _agent_seed(client).sandbox_access.connection.descriptor["sandbox_id"] == "sb-1"
    assert provider.rules == [("model_only", [MODEL_HOST])]
    # The capabilities are asked once per process.
    assert client.capability_requests == 1
    await server._agent_capabilities()
    assert client.capability_requests == 1


async def test_a_capabilities_request_that_fails_otherwise_fails_the_attempt_for_a_retry() -> None:
    server, client = _server(_Provider())
    client.capabilities_error = _http_error(503)

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("seed", False)


async def test_a_task_with_tools_never_asks_the_agent() -> None:
    server, client = _server(_Provider(), agent=IN_PROCESS_AGENT)

    result = await server.run_request(_with_tools(_request()))

    assert result.failure is None, result.failure
    assert _agent_seed(client).sandbox_access is not None
    assert client.capability_requests == 0


@pytest.mark.parametrize(
    "seed_fields, reason",
    [
        ({"verifier_mode": "separate"}, "declares verifier_mode 'separate' at seed, but environment sets no"),
        ({"sandbox_group": {"image": "verifier:1"}}, "declares a sandbox group at seed, but environment sets no"),
    ],
)
async def test_a_declaration_that_needs_a_group_without_a_sandbox_provider_fails_the_attempt_for_good(
    seed_fields: dict[str, Any], reason: str
) -> None:
    server, client = _environment_server()
    client.responses[0] = _seed_reply(**seed_fields)

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert reason in result.failure.failure_reason
    assert _paths(client) == ["/seed_session", "/close_session"]


async def test_without_a_sandbox_provider_an_undeclared_attempt_runs_as_before() -> None:
    server, client = _environment_server()

    result = await server.run_request(_with_tools(_request()))

    assert result.failure is None, result.failure
    assert result.result.reward == 1.0
    assert _agent_seed(client).sandbox_access is None
    assert NG_SANDBOX_ACCESS_KEY not in _verify_body(client)
    assert server._owned_sandboxes == {}
    assert client.capability_requests == 0


async def test_the_separate_verifier_verifies_in_the_group_its_resources_server_declared() -> None:
    provider = _Provider()
    server, client = _server(provider, agent=IN_PROCESS_AGENT)
    client.responses[0] = _seed_reply(
        verifier_mode="separate", sandbox_group={"image": "verifier:1", "files": {"tests/test_task.py": "assert True"}}
    )

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    spec = provider.boxes["sb-1"]["spec"]
    assert (spec.image, spec.files) == ("verifier:1", {"tests/test_task.py": "assert True"})
    assert NG_SANDBOX_ACCESS_KEY in _verify_body(client)
    assert provider.running() == []


async def test_a_separate_group_that_cannot_start_fails_the_attempt_for_a_retry_with_the_agents_evidence() -> None:
    provider = _Provider()
    provider.fail_create = SandboxCreateError("provider unreachable")
    server, client = _server(provider, agent=IN_PROCESS_AGENT)
    client.responses[0] = _seed_reply(verifier_mode="separate")

    result = await server.run_request(_request())

    # An infrastructure failure: masked and retried, never a reward of 0, and never verified on this host.
    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("verification", False)
    assert "provider unreachable" in result.failure.failure_reason
    assert "/verify" not in _paths(client)
    assert result.failure.partial_response is not None
    assert provider.running() == []


async def test_a_separate_verifier_on_a_provider_that_cannot_cut_egress_is_refused_before_the_agents_turn() -> None:
    provider = _ModelOnlyProvider()
    server, client = _server(provider, agent=IN_PROCESS_AGENT)
    client.responses[0] = _seed_reply(verifier_mode="separate")

    result = await server.run_request(_request())

    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "cannot apply network 'no_network'" in result.failure.failure_reason
    assert _paths(client) == ["/seed_session", "/close_session"]
    assert provider.boxes == {}


@pytest.mark.parametrize(
    "agent, verifier_mode", [({}, "shared"), (IN_PROCESS_AGENT, "separate")], ids=["agent's group", "verifier's group"]
)
async def test_the_resources_server_prepares_the_first_group_before_anyone_borrows_it(
    agent: dict[str, Any], verifier_mode: str
) -> None:
    provider = _Provider()
    server, client = _server(provider, agent=agent)
    client.responses[0] = _seed_reply(
        verifier_mode=verifier_mode, sandbox_group={"prepare": True, "env": {"TASK": "1"}}
    )
    late = verifier_mode == "separate"
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


async def test_a_group_no_attempt_can_start_is_refused_at_seed_before_the_agents_turn() -> None:
    provider = _Provider()
    server, client = _server(provider, agent=IN_PROCESS_AGENT)
    client.responses[0] = _seed_reply(verifier_mode="separate", sandbox_group={"image": "verifier:1"})
    server.config.sandbox_config.workdir = None

    result = await server.run_request(_request())

    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "has no workdir" in result.failure.failure_reason
    assert _paths(client) == ["/seed_session", "/close_session"]
    assert provider.boxes == {}
