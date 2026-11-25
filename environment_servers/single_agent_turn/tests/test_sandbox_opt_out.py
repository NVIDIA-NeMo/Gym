# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Who borrows the attempt's sandbox group: the sandbox opt-outs, a grader's own sandbox, and what Resources declares."""

import json
from typing import Any

import pytest

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServer
from environment_servers.single_agent_turn.tests.test_attempt_sandbox import _lending_server, _Provider
from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxAccess, SandboxCreateError
from nemo_gym.server_runtime import NG_SANDBOX_ACCESS_KEY, PREPARE_SANDBOX_PATH, ResourcesPrepareSandboxRequest
from tests.helpers.single_agent_turn import _environment_server, _request, _Response


def _recording(server: SingleAgentTurnEnvironmentServer, client: Any, provider: _Provider) -> list[tuple[str, list]]:
    """Record, at each call the server makes, the sandboxes then running."""
    seen: list[tuple[str, list]] = []
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


async def test_a_pure_qa_task_runs_its_turn_lent_nothing_and_its_grader_gets_a_sandbox_of_its_own() -> None:
    provider = _Provider()
    server, client = _lending_server(provider, sandbox_opt_out={"task": True})
    seen = _recording(server, client, provider)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert result.result.reward == 1.0
    # The agent was lent no sandbox, so its loop runs in the Agent Server process, and none ran during its turn.
    seed = AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"])
    assert seed.sandbox_access is None
    assert dict(seen)["/ng-rollout/rollout-a2/v1/responses"] == []
    assert dict(seen)["/v1/agent_sessions/close"] == []
    # The grader's sandbox started after the agent closed and was lent to /verify alone.
    assert dict(seen)["/verify"] == ["sb-1"]
    lent = SandboxAccess.model_validate(_verify_body(client)[NG_SANDBOX_ACCESS_KEY])
    assert lent.connection.descriptor["sandbox_id"] == "sb-1"
    assert lent.workdir == "/workspace"
    assert provider.running() == []


async def test_a_grader_that_executes_nothing_gets_no_sandbox_and_the_agents_stops_before_verify() -> None:
    provider = _Provider()
    server, client = _lending_server(provider, sandbox_opt_out={"grader": True})
    seen = _recording(server, client, provider)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"]).sandbox_access is not None
    assert dict(seen)["/ng-rollout/rollout-a2/v1/responses"] == ["sb-1"]
    # Nothing borrows the group after the agent closed, so it is stopped before the Resources Server grades.
    assert dict(seen)["/verify"] == []
    assert NG_SANDBOX_ACCESS_KEY not in _verify_body(client)
    assert list(provider.boxes) == ["sb-1"]


async def test_a_task_and_a_grader_that_both_opt_out_start_no_sandbox() -> None:
    server, client = _lending_server(None, sandbox_opt_out={"task": True, "grader": True})

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"]).sandbox_access is None
    assert NG_SANDBOX_ACCESS_KEY not in _verify_body(client)
    # No provider was built, not even by the cleanup that stops an attempt sandbox.
    assert server._owned_sandboxes == {}


async def test_a_task_that_declares_tools_cannot_opt_out_and_no_agent_runs() -> None:
    server, client = _lending_server(_Provider(), sandbox_opt_out={"task": True})
    request = _request()
    request.task.task_input.responses_create_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": "task", "tools": [{"type": "function", "name": "run_python", "parameters": {}, "strict": True}]}
    )

    result = await server.run_request(request)

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "pure QA task" in result.failure.failure_reason
    assert "run_python" in result.failure.failure_reason
    assert [path for _, path, _ in client.calls] == ["/seed_session", "/close_session"]


async def test_a_grader_that_runs_code_cannot_opt_out_and_no_agent_runs() -> None:
    provider = _Provider()
    server, client = _lending_server(provider, sandbox_opt_out={"grader": True})
    client.responses[0] = _seed_reply(sandbox_group={"grader_runs_code": True})

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "declares that its grader runs code" in result.failure.failure_reason
    assert [path for _, path, _ in client.calls] == ["/seed_session", "/close_session"]
    assert provider.boxes == {}


async def test_a_grader_that_runs_code_of_a_pure_qa_task_grades_in_the_group_it_declared() -> None:
    provider = _Provider()
    server, client = _lending_server(provider, sandbox_opt_out={"task": True})
    client.responses[0] = _seed_reply(
        sandbox_group={"grader_runs_code": True, "image": "grader:1", "files": {"tests/test_task.py": "assert True"}}
    )

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    spec = provider.boxes["sb-1"]["spec"]
    assert (spec.image, spec.files) == ("grader:1", {"tests/test_task.py": "assert True"})
    assert NG_SANDBOX_ACCESS_KEY in _verify_body(client)
    assert provider.running() == []


async def test_a_grader_sandbox_that_cannot_start_fails_the_attempt_for_a_retry_with_the_agents_evidence() -> None:
    provider = _Provider()
    provider.fail_create = SandboxCreateError("provider unreachable")
    server, client = _lending_server(provider, sandbox_opt_out={"task": True})

    result = await server.run_request(_request())

    # An infrastructure failure: masked and retried, never a reward of 0, and never graded on this host.
    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("verification", False)
    assert "provider unreachable" in result.failure.failure_reason
    assert "/verify" not in [path for _, path, _ in client.calls]
    assert result.failure.partial_response is not None
    assert provider.running() == []


async def test_the_resources_server_prepares_the_group_before_the_agent_borrows_it() -> None:
    provider = _Provider()
    server, client = _lending_server(provider)
    client.responses[0] = _seed_reply(sandbox_group={"prepare": True, "env": {"TASK": "1"}})
    client.responses.insert(1, _Response({"resources_session_id": "resources-session"}))
    seen = _recording(server, client, provider)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    paths = [path for path, _ in seen]
    assert paths.index(PREPARE_SANDBOX_PATH) < paths.index("/v1/agent_sessions")
    assert dict(seen)[PREPARE_SANDBOX_PATH] == ["sb-1"]
    _, _, kwargs = client.calls[paths.index(PREPARE_SANDBOX_PATH)]
    prepare = ResourcesPrepareSandboxRequest.model_validate(kwargs["json"])
    assert prepare.resources_session_id.startswith("resources-session-")
    assert prepare.sandbox_access.connection.descriptor["sandbox_id"] == "sb-1"
    # The Resources session's cookie names the episode, as on every other Resources call.
    assert kwargs["cookies"] == {"session": "cookie-value"}
    assert provider.boxes["sb-1"]["spec"].env["TASK"] == "1"
    assert provider.running() == []


async def test_a_resources_server_that_declares_a_group_with_no_sandbox_provider_fails_the_attempt_for_good() -> None:
    server, client = _environment_server()
    client.responses[0] = _seed_reply(sandbox_group={"grader_runs_code": True})

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("seed", True)
    assert "sets no sandbox_provider" in result.failure.failure_reason
    assert [path for _, path, _ in client.calls] == ["/seed_session", "/close_session"]


def test_startup_refuses_a_task_sandbox_opt_out_for_an_episode_that_grants_tools() -> None:
    server, client = _environment_server(resources_tool_transports=["mcp"])
    config = server.config.model_copy(
        update={"sandbox_opt_out": server.config.sandbox_opt_out.model_validate({"task": True})}
    )
    server = SingleAgentTurnEnvironmentServer(config=config, server_client=client)

    with pytest.raises(ValueError, match=r"pure QA task, but every episode grants the tools of \['resources'\]"):
        server.check_sandbox_provider()


@pytest.mark.parametrize("opt_out", [{}, {"task": True}, {"grader": True}, {"task": True, "grader": True}])
async def test_without_a_sandbox_provider_the_opt_outs_change_nothing(opt_out: dict[str, bool]) -> None:
    server, client = _environment_server()
    config = server.config.model_copy(
        update={"sandbox_opt_out": server.config.sandbox_opt_out.model_validate(opt_out)}
    )
    server = SingleAgentTurnEnvironmentServer(config=config, server_client=client)

    result = await server.run_request(_request())

    assert result.failure is None, result.failure
    assert AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"]).sandbox_access is None
    assert NG_SANDBOX_ACCESS_KEY not in _verify_body(client)
    assert json.loads(json.dumps(result.result.model_dump(mode="json")))["reward"] == 1.0
    assert server._owned_sandboxes == {}
