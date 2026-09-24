# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise mini-SWE without importing a benchmark or its resource models."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.miniswe_sandboxed_agent import app as module
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessOutcome
from responses_api_agents.miniswe_sandboxed_agent.models import MiniSWERunRequest, SeedSessionResponse


@pytest.fixture
async def fixture(tmp_path, monkeypatch):
    agent = module.MiniSWESandboxedAgent(
        config=module.MiniSWESandboxedConfig(
            host="localhost",
            port=1,
            name="agent",
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "other_resources"},
            model_server={"type": "responses_api_models", "name": "model"},
            artifacts_dir=tmp_path,
            shutdown_timeout_sec=0.02,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    request = SimpleNamespace(session={SESSION_ID_KEY: "owner"}, cookies={"session": "incoming"})
    body = MiniSWERunRequest(responses_create_params={"input": []}, problem={"id": 42}, rollout_id="rollout")
    seed = dict(
        session_id="resource-session",
        task_id="problem-42",
        sandbox_descriptor={"sandbox_id": "borrowed"},
        sandbox_provider={"local": {}},
        instruction="Solve this other benchmark's problem",
        agent_timeout_sec=60,
    )
    provider = SimpleNamespace(aclose=AsyncMock())
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="/workspace\n")))
    monkeypatch.setattr(module, "create_provider", MagicMock(return_value=provider))
    monkeypatch.setattr(module.AsyncSandbox, "connect", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(module, "get_response_json", AsyncMock(side_effect=lambda r: r.value))
    harnesses = []

    def harness(**kwargs):
        async def execute(budget):
            assert 0 < budget <= 60
            response = await kwargs["query"](kwargs["params"].model_dump(mode="json"))
            return response, HarnessOutcome(reason="completed"), {"harness_version": "test"}

        instance = SimpleNamespace(
            **kwargs, setup=AsyncMock(), close=AsyncMock(), execute=AsyncMock(side_effect=execute)
        )
        harnesses.append(instance)
        return instance

    monkeypatch.setattr(module, "MiniSWEHarness", harness)
    verification = {}

    async def post(*, server_name, url_path, json, cookies, **kwargs):
        if url_path == "/seed_session":
            assert server_name == "other_resources"
            assert json["problem"] == {"id": 42}
            assert cookies == {"session": "incoming"}
            value = seed
        elif url_path == "/v1/responses":
            assert server_name == "model"
            assert cookies == {"session": "seeded"}
            assert kwargs["headers"]["x-session-id"] == "resource-session"
            value = module.empty_response(body.responses_create_params, "model").model_dump(mode="json")
        elif url_path == "/verify":
            assert server_name == "other_resources"
            assert cookies == {"session": "seeded"}
            verification.update(json)
            # A different benchmark need not return TB4's termination, session,
            # evaluation_completed, artifacts, or timing fields.
            value = {
                "responses_create_params": json["responses_create_params"],
                "response": json["response"],
                "reward": 0.25,
                "problem_score": {"passed": 1, "total": 4},
            }
        else:
            raise AssertionError(url_path)
        return SimpleNamespace(value=value, cookies={"session": "seeded"})

    agent.server_client.post = AsyncMock(side_effect=post)
    yield SimpleNamespace(
        agent=agent,
        request=request,
        body=body,
        seed=seed,
        provider=provider,
        sandbox=sandbox,
        harnesses=harnesses,
        verification=verification,
    )
    await agent.shutdown()


async def test_run_with_unrelated_resource_schema(fixture):
    f = fixture
    response = await f.agent.run(f.request, f.body)
    assert response.reward == 0.25
    assert response.model_dump()["problem_score"] == {"passed": 1, "total": 4}
    assert "evaluation_completed" not in response.model_dump()
    context = f.harnesses[0].context
    assert context.task_id == "problem-42" and context.instruction == f.seed["instruction"]
    assert context.workdir == "/workspace"
    assert f.verification["session_id"] == "resource-session"
    assert f.verification["termination"]["reason"] == "completed"
    assert f.verification["agent_started"]
    assert f.verification["harness_metadata"] == {"harness_version": "test"}
    f.provider.aclose.assert_awaited_once()
    assert f.body.responses_create_params.input == []


async def test_run_agent_borrowed_session_without_resource_calls(fixture):
    f = fixture
    f.seed.pop("task_id")  # Optional for other resources and old seed responses.
    result = await f.agent._run_agent(
        SeedSessionResponse.model_validate(f.seed),
        f.body.responses_create_params.model_copy(deep=True),
        client_session_id="owner",
        rollout_id="activation",
        capture_model_calls=False,
        cookies={"session": "seeded"},
    )
    assert result.termination.reason == "completed" and result.agent_started
    f.agent.server_client.post.assert_awaited_once()
    assert f.agent.server_client.post.await_args.kwargs["server_name"] == "model"
    assert f.harnesses[0].context.task_id is None
    f.provider.aclose.assert_awaited_once()


async def test_seed_failure_still_requests_resource_cleanup(fixture):
    f = fixture
    f.seed.clear()
    f.seed.update(
        session_id="resource-session", termination={"reason": "infrastructure_error", "detail": "seed failed"}
    )
    await f.agent.run(f.request, f.body)
    assert not f.harnesses
    assert not f.verification["agent_started"]
    assert f.verification["termination"]["detail"] == "seed failed"
    f.provider.aclose.assert_not_awaited()


async def test_shutdown_while_seed_request_never_returns(fixture):
    f = fixture
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def blocked(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    f.agent.server_client.post.side_effect = blocked
    worker = asyncio.create_task(f.agent.run(f.request, f.body))
    await started.wait()
    await asyncio.wait_for(f.agent.shutdown(), timeout=0.5)
    await asyncio.wait_for(cancelled.wait(), timeout=0.5)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(worker, timeout=0.5)
    f.agent.server_client.post.assert_awaited_once()
    assert not f.harnesses


async def test_run_invokes_responses_with_session_state_and_releases_it(fixture, monkeypatch):
    f = fixture
    calls = []
    original = module.MiniSWESandboxedAgent.responses

    async def responses(self, request, body):
        key = (request.session[SESSION_ID_KEY], request.session[module._SESSION_KEY])
        state = self._sessions[key]
        assert state.sandbox is f.sandbox
        assert state.harness is f.harnesses[0]
        assert state.harness.context.instruction == f.seed["instruction"]
        calls.append(key)
        return await original(self, request, body)

    monkeypatch.setattr(module.MiniSWESandboxedAgent, "responses", responses)
    first, replay = await asyncio.gather(f.agent.run(f.request, f.body), f.agent.run(f.request, f.body))
    assert first == replay
    assert calls == [("owner", "resource-session")]
    assert not f.agent._sessions
    assert f.body.responses_create_params.input == []


async def test_responses_requires_matching_session_and_replays_one_execution(fixture):
    from fastapi import HTTPException, Request

    f = fixture
    params = f.body.responses_create_params
    response = module.empty_response(params, "model")
    execute = AsyncMock(return_value=(response, HarnessOutcome(reason="completed"), {}))
    state = module.MiniSWESession(
        sandbox=f.sandbox,
        harness=SimpleNamespace(params=params, execute=execute),
        budget=60,
    )
    f.agent._sessions["owner", "resource-session"] = state

    def request(owner, session_id):
        return Request({"type": "http", "session": {SESSION_ID_KEY: owner, module._SESSION_KEY: session_id}})

    for owner, session_id in [("other", "resource-session"), ("owner", "missing")]:
        with pytest.raises(HTTPException, match="No seeded") as error:
            await f.agent.responses(request(owner, session_id), params)
        assert error.value.status_code == 409
    with pytest.raises(HTTPException, match="bound to another"):
        await f.agent.responses(request("owner", "resource-session"), params.model_copy(update={"input": "different"}))
    first, replay = await asyncio.gather(
        *[f.agent.responses(request("owner", "resource-session"), params) for _ in range(2)]
    )
    assert first == replay == response
    execute.assert_awaited_once_with(60)
    assert state.result[1].reason == "completed"
    f.agent._sessions.clear()
