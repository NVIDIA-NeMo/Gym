# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise mini-SWE without importing a benchmark or its resource models."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from httpx import ASGITransport, AsyncClient
from starlette.middleware.sessions import SessionMiddleware

from nemo_gym.rollout_correlation import RolloutContextMiddleware
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
    request = Request(
        {"type": "http", "session": {SESSION_ID_KEY: "owner"}, "headers": [(b"cookie", b"session=incoming")]}
    )
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
    monkeypatch.setattr(module, "get_server_url", lambda name: "http://gym-model:8000")
    harnesses = []

    def harness(**kwargs):
        async def execute(budget):
            assert 0 < budget <= 60
            response = module.empty_response(kwargs["params"], "model")
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
    assert f.harnesses[0].model_base_url == "http://gym-model:8000/v1"
    f.provider.aclose.assert_awaited_once()
    assert f.body.responses_create_params.input == []


async def test_run_passes_rollout_prefixed_gym_model_url(fixture):
    f = fixture
    f.agent.server_client.global_config_dict = {"observability_enabled": True}
    body = f.body.model_copy(update={"capture_rollout_id": "rollout"})
    await f.agent.run(f.request, body)
    assert f.harnesses[0].model_base_url == "http://gym-model:8000/ng-rollout/rollout/v1"


async def test_run_uses_sandbox_reachable_model_url_with_rollout_capture(fixture, monkeypatch):
    f = fixture
    f.agent.config = module.MiniSWESandboxedConfig.model_validate(
        f.agent.config.model_dump() | {"sandbox_model_base_url": "https://sandbox-model:8443/gym/v1/"}
    )
    monkeypatch.setattr(module.MiniSWESandboxedAgent, "_token_id_capture_enabled", lambda self: True)
    body = f.body.model_copy(update={"capture_rollout_id": "rollout", "capture_token_ids": True})
    await f.agent.run(f.request, body)
    assert (
        f.harnesses[0].model_base_url == "https://sandbox-model:8443/gym/ng-rollout/rollout/training-token-capture/v1"
    )


@pytest.mark.parametrize("url", ["localhost:8000", "https://sandbox-model/gym?token=secret"])
def test_sandbox_model_url_rejects_non_http_roots(url):
    with pytest.raises(ValueError, match="sandbox_model_base_url"):
        module.MiniSWESandboxedConfig.model_validate(
            {
                "host": "localhost",
                "port": 1,
                "name": "agent",
                "entrypoint": "app.py",
                "resources_server": {"type": "resources_servers", "name": "resources"},
                "model_server": {"type": "responses_api_models", "name": "model"},
                "sandbox_model_base_url": url,
            }
        )


async def test_responses_sets_up_borrowed_session_without_resource_calls(fixture):
    f = fixture
    f.seed.pop("task_id")  # Optional for other resources and old seed responses.
    state = module.MiniSWESession(
        sandbox=f.sandbox,
        seed=SeedSessionResponse.model_validate(f.seed),
        original_params=f.body.responses_create_params.model_copy(deep=True),
        rollout_id="activation",
        capture_model_calls=False,
    )
    f.agent._sessions[("owner", state.rollout_id)] = state
    response = await f.agent.responses(f.request, f.body.responses_create_params)
    assert state.result.termination.reason == "completed" and state.result.agent_started
    assert response == state.result.response
    f.agent.server_client.post.assert_not_awaited()
    module.AsyncSandbox.connect.assert_not_awaited()
    f.harnesses[0].setup.assert_awaited_once()
    assert f.harnesses[0].context.task_id is None
    assert f.harnesses[0].params.input[0].content == f.seed["instruction"]
    assert f.body.responses_create_params.input == []
    f.provider.aclose.assert_not_awaited()


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
        key = (request.session[SESSION_ID_KEY], module.current_rollout_id())
        state = self._sessions[key]
        assert request is f.request
        assert request.session == {SESSION_ID_KEY: "owner"}
        assert state.seed.session_id == "resource-session"
        assert state.sandbox is f.sandbox
        assert not f.harnesses
        module.AsyncSandbox.connect.assert_awaited_once()
        assert body.input == []
        calls.append(key)
        return await original(self, request, body)

    monkeypatch.setattr(module.MiniSWESandboxedAgent, "responses", responses)
    first, replay = await asyncio.gather(f.agent.run(f.request, f.body), f.agent.run(f.request, f.body))
    assert first == replay
    assert calls == [("owner", "rollout")]
    assert not f.agent._sessions
    assert f.body.responses_create_params.input == []


async def test_responses_requires_matching_session_and_replays_one_execution(fixture, monkeypatch):
    f = fixture
    params = f.body.responses_create_params
    response = module.empty_response(params, "model")
    started, finish = asyncio.Event(), asyncio.Event()

    async def run_harness(budget):
        started.set()
        await finish.wait()
        return response, HarnessOutcome(reason="completed"), {}

    execute = AsyncMock(side_effect=run_harness)
    setup_started, setup_finish = asyncio.Event(), asyncio.Event()

    async def setup():
        setup_started.set()
        await setup_finish.wait()

    harness = SimpleNamespace(setup=AsyncMock(side_effect=setup), execute=execute, close=AsyncMock())
    constructor = MagicMock(return_value=harness)
    monkeypatch.setattr(module, "MiniSWEHarness", constructor)
    state = module.MiniSWESession(
        sandbox=f.sandbox,
        seed=SeedSessionResponse.model_validate(f.seed),
        original_params=params.model_copy(deep=True),
        rollout_id="rollout",
        capture_model_calls=False,
    )
    f.agent._sessions[("owner", state.rollout_id)] = state

    def request(owner):
        return Request({"type": "http", "session": {SESSION_ID_KEY: owner}})

    with pytest.raises(HTTPException, match="No seeded") as error:
        await f.agent.responses(request("other"), params)
    assert error.value.status_code == 409
    with pytest.raises(HTTPException, match="bound to another"):
        await f.agent.responses(request("owner"), params.model_copy(update={"input": "different"}))
    first_call = asyncio.create_task(f.agent.responses(request("owner"), params))
    await setup_started.wait()
    replay_call = asyncio.create_task(f.agent.responses(request("owner"), params.model_copy(deep=True)))
    await asyncio.sleep(0)
    constructor.assert_called_once()
    harness.setup.assert_awaited_once()
    execute.assert_not_awaited()
    setup_finish.set()
    await started.wait()
    assert execute.await_count == 1
    finish.set()
    first, replay = await asyncio.gather(first_call, replay_call)
    assert first == replay == response
    execute.assert_awaited_once()
    assert 0 < execute.call_args.args[0] <= 60
    assert state.result.termination.reason == "completed"
    f.agent._sessions.clear()


@pytest.mark.parametrize("prefixed", [False, True])
async def test_http_responses_uses_middleware_session_cookie(fixture, monkeypatch, prefixed):
    f = fixture
    params = f.body.responses_create_params
    response = module.empty_response(params, "model")
    execute = AsyncMock(return_value=(response, HarnessOutcome(reason="completed"), {}))
    harness = SimpleNamespace(setup=AsyncMock(), execute=execute, close=AsyncMock())
    constructor = MagicMock(return_value=harness)
    monkeypatch.setattr(module, "MiniSWEHarness", constructor)
    f.agent._sessions[("owner", "rollout")] = module.MiniSWESession(
        sandbox=f.sandbox,
        seed=SeedSessionResponse.model_validate(f.seed),
        original_params=params.model_copy(deep=True),
        rollout_id="rollout",
        capture_model_calls=False,
    )
    if prefixed:
        f.agent._sessions[("owner", "another")] = module.MiniSWESession(
            sandbox=f.sandbox,
            seed=SeedSessionResponse.model_validate(dict(f.seed, session_id="another-session")),
            original_params=params.model_copy(deep=True),
            rollout_id="another",
            capture_model_calls=False,
        )
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-key")
    app.add_middleware(RolloutContextMiddleware)

    @app.get("/session")
    async def start_session(request: Request):
        request.session[SESSION_ID_KEY] = "owner"
        return {}

    @app.post("/v1/responses")
    async def responses(request: Request, body: module.NeMoGymResponseCreateParamsNonStreaming):
        return await f.agent.responses(request, body)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.get("/session")
        path = "/ng-rollout/rollout/v1/responses" if prefixed else "/v1/responses"
        result = await client.post(path, json=params.model_dump(mode="json"))
        assert result.status_code == 200
        assert result.json()["id"] == response.id
        replay = await client.post(path, json=params.model_dump(mode="json"))
        assert replay.status_code == 200
        assert replay.json() == result.json()
    execute.assert_awaited_once()
    assert 0 < execute.call_args.args[0] <= 60
    constructor.assert_called_once()
    harness.setup.assert_awaited_once()
    module.AsyncSandbox.connect.assert_not_awaited()
    assert constructor.call_args.kwargs["context"].session_id == "resource-session"
    if prefixed:
        assert f.agent._sessions[("owner", "another")].worker is None


@pytest.mark.parametrize("shutdown", [False, True])
async def test_connection_timeout_or_shutdown_releases_transport_before_verification(fixture, shutdown):
    f = fixture
    entered = asyncio.Event()

    async def connect(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    module.AsyncSandbox.connect.side_effect = connect
    if not shutdown:
        f.agent.config.setup_timeout_sec = 0.01
    original_post = f.agent.server_client.post.side_effect

    async def post(**kwargs):
        if kwargs["url_path"] == "/verify":
            f.provider.aclose.assert_awaited_once()
            assert not f.agent._sessions
        return await original_post(**kwargs)

    f.agent.server_client.post.side_effect = post
    caller = asyncio.create_task(f.agent.run(f.request, f.body))
    await entered.wait()
    if shutdown:
        await f.agent.shutdown()
    await asyncio.wait_for(caller, timeout=1)
    assert f.verification["termination"]["reason"] == ("cancelled" if shutdown else "timeout")
    assert not f.verification["agent_started"]
    assert not f.harnesses


@pytest.mark.parametrize("capture", [False, True])
async def test_concurrent_rollouts_share_cookie_without_sharing_execution(fixture, monkeypatch, capture):
    f = fixture
    entered, release = asyncio.Event(), asyncio.Event()
    original_constructor = module.MiniSWEHarness
    original_post = f.agent.server_client.post.side_effect

    async def post(**kwargs):
        response = await original_post(**kwargs)
        if kwargs["url_path"] == "/seed_session":
            response.value = dict(response.value, session_id=kwargs["json"]["rollout_id"])
        return response

    async def setup():
        if len(f.harnesses) == 2:
            entered.set()
        await release.wait()

    def harness(**kwargs):
        instance = original_constructor(**kwargs)
        instance.setup.side_effect = setup
        return instance

    f.agent.server_client.post.side_effect = post
    monkeypatch.setattr(module, "MiniSWEHarness", harness)
    bodies = [
        f.body.model_copy(update={"rollout_id": rid, "capture_rollout_id": rid if capture else None})
        for rid in ("first", "second")
    ]
    callers = [asyncio.create_task(f.agent.run(f.request, body)) for body in bodies]
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert set(f.agent._sessions) == {("owner", "first"), ("owner", "second")}
        with pytest.raises(HTTPException, match="Multiple mini-SWE rollouts"):
            await f.agent.responses(f.request, f.body.responses_create_params)
        with module.rollout_context("missing"):
            with pytest.raises(HTTPException, match="No seeded"):
                await f.agent.responses(f.request, f.body.responses_create_params)
        retry = asyncio.create_task(f.agent.run(f.request, bodies[0]))
    finally:
        release.set()
        results = await asyncio.gather(*callers)
    assert await retry == results[0]
    assert results[0].response.id != results[1].response.id
    assert {h.context.session_id for h in f.harnesses} == {"first", "second"}
    for harness in f.harnesses:
        harness.setup.assert_awaited_once()
        harness.execute.assert_awaited_once()
    verifications = [
        c.kwargs["json"] for c in f.agent.server_client.post.await_args_list if c.kwargs["url_path"] == "/verify"
    ]
    assert {v["session_id"] for v in verifications} == {"first", "second"}
    assert all(v["termination"]["reason"] == "completed" for v in verifications)
    assert not f.agent._sessions


@pytest.mark.parametrize("during_setup", [False, True])
async def test_cancellation_during_cleanup_delays_transport_release_and_verification(
    fixture, monkeypatch, during_setup
):
    from responses_api_agents.miniswe_sandboxed_agent.harness import MiniSWEHarness

    f = fixture
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def cleanup_exec(*args, **kwargs):
        if args[0] == "pwd":
            return SimpleNamespace(return_code=0, stdout="/workspace\n")
        entered.set()
        await release.wait()
        finished.set()
        return SimpleNamespace(return_code=0)

    class Harness(MiniSWEHarness):
        async def setup(self):
            if during_setup:
                raise RuntimeError("setup failed")

        async def execute(self, budget):
            await self.close()
            return module.empty_response(self.params, "model"), HarnessOutcome(reason="completed"), {}

    f.sandbox.exec.side_effect = cleanup_exec
    monkeypatch.setattr(module, "MiniSWEHarness", Harness)
    caller = asyncio.create_task(f.agent.run(f.request, f.body))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        run = f.agent._runs[("owner", "rollout")][1]
        worker = f.agent._sessions[("owner", "rollout")].worker
        for _ in range(3):
            run.cancel()
            worker.cancel()
            await asyncio.sleep(0)
            assert not finished.is_set()
            assert not caller.done()
            assert not f.verification
            f.provider.aclose.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(caller, 1)
    assert finished.is_set()
    f.provider.aclose.assert_awaited_once()
    assert f.verification["termination"]["reason"] == "cancelled"
    assert not f.agent._sessions
