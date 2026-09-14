# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual TCP requests through SimpleAgent, GenRM and the production middleware.

Only model inference is deterministic; requests, retries, cancellation, and
Uvicorn shutdown cross real HTTP connections through Gym's ServerClient.
"""

import asyncio
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf

import nemo_gym.server_utils as http
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from resources_servers.genrm_compare.app import GenRMCompareResourcesServer
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import member
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


@asynccontextmanager
async def listening(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", timeout_graceful_shutdown=0.5))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        await until(lambda: server.started or task.done())
        assert server.started
        yield f"http://127.0.0.1:{port}", server, task
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        sock.close()


def production_app(instance):
    app = instance.setup_webserver()
    instance.setup_exception_middleware(app)
    instance.setup_cancellation_middleware(app)
    return app


@pytest.fixture
async def services(config, monkeypatch):
    # Each test owns the loop and the global client; restore any enclosing fixture.
    monkeypatch.setattr(http, "_GLOBAL_AIOHTTP_CLIENT", None)
    session = http.set_global_aiohttp_client(http.GlobalAIOHTTPAsyncClientConfig())
    client = http.ServerClient.model_construct(global_config_dict=OmegaConf.create({}))
    config.num_rollouts_per_prompt = 4
    config.cohort_timeout_s = 3
    resource = GenRMCompareResourcesServer(config=config, server_client=client)
    state = SimpleNamespace(
        policy_calls=0, judge_calls=0, judge_status=200, judge_empty=False, judge_release=asyncio.Event()
    )
    state.judge_release.set()
    policy_app, judge_app = FastAPI(), FastAPI()

    @policy_app.post("/v1/responses")
    async def policy(request: Request):
        await request.json()
        state.policy_calls += 1
        payload = member(0, response_id=f"policy-{state.policy_calls}").response.model_dump(mode="json")
        payload["output"] = [
            {
                "id": f"message-{state.policy_calls}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "4", "annotations": []}],
            }
        ]
        return payload

    @judge_app.post("/v1/responses")
    async def judge(request: Request):
        payload = await request.json()
        assert payload["metadata"]["response_1"] == payload["metadata"]["response_2"] == "4"
        state.judge_calls += 1
        await state.judge_release.wait()
        if state.judge_status != 200:
            return JSONResponse({"error": "judge unavailable"}, status_code=state.judge_status)
        return {
            "output": []
            if state.judge_empty
            else [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"score_1":4,"score_2":2,"ranking":1}'}],
                }
            ]
        }

    agent_config = SimpleAgentConfig(
        host="127.0.0.1",
        port=0,
        name="agent",
        entrypoint="app.py",
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        resources_server=ResourcesServerRef(type="resources_servers", name="resource"),
    )
    agent = SimpleAgent(config=agent_config, server_client=client)
    try:
        async with AsyncExitStack() as stack:
            for name, app in (
                ("policy", policy_app),
                ("judge", judge_app),
                ("resource", production_app(resource)),
                ("agent", production_app(agent)),
            ):
                url, server, task = await stack.enter_async_context(listening(app))
                client._server_base_urls[name] = url
                if name == "resource":
                    state.resource_http_server, state.resource_http_task = server, task
            state.resource, state.client = resource, client
            yield state
    finally:
        await session.close()


async def run(services, index, *, group="group", attempt=0):
    payload = member(index, group=group, attempt=attempt).model_dump(mode="json", by_alias=True)
    del payload["response"]
    result = await services.client.post(server_name="agent", url_path="/run", json=payload)
    return result.status, await result.json()


@pytest.mark.parametrize("judge_failure", [False, True])
async def test_run_returns_503_without_reward_on_incomplete_or_judge_failure(services, judge_failure):
    if judge_failure:
        services.judge_status = 500
    else:
        services.resource.config.cohort_timeout_s = 0.05
    results = await asyncio.gather(*(run(services, i) for i in range(4 if judge_failure else 1)))
    assert all(status == 503 and "reward" not in body for status, body in results)
    assert all(not c.members and c.phase == "failed" for c in services.resource._verify_cohorts.values())


async def test_empty_http_200_judge_fails_through_run(services):
    services.judge_empty = True
    results = await asyncio.gather(*(run(services, i) for i in range(4)))
    assert all(status == 503 and "reward" not in body for status, body in results)


async def test_run_completion_replay_and_invalid_member_preserve_status(services):
    results = await asyncio.gather(*(run(services, i) for i in (3, 0, 2, 1)))
    assert all(status == 200 and body["reward"] == 3.0 for status, body in results)
    assert [body["_ng_rollout_index"] for _, body in results] == [3, 0, 2, 1]
    assert services.judge_calls == 4
    assert (await run(services, 0))[0] == 409
    assert (await run(services, 4, group="invalid"))[0] == 422
    assert services.judge_calls == 4


@pytest.mark.parametrize("during_judging", [False, True])
async def test_run_disconnect_fails_group_then_new_attempt_succeeds(services, during_judging):
    services.judge_release.clear()
    count = 3 if during_judging else 1
    requests = [asyncio.create_task(run(services, i)) for i in range(count)]
    await until(lambda: any(len(c.members) == count for c in services.resource._verify_cohorts.values()))
    if during_judging:
        await until(lambda: services.judge_calls > 0)
    old = next(iter(services.resource._verify_cohorts.values()))
    requests[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await requests[0]
    await until(lambda: old.phase == "failed")
    assert all(status == 503 for status, _ in await asyncio.gather(*requests[1:]))
    # A transport retry of /run generates another answer. It must not be judged
    # against abandoned answers or keep its siblings waiting for the deadline.
    previous_policy_calls = services.policy_calls
    assert (await run(services, 0))[0] == 503
    assert services.policy_calls == previous_policy_calls + 1
    assert not old.members and not old.rewards
    services.judge_release.set()
    results = await asyncio.gather(*(run(services, i, attempt=1) for i in range(4)))
    assert all(status == 200 and body["reward"] == 3.0 for status, body in results)


async def test_production_graceful_shutdown_releases_active_state(services):
    services.judge_release.clear()
    requests = [asyncio.create_task(run(services, i)) for i in range(3)]
    await until(lambda: services.judge_calls > 0)
    services.resource_http_server.should_exit = True
    await asyncio.wait_for(services.resource_http_task, 3)
    # Uvicorn cancels handlers after its production 0.5s grace, before lifespan
    # cleanup. The connections can fail, but none may receive a successful score.
    results = await asyncio.wait_for(asyncio.gather(*requests, return_exceptions=True), 3)
    assert all(isinstance(result, Exception) or result[0] >= 500 for result in results)
    assert not services.resource._verify_cohorts and not services.resource._cohort_tasks
