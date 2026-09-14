# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual TCP requests through SimpleAgent, GenRM and the production middleware.

Only model inference is deterministic; requests, retries, cancellation, and
Uvicorn shutdown cross real HTTP connections through Gym's ServerClient.
"""

import asyncio
import json
import socket
import time
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
    client = http.ServerClient.model_construct(
        global_config_dict=OmegaConf.create({"agent": {"responses_api_agents": {"simple_agent": {}}}})
    )
    config.num_rollouts_per_prompt = 4
    config.cohort_collection_timeout_s = 3
    config.cohort_evaluation_timeout_s = 3
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
async def test_run_fails_without_reward_and_preserves_reason(services, judge_failure):
    if judge_failure:
        services.judge_status = 500
    else:
        services.resource.config.cohort_collection_timeout_s = 0.05
    results = await asyncio.gather(*(run(services, i) for i in range(4 if judge_failure else 1)))
    assert all(status == 500 and "reward" not in body for status, body in results)
    reason = "judge offline" if judge_failure else "did not collect 4 unique rollout indices"
    if judge_failure:
        assert all("500" in body for _, body in results)
    else:
        assert all(reason in body for _, body in results)
    assert all(c.phase == "failed" and not c.rewards for c in services.resource._verify_cohorts.values())


async def test_empty_http_200_judge_retries_then_fails_through_run(services):
    services.judge_empty = True
    results = await asyncio.gather(*(run(services, i) for i in range(4)))
    assert all(status == 500 and "no completed answer after 4 attempts" in body for status, body in results)
    assert services.judge_calls == 16


async def test_run_resampling_conflicts_while_exact_verify_replays(services):
    results = await asyncio.gather(*(run(services, i) for i in (3, 0, 2, 1)))
    assert all(status == 200 and body["reward"] == 3.0 for status, body in results)
    assert [body["_ng_rollout_index"] for _, body in results] == [3, 0, 2, 1]
    assert services.judge_calls == 4
    for _, body in results:
        response = await services.client.post(server_name="resource", url_path="/verify", json=body)
        assert response.status == 200 and (await response.json())["reward"] == body["reward"]
    status, body = await run(services, 0)
    assert status == 500 and "different response" in body
    assert services.judge_calls == 4


@pytest.mark.parametrize("during_judging", [False, True])
async def test_verify_disconnect_allows_exact_reattachment_over_tcp(services, during_judging):
    services.judge_release.clear()
    payloads = []
    for i in range(4):
        body = member(i).model_dump(mode="json", by_alias=True)
        policy = await services.client.post(server_name="policy", url_path="/v1/responses", json={})
        body["response"] = await policy.json()
        payloads.append(body)

    async def verify(i):
        response = await services.client.post(server_name="resource", url_path="/verify", json=payloads[i])
        return response.status, await response.json()

    count = 4 if during_judging else 1
    requests = [asyncio.create_task(verify(i)) for i in range(count)]
    await until(lambda: any(len(c.members) == count for c in services.resource._verify_cohorts.values()))
    if during_judging:
        await until(lambda: services.judge_calls == 4)
    old = next(iter(services.resource._verify_cohorts.values()))
    requests[0].cancel()
    await asyncio.gather(requests[0], return_exceptions=True)
    await until(lambda: not old.members[0].waiters)
    assert old.phase in ("collecting", "evaluating") and old.members[0].body is not None
    requests[0] = asyncio.create_task(verify(0))
    requests += [asyncio.create_task(verify(i)) for i in range(count, 4)]
    services.judge_release.set()
    results = await asyncio.gather(*requests)
    assert all(status == 200 and body["reward"] == 3.0 for status, body in results)
    assert services.policy_calls == services.judge_calls == 4


async def test_closed_judge_port_is_bounded_despite_transport_retries(services):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    # Keep the unlistening port reserved so another service cannot reuse it.
    services.client._server_base_urls["judge"] = f"http://127.0.0.1:{sock.getsockname()[1]}"
    try:
        started = time.monotonic()
        results = await asyncio.gather(*(run(services, i) for i in range(4)))
        assert time.monotonic() - started < 2
        assert all(status == 500 and "TimeoutError" in body for status, body in results)
    finally:
        sock.close()


async def test_production_graceful_shutdown_releases_active_state(services):
    services.judge_release.clear()
    requests = [asyncio.create_task(run(services, i)) for i in range(4)]
    await until(lambda: services.judge_calls > 0)
    services.resource_http_server.should_exit = True
    await asyncio.wait_for(services.resource_http_task, 3)
    results = await asyncio.wait_for(asyncio.gather(*requests, return_exceptions=True), 3)
    assert all(isinstance(result, Exception) or result[0] >= 500 for result in results)
    assert not services.resource._verify_cohorts and not services.resource._cohort_tasks


@pytest.mark.parametrize("judge_failure", [False, True])
async def test_collector_saves_actual_cohort_failure_class_and_reason(services, tmp_path, monkeypatch, judge_failure):
    import nemo_gym.rollout_collection as collection

    monkeypatch.setattr(collection, "setup_server_client_utils", lambda *a, **k: services.client)
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: OmegaConf.create({}))
    # Use the real collector, agent, resource and aggregate endpoints. Only the
    # config/head-server discovery is replaced by the fixture's bound addresses.
    body = member(0).model_dump(mode="json", by_alias=True)
    del body["response"]
    body["agent_ref"] = {"name": "agent"}
    input_path, output_path = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    input_path.write_text(json.dumps(body) + "\n")
    if judge_failure:
        services.judge_status = 500
    else:
        services.resource.config.cohort_collection_timeout_s = 0.05
    config = collection.RolloutCollectionConfig(
        input_jsonl_fpath=str(input_path),
        output_jsonl_fpath=str(output_path),
        num_repeats=4 if judge_failure else 1,
        num_samples_in_parallel=4,
        route_failures_to_sidecar=True,
        disable_health_check=True,
        count_failure_classes_as_zero=["agent_run_error"],
    )
    with pytest.raises(RuntimeError, match="produced a result"):
        await collection.RolloutCollectionHelper().run_from_config(config)
    assert not output_path.read_text().strip()
    failures = [json.loads(line) for line in (tmp_path / "output_failures.jsonl").read_text().splitlines()]
    assert len(failures) == (4 if judge_failure else 1)
    for row in failures:
        assert row["_ng_failure_class"] == "agent_run_error"
        assert row["_ng_failure_http_status"] == 500
        assert "reward" not in row and "response" not in row
        assert row["_ng_group_id"] == "group"
        assert (
            "evaluation failed" in row["_ng_failure_response_body"]
            if judge_failure
            else "did not collect" in row["_ng_failure_response_body"]
        )
    assert not (tmp_path / "output_aggregate_metrics.json").exists()
