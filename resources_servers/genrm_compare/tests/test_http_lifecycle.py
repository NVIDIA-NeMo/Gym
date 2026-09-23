# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from fastapi.responses import JSONResponse, Response
from omegaconf import OmegaConf

import nemo_gym.server_utils as http
from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.reward_profile import RewardProfiler
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
        policy_calls=0,
        judge_calls=0,
        judge_status=200,
        judge_statuses=[],
        judge_empty=False,
        judge_media_type="application/json",
        judge_texts=[],
        truncated_judge_responses=0,
        judge_release=asyncio.Event(),
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
        status = state.judge_statuses.pop(0) if state.judge_statuses else state.judge_status
        if status != 200:
            return JSONResponse({"error": "judge unavailable"}, status_code=status)
        if state.truncated_judge_responses:
            state.truncated_judge_responses -= 1
            # Send successful headers and part of the body, then break the TCP response.
            # Uvicorn closes the connection when the advertised length is not fulfilled.
            return Response(b'{"output":', headers={"content-length": "1000"}, media_type="application/json")
        text = state.judge_texts.pop(0) if state.judge_texts else '{"score_1":4,"score_2":2,"ranking":1}'
        response = {
            "output": []
            if state.judge_empty
            else [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }
            ]
        }
        return Response(json.dumps(response), media_type=state.judge_media_type)

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
    payload["_ng_task_index"] = 0
    result = await services.client.post(server_name="agent", url_path="/run", json=payload)
    return result.status, await result.json()


async def test_conversion_failure_releases_http_peer_and_requires_new_attempt(services, monkeypatch):
    async def verify(index, *, attempt=0):
        payload = member(index, attempt=attempt).model_dump(mode="json", by_alias=True)
        payload["response"]["output"] = [
            {
                "id": f"message-{index}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "4", "annotations": []}],
            }
        ]
        result = await services.client.post(server_name="resource", url_path="/verify", json=payload)
        return result.status, await result.json()

    # These direct verify requests use fixed IDs, so fault injection targets one member.
    original = services.resource._comparison_response
    first = asyncio.create_task(verify(0))
    await until(lambda: bool(services.resource._verify_cohorts))
    cohort = next(iter(services.resource._verify_cohorts.values()))
    assert list(cohort.members) == [0] and len(cohort.members[0].waiters) == 1

    def fail_one(response):
        if response.id == "answer-1":
            raise ValueError("injected conversion failure")
        return original(response)

    monkeypatch.setattr(services.resource, "_comparison_response", fail_one)
    status, body = await verify(1)
    assert status == 503 and "injected conversion failure" in body["detail"]
    peer_status, peer_body = await asyncio.wait_for(first, 0.5)
    assert peer_status == 503 and "injected conversion failure" in peer_body["detail"]
    assert cohort.phase == "failed" and all(not m.waiters for m in cohort.members.values())
    assert services.resource._active_group_count == 0 and services.judge_calls == 0
    monkeypatch.setattr(services.resource, "_comparison_response", original)
    assert (await verify(1))[0] == 503
    results = await asyncio.gather(*(verify(i, attempt=1) for i in range(4)))
    assert all(status == 200 and body["reward"] == 3.0 for status, body in results)
    assert services.judge_calls == 4


async def test_judge_task_start_failure_releases_all_http_waiters(services, monkeypatch):
    create_task = asyncio.create_task
    failures = []

    def fail_judging(coro, **kwargs):
        if kwargs.get("name", "").startswith("genrm-cohort-evaluation"):
            failures.append(kwargs["name"])
            raise RuntimeError("injected task startup failure")
        return create_task(coro, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", fail_judging)
    results = await asyncio.gather(*(run(services, i) for i in range(4)))
    # SimpleAgent translates the resources server's 503 into a failed /run (500).
    for status, body in results:
        assert status == 500 and "reward" not in body
        assert "GenRM cohort task startup failed: RuntimeError: injected task startup failure" in body
    assert len(failures) == 1 and services.judge_calls == 0
    cohort = next(iter(services.resource._verify_cohorts.values()))
    assert cohort.phase == "failed" and services.resource._active_group_count == 0
    assert all(not member.waiters and member.response_obj is None for member in cohort.members.values())


async def test_incomplete_run_fails_without_reward(services):
    services.resource.config.cohort_collection_timeout_s = 0.05
    status, body = await run(services, 0)
    assert status == 500 and "reward" not in body
    assert "did not collect 4 unique rollout indices" in body
    assert all(c.phase == "failed" and not c.rewards for c in services.resource._verify_cohorts.values())


def assert_judge_failure(status, body, reason):
    assert status == 200
    assert body["_ng_failure_class"] == "judge_failed"
    assert reason in body["_ng_failure_judge_error"]
    assert body["response"]["id"].startswith("policy-")
    assert body["response"]["output"][0]["content"][0]["text"] == "4"
    assert body["instance_config"]["mask_sample"] is True
    assert body["mask_sample"] is True
    assert body["failure_kind"] == "judge_failed"
    assert body["failure_reason"] == body["_ng_failure_judge_error"]
    assert len(body["failure_reason"]) <= 2000
    validated = BaseVerifyResponse.model_validate(body)
    assert validated.mask_sample is True and validated.failure_kind == "judge_failed"
    assert body["reward"] == 0  # Failsafe placeholder; never a scored result.


async def test_judge_failure_preserves_answer_and_diagnostics_through_run(services):
    services.judge_status = 500
    results = await asyncio.gather(*(run(services, i) for i in range(4)))
    for index, (status, body) in enumerate(results):
        assert_judge_failure(status, body, "judge unavailable")
        assert body["_ng_group_id"] == "group" and body["_ng_group_attempt"] == 0
        assert body["_ng_task_index"] == 0 and body["_ng_rollout_index"] == index
        identity_fields = {"task_index", "rollout_index", "group_id", "group_attempt"}
        assert identity_fields.isdisjoint(body)
        metrics = RewardProfiler().rollout_info_from_result(body)
        assert identity_fields.isdisjoint(metrics) and "_ng_group_attempt" not in metrics
        error = body["_ng_failure_judge_error"]
        assert "judge /v1/responses pair=" in error and "500" in error and "deadline=" in error
    assert len({body["response"]["id"] for _, body in results}) == 4
    assert all(c.phase == "failed" and not c.rewards for c in services.resource._verify_cohorts.values())


async def test_empty_http_200_judge_retries_then_fails_through_run(services):
    services.judge_empty = True
    results = await asyncio.gather(*(run(services, i) for i in range(4)))
    for status, body in results:
        assert_judge_failure(status, body, "no completed answer after 4 attempts")
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
    assert old.phase in ("collecting", "evaluating") and old.members[0].response_obj is not None
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
        for status, body in results:
            assert_judge_failure(status, body, "TimeoutError")
            assert "deadline=0.2s" in body["_ng_failure_judge_error"]
    finally:
        sock.close()


async def test_production_graceful_shutdown_releases_active_state(services):
    services.resource.config.judge_request_timeout_s = 10
    services.resource.config.cohort_evaluation_timeout_s = 10
    services.judge_release.clear()
    requests = [asyncio.create_task(run(services, i)) for i in range(4)]
    await until(lambda: services.judge_calls > 0)
    cohort = next(iter(services.resource._verify_cohorts.values()))
    services.resource_http_server.should_exit = True
    await asyncio.wait_for(services.resource_http_task, 3)
    results = await asyncio.wait_for(asyncio.gather(*requests, return_exceptions=True), 3)
    assert all(isinstance(result, Exception) or result[0] >= 500 for result in results)
    assert cohort.failure == "GenRM server is shutting down"
    assert not services.resource._verify_cohorts and not services.resource._cohort_tasks


@pytest.mark.parametrize("judge_failure", [False, True])
@pytest.mark.parametrize("count_failure", [False, True])
async def test_collector_saves_actual_cohort_failure_class_and_reason(
    services, tmp_path, monkeypatch, judge_failure, count_failure
):
    import nemo_gym.rollout_collection as collection

    monkeypatch.setattr(collection, "setup_server_client_utils", lambda *a, **k: services.client)
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: OmegaConf.create({}))
    # Use the real collector, agent, resource and aggregate endpoints. Only the
    # config/head-server discovery is replaced by the fixture's bound addresses.
    body = member(0).model_dump(mode="json", by_alias=True)
    del body["response"]
    body.pop("_ng_task_index", None)  # Let the collector assign the task index.
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
        count_failure_classes_as_zero=["judge_failed" if judge_failure else "agent_run_error"]
        if count_failure
        else [],
    )
    if count_failure:
        await collection.RolloutCollectionHelper().run_from_config(config)
    else:
        with pytest.raises(RuntimeError, match="produced a result"):
            await collection.RolloutCollectionHelper().run_from_config(config)
    assert not output_path.read_text().strip()
    sidecar_path = tmp_path / "output_failures.jsonl"
    original_sidecar = sidecar_path.read_bytes()
    failures = [json.loads(line) for line in original_sidecar.splitlines()]
    assert len(failures) == (4 if judge_failure else 1)
    for row in failures:
        if judge_failure:
            assert_judge_failure(200, row, "judge unavailable")
        else:
            assert row["_ng_failure_class"] == "agent_run_error"
            assert row["_ng_failure_http_status"] == 500
            assert "reward" not in row and "response" not in row
            assert "did not collect" in row["_ng_failure_response_body"]
        assert row["_ng_task_index"] == 0
        assert 0 <= row["_ng_rollout_index"] < config.num_repeats
    inputs = [json.loads(line) for line in (tmp_path / "output_materialized_inputs.jsonl").read_text().splitlines()]
    assert all(row["_ng_group_id"] == "group" for row in inputs)
    assert {(r["_ng_task_index"], r["_ng_rollout_index"]) for r in failures} == {
        (r["_ng_task_index"], r["_ng_rollout_index"]) for r in inputs
    }
    online_path = tmp_path / "output_aggregate_metrics.json"
    if count_failure:
        online = json.loads(online_path.read_text())
        assert online[0]["key_metrics"] == {"mean/reward": 0.0}
        offline_path = await collection.RolloutAggregationHelper().run_from_config(
            collection.RolloutAggregationConfig(
                input_glob=str(output_path),
                output_jsonl_fpath=str(tmp_path / "merged.jsonl"),
                count_failure_classes_as_zero=config.count_failure_classes_as_zero,
                disable_health_check=True,
            )
        )
        assert json.loads(offline_path.read_text()) == online
        assert output_path.read_bytes() == (tmp_path / "merged.jsonl").read_bytes() == b""
        assert sidecar_path.read_bytes() == original_sidecar
    else:
        assert not online_path.exists()


async def test_collector_can_repeat_legacy_task_on_same_live_server(services, tmp_path, monkeypatch):
    import nemo_gym.rollout_collection as collection

    monkeypatch.setattr(collection, "setup_server_client_utils", lambda *a, **k: services.client)
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: OmegaConf.create({}))
    input_path = tmp_path / "legacy-task.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "2+2?"}]},
                "agent_ref": {"name": "agent"},
            }
        )
        + "\n"
    )
    response_ids = []
    for invocation in range(2):
        output_path = tmp_path / f"legacy-run-{invocation}.jsonl"
        config = collection.RolloutCollectionConfig(
            input_jsonl_fpath=str(input_path),
            output_jsonl_fpath=str(output_path),
            num_repeats=4,
            num_samples_in_parallel=4,
            disable_health_check=True,
        )
        await collection.RolloutCollectionHelper().run_from_config(config)
        rows = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert len(rows) == 4
        assert all(row["reward"] == 3.0 for row in rows)
        response_ids.append({row["response"]["id"] for row in rows})
        assert not services.resource._verify_cohorts
    assert response_ids[0].isdisjoint(response_ids[1])
    assert services.policy_calls == services.judge_calls == 8


async def test_failed_legacy_collector_requires_fresh_explicit_group(services, tmp_path, monkeypatch):
    import nemo_gym.rollout_collection as collection

    monkeypatch.setattr(collection, "setup_server_client_utils", lambda *a, **k: services.client)
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: OmegaConf.create({}))
    input_path, output_path = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "2+2?"}]},
                "agent_ref": {"name": "agent"},
            }
        )
        + "\n"
    )
    config = collection.RolloutCollectionConfig(
        input_jsonl_fpath=str(input_path),
        output_jsonl_fpath=str(output_path),
        num_repeats=4,
        num_samples_in_parallel=4,
        disable_health_check=True,
    )
    services.judge_status = 500
    with pytest.raises(RuntimeError, match="produced a result"):
        await collection.RolloutCollectionHelper().run_from_config(config)
    failures = [json.loads(line) for line in (tmp_path / "output_failures.jsonl").read_text().splitlines()]
    assert len(failures) == 4
    for row in failures:
        assert_judge_failure(200, row, "judge unavailable")
    assert all(c.phase == "failed" for c in services.resource._verify_cohorts.values())

    services.judge_status = 200
    judge_calls = services.judge_calls
    config.resume_from_cache = True
    with pytest.raises(RuntimeError, match="produced a result"):
        await collection.RolloutCollectionHelper().run_from_config(config)
    assert not output_path.read_text().strip()
    assert services.judge_calls == judge_calls
    retried_failures = [json.loads(line) for line in (tmp_path / "output_failures.jsonl").read_text().splitlines()]
    assert all("fresh _ng_group_id" in row["_ng_failure_judge_error"] for row in retried_failures)

    # Recovery is explicitly coordinated as a fresh complete group, not a legacy-key restart.
    replacement_input = tmp_path / "replacement.jsonl"
    replacement_input.write_text(
        json.dumps(json.loads(input_path.read_text()) | {"_ng_group_id": "replacement", "_ng_group_attempt": 0}) + "\n"
    )
    output_path = tmp_path / "replacement-output.jsonl"
    config = config.model_copy(
        update={
            "input_jsonl_fpath": str(replacement_input),
            "output_jsonl_fpath": str(output_path),
            "resume_from_cache": False,
        }
    )
    await collection.RolloutCollectionHelper().run_from_config(config)
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 4 and all(row["reward"] == 3.0 for row in rows)
    assert {r["response"]["id"] for r in rows}.isdisjoint(r["response"]["id"] for r in failures)
    assert all("_ng_failure_class" not in row for row in rows)
    assert all(row["_ng_group_id"] == "replacement" for row in rows)
    assert services.policy_calls == 12
    judge_calls = services.judge_calls
    config.resume_from_cache = True
    await collection.RolloutCollectionHelper().run_from_config(config)
    assert services.policy_calls == 12 and services.judge_calls == judge_calls
    assert [json.loads(line) for line in output_path.read_text().splitlines()] == rows


@pytest.mark.parametrize("status", [408, 429, 503])
async def test_transient_http_failure_recovers_without_regenerating_answers(services, status):
    services.judge_statuses = [status]
    services.judge_media_type = "text/plain"
    results = await asyncio.gather(*(run(services, i) for i in range(4)))
    assert all(code == 200 and body["reward"] == 3.0 for code, body in results)
    assert all(
        "_ng_failure_class" not in body and not (body.get("instance_config") or {}).get("mask_sample")
        for _, body in results
    )
    assert services.policy_calls == 4 and services.judge_calls == 5
    assert len({body["response"]["id"] for _, body in results}) == 4


@pytest.mark.parametrize("recovers", [True, False])
async def test_interrupted_judge_body_retries_without_regenerating_answers(services, recovers):
    # Allow Uvicorn to close the truncated response before the independent request
    # deadline can win on a loaded runner. Keep both failure paths bounded.
    services.resource.config.judge_request_timeout_s = 2.0
    services.resource.config.cohort_evaluation_timeout_s = 10.0
    services.truncated_judge_responses = 1 if recovers else 100
    results = await asyncio.gather(*(run(services, i) for i in range(4)))
    assert services.policy_calls == 4
    if recovers:
        assert services.judge_calls == 5
        assert all(status == 200 and body["reward"] == 3.0 and not body["mask_sample"] for status, body in results)
    else:
        assert 4 <= services.judge_calls <= 4 * (services.resource.config.genrm_parse_retries + 1)
        for status, body in results:
            assert_judge_failure(status, body, "ClientPayloadError")
            assert body["mask_sample"] is True
        assert all(c.phase == "failed" and not c.rewards for c in services.resource._verify_cohorts.values())


@pytest.mark.parametrize("value", ["NaN", "Infinity"])
@pytest.mark.parametrize("recovers", [False, True])
async def test_nonfinite_judge_output_uses_parse_retries_over_http(services, value, recovers):
    invalid = json.dumps({"score_1": value, "score_2": 2, "ranking": 1})
    services.judge_texts = [invalid] * (1 if recovers else 16)
    results = await asyncio.gather(*(run(services, i) for i in range(4)))
    assert all(status == 200 and body["reward"] == 3.0 for status, body in results)
    assert all("_ng_failure_class" not in body for _, body in results)
    assert services.policy_calls == 4 and services.judge_calls == (5 if recovers else 16)


async def test_explicit_judge_failure_requires_new_shared_attempt_over_http(services):
    services.judge_status = 500
    failures = await asyncio.gather(*(run(services, i) for i in range(4)))
    calls = services.judge_calls
    services.judge_status = 200
    retries = await asyncio.gather(*(run(services, i) for i in range(4)))
    assert services.judge_calls == calls
    for original, retry in zip(failures, retries):
        assert_judge_failure(*retry, "judge unavailable")
        assert retry[1]["_ng_failure_judge_error"] == original[1]["_ng_failure_judge_error"]
    recovered = await asyncio.gather(*(run(services, i, attempt=1) for i in range(4)))
    assert all(
        status == 200 and body["reward"] == 3.0 and "_ng_failure_class" not in body for status, body in recovered
    )
    assert services.judge_calls == calls + 4
