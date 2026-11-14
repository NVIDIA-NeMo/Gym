# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Environment Server uses real HTTP agent sessions and preserves native grading."""

import asyncio
from http.cookies import SimpleCookie
from types import SimpleNamespace

import httpx
import pytest
from omegaconf import OmegaConf

from environment_servers.saas_bench_environment import app as environment
from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import AgentContext
from responses_api_agents.saas_bench_agent.app import SaasBenchAgent
from responses_api_agents.saas_bench_agent.tests.test_app import agent, grader, request, result  # noqa: F401


def client_for(actor):
    config = OmegaConf.create(
        {
            actor.config.name: {
                "responses_api_agents": {
                    "saas_bench_agent": {
                        "name": actor.config.name,
                        "host": "127.0.0.1",
                        "port": 1,
                        "entrypoint": "app.py",
                    }
                }
            }
        }
    )
    return ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=config)


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["opencode", "browser_use"])
@pytest.mark.parametrize("typed", [False, True])
async def test_environment_uses_seed_activate_close_and_preserves_grade(agent, grader, monkeypatch, harness, typed):
    agent.config.harness = harness
    grader.config.harness = harness
    agent.server_client = client_for(agent)
    grader.server_client = client_for(agent)
    calls = []
    native = result(0.375)
    native["execution_health"] = {"termination": "completed"}

    async def execute(body):
        return native

    monkeypatch.setattr(agent, "_execute", execute)
    transport = httpx.ASGITransport(app=agent.setup_webserver())
    async with httpx.AsyncClient(transport=transport, base_url="http://actor") as http:

        async def post(server_name, url_path, *, json, cookies, **kwargs):
            calls.append(url_path)
            value = json.model_dump(mode="json") if hasattr(json, "model_dump") else json
            response = await http.post(url_path, json=value, cookies=cookies)
            jar = SimpleCookie()
            for name, value in response.cookies.items():
                jar[name] = value
            return SimpleNamespace(payload=response.json(), status=response.status_code, cookies=jar)

        async def check(reply):
            assert reply.status == 200, reply.payload

        async def decode(reply):
            return reply.payload

        monkeypatch.setattr(type(grader.server_client), "post", lambda self, *args, **kwargs: post(*args, **kwargs))
        monkeypatch.setattr(grader, "_prepare_runtime", lambda: None)
        monkeypatch.setattr(environment, "raise_for_status", check)
        monkeypatch.setattr(environment, "get_response_json", decode)
        row = {**request().model_dump(mode="json", by_alias=True), "_ng_rollout_id": "repeat-0"}
        if harness == "browser_use":
            row["verifier_metadata"]["harness_version"] = "0.13.7"
        if typed:
            row.pop("_ng_rollout_id")
            row = {
                "episode_id": {"rollout_id": "repeat-0", "attempt": 1},
                "task": {"task_id": {"taskset": "saas", "task_id": "fixture"}, "task_input": row},
            }
        response = await grader.run_flat(row)
        if typed:
            response = response["result"]
    expected_capture = "repeat-0-a1" if typed else "repeat-0"
    assert response["native_diagnostics"]["rollout_id"] == expected_capture
    assert calls == ["/v1/agent_sessions", f"/ng-rollout/{expected_capture}/v1/responses", "/v1/agent_sessions/close"]
    assert response["reward"] == 0.375 and response["mask_sample"] is False
    assert "saas_bench_native_result" not in response["response"]
    assert all(
        record.state is None and record.close_response.cleanup_confirmed for record in agent._session_records.values()
    )
    assert not {"run", "aggregate_metrics"}.intersection(vars(SaasBenchAgent))


@pytest.mark.asyncio
async def test_session_close_cancels_and_waits_for_activation_cleanup(agent, monkeypatch):
    agent.server_client = client_for(agent)
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def execute(body):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    monkeypatch.setattr(agent, "_execute", execute)
    episode = EpisodeId(rollout_id="repeat-0")
    seed = AgentSeedSessionRequest(
        agent_session_id="native-session",
        episode_id=episode,
        task_id=TaskId(taskset="saas", task_id="fixture"),
        agent_context=AgentContext(
            values={"episode": {**request().model_dump(mode="json", by_alias=True), "_ng_rollout_id": "repeat-0"}}
        ),
    )
    transport = httpx.ASGITransport(app=agent.setup_webserver())
    async with httpx.AsyncClient(transport=transport, base_url="http://actor") as http:
        seeded = await http.post("/v1/agent_sessions", json=seed.model_dump(mode="json", by_alias=True))
        assert seeded.status_code == 200, seeded.text
        task = asyncio.create_task(
            http.post(
                "/ng-rollout/repeat-0/v1/responses", json=request().responses_create_params.model_dump(mode="json")
            )
        )
        await asyncio.wait_for(started.wait(), 10)
        closed = await http.post(
            "/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(agent_session_id="native-session", episode_id=episode).model_dump(
                mode="json"
            ),
        )
        assert closed.status_code == 200 and closed.json()["cleanup_confirmed"] is True
        assert cleaned.is_set()
        await asyncio.gather(task, return_exceptions=True)
        stale = await http.post("/v1/agent_sessions", json=seed.model_dump(mode="json", by_alias=True))
        assert stale.status_code == 409
