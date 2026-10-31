# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model -> agent -> legacy environment server -> rollout collection over real local sockets."""

import asyncio
from unittest.mock import MagicMock

import uvicorn
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from fastapi import Body
from omegaconf import OmegaConf

import nemo_gym.client.http_client
import nemo_gym.openai_utils as openai_utils_module
import nemo_gym.server_utils
from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.artifacts.rollout_rows import NG_FAILURE_CLASS_KEY
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import AgentServerRef, BaseServerConfig
from nemo_gym.config.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.failure_transport import FailureReportRow
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.rollout_collection import RolloutCollectionHelper


MODEL_REPLY = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "m",
    "status": "completed",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}


class _ModelAgent(SimpleResponsesAPIAgent):
    model_url: str = ""

    async def responses(self, body: dict = Body(default_factory=dict)) -> dict:
        client = NeMoGymAsyncOpenAI(api_key="k", base_url=self.model_url, max_http_attempts=2)
        return await client.create_response(model="m", input="hi")

    async def run(self, body: dict = Body(default_factory=dict)) -> dict:
        await self.responses({})  # In process, as agents call their own model loop.
        return {"reward": 1.0, "responses_create_params": {"input": "hi"}}


async def _serve(app) -> tuple[uvicorn.Server, asyncio.Task, int]:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="off", log_level="warning"))
    task = asyncio.create_task(server.serve())
    async with asyncio.timeout(5):
        while not server.started:
            await asyncio.sleep(0.01)
    return server, task, server.servers[0].sockets[0].getsockname()[1]


async def _collect(monkeypatch, statuses: list[int]):
    model_hits = []
    observed = []

    async def model(request):
        model_hits.append(request.path)
        status = statuses[min(len(model_hits) - 1, len(statuses) - 1)]
        if status != 200:
            return web.json_response({"error": "busy"}, status=status)
        return web.json_response(MODEL_REPLY)

    real_observe = openai_utils_module.observe_failure
    monkeypatch.setattr(openai_utils_module, "observe_failure", lambda e: (observed.append(e), real_observe(e)))
    monkeypatch.setattr(openai_utils_module, "_retry_wait_seconds", lambda *args: 0)

    app = web.Application()
    app.router.add_route("POST", "/{tail:.*}", model)
    servers = []
    async with TestServer(app) as model_server, ClientSession() as session:
        monkeypatch.setattr(nemo_gym.client.http_client, "_GLOBAL_AIOHTTP_CLIENT", session)
        agent = _ModelAgent(
            config=BaseResponsesAPIAgentConfig(host="127.0.0.1", port=0, entrypoint="", name="my_agent"),
            server_client=MagicMock(spec=ServerClient),
            model_url=str(model_server.make_url("/v1")),
        )
        agent_app = agent.setup_webserver()
        agent.setup_exception_middleware(agent_app)
        servers.append(await _serve(agent_app))
        agent_port = servers[-1][2]
        global_config = {
            "my_agent": {"responses_api_agents": {"impl": {"host": "127.0.0.1", "port": agent_port}}},
            "environment": {
                "environment_servers": {
                    "legacy_agent": {"agent_server": {"name": "my_agent"}, "host": "127.0.0.1", "port": 0}
                }
            },
        }
        env_client = ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
            global_config_dict=OmegaConf.create(global_config),
        )
        env = LegacyAgentEnvironmentServer(
            config=LegacyAgentEnvironmentServerConfig(
                name="environment",
                host="127.0.0.1",
                port=0,
                entrypoint="app.py",
                agent_server=AgentServerRef(type="responses_api_agents", name="my_agent"),
            ),
            server_client=env_client,
        )
        env_app = env.setup_webserver()
        env.setup_exception_middleware(env_app)
        servers.append(await _serve(env_app))
        global_config["environment"]["environment_servers"]["legacy_agent"]["port"] = servers[-1][2]
        collector_client = ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
            global_config_dict=OmegaConf.create(global_config),
        )
        monkeypatch.setattr(RolloutCollectionHelper, "setup_server_client", lambda *args: collector_client)
        try:
            row = {
                AGENT_REF_KEY_NAME: {"name": "my_agent"},
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "responses_create_params": {"input": "hi"},
            }
            (future,) = RolloutCollectionHelper()._run_examples_with_metadata([row], route_failures_to_sidecar=True)
            completed = await asyncio.wait_for(future, timeout=20)
        finally:
            for server, task, _ in servers:
                server.should_exit = True
                await task
    merged = FailureReportRow.merged_into(dict(completed.result), completed.failure_report)
    print("E2E_ROW", statuses, merged)
    return merged, model_hits, observed


async def test_a_recovered_model_call_is_a_success_row_with_the_observation_and_no_failure_id(monkeypatch) -> None:
    row, model_hits, observed = await _collect(monkeypatch, [503, 200])

    assert row["reward"] == 1.0 and NG_FAILURE_CLASS_KEY not in row and "_ng_failure_id" not in row
    assert len(model_hits) == 2  # One retry by the model client; one episode, no collection retry.
    assert [e.event.event_id for e in observed] == list(
        row[FailureReportRow.EVENTS]
    )  # Same id at origin and in the row.


async def test_an_exhausted_model_call_reaches_the_row_through_agent_and_environment(monkeypatch) -> None:
    row, model_hits, observed = await _collect(monkeypatch, [502])

    assert row[NG_FAILURE_CLASS_KEY] and "reward" not in row
    assert len(model_hits) == 2  # max_http_attempts; no extra call from any boundary.
    events = row[FailureReportRow.EVENTS]
    own = events[row["_ng_failure_id"]]
    assert own["boundary"] == "collection"
    agents = [e for e in events.values() if e.get("boundary") == "agent"]
    assert (
        len(agents) == 1
        and agents[0]["operation"] == "agent.run"
        and agents[0]["failure_kind"] == "agent_route_failed"
    )
    for failure in observed:
        assert failure.event.event_id in events  # The recorded model attempts keep their ids across every hop.
    chain, current = [], row["_ng_failure_id"]
    while current is not None:
        chain.append(current)
        current = events[current].get("cause_id")
    # The explicit chain ends at the final attempt's own ModelFailure; the earlier attempt stays a sibling.
    assert chain[-1] == observed[-1].event.event_id and observed[0].event.event_id not in chain
    assert sum(1 for e in events.values() if e.get("boundary") == "model") == len(observed) == 2
