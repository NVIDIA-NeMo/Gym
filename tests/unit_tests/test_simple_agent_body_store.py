# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scripted provider sockets, real SimpleAgent/environment routes and durable Collection/Store; no model inference."""

import asyncio
import json

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from fastapi import Body, FastAPI, Response
from omegaconf import OmegaConf

import nemo_gym.client.http_client as http_client
import nemo_gym.openai_utils as model_client
import nemo_gym.rollout_collection as collection
from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.failure_transport import FailureReportRow
from nemo_gym.rollout_store import RolloutStore
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig
from tests.unit_tests.test_agent_environment_collection_failures import MODEL_REPLY, _serve
from tests.unit_tests.test_responses_api_model_streaming import _EchoModel


class _ProviderModel(_EchoModel):
    provider_url: str = ""

    async def responses(self, body=Body()):
        client = model_client.NeMoGymAsyncOpenAI(
            api_key="test", base_url=self.provider_url, max_http_attempts=2, request_timeout_seconds=0.04
        )
        return NeMoGymResponse.model_validate(await client.create_response(input=[]))


@pytest.mark.parametrize("legacy", [False, True, "agent_run"])
@pytest.mark.parametrize("mode", ["truncated", "timeout", "omitted", "recovered_503"])
async def test_body_failure_survives_actual_simple_agent_environment_store(tmp_path, monkeypatch, legacy, mode):
    hits = []
    observed = []

    async def provider(request):
        hits.append(1)
        if mode == "omitted":
            return web.json_response(
                {"error": "unknown"}, status=500, headers={"X-NG-Failure-Report": '{"omitted":true}'}
            )
        if mode == "recovered_503":
            if len(hits) == 1:
                return web.json_response({"error": "busy"}, status=503)
            return web.json_response(MODEL_REPLY)
        response = web.StreamResponse(headers={"Content-Type": "application/json", "Content-Length": "1000"})
        await response.prepare(request)
        await response.write(b'{"id":')
        if mode == "truncated":
            request.transport.close()
        else:
            await asyncio.sleep(0.12)
        return response

    app = web.Application()
    app.router.add_post("/v1/responses", provider)
    real_observe = model_client.observe_failure
    monkeypatch.setattr(model_client, "observe_failure", lambda e: (observed.append(e), real_observe(e)))
    monkeypatch.setattr(model_client, "_retry_wait_seconds", lambda *a: 0)
    servers = []
    config = {"observability_enabled": True, "model_call_capture_dir": str(tmp_path / "captures")}

    def client():
        return ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=OmegaConf.create(config)
        )

    async def start(server, name, kind, impl):
        application = server.setup_webserver()
        server.setup_exception_middleware(application)
        servers.append(await _serve(application))
        port = servers[-1][2]
        config[name] = {kind: {impl: {"host": "127.0.0.1", "port": port}}}
        return port

    async with TestServer(app) as upstream, ClientSession() as session:
        monkeypatch.setattr(http_client, "_GLOBAL_AIOHTTP_CLIENT", session)
        try:
            model = _ProviderModel(
                config=BaseResponsesAPIModelConfig(host="127.0.0.1", port=0, entrypoint="", name="model"),
                server_client=client(),
                provider_url=str(upstream.make_url("/v1")),
            )
            await start(model, "model", "responses_api_models", "scripted")
            resource_app = FastAPI()

            @resource_app.post("/ng-rollout/{rollout_id}/seed_session")
            @resource_app.post("/seed_session")
            async def seed(body: dict, response: Response):
                response.set_cookie("session", "scripted-resource")
                return {"resources_session_id": body.get("resources_session_id", "legacy-resource")}

            @resource_app.post("/ng-rollout/{rollout_id}/close_session")
            @resource_app.post("/close_session")
            async def close(body: dict):
                return {}

            @resource_app.post("/ng-rollout/{rollout_id}/verify")
            @resource_app.post("/verify")
            async def verify(body: dict):
                return body | {"reward": 1.0}

            servers.append(await _serve(resource_app))
            config["resources"] = {"resources_servers": {"scripted": {"host": "127.0.0.1", "port": servers[-1][2]}}}
            agent = SimpleAgent(
                config=SimpleAgentConfig(
                    host="127.0.0.1",
                    port=0,
                    entrypoint="",
                    name="agent",
                    model_server={"name": "model", "type": "responses_api_models"},
                    resources_server={"name": "resources", "type": "resources_servers"},
                ),
                server_client=client(),
            )
            await start(agent, "agent", "responses_api_agents", "simple_agent")
            kind = "single_agent_turn_legacy" if legacy else "single_agent_turn"
            envtype = SingleAgentTurnLegacyEnvironmentServer if legacy else SingleAgentTurnEnvironmentServer
            env = envtype(
                config=SingleAgentTurnEnvironmentServerConfig(
                    host="127.0.0.1",
                    port=0,
                    entrypoint="",
                    name="environment",
                    cleanup_timeout_seconds=1,
                    agent_server={"name": "agent", "type": "responses_api_agents"},
                    resources_server={"name": "resources", "type": "resources_servers"},
                ),
                server_client=client(),
            )
            if legacy == "agent_run":
                kind = "legacy_agent"
                env = LegacyAgentEnvironmentServer(
                    config=LegacyAgentEnvironmentServerConfig(
                        host="127.0.0.1",
                        port=0,
                        entrypoint="",
                        name="environment",
                        agent_server={"name": "agent", "type": "responses_api_agents"},
                    ),
                    server_client=client(),
                )
            await start(env, "environment", "environment_servers", kind)
            config["environment"]["environment_servers"][kind].update(
                {"agent_server": {"name": "agent"}, "resources_server": {"name": "resources"}}
            )
            agent.server_client.global_config_dict = OmegaConf.create(config)
            collector = client()
            monkeypatch.setattr(collection, "setup_server_client_utils", lambda *a, **kw: collector)
            monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})
            source = tmp_path / "input.jsonl"
            out = tmp_path / "rollouts.jsonl"
            row = (
                {
                    "agent_ref": {"name": "agent"},
                    "task_source": "resources",
                    "instance_id": "task",
                    "responses_create_params": {"input": "fixture"},
                }
                if legacy
                else {
                    "task_id": {"taskset": "resources", "task_id": "task"},
                    "task_input": {
                        "responses_create_params": {"input": "fixture"},
                        "task_data": {"instance_id": "task"},
                    },
                }
            )
            source.write_text(json.dumps(row) + "\n")
            cfg = collection.RolloutCollectionConfig(
                input_jsonl_fpath=str(source),
                output_jsonl_fpath=str(out),
                environment_server_routes={"resources": "environment"} if not legacy else {},
                route_failures_to_sidecar=True,
                disable_aggregation=True,
                disable_health_check=True,
                require_complete=False,
            )

            async def collect(value):
                try:
                    await collection.RolloutCollectionHelper().run_from_config(value)
                except RuntimeError as e:
                    assert mode != "recovered_503" and "has no score to report" in str(e), str(e)

            await collect(cfg)
            store = RolloutStore.read(out)
            if mode == "recovered_503":
                [saved] = store.selected("success")
                assert saved["reward"] == 1.0 and saved.get("_ng_failure_id") is None
                assert len(hits) == 2
            else:
                [saved] = store.failures()
                assert "reward" not in saved
                assert len(hits) == (2 if mode == "omitted" else 1)
                report = FailureReportRow.report(saved)
                assert report is not None and report.failure_id
                chain = []
                current = report.failure_id
                while current is not None:
                    assert current not in chain
                    chain.append(current)
                    current = report.events[current].cause_id
                assert observed[-1].event_id in chain
                assert report.events[observed[-1].event_id].http_status == (500 if mode == "omitted" else 200)
                if mode == "omitted":
                    assert report.omitted
                assert store.pending() == []
            before = len(hits)
            await collect(cfg.model_copy(update={"resume_from_cache": True}))
            assert len(hits) == before
            (tmp_path / "persisted-proof.json").write_text(
                json.dumps(
                    {
                        "legacy": legacy,
                        "mode": mode,
                        "wire_calls": len(hits),
                        "selected_failure_id": saved.get("_ng_failure_id"),
                        "observed_event_ids": [x.event_id for x in observed],
                        "pending": len(store.pending()),
                        "resume_no_new_request": True,
                        "saved_reward": saved.get("reward"),
                    },
                    indent=2,
                )
            )
        finally:
            for srv, task, _ in reversed(servers):
                srv.should_exit = True
                await task
