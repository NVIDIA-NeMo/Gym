# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real adapter routes and durable collection with a scripted model; no inference or sandbox provider."""

import json
import sqlite3
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientSession
from fastapi import FastAPI, Request
from omegaconf import OmegaConf

import nemo_gym.client.http_client as http_client
import nemo_gym.rollout_collection as collection
from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.failure_transport import FailureReportRow
from nemo_gym.rollout_recovery import has_saved_generation
from nemo_gym.rollout_store import RolloutStore
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.typed_exceptions import NgFailureEvent
from resources_servers.bird_sql.app import BirdSqlResourcesServer, BirdSqlResourcesServerConfig
from resources_servers.deepswe.app import DeepSWEResourcesServer, DeepSWESeedSessionRequest, VerifierResult
from resources_servers.deepswe.tests.conftest import task_assets
from resources_servers.deepswe.tests.test_app import _config as deepswe_config
from responses_api_agents.gdp_pdf_agent.app import GdpPdfAgent, GdpPdfAgentConfig
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig
from tests.unit_tests.test_agent_environment_collection_failures import MODEL_REPLY, _serve


class _UnavailableVerifier(SimpleResourcesServer):
    async def verify(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
        raise NgFailureEvent(
            boundary="resources_server",
            failure_kind="verifier_error",
            failure_reason="scripted verifier unavailable",
            operation="verify",
            stage="verification",
        )


@pytest.mark.parametrize(
    "case",
    [
        "bird_missing",
        "bird_corrupt",
        "bird_wrong",
        "bird_correct",
        "gdp",
        "deep_verifier",
        "deep_no_patch",
        "deep_missing",
    ],
)
async def test_adapter_failure_keeps_saved_answer_and_resume_never_regenerates(tmp_path, monkeypatch, case):
    servers = []
    hits = []
    config = {"observability_enabled": True, "model_call_capture_dir": str(tmp_path / "captures")}
    answer = dict(MODEL_REPLY) | {
        "output": [
            {
                "type": "message",
                "id": "answer",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "```sql\nSELECT 1\n```", "annotations": []}],
            }
        ]
    }

    def client():
        return ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=OmegaConf.create(config)
        )

    async def start(server, name, kind, impl):
        app = server.setup_webserver()
        server.setup_exception_middleware(app)
        servers.append(await _serve(app))
        config[name] = {kind: {impl: {"host": "127.0.0.1", "port": servers[-1][2]}}}

    row = {
        "agent_ref": {"name": "agent"},
        "task_source": "resources",
        "instance_id": "task",
        "responses_create_params": {"input": [{"role": "user", "content": "What is the answer?"}]},
    }
    model_app = FastAPI()

    @model_app.post("/ng-rollout/{rollout_id}/v1/responses")
    @model_app.post("/v1/responses")
    @model_app.post("/ng-rollout/{rollout_id}/responses")
    @model_app.post("/responses")
    async def model(body: dict):
        hits.append(body)
        return answer

    async with ClientSession() as session:
        monkeypatch.setattr(http_client, "_GLOBAL_AIOHTTP_CLIENT", session)
        try:
            servers.append(await _serve(model_app))
            config["model"] = {"responses_api_models": {"scripted": {"host": "127.0.0.1", "port": servers[-1][2]}}}
            agent_type, cfg_type = SimpleAgent, SimpleAgentConfig
            extra = {}
            if case.startswith("bird"):
                monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _: tmp_path)
                resource = BirdSqlResourcesServer(
                    config=BirdSqlResourcesServerConfig(
                        host="127.0.0.1", port=0, entrypoint="", name="resources", bird_sql_dir=str(tmp_path)
                    ),
                    server_client=client(),
                )
                row.update(
                    db_id="fixture",
                    gt_sql="SELECT 2" if case == "bird_wrong" else "SELECT 1",
                    question="q",
                    difficulty="simple",
                    id=0,
                )
                if case != "bird_missing":
                    directory = tmp_path / "fixture"
                    directory.mkdir()
                    database = directory / "fixture.sqlite"
                    if case == "bird_corrupt":
                        database.write_bytes(b"not a sqlite database")
                        row["gt_sql"] = "SELECT * FROM missing"
                    else:
                        sqlite3.connect(database).close()
            elif case == "gdp":
                resource = _UnavailableVerifier(
                    config=BaseResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="", name="resources"),
                    server_client=client(),
                )
                document = tmp_path / "document.json"
                document.write_text(
                    json.dumps({"source_dpi": 150, "pages": [{"page_number": 1, "text": "One answer."}]})
                )
                row["verifier_metadata"] = {
                    "document_manifest": document.name,
                    "task_id": "task",
                    "criteria": [{"index": 1}],
                }
                agent_type, cfg_type = GdpPdfAgent, GdpPdfAgentConfig
                extra = {
                    "documents_base_dir": str(tmp_path),
                    "include_images": False,
                    "strip_images_from_output": False,
                }
            else:
                assets = task_assets.__wrapped__(tmp_path)
                cfg = deepswe_config(assets, golden=False)
                cfg.logs_dir = tmp_path / "logs"
                cfg.include_model_patch_in_response = case != "deep_no_patch"
                resource = DeepSWEResourcesServer(config=cfg, server_client=client())
                sandbox = AsyncMock()
                sandbox.serialize.return_value = {"sandbox_id": "scripted-sandbox"}
                monkeypatch.setattr(resource, "_create_sandbox", AsyncMock(return_value=sandbox))
                monkeypatch.setattr(
                    resource, "_collect_model_patch", AsyncMock(return_value=b"diff --git a/a b/a\n+real answer\n")
                )
                monkeypatch.setattr(
                    resource,
                    "_run_verifier",
                    AsyncMock(
                        return_value=VerifierResult(
                            evaluation_completed=False, reward=0, verifier_error="missing reward artifact"
                        )
                    ),
                )
                row.update(
                    task_id="example-task",
                    image="public.example/project/example-task:v1.1",
                    verifier_metadata={"task_id": "example-task"},
                )
                if case == "deep_missing":
                    original = resource.seed_session

                    async def lose_session(self, request: Request, body: DeepSWESeedSessionRequest):
                        result = await original(request, body)
                        resource._agent_sessions.clear()
                        return result

                    monkeypatch.setattr(DeepSWEResourcesServer, "seed_session", lose_session)
            await start(resource, "resources", "resources_servers", "scripted")
            agent = agent_type(
                config=cfg_type(
                    host="127.0.0.1",
                    port=0,
                    entrypoint="",
                    name="agent",
                    max_steps=1,
                    model_server={"name": "model", "type": "responses_api_models"},
                    resources_server={"name": "resources", "type": "resources_servers"},
                    **extra,
                ),
                server_client=client(),
            )
            await start(agent, "agent", "responses_api_agents", "scripted")
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
            await start(env, "environment", "environment_servers", "legacy_agent")
            config["environment"]["environment_servers"]["legacy_agent"]["agent_server"] = {"name": "agent"}
            for producer in (resource, agent, env):
                producer.server_client.global_config_dict = OmegaConf.create(config)
            collector = client()
            monkeypatch.setattr(collection, "setup_server_client_utils", lambda *a, **kw: collector)
            monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})
            source, out = tmp_path / "input.jsonl", tmp_path / "rollouts.jsonl"
            source.write_text(json.dumps(row) + "\n")
            cfg = collection.RolloutCollectionConfig(
                input_jsonl_fpath=str(source),
                output_jsonl_fpath=str(out),
                route_failures_to_sidecar=True,
                disable_aggregation=True,
                disable_health_check=True,
                require_complete=False,
            )

            async def collect(value):
                try:
                    await collection.RolloutCollectionHelper().run_from_config(value)
                except RuntimeError as error:
                    assert "has no score to report" in str(error), str(error)

            await collect(cfg)
            assert len(hits) == 1
            store = RolloutStore.read(out)
            if case in {"bird_correct", "bird_wrong"}:
                [saved] = store.selected("success")
                assert saved["reward"] == (1 if case == "bird_correct" else 0)
                assert not list(store.failures())
            else:
                [saved] = store.failures() or store.selected("success")
                assert saved.get("mask_sample") is True
                assert not saved.get("_ng_failure_counts_as_zero", False)
                retained = (
                    saved.get("response")
                    or saved.get("_ng_failure_response")
                    or saved.get("_ng_failure_partial_response")
                )
                assert NeMoGymResponse.model_validate(retained) == NeMoGymResponse.model_validate(answer)
                assert retained["id"] == answer["id"]
                assert has_saved_generation(saved)
                coverage = json.loads(out.with_name("rollouts_coverage.json").read_text())
                assert coverage["measured"] == 0 and coverage["masked"] == 1
                assert coverage["final_zeros"] == 0
                report = FailureReportRow.report(saved)
                assert report.events
                kinds = {event.failure_kind for event in report.events.values()}
                assert ("session_lost" if case == "deep_missing" else "verifier_error") in kinds
                if case == "gdp":
                    assert any(
                        e.failure_kind == "agent_route_failed" and e.cause_id in report.events
                        for e in report.events.values()
                    )
                    assert saved.get("ng_trajectory") is not None
                if case.startswith("deep"):
                    assert saved.get("model_patch") == (
                        "diff --git a/a b/a\n+real answer\n" if case == "deep_verifier" else None
                    )
                    if case == "deep_missing":
                        resource._run_verifier.assert_not_awaited()
            before = out.read_bytes() if out.exists() else b""
            await collect(cfg.model_copy(update={"resume_from_cache": True}))
            assert len(hits) == 1
            assert (out.read_bytes() if out.exists() else b"") == before
        finally:
            for server, task, _ in reversed(servers):
                server.should_exit = True
                await task
