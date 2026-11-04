# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real adapter routes and durable collection with a scripted model; no inference or sandbox provider."""

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import uvicorn
from aiohttp import ClientSession
from fastapi import FastAPI, Request
from omegaconf import OmegaConf

import nemo_gym.client.http_client as http_client
import nemo_gym.rollouts.rollout_collection as collection
from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from nemo_gym.core import FailureReportRow, NeMoGymResponse, NgFailureEvent
from nemo_gym.rollouts import RolloutStore
from nemo_gym.rollouts.rollout_recovery import has_saved_generation
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.bird_sql.app import BirdSqlResourcesServer, BirdSqlResourcesServerConfig
from resources_servers.deepswe.app import (
    DeepSWEResourcesServer,
    DeepSWEResourcesServerConfig,
    DeepSWESeedSessionRequest,
    VerifierResult,
)
from resources_servers.false_statement_judge.app import FalseStatementJudgeConfig, FalseStatementJudgeServer
from responses_api_agents.gdp_pdf_agent.app import GdpPdfAgent, GdpPdfAgentConfig
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


MODEL_REPLY = {
    "id": "scripted-response",
    "object": "response",
    "created_at": 0,
    "model": "scripted",
    "status": "completed",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}


async def _serve(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="off", log_level="warning"))
    task = asyncio.create_task(server.serve())
    async with asyncio.timeout(5):
        while not server.started:
            await asyncio.sleep(0.01)
    return server, task, server.servers[0].sockets[0].getsockname()[1]


def _task_assets(tmp_path: Path) -> Path:
    task_id = "example-task"
    task_dir = tmp_path / "tasks" / task_id
    for directory in (task_dir / "tests", task_dir / "solution"):
        directory.mkdir(parents=True)

    (task_dir / "task.toml").write_text(
        """schema_version = "1.3"
[task]
name = "datacurve/example-task"
description = ""
authors = []
keywords = []
[metadata]
task_id = "example-task"
repository_url = "https://github.com/example/repo"
base_commit_hash = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
language = "python"
[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 1800.0
[verifier.environment]
cpus = 3
memory_mb = 12288
storage_mb = 25600
[[verifier.collect]]
command = "cd /app && mkdir -p /logs/artifacts && git config --global --add safe.directory /app && git diff --binary 0123456789abcdef0123456789abcdef01234567 HEAD > /logs/artifacts/model.patch"
timeout_sec = 300.0
[agent]
network_mode = "no-network"
timeout_sec = 5400.0
[environment]
docker_image = "public.example/project/example-task:v1.1"
cpus = 2
memory_mb = 8192
storage_mb = 20480
""",
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Implement the feature and commit it.\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (task_dir / "tests" / "test.patch").write_text("", encoding="utf-8")
    (task_dir / "tests" / "grader.py").write_text("", encoding="utf-8")
    (task_dir / "tests" / "config.json").write_text("{}\n", encoding="utf-8")
    (task_dir / "solution" / "solution.patch").write_text("golden patch\n", encoding="utf-8")

    return task_dir.parent


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
        "broken_missing",
        "broken_zero",
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
            elif case.startswith("broken"):
                resource = FalseStatementJudgeServer(
                    config=FalseStatementJudgeConfig(
                        host="127.0.0.1",
                        port=0,
                        entrypoint="",
                        name="resources",
                        judge_model_server={"name": "judge", "type": "responses_api_models"},
                        judge_responses_create_params={"input": []},
                    ),
                    server_client=client(),
                )
                monkeypatch.setattr(
                    resource,
                    "_call_judge",
                    AsyncMock(return_value="<points>0</points>" if case == "broken_zero" else "unusable verdict"),
                )
                row.update(problem="False claim", original_problem="Original claim")
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
                assets = _task_assets(tmp_path)
                cfg = DeepSWEResourcesServerConfig(
                    host="127.0.0.1",
                    port=0,
                    entrypoint="",
                    name="resources",
                    tasks_dir=assets,
                    expected_task_count=1,
                    sandbox_provider="scripted",
                    sandbox_config={},
                )
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
            if case in {"bird_correct", "bird_wrong", "broken_zero"}:
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
                assert coverage["measured"] == 0
                assert coverage["masked"] == (0 if case == "broken_missing" else 1)
                assert coverage["final_zeros"] == 0
                report = FailureReportRow.report(saved)
                assert report.events
                kinds = {event.failure_kind for event in report.events.values()}
                expected_kind = (
                    "session_lost"
                    if case == "deep_missing"
                    else "judge_failed"
                    if case == "broken_missing"
                    else "verifier_error"
                )
                assert expected_kind in kinds
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
            retained = (
                saved.get("response") or saved.get("_ng_failure_response") or saved.get("_ng_failure_partial_response")
            )
            canonical = NeMoGymResponse.model_validate(retained).model_dump(mode="json")
            expected = NeMoGymResponse.model_validate(answer).model_dump(mode="json")
            assert canonical == expected
            answer_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
            before = out.read_bytes() if out.exists() else b""
            await collect(cfg.model_copy(update={"resume_from_cache": True}))
            assert len(hits) == 1
            assert (out.read_bytes() if out.exists() else b"") == before
            (tmp_path / "saved-answer-proof.json").write_text(
                json.dumps(
                    {
                        "case": case,
                        "model_calls_before_resume": 1,
                        "model_calls_after_resume": len(hits),
                        "canonical_response_sha256": answer_hash,
                        "response_unchanged": True,
                        "rows_byte_unchanged_on_resume": True,
                        "mask_sample": saved.get("mask_sample"),
                        "reward": saved.get("reward"),
                        "failure_kind": saved.get("failure_kind"),
                        "has_saved_generation": has_saved_generation(saved),
                    }
                )
            )
        finally:
            for server, task, _ in reversed(servers):
                server.should_exit = True
                await task
