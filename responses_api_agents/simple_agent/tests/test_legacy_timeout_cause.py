# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scripted provider sockets through VLLM, SimpleAgent, MCQA, environment and Collection/Store; no inference."""

import asyncio
import json
import socket
from pathlib import Path

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from omegaconf import OmegaConf

import nemo_gym.client.http_client as http_client
import nemo_gym.openai_utils as model_client
import nemo_gym.rollout_collection as collection
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from nemo_gym.failure_transport import FailureReportRow
from nemo_gym.rollout_store import RolloutStore
from resources_servers.equivalence_llm_judge.app import LLMJudgeResourcesServer, LLMJudgeResourcesServerConfig
from resources_servers.mcqa.app import MCQAResourcesServer, MCQAResourcesServerConfig
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
from tests.unit_tests.test_agent_environment_collection_failures import _serve


@pytest.mark.parametrize(
    "legacy,resource_kind,mode",
    [
        (legacy, "mcqa", mode)
        for legacy in (False, True)
        for mode in (
            "503",
            "503_retried",
            "503_agent_retried",
            "timeout",
            "disconnect",
            "connect",
            "timeout_retry",
            "timeout_env_fail",
            "wrong",
            "healthy",
            "length",
        )
    ]
    + [(legacy, "equivalence", mode) for legacy in (False, True) for mode in ("503", "timeout")],
)
async def test_policy_failure_survives_vllm_agent_environment_store(
    tmp_path, monkeypatch, legacy, resource_kind, mode
):
    hits = []
    verifier_calls = []

    async def provider(request):
        await request.json()
        hits.append(1)
        if mode.startswith("timeout"):
            await asyncio.sleep(4.3)
        if mode == "disconnect":
            request.transport.close()
            return web.Response()
        if mode.startswith("503"):
            return web.json_response(
                {"error": {"message": "scripted unavailable", "type": "server_error"}}, status=503
            )
        return web.json_response(
            {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1,
                "model": "fake-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": ("\\boxed{A}" if mode == "healthy" else "\\boxed{B}"),
                        },
                        "finish_reason": ("length" if mode == "length" else "stop"),
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", provider)
    monkeypatch.setattr(model_client, "_retry_wait_seconds", lambda *a: 0)
    servers = []
    config = {"observability_enabled": True, "model_call_capture_dir": str(tmp_path / "captures")}

    def client():
        return ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=OmegaConf.create(config)
        )

    async def start(server, name, kind, impl):
        application = server.setup_webserver()
        if name == "resources":

            @application.middleware("http")
            async def record_verifier(request, call_next):
                if request.url.path.endswith("/verify"):
                    verifier_calls.append(request.url.path)
                return await call_next(request)

        server.setup_exception_middleware(application)
        servers.append(await _serve(application))
        port = servers[-1][2]
        config[name] = {kind: {impl: {"host": "127.0.0.1", "port": port}}}
        return port

    async with TestServer(app) as upstream, ClientSession() as session:
        monkeypatch.setattr(http_client, "_GLOBAL_AIOHTTP_CLIENT", session)
        try:
            provider_url = str(upstream.make_url("/v1"))
            if mode == "connect":
                with socket.socket() as reserved:
                    reserved.bind(("127.0.0.1", 0))
                    port = reserved.getsockname()[1]
                provider_url = f"http://127.0.0.1:{port}/v1"
            model = VLLMModel(
                config=VLLMModelConfig(
                    host="127.0.0.1",
                    port=0,
                    entrypoint="",
                    name="model",
                    base_url=provider_url,
                    api_key="dummy",
                    model="fake-model",
                    return_token_id_information=False,
                    uses_reasoning_parser=False,
                    uses_interleaved_reasoning=False,
                    max_http_attempts=2 if mode == "503_retried" else 1,
                    max_num_tries=None if mode == "503_retried" else 1,
                    request_timeout_seconds=4,
                ),
                server_client=client(),
            )
            await start(model, "model", "responses_api_models", "scripted")
            resource = MCQAResourcesServer(
                config=MCQAResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="", name="resources"),
                server_client=client(),
            )
            if resource_kind == "equivalence":
                resource = LLMJudgeResourcesServer(
                    config=LLMJudgeResourcesServerConfig(
                        host="127.0.0.1",
                        port=0,
                        entrypoint="",
                        name="resources",
                        judge_model_server={"name": "not_called", "type": "responses_api_models"},
                        judge_responses_create_params={"input": []},
                        judge_prompt_template_fpath=str(
                            Path(__file__).parents[3]
                            / "resources_servers/equivalence_llm_judge/prompt_templates/equivalence_llm_judge.txt"
                        ),
                    ),
                    server_client=client(),
                )
            await start(
                resource,
                "resources",
                "resources_servers",
                "mcqa" if resource_kind == "mcqa" else "equivalence_llm_judge",
            )
            agent = SimpleAgent(
                config=SimpleAgentConfig(
                    host="127.0.0.1",
                    port=0,
                    entrypoint="",
                    name="agent",
                    final_on_model_timeout=mode != "timeout_retry",
                    max_model_retries=1 if mode == "503_agent_retried" else 0,
                    model_server={"name": "model", "type": "responses_api_models"},
                    resources_server={"name": "resources", "type": "resources_servers"},
                ),
                server_client=client(),
            )
            await start(agent, "agent", "responses_api_agents", "simple_agent")
            config["agent"]["responses_api_agents"]["simple_agent"].update(
                {k: v for k, v in agent.config.model_dump(mode="json").items() if k not in {"host", "port"}}
            )
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
            if mode == "timeout_env_fail":
                env.config.agent_outcome_policy = env.config.agent_outcome_policy.model_validate(
                    {"by_failure_kind": {"model_policy_response_timeout": "fail"}}
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
            (tmp_path / "artifacts").mkdir()
            out = tmp_path / "artifacts" / "rollouts.jsonl"
            row = (
                {
                    "agent_ref": {"name": "agent"},
                    "task_source": "resources",
                    "instance_id": "task",
                    "responses_create_params": {"input": "fixture"},
                    "expected_answer": "A",
                    "options": [{"A": "first"}, {"B": "second"}],
                }
                if legacy
                else {
                    "task_id": {"taskset": "resources", "task_id": "task"},
                    "task_input": {
                        "responses_create_params": {"input": "fixture"},
                        "task_data": {
                            "instance_id": "task",
                            "expected_answer": "A",
                            "options": [{"A": "first"}, {"B": "second"}],
                        },
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
                    assert "has no score to report" in str(e), str(e)

            await collect(cfg)
            store = RolloutStore.read(out)
            rows = list(store.selected("success")) + list(store.failures())
            assert len(rows) == 1
            saved = rows[0]
            report = FailureReportRow.report(saved)
            before = len(hits)
            await collect(cfg.model_copy(update={"resume_from_cache": True}))
            proof = {
                "mode": mode,
                "legacy": legacy,
                "resources": resource_kind,
                "calls_before": before,
                "calls_after": len(hits),
                "reward": saved.get("reward"),
                "failure_kind": saved.get("_ng_failure_kind"),
                "failure_id": saved.get("_ng_failure_id"),
                "counts_as_zero": saved.get("_ng_failure_counts_as_zero"),
                "events": {k: v.model_dump(mode="json") for k, v in report.events.items()},
                "pending": len(store.pending()),
                "verifier_calls": len(verifier_calls),
            }
            (tmp_path / "persisted-proof.json").write_text(json.dumps(proof, indent=2))
            assert len(verifier_calls) == (1 if mode in {"wrong", "healthy", "length"} else 0)
            if mode in {"wrong", "healthy", "length"}:
                assert saved["reward"] == (1 if mode == "healthy" else 0) and len(hits) == 1
            elif mode in {"disconnect", "connect"}:
                assert saved.get("_ng_failure_counts_as_zero") is not True
                assert "reward" not in saved and len(hits) == (1 if mode == "disconnect" else 0)
                assert len(store.pending()) == (1 if mode == "connect" else 0)
                assert any(event.request_upload_complete is (mode == "disconnect") for event in report.events.values())
                coverage = json.loads(out.with_name("rollouts_coverage.json").read_text())
                assert coverage["attempts"] == (2 if mode == "connect" else 1)
                assert coverage["max_rollout_attempts"] == 3
            elif mode in {"timeout_retry", "timeout_env_fail"}:
                assert saved.get("_ng_failure_kind") == "model_policy_response_timeout"
                assert saved.get("_ng_failure_counts_as_zero") is not True and "reward" not in saved
                assert len(hits) == (2 if mode == "timeout_retry" else 1)
                assert len(store.pending()) == (1 if mode == "timeout_retry" else 0)
            elif mode == "timeout":
                assert saved.get("failure_kind") == "model_policy_response_timeout", proof
                assert saved["_ng_failure_counts_as_zero"] is True
                [metric_row] = collection._counted_failure_rows([saved], [])
                assert metric_row["reward"] == 0
                coverage = json.loads(out.with_name("rollouts_coverage.json").read_text())
                assert coverage["scored"] == coverage["expected"] == 1
                assert coverage["final_zeros"] == 1 and coverage["complete"] is True
                assert len(hits) == 1
            else:
                assert saved.get("_ng_failure_id") is not None, proof
                assert saved.get("_ng_failure_kind") == "model_request_error", proof
                assert len(hits) == (4 if mode in {"503_retried", "503_agent_retried"} else 2)
                if mode in {"503_retried", "503_agent_retried"}:
                    attempts = [event for event in report.events.values() if event.operation == "model_request"]
                    assert len(attempts) == 2 and len({event.attempt_id for event in attempts}) == 2
                    assert len({event.operation_id for event in attempts}) == (2 if mode == "503_agent_retried" else 1)
                    assert all(event.http_status == 503 for event in attempts)
        finally:
            for srv, task, _ in reversed(servers):
                srv.should_exit = True
                await task
