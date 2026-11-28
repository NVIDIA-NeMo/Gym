# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the configured adaptive stack over HTTP, including session cleanup."""

import asyncio
import copy
import json
from contextlib import AsyncExitStack, asynccontextmanager

import pytest
import uvicorn
from aiohttp import web
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf

from benchmarks.agentic_if_realworld import rollout_driver
from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.client import ServerClient, http_client
from nemo_gym.config import BaseServerConfig, GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper, run_health_checks
from resources_servers.agentic_if_realworld.app import ResourcesConfig, ResourcesServer
from resources_servers.agentic_if_realworld.benchmark import core
from resources_servers.agentic_if_realworld.benchmark.synthetic import create
from resources_servers.agentic_if_realworld.benchmark.transport import RetryableTransportError
from resources_servers.agentic_if_realworld.tests.test_transport import indexed_review
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


@asynccontextmanager
async def serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["complete", "incomplete", "disconnect", "overflow", "judge_outage"])
async def test_configured_session_driver(tmp_path, monkeypatch, unused_tcp_port_factory, delivery):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEMO_GYM_EXTRA_ROOTS", raising=False)
    monkeypatch.setenv("AGENTIC_IF_JUDGE_API_KEY", "synthetic")
    runtime = core.runtime_identity()
    monkeypatch.setattr(rollout_driver, "runtime_identity", lambda: copy.deepcopy(runtime))
    create(tmp_path / "data")
    dataset = core.Dataset(tmp_path / "data", synthetic=True)
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("".join(json.dumps(row) + "\n" for row in dataset.prepared()))
    receipts = tmp_path / "receipts"
    captures = tmp_path / "captures"
    policy_calls, judge_calls, events = [], [], []

    async def upstream(request):
        body = await request.json()
        if request.path == "/judge":
            judge_calls.append(body)
            payload = core.read(next((receipts / "judges").glob("*/payload.json")))
            return web.json_response(
                {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(indexed_review(payload))}}]}
            )
        policy_calls.append(body)
        assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
        content = body["messages"][0]["content"]
        if isinstance(content, list):
            content = "".join(part["text"] for part in content)
        assert content == dataset.request(next(iter(dataset.inputs)))["messages"][0]["content"]
        assert body["temperature"] == 0.7 and not body.get("tools")
        assert body["chat_template_kwargs"] == {"reasoning_effort": "max"}
        if delivery == "disconnect":
            request.transport.close()
            return web.Response()
        if delivery == "overflow":
            return web.json_response(
                {
                    "error": {
                        "message": "maximum context length is 65536 tokens, but requested 72768 tokens (40000 in the messages, 32768 in the completion)"
                    }
                },
                status=400,
            )
        incomplete = delivery == "incomplete" and len(policy_calls) == 1
        return web.json_response(
            {
                "id": "synthetic-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "synthetic-policy",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length" if incomplete else "stop",
                        "message": {
                            "role": "assistant",
                            "content": "partial" if incomplete else "OK.",
                            "reasoning": "Think first",
                            "prompt_token_ids": [1, 2],
                            "generation_token_ids": [3, 4],
                            "generation_log_probs": [-0.1, -0.2],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            }
        )

    upstream_port = unused_tcp_port_factory()
    backend = web.Application()
    backend.router.add_post("/v1/chat/completions", upstream)
    backend.router.add_post("/judge", upstream)
    runner = web.AppRunner(backend)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", upstream_port).start()
    initial = {
        "observability_enabled": True,
        "model_call_capture_dir": str(captures),
        "config_paths": [
            "benchmarks/agentic_if_realworld/config.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        "policy_base_url": f"http://127.0.0.1:{upstream_port}/v1",
        "policy_model_name": "synthetic-policy",
        "policy_api_key": "synthetic",
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "chat_template_kwargs": {"reasoning_effort": "max"},
                    "return_token_id_information": True,
                }
            }
        },
        "agentic_if_realworld_resources": {
            "resources_servers": {
                "agentic_if_realworld": {
                    "dataset_dir": str(dataset.directory),
                    "synthetic": True,
                    "receipt_dir": str(receipts),
                    "judge_endpoint": f"http://127.0.0.1:{upstream_port}/judge",
                    "judge_model": "synthetic-judge",
                }
            }
        },
    }
    settings = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(initial),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    components = [
        (
            "agentic_if_realworld_resources",
            "resources_servers",
            "agentic_if_realworld",
            ResourcesConfig,
            ResourcesServer,
        ),
        ("policy_model", "responses_api_models", "vllm_model", VLLMModelConfig, VLLMModel),
        ("agentic_if_realworld_agent", "responses_api_agents", "simple_agent", SimpleAgentConfig, SimpleAgent),
        (
            "agentic_if_realworld_environment",
            "environment_servers",
            "single_agent_turn_legacy",
            SingleAgentTurnLegacyEnvironmentServerConfig,
            SingleAgentTurnLegacyEnvironmentServer,
        ),
    ]
    for name, group, kind, _, _ in components:
        settings[name][group][kind].update(host="127.0.0.1", port=unused_tcp_port_factory())
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=0), global_config_dict=settings)
    monkeypatch.setattr(RolloutCollectionHelper, "setup_server_client", lambda self, *args: client)
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(prepared), output_jsonl_fpath=str(tmp_path / "out.jsonl"), num_samples_in_parallel=1
    )

    def observe(app, name):
        @app.middleware("http")
        async def record(request, call_next):
            if request.method == "POST":
                body = await request.json()
                events.append((name, request.url.path, body))
                if (
                    delivery == "judge_outage"
                    and request.url.path.endswith("/verify")
                    and sum(path.endswith("/verify") for _, path, _ in events) == 1
                ):
                    return JSONResponse(
                        {"failure_kind": "judge_failed", "failure_reason": "synthetic judge outage before dispatch"}
                    )
            return await call_next(request)

    try:
        async with AsyncExitStack() as stack:
            for name, group, kind, config_type, server_type in components:
                server_config = config_type.model_validate(
                    {**OmegaConf.to_container(settings[name][group][kind], resolve=True), "name": name}
                )
                app = server_type(config=server_config, server_client=client).setup_webserver()
                observe(app, name)
                await stack.enter_async_context(serve(app, server_config.port))
            if delivery == "disconnect":
                with pytest.raises(ValueError, match="[Aa]mbiguous|reconcil"):
                    await rollout_driver.run_rollout_collection(config, settings)
                assert len(policy_calls) == 1 and not judge_calls
                saved = {
                    p: p.read_bytes()
                    for parent in [receipts, captures]
                    for p in parent.rglob("*")
                    if p.is_file() and p.name != ".lock"
                }
                config.resume_from_cache = True
                with pytest.raises(ValueError, match="[Aa]mbiguous|reconcil"):
                    await rollout_driver.run_rollout_collection(config, settings)
                assert len(policy_calls) == 1 and not judge_calls
                assert all(p.read_bytes() == value for p, value in saved.items())
            else:
                if delivery == "judge_outage":
                    # A typed judge failure defers the task without failing the whole run.
                    with pytest.raises(RetryableTransportError, match="synthetic judge outage"):
                        await rollout_driver.run_rollout_collection(config, settings)
                    assert len(policy_calls) == 1 and not judge_calls
                    config.resume_from_cache = True
                await rollout_driver.run_rollout_collection(config, settings)
                expected = core.PHASES[:2] if delivery == "incomplete" else core.PHASES[:1]
                assert [(b["seed"], b["max_tokens"]) for b in policy_calls] == list(expected)
                record = core.read(tmp_path / "out.jsonl")
                assert record["mask_sample"] is (delivery == "overflow")
                assert len(judge_calls) == (0 if delivery == "overflow" else 1)
                if delivery != "overflow":
                    assert record["response"]["output"][-1]["generation_token_ids"] == [3, 4]
                health = run_health_checks(tmp_path / "out.jsonl", workers=1).summary["run"]["artifacts"]
                assert health["healthy"] and health["missing"] == 0
                before = (len(policy_calls), len(judge_calls), len(events))
                saved = {
                    p: p.read_bytes()
                    for parent in [receipts, captures]
                    for p in parent.rglob("*")
                    if p.is_file() and p.name != ".lock"
                }
                config.resume_from_cache = True
                await rollout_driver.run_rollout_collection(config, settings)
                assert (len(policy_calls), len(judge_calls)) == before[:2]
                assert not any(path.endswith("/run") for _, path, _ in events[before[2] :])
                assert all(p.read_bytes() == value for p, value in saved.items())
            for name, seed_path, close_path, key in [
                ("agentic_if_realworld_resources", "/seed_session", "/close_session", "resources_session_id"),
                ("agentic_if_realworld_agent", "/v1/agent_sessions", "/v1/agent_sessions/close", "agent_session_id"),
            ]:
                seeds = [body[key] for server, path, body in events if server == name and path.endswith(seed_path)]
                closes = [body[key] for server, path, body in events if server == name and path.endswith(close_path)]
                assert len(seeds) == len(policy_calls) and sorted(seeds) == sorted(closes)
            for _, path, body in events:
                if path.endswith("/v1/agent_sessions"):
                    assert body["tool_accesses"] == []
            assert not any(name == "agentic_if_realworld_agent" and path.endswith("/run") for name, path, _ in events)
    finally:
        await http_client.close_global_aiohttp_client()
        await runner.cleanup()
