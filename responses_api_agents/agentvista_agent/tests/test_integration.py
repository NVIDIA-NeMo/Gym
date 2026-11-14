# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate adapters against actual Gym schemas; no hosted inference or GPU work."""

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest, ModelServerRef
from resources_servers.agentvista.app import AgentVistaResources, AgentVistaResourcesConfig
from responses_api_agents.agentvista_agent.app import (
    AgentVistaAgent,
    AgentVistaConfig,
    AgentVistaRunRequest,
)
from responses_api_agents.agentvista_agent.runtime import HARNESS_VERSION
from responses_api_agents.agentvista_agent.runtime.results import (
    AgentVistaResults,
    AgentVistaVerifyRequest,
    policy_response,
)


def formatted(agent, body, result):
    response = policy_response(body, result, agent.config.model_name)
    verify = AgentVistaVerifyRequest(**body.model_dump(), response=response)
    return AgentVistaResults.format_result(agent, verify, result)


def metrics_owner(agent):
    config = AgentVistaResourcesConfig(
        host="127.0.0.1",
        port=1235,
        entrypoint="app.py",
        name="agentvista_resources",
        source_root=agent.config.source_root,
        dataset=agent.config.dataset,
        image_folder=agent.config.image_folder,
        output_root=agent.config.output_root,
        harness=agent.config.harness,
    )
    return AgentVistaResources(config=config, server_client=agent.server_client)


@pytest.fixture
def agent(tmp_path):
    (tmp_path / "infer.py").write_text("")
    config = AgentVistaConfig(
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        name="agentvista_test",
        source_root=str(tmp_path),
        dataset=str(tmp_path / "tasks.json"),
        image_folder=str(tmp_path),
        output_root=str(tmp_path / "episodes"),
        model_name="fixture",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    return AgentVistaAgent(config=config, server_client=MagicMock(spec=ServerClient))


def request(identity="repeat-0", task="fixture"):
    return AgentVistaRunRequest.model_validate(
        {
            "responses_create_params": {"input": "Read image"},
            "_ng_rollout_id": identity,
            "verifier_metadata": {
                "task_id": task,
                "source_sha256": "hash",
                "category_id": "test",
                "harness_version": HARNESS_VERSION,
            },
        }
    )


def result(masked=False):
    return {
        "artifacts": "/fixture",
        "mask_sample": masked,
        "verify": {"status": "FAIL", "reward": 0.0},
        "native_trajectory": [
            {
                "type": "tool_use",
                "part": {
                    "callID": "one",
                    "tool": "agentvista_code_interpreter",
                    "state": {
                        "input": {"code": "print(42)"},
                        "output": "42",
                        "status": "completed",
                    },
                },
            },
            {"type": "text", "part": {"text": "<answer>42</answer>"}},
        ],
    }


def test_real_gym_tool_transcript_and_measured_zero(agent):
    response = formatted(agent, request(), result())
    assert response.reward == 0 and response.mask_sample is False
    assert response.response.output[0].type == "function_call"
    assert response.response.output[1].call_id == "one"
    assert response.response.usage is None
    assert response.reward_kind == "llm_judged" and response.deterministically_verified is False
    assert formatted(agent, request(), result(True)).mask_sample is True


@pytest.mark.asyncio
async def test_judge_evidence_and_counted_zero_rows_are_retained(agent):
    value = result()
    value["verify"] = {
        "status": "PASS",
        "reward": 1.0,
        "prediction": "42",
        "analysis": "Score: 1",
        "model": "fixture-judge",
    }
    response = formatted(agent, request(), value)
    assert response.verifier_diagnostics == value["verify"]
    assert response.judge_response == "Score: 1"
    # The collector's counted failures retain task fields but have no verifier-produced fields.
    zero = {"reward": 0.0, "mask_sample": False, "verifier_metadata": {"category_id": "test"}}
    metrics = await metrics_owner(agent).aggregate_metrics(
        AggregateMetricsRequest(verify_responses=[response.model_dump(), zero])
    )
    assert metrics.agent_metrics["agentvista/num_scored"] == 2
    assert metrics.agent_metrics["agentvista/mean_score"] == 0.5
    assert metrics.agent_metrics["agentvista/pass_rate"] == 0.5
    assert metrics.agent_metrics["agentvista/test/mean_score"] == 0.5


@pytest.mark.asyncio
async def test_budget_exhaustion_keeps_judged_zero_and_explicit_metrics(agent):
    value = result()
    value["native_trajectory"] = []
    value["execution_health"] = {
        "termination": "output_budget_exhausted",
        "finish_reason": "length",
        "tool_calls": 0,
        "tool_errors": 0,
        "execution_healthy": True,
        "task_success": False,
    }
    response = formatted(agent, request(), value)
    assert not response.mask_sample and response.reward == 0
    assert response.response.status == "incomplete"
    metrics = await metrics_owner(agent).aggregate_metrics(
        AggregateMetricsRequest(verify_responses=[response.model_dump()])
    )
    assert metrics.agent_metrics["agentvista/num_scored"] == 1
    assert metrics.agent_metrics["agentvista/num_masked"] == 0
    assert metrics.agent_metrics["agentvista/output_budget_exhausted"] == 1


@pytest.mark.asyncio
async def test_cached_rollout_retry_does_not_rerun_actor(agent):
    body = request()
    output = Path(agent.config.output_root) / hashlib.sha256(b"repeat-0").hexdigest()
    output.mkdir(parents=True)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "request": body.model_dump(),
                "config": agent.config.model_dump(),
                "harness_version": HARNESS_VERSION,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    (output / "binding.json").write_text(json.dumps({"fingerprint": fingerprint}))
    (output / "result.json").write_text(json.dumps(result()))
    first, second = await asyncio.gather(agent._execute(body), agent._execute(body))
    assert first["verify"]["reward"] == second["verify"]["reward"] == 0 and first["mask_sample"] is False
    with pytest.raises(Exception, match="different request"):
        await agent._execute(request(task="other"))


@pytest.mark.asyncio
async def test_missing_rollout_identity_rejected(agent):
    with pytest.raises(Exception, match="_ng_rollout_id"):
        await agent._execute(request().model_copy(update={"capture_rollout_id": None}))


@pytest.mark.asyncio
async def test_episode_output_stays_in_mounted_path(agent, monkeypatch, tmp_path):
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    mounted = tmp_path / "mounted"
    mounted.symlink_to(persistent, target_is_directory=True)
    agent.config.output_root = str(mounted / "episodes")
    observed = []

    async def episode(self, body, output, fingerprint, rollout_id):
        observed.append(output)
        return result()

    monkeypatch.setattr(AgentVistaAgent, "_episode", episode)
    await agent._execute(request())
    assert observed == [mounted / "episodes" / hashlib.sha256(b"repeat-0").hexdigest()]


def test_actual_gym_http_model_route_and_sampling_capture(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from omegaconf import OmegaConf

    import nemo_gym.config.global_config as global_config
    from nemo_gym.config import ROLLOUT_PATH_PREFIX, BaseServerConfig
    from nemo_gym.core import NeMoGymChatCompletion
    from responses_api_models.vllm_model.agentvista_opencode import AgentVistaOpenCodeModel
    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

    cfg = OmegaConf.create(
        {
            "observability_enabled": True,
            "model_call_capture_dir": str(tmp_path / "captures"),
        }
    )
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", cfg)
    identity = "episode-fixture"
    seen = []

    async def complete(self, request, body):
        seen.append(body.model_dump(exclude_unset=True))
        return NeMoGymChatCompletion.model_validate(
            {
                "id": "served-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "<think>reasoning</think>",
                            "tool_calls": [
                                {
                                    "id": "call-fixture",
                                    "type": "function",
                                    "function": {
                                        "name": "agentvista_code_interpreter",
                                        "arguments": '{"code":"print(42)"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr(VLLMModel, "chat_completions", complete)
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=23480), global_config_dict=cfg)
    model_config = VLLMModelConfig(
        host="127.0.0.1",
        port=23482,
        entrypoint="test",
        name="policy_model",
        base_url="http://127.0.0.1:1/v1",
        api_key="fixture",
        model="fixture",
        return_token_id_information=False,
        uses_reasoning_parser=False,
    )
    policy = AgentVistaOpenCodeModel(config=model_config, server_client=client)
    with TestClient(policy.setup_webserver()) as api:
        response = api.post(
            f"/{ROLLOUT_PATH_PREFIX}/{identity}/v1/chat/completions",
            json={
                "model": "fixture",
                "messages": [{"role": "user", "content": "Act"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "agentvista_code_interpreter",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                            },
                        },
                    }
                ],
                "stream": False,
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["choices"][0]["message"]["reasoning_content"] == "reasoning"
        assert payload["choices"][0]["message"]["tool_calls"][0]["id"] == "call-fixture"
    assert list((tmp_path / "captures").rglob("*.json*"))


@pytest.mark.parametrize("stream_indices", [False, True])
def test_default_tool_history_replays_through_gym_without_changing_capture(stream_indices):
    from pydantic import ValidationError

    from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming
    from responses_api_agents.agentvista_agent.runtime.default_worker import _gym_messages

    call = {"id": "call-fixture", "type": "function", "function": {"name": "index_tool", "arguments": '{"index":7}'}}
    if stream_indices:
        call["index"] = 0
        call["function"]["index"] = 0
    messages = [
        {"role": "assistant", "content": "Original explanation", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call-fixture", "content": "Original result"},
    ]
    original = json.loads(json.dumps(messages))
    if stream_indices:
        with pytest.raises(ValidationError):
            NeMoGymChatCompletionCreateParamsNonStreaming.model_validate({"model": "fixture", "messages": messages})
    normalized = _gym_messages(messages)
    parsed = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate({"model": "fixture", "messages": normalized})
    assert parsed.messages and messages == original
    assert normalized[0]["tool_calls"][0] == {
        "id": "call-fixture",
        "type": "function",
        "function": {"name": "index_tool", "arguments": '{"index":7}'},
    }
    assert normalized[0]["content"] == "Original explanation"
    assert normalized[1] == original[1]
