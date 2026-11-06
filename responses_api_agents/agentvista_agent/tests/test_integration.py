# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://gitlab-master.nvidia.com/pritamb/agentvista
# Revision: 06f6a8f1532766d876415757107a11731b6148ae (Apache-2.0 adapter files).
"""Validate adapters against actual Gym schemas; no hosted inference or GPU work."""

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


pytest.importorskip("nemo_gym")
if not os.environ.get("AGENTVISTA_ROOT"):
    pytest.skip("Set AGENTVISTA_ROOT to run the external benchmark integration contracts", allow_module_level=True)
ROOT = Path(os.environ["AGENTVISTA_ROOT"]).resolve()
sys.path.insert(0, str(ROOT))
from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.agentvista_agent.app import (
    AgentVistaAgent,
    AgentVistaConfig,
    AgentVistaRunRequest,
)
from responses_api_agents.agentvista_agent.training import AgentVistaTrainingAgent


@pytest.fixture
def agent(tmp_path):
    config = AgentVistaConfig(
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        name="agentvista_test",
        source_root=str(ROOT),
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
                "harness_version": "2.1.1",
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
    response = agent._response(request(), result())
    assert response.reward == 0 and response.mask_sample is False
    assert response.response.output[0].type == "function_call"
    assert response.response.output[1].call_id == "one"
    assert response.response.usage is None
    assert response.reward_kind == "llm_judged" and response.deterministically_verified is False
    assert agent._response(request(), result(True)).mask_sample is True


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
                "harness_version": "2.1.1",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    (output / "binding.json").write_text(json.dumps({"fingerprint": fingerprint}))
    (output / "result.json").write_text(json.dumps(result()))
    first, second = await asyncio.gather(agent.run(body), agent.run(body))
    assert first.reward == second.reward == 0 and first.mask_sample is False
    with pytest.raises(Exception, match="different request"):
        await agent.run(request(task="other"))


@pytest.mark.asyncio
async def test_missing_rollout_identity_rejected(agent):
    with pytest.raises(Exception, match="_ng_rollout_id"):
        await agent.run(request().model_copy(update={"capture_rollout_id": None}))


def test_training_terminal_is_witnessed_and_absence_masks(agent, tmp_path):
    from types import SimpleNamespace

    training = object.__new__(AgentVistaTrainingAgent)
    object.__setattr__(
        training,
        "config",
        SimpleNamespace(training_capture_root=str(tmp_path), model_name="fixture"),
    )
    journal = tmp_path / (hashlib.sha256(b"repeat-0").hexdigest() + ".jsonl")
    journal.write_text(
        json.dumps(
            {
                "rollout_id": "repeat-0",
                "response": {
                    "id": "served-terminal",
                    "choices": [{"message": {"content": "<answer>42</answer>"}}],
                },
            }
        )
        + "\n"
    )
    response = training._response(request(), result())
    assert response.terminal_response_id == "served-terminal" and response.mask_sample is False
    journal.unlink()
    assert training._response(request(), result()).mask_sample is True


@pytest.mark.parametrize("training", [False, True])
def test_actual_gym_http_model_route_and_sampling_capture(tmp_path, monkeypatch, training):
    from fastapi.testclient import TestClient
    from omegaconf import OmegaConf

    import nemo_gym.global_config as global_config
    from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, BaseServerConfig
    from nemo_gym.openai_utils import NeMoGymChatCompletion
    from responses_api_models.vllm_model.agentvista_opencode import AgentVistaOpenCodeModel
    from responses_api_models.vllm_model.agentvista_training import AgentVistaTrainingModel
    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

    cfg = OmegaConf.create(
        {
            "observability_enabled": True,
            "model_call_capture_dir": str(tmp_path / "captures"),
            "agentvista_training_capture_root": str(tmp_path),
        }
    )
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", cfg)
    identity = "episode-fixture"
    journal = tmp_path / (hashlib.sha256(identity.encode()).hexdigest() + ".request.json")
    journal.write_text(
        json.dumps(
            {
                "rollout_id": identity,
                "sampling": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512},
            }
        )
    )
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
    cls = AgentVistaTrainingModel if training else AgentVistaOpenCodeModel
    policy = cls(config=model_config, server_client=client)
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
    if training:
        assert seen[0]["temperature"] == 0.7 and seen[0]["max_tokens"] == 512
        capture = json.loads(journal.with_name(journal.name.replace(".request.json", ".jsonl")).read_text())
        assert capture["rollout_id"] == identity and capture["response"]["id"] == "served-fixture"
    assert list((tmp_path / "captures").rglob("*.json*"))
