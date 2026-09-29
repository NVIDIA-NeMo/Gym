# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise overflow capture through the adapter, trajectory projection, and health runner."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import merge_model_call_capture_into_record
from nemo_gym.rollout_collection import _build_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


@pytest.mark.parametrize("completions", [False, True])
@pytest.mark.parametrize("propagate", [False, True])
@pytest.mark.parametrize("dialect", ["chat/completions", "responses"])
@pytest.mark.parametrize("recover", [False, True])
def test_overflow_health_uses_final_outcome(tmp_path, completions, propagate, dialect, recover):
    server = VLLMModel(
        config=VLLMModelConfig(
            host="localhost",
            port=8081,
            entrypoint="",
            name="policy",
            model="dummy",
            base_url="http://unused/v1",
            api_key="dummy",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            use_completions_api=completions,
            propagate_context_overflow_errors=propagate,
        ),
        server_client=MagicMock(
            spec=ServerClient,
            global_config_dict={
                "observability_enabled": True,
                "model_call_capture_dir": str(tmp_path),
            },
        ),
    )
    error = ClientResponseError(MagicMock(real_url="http://unused"), (), status=400, message="Bad Request")
    error.response_content = b'{"error":{"message":"maximum context length exceeded"}}'
    success = {
        "id": "provider-success",
        "object": "text_completion" if completions else "chat.completion",
        "created": 123,
        "model": "dummy",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                **(
                    {"text": "answer", "logprobs": None}
                    if completions
                    else {"message": {"role": "assistant", "content": "answer"}}
                ),
            }
        ],
    }
    client = MagicMock()
    method = AsyncMock(side_effect=[error, success])
    client.create_completion = method
    client.create_chat_completion = method
    server._clients = [client]
    app = server.setup_webserver()
    server.setup_exception_middleware(app)
    body = {"model": "dummy", "input" if dialect == "responses" else "messages": [{"role": "user", "content": "hi"}]}
    with TestClient(app) as http:
        first = http.post(f"/ng-rollout/0-0/v1/{dialect}", json=body)
        assert first.status_code == (400 if propagate else 200)
        if recover:
            assert http.post(f"/ng-rollout/0-0/v1/{dialect}", json=body).status_code == 200
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    result = merge_model_call_capture_into_record(dict(row), [tmp_path], include_payloads=True)
    calls = result["ng_model_call_capture"]["calls"]
    assert calls[0]["upstream_status_code"] == 400
    assert calls[0]["error_category"] == "context_length_exceeded"
    trajectory = _build_trajectory_record(row, result).model_dump(mode="json")
    trajectory["invocations"] = [
        {
            "invocation_id": "root",
            "status": "completed" if recover else "failed",
            "model_calls": [{"model_call_id": call["model_call_id"]} for call in calls],
        }
    ]
    trajectory["gaps"] = [{"code": "turns_unavailable"}]
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(row | {"ng_trajectory": trajectory}) + "\n")
    [digest] = run_health_checks(path, workers=1).rollouts
    assert {finding.check for finding in digest.findings} == (set() if recover else {"model_call_last_failed"})
    assert digest.model_call_errors == 1
    assert "model_call_last_failed" not in digest.unobserved
    assert digest.verdict == ("unobserved" if recover else "unhealthy")
