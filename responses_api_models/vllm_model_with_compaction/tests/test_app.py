# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
from responses_api_models.vllm_model_with_compaction.app import (
    VLLMContextCompactionResponseCreateParams,
    VLLMModelWithCompaction,
)


def _make_model() -> VLLMModelWithCompaction:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="policy_model_for_compaction",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy_model",
        return_token_id_information=True,
        request_prompt_and_generation_token_ids=True,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
    )
    return VLLMModelWithCompaction(
        config=config,
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )


def test_base_vllm_model_has_no_compaction_routes() -> None:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="policy_model",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy_model",
        return_token_id_information=False,
        uses_reasoning_parser=False,
    )
    model = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    app = model.setup_webserver()
    paths = {route.path for route in app.routes}

    assert "/tokenize" not in paths
    assert "/v1/responses/context-compaction" not in paths
    client = TestClient(app)
    assert client.post("/tokenize", json={"input": "hi"}).status_code == 404
    assert (
        client.post(
            "/v1/responses",
            json={"input": "hi", "required_prefix_token_ids": [10, 11]},
        ).status_code
        == 422
    )


def test_context_compaction_conversion_keeps_prefix_out_of_shared_schema() -> None:
    model = _make_model()
    standard_body, chat_params = model._context_compaction_chat_params(
        VLLMContextCompactionResponseCreateParams(
            input="hi",
            required_prefix_token_ids=[10, 11],
        )
    )

    assert "required_prefix_token_ids" not in standard_body.model_dump()
    assert chat_params.required_prefix_token_ids == [10, 11]


def test_plain_responses_endpoint_forwards_exact_prefix() -> None:
    model = _make_model()
    captured_kwargs: dict[str, Any] = {}

    async def mock_create_chat_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "dummy_model",
            "prompt_token_ids": [10, 11, 12],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "token_ids": [13],
                    "message": {"role": "assistant", "content": "ok", "tool_calls": None},
                    "logprobs": {
                        "content": [
                            {
                                "token": "token_id:13",
                                "logprob": -0.1,
                                "bytes": None,
                                "top_logprobs": [],
                            }
                        ]
                    },
                }
            ],
        }

    mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
    mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
    model._clients = [mock_client]

    response = TestClient(model.setup_webserver()).post(
        "/v1/responses",
        json={"input": "hi", "required_prefix_token_ids": [10, 11]},
    )

    assert response.status_code == 200
    assert captured_kwargs["required_prefix_token_ids"] == [10, 11]
    assert captured_kwargs["return_token_ids"] is True


def test_generation_fails_closed_instead_of_retokenizing_missing_exact_ids() -> None:
    model = _make_model()

    async def mock_create_chat_completion(**kwargs):
        assert kwargs["return_token_ids"] is True
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "dummy_model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok", "tool_calls": None},
                    "logprobs": {
                        "content": [
                            {
                                "token": "token_id:13",
                                "logprob": -0.1,
                                "bytes": None,
                                "top_logprobs": [],
                            }
                        ]
                    },
                }
            ],
        }

    mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
    mock_client.create_chat_completion = AsyncMock(side_effect=mock_create_chat_completion)
    mock_client.create_tokenize = AsyncMock(
        side_effect=AssertionError("missing generation evidence must not fall back to /tokenize")
    )
    model._clients = [mock_client]

    with pytest.raises(RuntimeError, match="Refusing to reconstruct on-policy training evidence"):
        TestClient(model.setup_webserver()).post(
            "/v1/responses",
            json={"input": "hi", "required_prefix_token_ids": [10, 11]},
        )
    mock_client.create_tokenize.assert_not_awaited()


def test_tokenize_endpoint_forwards_exact_prefix_without_sampling() -> None:
    model = _make_model()
    captured_kwargs: dict[str, Any] = {}

    async def mock_create_tokenize(**kwargs):
        captured_kwargs.update(kwargs)
        return {"tokens": [10, 11, 12]}

    mock_client = MagicMock(spec=NeMoGymAsyncOpenAI)
    mock_client.create_tokenize = AsyncMock(side_effect=mock_create_tokenize)
    model._clients = [mock_client]

    response = TestClient(model.setup_webserver()).post(
        "/tokenize",
        json={"input": "hi", "required_prefix_token_ids": [10, 11]},
    )

    assert response.status_code == 200
    assert response.json() == {"tokens": [10, 11, 12]}
    assert captured_kwargs["required_prefix_token_ids"] == [10, 11]
    assert "logprobs" not in captured_kwargs
    assert "return_token_ids" not in captured_kwargs
