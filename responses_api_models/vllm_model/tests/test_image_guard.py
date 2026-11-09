# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi import Request

from nemo_gym.client import ServerClient
from nemo_gym.models.token_capture import assistant_fingerprint, conversation_digest
from responses_api_models.vllm_model.app import VLLMModel
from responses_api_models.vllm_model.app_opencode import OpenCodeVLLMModel
from responses_api_models.vllm_model.hosted_visual_eval import HostedVisualEvalModel
from responses_api_models.vllm_model.image_guard import IMAGE_LIMIT_NOTICE, limit_request_images
from responses_api_models.vllm_model.visual_eval import VisualEvalModel, VisualEvalModelConfig


def conversation():
    return {
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "first"}}]},
            {"role": "assistant", "content": "Inspecting", "reasoning_content": "Keep this reasoning"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "crop"}}]},
        ]
    }


def test_guard_preserves_history_and_is_stable_across_turns():
    body = conversation()
    original = deepcopy(body)
    guarded = limit_request_images(body, max_images=1)
    assert guarded["messages"][-1]["content"] == [{"type": "text", "text": IMAGE_LIMIT_NOTICE}]
    assert body == original
    assert guarded["messages"][:2] == body["messages"][:2]
    assert assistant_fingerprint(guarded["messages"]) == assistant_fingerprint(body["messages"])
    assert limit_request_images(guarded, max_images=1) == guarded
    body["messages"].extend(
        [
            {"role": "assistant", "content": "Checking again", "reasoning_content": "Still retained"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "another crop"}}]},
        ]
    )
    next_turn = limit_request_images(body, max_images=1)
    assert conversation_digest(next_turn["messages"][:3]) == conversation_digest(guarded["messages"])
    assert next_turn["messages"][-1]["content"] == guarded["messages"][-1]["content"]


def test_disabled_guard_does_not_change_requests():
    body = conversation()
    assert limit_request_images(body, max_images=None) is body
    with pytest.raises(ValueError):
        limit_request_images(body, max_images=0)


@pytest.mark.parametrize("model_class", [VisualEvalModel, HostedVisualEvalModel])
@pytest.mark.parametrize("explicit_cap", [False, True])
def test_eval_gateway_preserves_all_images_unless_cap_is_configured(model_class, explicit_cap):
    settings = {"max_images": 33} if explicit_cap else {}
    config = VisualEvalModelConfig(
        host="localhost",
        port=8081,
        name="policy",
        entrypoint="visual_eval.py",
        base_url="http://localhost/v1",
        api_key="unused",
        model="policy",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        **settings,
    )
    assert config.max_images == (33 if explicit_cap else None)
    with patch.object(VLLMModel, "_post_init"):
        model = model_class(config=config, server_client=MagicMock(spec=ServerClient))
    content = [{"type": "image_url", "image_url": {"url": f"image-{i}"}} for i in range(35)]
    body = {"messages": [{"role": "user", "content": deepcopy(content)}]}
    result = model._preprocess_chat_completion_create_params(Request({"type": "http", "headers": []}), body)
    expected = content if not explicit_cap else content[:33] + [{"type": "text", "text": IMAGE_LIMIT_NOTICE}] * 2
    assert result["messages"][0]["content"] == expected
    assert body["messages"][0]["content"] == content


@pytest.mark.parametrize("preset", ["eval_55_grace10.yaml", "eval_advisory.yaml"])
def test_standalone_eval_presets_preserve_explicit_image_capacity(preset):
    root = Path(__file__).resolve().parents[3]
    settings = yaml.safe_load((root / "responses_api_agents/opencode_multimodal_agent/configs" / preset).read_text())[
        "policy_model"
    ]["responses_api_models"]["vllm_model"]
    config = VisualEvalModelConfig(
        host="localhost",
        port=8081,
        name="policy",
        base_url="http://localhost/v1",
        api_key="unused",
        model="policy",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        **settings,
    )
    assert config.max_images == 33


@pytest.mark.asyncio
async def test_training_omits_images_before_lineage_and_capture_dispatch():
    with patch.object(OpenCodeVLLMModel, "_post_init"):
        server = OpenCodeVLLMModel.model_construct(config=SimpleNamespace(max_images=1))
    body = conversation()
    with patch.object(VLLMModel, "chat_completions_dispatch", new_callable=AsyncMock) as dispatch:
        await server.chat_completions_dispatch(object(), body)
    actual = dispatch.call_args.args[1]
    assert actual == limit_request_images(body, max_images=1)
    assert body["messages"][-1]["content"][0]["type"] == "image_url"
