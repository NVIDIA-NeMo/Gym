# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint, conversation_digest
from responses_api_models.vllm_model.app import VLLMModel
from responses_api_models.vllm_model.app_opencode import OpenCodeVLLMModel
from responses_api_models.vllm_model.image_guard import IMAGE_LIMIT_NOTICE, limit_request_images


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
