# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

from responses_api_models.vllm_model.app import VLLMModel
from responses_api_models.vllm_model.visual_eval import IMAGE_LIMIT_NOTICE, VisualEvalModel


def test_image_capacity_preserves_first_images_and_explains_omissions(monkeypatch):
    monkeypatch.setattr(VLLMModel, "_preprocess_chat_completion_create_params", lambda self, req, body: body)
    images = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i}"}} for i in range(35)]
    body = {
        "messages": [
            {"role": "user", "content": images[:1]},
            {"role": "assistant", "content": "reasoning"},
            {"role": "user", "content": images[1:]},
        ]
    }
    monkeypatch.setattr(VisualEvalModel, "model_post_init", lambda self, context: None)
    # Construct without server startup; this tests the actual eval override.
    agent = VisualEvalModel.model_construct(config=SimpleNamespace(max_images=33))
    result = agent._preprocess_chat_completion_create_params(MagicMock(), body)
    blocks = result["messages"][0]["content"] + result["messages"][2]["content"]
    assert blocks[:33] == images[:33]
    assert blocks[33:] == [{"type": "text", "text": IMAGE_LIMIT_NOTICE}] * 2
    assert "33" not in IMAGE_LIMIT_NOTICE
    assert result["messages"][1]["content"] == "reasoning"


def test_disabled_image_guard_preserves_all_attachments(monkeypatch):
    monkeypatch.setattr(VLLMModel, "_preprocess_chat_completion_create_params", lambda self, req, body: body)
    monkeypatch.setattr(VisualEvalModel, "model_post_init", lambda self, context: None)
    images = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i}"}} for i in range(128)]
    body = {"messages": [{"role": "user", "content": images}]}
    agent = VisualEvalModel.model_construct(config=SimpleNamespace(max_images=None))
    result = agent._preprocess_chat_completion_create_params(MagicMock(), body)
    assert result["messages"][0]["content"] == images
    assert len(result["messages"][0]["content"]) == 128
