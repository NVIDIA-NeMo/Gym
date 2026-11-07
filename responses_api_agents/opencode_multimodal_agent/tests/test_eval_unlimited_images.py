# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image
from pydantic import ValidationError

from responses_api_agents.opencode_multimodal_agent.eval import OpenCodeMultimodalEvalConfig
from responses_api_agents.opencode_multimodal_agent.media import stage_inputs
from responses_api_agents.opencode_multimodal_agent.tests.test_app import body, config
from responses_api_agents.opencode_multimodal_agent.tests.test_eval import evaluation
from responses_api_models.vllm_model.app import VLLMModel
from responses_api_models.vllm_model.visual_eval import VisualEvalModel


@pytest.mark.parametrize("count", [40, 42])
def test_unlimited_eval_stages_every_frame_in_order(tmp_path, count):
    raw_frames = []
    for i in range(count):
        stream = io.BytesIO()
        Image.new("RGB", (2, 2), (i, 0, 0)).save(stream, format="PNG")
        raw_frames.append(stream.getvalue())
    request = body(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Compare all frames"},
                    *[
                        {
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": "data:image/png;base64," + base64.b64encode(raw).decode(),
                        }
                        for raw in raw_frames
                    ],
                ],
            }
        ]
    )
    _, _, staged = stage_inputs(request, tmp_path, max_source_pixels=100, max_images=None, preserve_format=True)
    assert [p.read_bytes() for p in staged] == raw_frames
    assert [p.name for p in staged] == [f"input-{i}.png" for i in range(count)]
    with pytest.raises(ValueError, match="Initial image count exceeds max_images"):
        stage_inputs(request, tmp_path, max_source_pixels=100, max_images=33)


def test_only_eval_accepts_unlimited_image_configuration(monkeypatch, tmp_path):
    bounded = evaluation(monkeypatch, tmp_path)
    unlimited = OpenCodeMultimodalEvalConfig.model_validate(bounded.config.model_dump() | {"max_images": None})
    assert unlimited.max_images is None
    assert bounded.config.max_images == config().max_images == 33
    with pytest.raises(ValidationError):
        config(max_images=None)


def test_unlimited_model_preserves_initial_and_tool_images(monkeypatch):
    monkeypatch.setattr(VLLMModel, "_preprocess_chat_completion_create_params", lambda self, req, body: body)
    images = [{"type": "image_url", "image_url": {"url": f"file:///frame-{i}.png"}} for i in range(50)]
    body_dict = {
        "messages": [
            {"role": "user", "content": images[:42]},
            {"role": "assistant", "content": "Inspect more frames"},
            {"role": "user", "content": images[42:]},
        ]
    }
    monkeypatch.setattr(VisualEvalModel, "model_post_init", lambda self, context: None)
    model = VisualEvalModel.model_construct(config=SimpleNamespace(max_images=None))
    result = model._preprocess_chat_completion_create_params(MagicMock(), body_dict)
    assert result["messages"][0]["content"] + result["messages"][2]["content"] == images
    assert result["messages"][1]["content"] == "Inspect more frames"
