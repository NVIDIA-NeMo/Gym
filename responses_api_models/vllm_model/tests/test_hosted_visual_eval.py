# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request

from nemo_gym.client import ServerClient
from responses_api_models.vllm_model.hosted_visual_eval import HostedVisualEvalModel
from responses_api_models.vllm_model.visual_eval import VisualEvalModelConfig


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("tools", [None, []])
def test_hosted_defaults_and_explicit_server_settings(configured, tools):
    config = VisualEvalModelConfig(
        host="localhost",
        port=8081,
        name="policy",
        entrypoint="hosted_visual_eval.py",
        base_url="http://localhost/v1",
        api_key="unused",
        model="policy",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        max_images=None,
        sampling_overrides={"temperature": 0.2, "top_p": 0.8} if configured else None,
        chat_template_kwargs={"truncate_history_thinking": False} if configured else None,
        extra_body={"custom_flag": 1} if configured else None,
    )
    with patch.object(HostedVisualEvalModel, "_post_init"):
        model = HostedVisualEvalModel(config=config, server_client=MagicMock(spec=ServerClient))
    result = model._preprocess_chat_completion_create_params(
        Request({"type": "http", "headers": []}),
        {"messages": [{"role": "user", "content": "question"}], "temperature": 0.6, "top_p": 0.95, "tools": tools},
    )
    if configured:
        assert result["temperature"] == 0.2
        assert result["top_p"] == 0.8
        assert result["custom_flag"] == 1
        assert result["chat_template_kwargs"]["truncate_history_thinking"] is False
    else:
        assert "temperature" not in result
        assert "top_p" not in result
