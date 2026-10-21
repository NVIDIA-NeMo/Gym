# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regress the stock runtime's synthetic assistant directive at the turn cap."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint, conversation_digest
from responses_api_models.vllm_model.app import VLLMModel
from responses_api_models.vllm_model.app_opencode import (
    OPENCODE_1_17_11_MAX_STEPS,
    OpenCodeVLLMModel,
    OpenCodeVLLMModelConfig,
    normalize_opencode_final_instruction,
)


@pytest.mark.parametrize("extra", [{}, {"reasoning_content": ""}, {"reasoning_content": None}])
def test_final_directive_preserves_model_history_and_parent_context(extra):
    prefix = [{"role": "user", "content": "Count the points in this image."}]
    generated = {"role": "assistant", "content": "I will inspect the image."}
    directive = {"role": "assistant", "content": OPENCODE_1_17_11_MAX_STEPS, **extra}
    body = {"messages": prefix + [generated, directive], "stream": True}
    original = deepcopy(body)
    assert assistant_fingerprint(body["messages"]) != assistant_fingerprint(prefix + [generated])
    normalized = normalize_opencode_final_instruction(body)
    assert assistant_fingerprint(normalized["messages"]) == assistant_fingerprint(prefix + [generated])
    assert conversation_digest(normalized["messages"][: len(prefix)]) == conversation_digest(prefix)
    assert normalized["messages"][-1] == {"role": "user", "content": OPENCODE_1_17_11_MAX_STEPS}
    assert body == original


@pytest.mark.parametrize(
    "tail",
    [
        {"role": "assistant", "content": "CRITICAL - MAXIMUM STEPS REACHED"},
        {"role": "assistant", "content": OPENCODE_1_17_11_MAX_STEPS + " altered"},
        {"role": "assistant", "content": OPENCODE_1_17_11_MAX_STEPS, "reasoning_content": "Generated reasoning"},
        {"role": "user", "content": OPENCODE_1_17_11_MAX_STEPS},
    ],
)
def test_other_history_is_not_reclassified(tail):
    body = {"messages": [tail]}
    assert normalize_opencode_final_instruction(body) is body


@pytest.mark.asyncio
async def test_dispatch_normalizes_before_upstream_capture():
    body = {"messages": [{"role": "assistant", "content": OPENCODE_1_17_11_MAX_STEPS}], "stream": True}
    with patch.object(OpenCodeVLLMModel, "_post_init"):
        server = OpenCodeVLLMModel.model_construct(config=SimpleNamespace(max_images=None))
    request = object()
    with patch.object(VLLMModel, "chat_completions_dispatch", new_callable=AsyncMock) as dispatch:
        await server.chat_completions_dispatch(request, body)
    assert dispatch.call_args.args[0] is request
    assert dispatch.call_args.args[1]["messages"][-1]["role"] == "user"


@pytest.mark.parametrize("reserve", [0, 2048])
def test_final_answer_budget_metadata_reaches_engine(reserve):
    import json
    from unittest.mock import MagicMock

    from starlette.requests import Request

    from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
    from nemo_gym.server_utils import ServerClient

    config = OpenCodeVLLMModelConfig(
        host="localhost",
        port=8081,
        base_url="http://localhost/v1",
        api_key="unused",
        model="policy",
        name="policy",
        entrypoint="app_opencode.py",
        return_token_id_information=False,
        uses_reasoning_parser=False,
    )
    with patch.object(OpenCodeVLLMModel, "_post_init"):
        server = OpenCodeVLLMModel(config=config, server_client=MagicMock(spec=ServerClient))
    body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {
            "model": "policy",
            "messages": [{"role": "user", "content": "answer"}],
            "metadata": {"extra_body": json.dumps({"nemo_final_answer_reserve": reserve})},
            "max_tokens": 49152,
        }
    )
    result = server._preprocess_chat_completion_create_params(
        Request({"type": "http", "headers": []}), body.model_dump(exclude_unset=True)
    )
    assert result["nemo_final_answer_reserve"] == reserve
    assert result["messages"] == body.messages
