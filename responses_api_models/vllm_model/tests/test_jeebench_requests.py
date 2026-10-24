# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep adapter imports in server CI, outside the core coverage gate."""

import json
from unittest.mock import MagicMock

import pytest

from benchmarks.indic.jee_bench.prepare import build_rows
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


@pytest.mark.parametrize("override", [False, True])
def test_jeebench_sampling_and_prompt_reach_vllm(override: bool) -> None:
    row = build_rows(
        [
            {
                "description": "Synthetic",
                "index": 1,
                "subject": "math",
                "type": "MCQ",
                "gold": "B",
                "question": "एक और एक?\n(A) 1\n(B) 2\n(C) 3\n(D) 4",
            }
        ],
        language="hi",
        revision="fixture",
    )[0]
    prompt = load_prompt_config("benchmarks/indic/jee_bench/prompts/default.yaml")
    params = apply_prompt_to_row(row, prompt)["responses_create_params"]
    if override:
        params.update(temperature=1.0, top_p=0.95, max_output_tokens=50000)
        params["metadata"] = {
            "extra_body": json.dumps({"top_k": -1, "seed": 42}),
            "chat_template_kwargs": json.dumps({"enable_thinking": True}),
        }
    model = VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="policy_model",
            base_url="http://unused/v1",
            api_key="dummy",
            model="test-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    request = NeMoGymResponseCreateParamsNonStreaming.model_validate(params)
    chat = model._converter.responses_to_chat_completion_create_params(request)
    outgoing = model._preprocess_chat_completion_create_params(MagicMock(), chat.model_dump(exclude_unset=True))
    assert outgoing["model"] == "test-model"
    assert outgoing["messages"][-1] == {"role": "user", "content": row["instruction"]}
    assert (outgoing["temperature"], outgoing["max_tokens"]) == ((1.0, 50000) if override else (0.0, 2048))
    if override:
        assert outgoing["top_p"] == 0.95
        assert outgoing["top_k"] == -1
        assert outgoing["seed"] == 42
        assert outgoing["chat_template_kwargs"] == {"enable_thinking": True}
    assert "verifier_metadata" not in outgoing
