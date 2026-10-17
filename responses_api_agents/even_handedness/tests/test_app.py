# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired-agent tests."""

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.even_handedness.app import EvenHandednessAgent, EvenHandednessRunRequest


def test_prompt_pair_uses_independent_inputs_without_losing_sampling() -> None:
    body = EvenHandednessRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "placeholder"}],
            temperature=0.7,
            top_p=0.9,
        ),
        prompt_a="argue A",
        prompt_b="argue B",
    )
    params_a = EvenHandednessAgent._params_for_prompt(body, body.prompt_a)
    params_b = EvenHandednessAgent._params_for_prompt(body, body.prompt_b)
    assert params_a.input[0].content == "argue A"
    assert params_b.input[0].content == "argue B"
    assert params_a.temperature == params_b.temperature == 0.7
    assert params_a.top_p == params_b.top_p == 0.9
    assert params_a is not params_b


def test_reverification_metadata_key_is_shared_with_the_resources_server() -> None:
    from resources_servers.even_handedness.app import EvenHandednessServer

    assert EvenHandednessAgent._RESPONSE_B_METADATA_KEY == EvenHandednessServer._RESPONSE_B_METADATA_KEY
