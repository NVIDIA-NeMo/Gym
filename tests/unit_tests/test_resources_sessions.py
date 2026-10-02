# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.base_resources_server import BaseSeedSessionResponse, BaseVerifyResponse
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.single_agent_turn_types import SingleAgentTurnResult


def test_empty_seed_response_preserves_legacy_wire_shape() -> None:
    assert BaseSeedSessionResponse().model_dump() == {}


def test_base_verify_response_preserves_legacy_extra_field_behavior() -> None:
    response = BaseVerifyResponse.model_validate(
        {
            "responses_create_params": {"input": "hi"},
            "response": NeMoGymResponse.model_construct(id="response", output=[]),
            "reward": 0.5,
            "request_only_field": "ignored",
        }
    )
    assert "request_only_field" not in response.model_dump()


def test_single_agent_turn_result_preserves_benchmark_fields() -> None:
    response = SingleAgentTurnResult.model_validate(
        {
            "responses_create_params": {"input": "hi"},
            "response": NeMoGymResponse.model_construct(id="response", output=[]),
            "reward": 0.5,
            "benchmark_diagnostic": "kept",
        }
    )
    assert response.model_dump()["benchmark_diagnostic"] == "kept"
