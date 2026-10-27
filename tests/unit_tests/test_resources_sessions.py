# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.base_resources_server import BaseSeedSessionResponse, BaseVerifyResponse, ResourcesSeedSessionResponse
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


@pytest.mark.parametrize("observation", [None, "", " \nReset prompt\t"])
def test_resources_seed_response_accepts_an_optional_observation(observation: str | None) -> None:
    payload = {"resources_session_id": "resources-session", "observation": observation}
    response = ResourcesSeedSessionResponse.model_validate(payload)
    assert response.model_dump(exclude_unset=True) == payload
    assert ResourcesSeedSessionResponse(resources_session_id="resources-session").observation is None


@pytest.mark.parametrize(
    "fields",
    [{"observation": 42}, {"observation": True}, {"observation": []}, {"observation": {}}, {"unknown": "value"}],
)
def test_resources_seed_response_keeps_observation_types_and_extra_fields_strict(fields: dict) -> None:
    with pytest.raises(ValidationError):
        ResourcesSeedSessionResponse.model_validate({"resources_session_id": "resources-session", **fields})
