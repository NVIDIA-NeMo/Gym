# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_observability import ModelCallRef


@pytest.mark.parametrize(
    "value",
    ({}, {"response_id": "resp-1"}, {"model_ref": {"name": "policy", "type": "responses_api_models"}}),
)
def test_model_call_ref_rejects_incomplete_join_keys(value: dict) -> None:
    with pytest.raises(ValidationError, match="model_call_id or both model_ref and response_id"):
        ModelCallRef.model_validate(value)


def test_model_call_ref_accepts_supported_join_keys() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")

    assert ModelCallRef(model_call_id="call-1").model_call_id == "call-1"
    assert ModelCallRef(model_ref=model_ref, response_id="resp-1").model_ref == model_ref
