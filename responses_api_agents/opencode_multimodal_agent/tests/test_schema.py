# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming


@pytest.mark.parametrize("reasoning", [None, "", "Inspect the image first."])
def test_assistant_reasoning_history_round_trip(reasoning: str | None) -> None:
    assistant = {"role": "assistant", "content": "checking", "reasoning_content": reasoning}
    payload = {"messages": [assistant]}
    parsed = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(payload)
    assert parsed.model_dump(mode="json", exclude_unset=True)["messages"] == payload["messages"]
    assistant["unknown_provider_field"] = "unexpected"
    # Keep the target branch's strict schema while using the upstream provider field.
    with pytest.raises(ValidationError, match="unknown_provider_field"):
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(payload)
