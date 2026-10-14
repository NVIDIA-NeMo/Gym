# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Missing reasoning summaries retain encrypted Responses output and continuation."""

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseReasoningItem,
)


def reasoning():
    return {
        "type": "reasoning",
        "id": "rs_synthetic",
        "encrypted_content": "synthetic-opaque-content",
    }


def response(item):
    return NeMoGymResponse.model_validate(
        {
            "id": "resp_synthetic",
            "created_at": 0,
            "model": "synthetic",
            "object": "response",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                item,
                {
                    "type": "function_call",
                    "id": "fc_synthetic",
                    "call_id": "call_synthetic",
                    "name": "shell",
                    "arguments": "{}",
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 10,
                "total_tokens": 20,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


def test_missing_summary_preserves_encrypted_response_and_replay():
    item = reasoning()
    parsed = NeMoGymResponseReasoningItem.model_validate(item)
    assert parsed.summary == []
    serialized = response(item).model_dump(mode="json")
    assert serialized["output"][0] == parsed.model_dump(mode="json")
    assert serialized["output"][0]["encrypted_content"] == item["encrypted_content"]
    assert serialized["output"][0]["id"] == item["id"]
    assert serialized["output"][1]["call_id"] == "call_synthetic"
    assert serialized["model"] == "synthetic"
    assert serialized["usage"]["total_tokens"] == 20
    request = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"model": "synthetic", "input": serialized["output"]}
    )
    assert request.model_dump(mode="json")["input"] == serialized["output"]
    omitted = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": [item]})
    assert omitted.model_dump(mode="json")["input"][0] == serialized["output"][0]


def test_supplied_summary_is_unchanged():
    summary = [{"type": "summary_text", "text": "Synthetic summary."}]
    item = reasoning() | {"summary": summary}
    assert response(item).model_dump(mode="json")["output"][0]["summary"] == summary


@pytest.mark.parametrize("summary", [None, "invalid", {}, [{"type": "invalid", "text": "invalid"}]])
def test_null_and_malformed_summaries_remain_rejected(summary):
    item = reasoning() | {"summary": summary}
    with pytest.raises(ValidationError):
        NeMoGymResponseReasoningItem.model_validate(item)
    with pytest.raises(ValidationError):
        response(item)
    with pytest.raises(ValidationError):
        NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": [item]})


def test_summary_default_is_independent():
    first = NeMoGymResponseReasoningItem.model_validate(reasoning())
    second = NeMoGymResponseReasoningItem.model_validate(reasoning())
    assert first.summary is not second.summary
