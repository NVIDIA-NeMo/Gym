# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.agentic_if_realworld.benchmark.core import Dataset
from resources_servers.agentic_if_realworld.benchmark.synthetic import create
from responses_api_models.agentic_if_text import app


@pytest.mark.parametrize("reported", [None, {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}])
def test_adapter_preserves_usage_and_does_not_invent_missing_counts(tmp_path, monkeypatch, reported):
    create(tmp_path / "data")
    dataset = Dataset(tmp_path / "data", synthetic=True)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=dataset.request("synthetic")["messages"],
        temperature=0.7,
        max_output_tokens=32768,
        metadata={"seed": "17", "receipt_id": "a" * 64},
    )
    requests = []

    async def generate(request, **kwargs):
        requests.append(request)
        return {
            "response": {
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "OK."}}],
                "usage": reported,
            }
        }

    monkeypatch.setattr(app, "wire_call", generate)
    model = app.TextModel.model_construct(
        config=SimpleNamespace(
            model="synthetic-policy",
            endpoint="http://synthetic.invalid",
            api_key_env="SYNTHETIC",
            receipt_dir=str(tmp_path / "receipts"),
        )
    )
    result = asyncio.run(model.responses(body))
    assert result.output_text == "OK."
    assert requests[0]["messages"] == dataset.request("synthetic")["messages"]
    assert requests[0]["stream_options"] == {"include_usage": True}
    if reported is None:
        assert result.usage is None
    else:
        assert result.usage.input_tokens == 10 and result.usage.output_tokens == 2
        assert result.usage.output_tokens_details.reasoning_tokens is None
    body.instructions = "injected instruction"
    with pytest.raises(ValueError, match="additional prompt"):
        asyncio.run(model.responses(body))
    assert len(requests) == 1
