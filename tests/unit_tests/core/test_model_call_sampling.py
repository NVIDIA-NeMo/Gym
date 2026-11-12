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
import base64

import pytest
from omegaconf import OmegaConf
from pydantic import ConfigDict, ValidationError

from nemo_gym.core.model_call_sampling import (
    MODEL_SAMPLING_NULL,
    ModelCallSampling,
    current_model_sampling_segment,
    decode_model_sampling,
    encode_model_sampling,
    model_call_sampling_config,
    model_sampling_context,
    model_sampling_from_run_body,
    with_model_sampling_segment,
)
from nemo_gym.server_runtime.base_resources_server import BaseRunRequest


def _encoded(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def test_values_survive_the_url_and_null_stays_distinct_from_values() -> None:
    sampling = ModelCallSampling(temperature=0.0, top_p=0.95, max_output_tokens=8192)

    encoded = encode_model_sampling(sampling)

    assert "=" not in encoded and "/" not in encoded and "+" not in encoded
    assert decode_model_sampling(encoded) == sampling
    assert encode_model_sampling(None) == MODEL_SAMPLING_NULL
    assert decode_model_sampling(MODEL_SAMPLING_NULL) is None
    # Unset fields stay unset rather than travelling as nulls that would erase the call's own values.
    assert decode_model_sampling(encode_model_sampling(ModelCallSampling(top_p=0.5))).model_fields_set == {"top_p"}


@pytest.mark.parametrize(
    "values",
    [
        {"temperature": True},
        {"temperature": "0.7"},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"temperature": float("nan")},
        {"top_p": float("inf")},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"max_output_tokens": True},
        {"max_output_tokens": 1.5},
        {"max_output_tokens": "128"},
        {"max_output_tokens": 0},
        {"max_output_tokens": 2**53},
        {"top_k": 20},
    ],
)
def test_values_a_generation_server_would_reject_fail_validation(values: dict) -> None:
    with pytest.raises(ValidationError):
        ModelCallSampling(**values)


def test_the_cap_accepts_the_largest_integer_json_doubles_represent() -> None:
    assert ModelCallSampling(max_output_tokens=2**53 - 1).max_output_tokens == 2**53 - 1


@pytest.mark.parametrize(
    "encoded",
    [
        "",
        "not+urlsafe",
        _encoded(b'{"top_p":0.5}') + "=",
        "a" * 1025,
        _encoded(b'{"top_p":0.1,"top_p":0.9}'),
        _encoded(b'{"temperature":NaN}'),
        _encoded(b'{"top_p":Infinity}'),
        _encoded(b"[]"),
        _encoded(b"not json"),
        _encoded(b'{"temperature":0.5,"seed":1}'),
        _encoded(b"\xff\xfe"),
    ],
)
def test_malformed_or_ambiguous_segments_are_refused(encoded: str) -> None:
    with pytest.raises(ValueError):
        decode_model_sampling(encoded)


def test_run_bodies_carry_values_under_the_row_key() -> None:
    values = {"temperature": 1.0, "top_p": 1.0, "max_output_tokens": 64}

    class ExtraRunRequest(BaseRunRequest):
        model_config = ConfigDict(extra="allow")

    run_request = BaseRunRequest.model_validate(
        {"responses_create_params": {"input": "task"}, "_ng_model_sampling": values}
    )

    assert model_sampling_from_run_body({"_ng_model_sampling": values}) == ModelCallSampling(**values)
    assert model_sampling_from_run_body(run_request) == ModelCallSampling(**values)
    assert model_sampling_from_run_body(
        ExtraRunRequest.model_validate({"responses_create_params": {"input": "task"}, "_ng_model_sampling": values})
    ) == ModelCallSampling(**values)
    # The values never leave the agent in a body it forwards, such as a verify request.
    assert "_ng_model_sampling" not in run_request.model_dump(by_alias=True)
    assert "model_sampling" not in run_request.model_dump()
    assert model_sampling_from_run_body({"responses_create_params": {"input": "task"}}) is None
    assert model_sampling_from_run_body(None) is None
    with pytest.raises(ValidationError):
        model_sampling_from_run_body({"_ng_model_sampling": {"temperature": -1}})


@pytest.mark.parametrize(
    "url_path, expected",
    [
        ("/v1/responses", "/ng-sampling/SEG/v1/responses"),
        ("/v1/chat/completions?trace=kept", "/ng-sampling/SEG/v1/chat/completions?trace=kept"),
        ("/v1/messages", "/ng-sampling/SEG/v1/messages"),
        ("/ng-rollout/4-2/v1/responses", "/ng-rollout/4-2/ng-sampling/SEG/v1/responses"),
        (
            "/ng-rollout/4-2-a1/training-token-capture/v1/chat/completions",
            "/ng-rollout/4-2-a1/training-token-capture/ng-sampling/SEG/v1/chat/completions",
        ),
        (
            "/ng-rollout/4-2/ng-agent-instance/judge/training-token-capture/v1/messages",
            "/ng-rollout/4-2/ng-agent-instance/judge/training-token-capture/ng-sampling/SEG/v1/messages",
        ),
        # Already carries a segment, is not a generation call, or has a malformed prefix: unchanged.
        ("/ng-rollout/4-2/training-token-capture/ng-agent-instance/judge/v1/responses",) * 2,
        ("/ng-rollout/4-2/ng-sampling/OTHER/v1/responses", "/ng-rollout/4-2/ng-sampling/OTHER/v1/responses"),
        ("/v1/embeddings", "/v1/embeddings"),
        ("/tokenize", "/tokenize"),
        ("/ng-control/v1/checkpoint/prepare", "/ng-control/v1/checkpoint/prepare"),
    ],
)
def test_the_segment_goes_after_the_rollout_prefix_of_generation_calls_only(url_path: str, expected: str) -> None:
    assert with_model_sampling_segment(url_path, "SEG") == expected
    assert with_model_sampling_segment(url_path, None) == url_path


def test_the_context_scopes_a_segment_to_its_block() -> None:
    assert current_model_sampling_segment() is None
    with model_sampling_context(MODEL_SAMPLING_NULL):
        assert current_model_sampling_segment() == MODEL_SAMPLING_NULL
        with model_sampling_context("values"):
            assert current_model_sampling_segment() == "values"
        assert current_model_sampling_segment() == MODEL_SAMPLING_NULL
    assert current_model_sampling_segment() is None


def test_the_global_block_marks_a_run_whose_rows_carry_values() -> None:
    config = model_call_sampling_config(
        OmegaConf.create(
            {
                "policy_model": {"responses_api_models": {"vllm_model": {}}},
                "model_call_sampling": {"required_model_servers": ["policy_model"]},
            }
        )
    )

    assert config is not None and config.required_model_servers == ["policy_model"]
    assert model_call_sampling_config(OmegaConf.create({"model_call_sampling": {}})).required_model_servers == []
    assert model_call_sampling_config(OmegaConf.create({})) is None
    assert model_call_sampling_config(None) is None
    with pytest.raises(ValidationError):
        model_call_sampling_config({"model_call_sampling": {"required_model_server": ["policy_model"]}})


@pytest.mark.parametrize("name", ["polcy_model", "policy_agent"])
def test_a_required_server_that_is_not_a_model_server_fails(name: str) -> None:
    """A misspelled or wrong name would otherwise leave the trained agent's Model Server unenforced."""
    with pytest.raises(ValueError, match=rf"not Model Servers in this run: \['{name}'\]"):
        model_call_sampling_config(
            OmegaConf.create(
                {
                    "policy_model": {"responses_api_models": {"vllm_model": {}}},
                    "policy_agent": {"responses_api_agents": {"simple_agent": {}}},
                    "model_call_sampling": {"required_model_servers": [name]},
                }
            )
        )
