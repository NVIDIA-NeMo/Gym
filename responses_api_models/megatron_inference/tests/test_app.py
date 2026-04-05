# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import MagicMock

from pytest import mark

from nemo_gym.openai_utils import (
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_models.megatron_inference.app import (
    MEGATRON_RESPONSES_TO_TRAIN,
    MegatronInferenceConverter,
    MegatronInferenceModel,
)
from responses_api_models.vllm_model.app import VLLMConverter, VLLMModelConfig


@mark.parametrize(
    "megatron_metadata,expected_policy_epoch,expected_kv_cache_epoch,expected_num_evictions",
    [
        (
            dict(policy_epoch=[[(1, 2)]], kv_cache_epoch=[[(3, 4)]], num_evictions=[7]),
            [[(1, 2)]],
            [[(3, 4)]],
            [7],
        ),
        (
            dict(),
            [[(0, 0)]],
            [[(0, 0)]],
            [0],
        ),
    ],
    ids=["with_megatron_metadata", "defaults_when_missing"],
)
def test_simple(
    megatron_metadata: dict,
    expected_policy_epoch: list,
    expected_kv_cache_epoch: list,
    expected_num_evictions: list,
) -> None:
    config = VLLMModelConfig(
        entrypoint="",
        name="",
        base_url="http://localhost:8000/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy_model",
        return_token_id_information=True,
        uses_reasoning_parser=False,
    )
    server = MegatronInferenceModel(config=config, server_client=MagicMock(spec=ServerClient))
    assert isinstance(server._converter, MegatronInferenceConverter)
    assert isinstance(server._converter, VLLMConverter)

    choice = NeMoGymChoice(
        index=0,
        finish_reason="stop",
        message=NeMoGymChatCompletionMessage(
            role="assistant",
            content="hi",
            tool_calls=None,
            prompt_token_ids=[1, 2, 3],
            generation_token_ids=[4, 5],
            generation_log_probs=[-0.1, -0.2],
            **megatron_metadata,
        ),
    )
    response_output = server._converter.postprocess_chat_response(choice)

    assert len(response_output) == 1
    last = response_output[-1]
    expected_cls = MEGATRON_RESPONSES_TO_TRAIN[NeMoGymResponseOutputMessage]
    assert isinstance(last, expected_cls)
    assert last.prompt_token_ids == [1, 2, 3]
    assert last.generation_token_ids == [4, 5]
    assert last.generation_log_probs == [-0.1, -0.2]
    assert last.policy_epoch == expected_policy_epoch
    assert last.kv_cache_epoch == expected_kv_cache_epoch
    assert last.num_evictions == expected_num_evictions
