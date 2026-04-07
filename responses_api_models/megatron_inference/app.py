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
from typing import Any, ClassVar, Dict, List, Tuple, Union

from pydantic import BaseModel, Field

from nemo_gym.openai_utils import (
    RESPONSES_TO_TRAIN,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionMessage,
    NeMoGymChatCompletionMessageForTraining,
    NeMoGymChoice,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.server_utils import is_nemo_gym_fastapi_worker
from responses_api_models.vllm_model.app import VLLMConverter, VLLMModel


class MegatronMetadataMixin(BaseModel):
    policy_epoch: List[List[Tuple[int, int]]] = Field(default_factory=lambda: [[(0, 0)]])
    kv_cache_epoch: List[List[Tuple[int, int]]] = Field(default_factory=lambda: [[(0, 0)]])
    num_evictions: List[int] = Field(default_factory=lambda: [0])


class MegatronChatCompletionMessageForTraining(
    NeMoGymChatCompletionMessageForTraining, MegatronMetadataMixin
):
    pass


class MegatronChoice(NeMoGymChoice):
    message: Union[NeMoGymChatCompletionMessage, MegatronChatCompletionMessageForTraining]


class MegatronChatCompletion(NeMoGymChatCompletion):
    choices: List[MegatronChoice]


MEGATRON_RESPONSES_TO_TRAIN = {
    base: type(f"Megatron{train.__name__}", (train, MegatronMetadataMixin), {})
    for base, train in RESPONSES_TO_TRAIN.items()
}

MegatronResponseOutputMessageForTraining = MEGATRON_RESPONSES_TO_TRAIN[NeMoGymResponseOutputMessage]


class MegatronResponse(NeMoGymResponse):
    output: List[Union[
        # Megatron training variants first so Pydantic prefers them when training fields are present
        MegatronResponseOutputMessageForTraining,
        MEGATRON_RESPONSES_TO_TRAIN[NeMoGymResponseFunctionToolCall],
        MEGATRON_RESPONSES_TO_TRAIN[NeMoGymResponseReasoningItem],
        # Non-training base types for output items that don't carry token IDs (e.g. reasoning)
        NeMoGymResponseOutputMessage,
        NeMoGymResponseFunctionToolCall,
        NeMoGymFunctionCallOutput,
        NeMoGymResponseReasoningItem,
    ]]


class MegatronInferenceConverter(VLLMConverter):
    def get_train_response_output_item_cls(
        self, response_output_item_cls: type[BaseModel]
    ) -> type[BaseModel]:
        return MEGATRON_RESPONSES_TO_TRAIN[response_output_item_cls]

    def get_extra_training_fields(self, message_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "policy_epoch": message_dict.get("policy_epoch", [[(0, 0)]]),
            "kv_cache_epoch": message_dict.get("kv_cache_epoch", [[(0, 0)]]),
            "num_evictions": message_dict.get("num_evictions", [0]),
        }


class MegatronInferenceModel(VLLMModel):
    _chat_completion_cls: ClassVar[type] = MegatronChatCompletion
    _response_cls: ClassVar[type] = MegatronResponse

    def model_post_init(self, context):
        super().model_post_init(context)
        self._converter = MegatronInferenceConverter(
            return_token_id_information=self.config.return_token_id_information,
        )


if __name__ == "__main__":
    MegatronInferenceModel.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = MegatronInferenceModel.run_webserver()  # noqa: F401
