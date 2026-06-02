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
from typing import Any, Dict

from fastapi import Request

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_models.vllm_model.app import VLLMConverter, VLLMModel, VLLMModelConfig


class SGLangModelConfig(VLLMModelConfig):
    """OpenAI-compatible SGLang endpoint wrapped as a NeMo Gym model server."""

    # SGLang exposes OpenAI-compatible chat completions but not the vLLM-specific
    # token-id extensions Gym uses for training traces.
    return_token_id_information: bool = False

    # SGLang is not a Responses API native server; Gym maps Responses requests
    # through Chat Completions with the shared converter.
    is_responses_native: bool = False

    # SGLang's OpenAI-compatible chat endpoint rejects the Responses API
    # `developer` role. Map it to `system` before forwarding.
    replace_developer_role_with_system: bool = True


class SGLangModel(VLLMModel):
    config: SGLangModelConfig

    def get_converter(self) -> VLLMConverter:
        return VLLMConverter(return_token_id_information=self.config.return_token_id_information)

    async def _responses_native(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        raise NotImplementedError("SGLangModel does not support Responses-native forwarding.")

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.return_token_id_information:
            raise NotImplementedError(
                "SGLangModel does not support return_token_id_information. "
                "Use vllm_model/local_vllm_model for Gym training token IDs."
            )

        return super()._preprocess_chat_completion_create_params(request, body_dict)


if __name__ == "__main__":
    SGLangModel.run_webserver()
