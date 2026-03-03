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

"""
GenRM Response API Model (local vLLM server).

A specialized Response API Model for GenRM (Generative Reward Model) that supports
custom roles for pairwise comparison (response_1, response_2, principle).
Downloads the model and starts a vLLM server (e.g. via Ray).
"""

from typing import Any, Dict

from fastapi import Request

from responses_api_models.local_vllm_model.app import (
    LocalVLLMModel,
    LocalVLLMModelConfig,
)
from responses_api_models.vllm_model.app import VLLMConverter


class GenRMModelMixin:
    """Mixin that provides GenRM preprocessing for the local vLLM backend.

    Expects config to have return_token_id_information and supports_principle_role.
    """

    def get_converter(self) -> VLLMConverter:
        return VLLMConverter(
            return_token_id_information=self.config.return_token_id_information,
        )

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extend base preprocessing to remap standard OpenAI roles to GenRM custom roles.

        The resources server sends comparison messages using standard OpenAI roles:

        - ``messages[-2]`` (role ``"user"``) → ``"response_1"``
        - ``messages[-1]`` (role ``"user"``) → ``"response_2"``
        - ``messages[-3]`` (role ``"system"``, when ``supports_principle_role=True``)
          → ``"principle"``

        This positional convention holds because the resources server always
        appends the two response messages at the end, after the conversation
        history (which ends with an ``assistant`` turn).
        """
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)

        messages = body_dict.get("messages", [])
        if len(messages) >= 2:
            messages[-2]["role"] = "response_1"
            messages[-1]["role"] = "response_2"

            if self.config.supports_principle_role and len(messages) >= 3:
                if messages[-3].get("role") == "system":
                    messages[-3]["role"] = "principle"

        return body_dict


class GenRMModelConfig(LocalVLLMModelConfig):
    """Configuration for GenRM with a locally managed vLLM server."""

    supports_principle_role: bool = True


class GenRMModel(GenRMModelMixin, LocalVLLMModel):
    """GenRM Response API Model (local vLLM server).

    Specialized Response API Model for GenRM inference. Downloads the model,
    starts a vLLM server (e.g. via Ray), and uses GenRM message formatting
    for response_1/response_2/principle roles.
    """

    config: GenRMModelConfig
