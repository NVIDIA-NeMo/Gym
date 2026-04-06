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

from nemo_gym.responses_api_models.local_vllm_model.app import (
    LocalVLLMModel,
    LocalVLLMModelConfig,
)
from nemo_gym.responses_api_models.vllm_model.app import VLLMConverter


class GenRMModelMixin:
    """Mixin that provides GenRM preprocessing for the local vLLM backend.

    Expects config to have return_token_id_information and supports_principle_role.
    """

    def get_converter(self) -> VLLMConverter:
        return VLLMConverter(
            return_token_id_information=self.config.return_token_id_information,
        )

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extend base preprocessing to inject GenRM custom roles from metadata.

        The resources server passes the comparison payload via ``metadata`` so
        that the ``input`` field carries only the conversation history and the
        request schema stays generic:

        - ``metadata["response_1"]`` → appended as a ``"response_1"`` message
        - ``metadata["response_2"]`` → appended as a ``"response_2"`` message
        - ``metadata["principle"]``  → appended as a ``"principle"`` message
          (only when ``supports_principle_role=True``)

        ``metadata`` is consumed here and not forwarded to vLLM.
        """
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)

        metadata = body_dict.pop("metadata", None) or {}
        response_1 = metadata.get("response_1")
        response_2 = metadata.get("response_2")
        principle = metadata.get("principle")

        messages = body_dict["messages"]

        if self.config.supports_principle_role and principle:
            messages.append({"role": "principle", "content": principle})

        if response_1 is not None:
            messages.append({"role": "response_1", "content": response_1})

        if response_2 is not None:
            messages.append({"role": "response_2", "content": response_2})

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


if __name__ == "__main__":
    GenRMModel.run_webserver()
