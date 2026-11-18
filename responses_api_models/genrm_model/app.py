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
Downloads the model and starts a vLLM server (e.g. via Ray). To use a GenRM vLLM pool that runs elsewhere, use
`external_app.py`, which needs no Ray.
"""

from typing import Any

from fastapi import Request

from responses_api_models.genrm_model.common import GenRMModelConfig, append_genrm_roles
from responses_api_models.local_vllm_model.app import LocalVLLMModel


class GenRMModel(LocalVLLMModel):
    """GenRM Response API Model (local vLLM server).

    Specialized Response API Model for GenRM inference. Downloads the model,
    starts a vLLM server (e.g. via Ray), and uses GenRM message formatting
    for response_1/response_2/principle roles.
    """

    ray_enabled = True

    config: GenRMModelConfig

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: dict[str, Any]) -> dict[str, Any]:
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)
        return append_genrm_roles(body_dict, supports_principle_role=self.config.supports_principle_role)


if __name__ == "__main__":
    GenRMModel.run_webserver()
