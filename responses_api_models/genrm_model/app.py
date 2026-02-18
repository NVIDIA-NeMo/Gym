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

from nemo_gym.server_utils import is_nemo_gym_fastapi_worker
from responses_api_models.local_vllm_model.app import (
    LocalVLLMModel,
    LocalVLLMModelConfig,
)
from responses_api_models.vllm_model.app import (
    VLLMConverter,
    VLLMConverterResponsesToChatCompletionsState,
)


class GenRMConverter(VLLMConverter):
    """Message converter for GenRM models.

    Extends VLLMConverter to handle GenRM-specific custom roles:
    - response_1: First candidate response for comparison
    - response_2: Second candidate response for comparison
    - principle: Optional judging principle for principle-based comparison
    """

    supports_principle_role: bool = True

    def _format_message(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        if m["role"] in ("response_1", "response_2", "principle"):
            state.flush_assistant()
            content = m["content"]
            if isinstance(content, list):
                content = "".join([part.get("text", "") for part in content])
            from nemo_gym.openai_utils import NeMoGymChatCompletionCustomRoleMessageParam

            converted = [NeMoGymChatCompletionCustomRoleMessageParam(role=m["role"], content=content)]
            state.messages.extend(converted)
            return
        super()._format_message(m, state)


class GenRMModelMixin:
    """Mixin that provides GenRM converter for the local vLLM backend.

    Expects config to have return_token_id_information and supports_principle_role.
    """

    def get_converter(self) -> GenRMConverter:
        return GenRMConverter(
            return_token_id_information=self.config.return_token_id_information,
            supports_principle_role=self.config.supports_principle_role,
        )


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
elif is_nemo_gym_fastapi_worker():
    app = GenRMModel.run_webserver()  # noqa: F401
