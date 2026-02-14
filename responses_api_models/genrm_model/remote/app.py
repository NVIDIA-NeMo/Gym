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
GenRM Response API Model (remote vLLM endpoint).

A specialized Response API Model for GenRM (Generative Reward Model) that supports
custom roles for pairwise comparison:
- response_1: First candidate response
- response_2: Second candidate response
- principle: Judging principle (optional, for principle-based comparison)

Extends VLLMModel via GenRMModelMixin. For a local vLLM server with GenRM use
genrm_model.local (same app.py/configs/tests/README structure).
Message formatting is handled by GenRMConverter.
"""

from nemo_gym.server_utils import is_nemo_gym_fastapi_worker
from responses_api_models.vllm_model.app import (
    VLLMConverter,
    VLLMConverterResponsesToChatCompletionsState,
    VLLMModel,
    VLLMModelConfig,
)


class GenRMModelConfig(VLLMModelConfig):
    """Configuration for GenRM model.

    Inherits all VLLMModelConfig parameters since GenRM is a vLLM-based model,
    but specialized for GenRM's custom roles.

    Attributes:
        supports_principle_role: Enable principle-based comparison mode
    """

    supports_principle_role: bool = True


class GenRMConverter(VLLMConverter):
    """Message converter for GenRM models.

    Extends VLLMConverter to handle GenRM-specific custom roles:
    - response_1: First candidate response for comparison
    - response_2: Second candidate response for comparison
    - principle: Optional judging principle for principle-based comparison

    These custom roles are passed through to the GenRM model's chat template.
    """

    supports_principle_role: bool = True

    def _format_message(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        """Override to handle GenRM-specific roles before delegating to parent.

        Args:
            m: Message dictionary with 'role' and 'content' fields
            state: Conversion state for tracking message accumulation
        """
        # Handle GenRM-specific custom roles
        if m["role"] in ("response_1", "response_2", "principle"):
            state.flush_assistant()
            content = m["content"]

            # Convert content to string if it's a list
            if isinstance(content, list):
                content = "".join([part.get("text", "") for part in content])

            from nemo_gym.openai_utils import NeMoGymChatCompletionCustomRoleMessageParam

            converted = [NeMoGymChatCompletionCustomRoleMessageParam(role=m["role"], content=content)]

            state.messages.extend(converted)
            return

        # Delegate standard roles to parent VLLMConverter
        super()._format_message(m, state)


class GenRMModelMixin:
    """Mixin that provides GenRM converter for both remote and local vLLM backends.

    Use with VLLMModel for remote endpoints (GenRMModel) or with LocalVLLMModel
    for a locally managed vLLM server (LocalGenRMModel in genrm_model.local).
    Expects config to have return_token_id_information and supports_principle_role.
    """

    def get_converter(self) -> GenRMConverter:
        return GenRMConverter(
            return_token_id_information=self.config.return_token_id_information,
            supports_principle_role=self.config.supports_principle_role,
        )


class GenRMModel(GenRMModelMixin, VLLMModel):
    """GenRM Response API Model (remote vLLM endpoint).

    Specialized Response API Model for GenRM (Generative Reward Model) inference.
    Inherits from VLLMModel for code reuse while specializing message formatting
    to support GenRM's custom roles for pairwise comparison.

    Use this model for:
    - Pairwise response comparison with GenRM models (remote endpoint)
    - Principle-based reward modeling
    - Any task requiring response_1/response_2 custom roles

    Configuration is handled via GenRMModelConfig. For a locally managed vLLM
    server with GenRM, use genrm_model.local instead.
    """

    config: GenRMModelConfig


if __name__ == "__main__":
    GenRMModel.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = GenRMModel.run_webserver()  # noqa: F401
