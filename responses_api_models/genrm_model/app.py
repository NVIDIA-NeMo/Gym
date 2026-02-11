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
GenRM Response API Model.

A specialized Response API Model for GenRM (Generative Reward Model) that supports
custom roles for pairwise comparison:
- response_1: First candidate response
- response_2: Second candidate response
- principle: Judging principle (optional, for principle-based comparison)

Inherits from VLLMModel for code reuse, with specialized message formatting via GenRMConverter.
"""

from typing import Dict

from nemo_gym.server_utils import is_nemo_gym_fastapi_worker

# Import from vllm_model using proper imports
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

            # Import custom role type
            from nemo_gym.openai_utils import NeMoGymChatCompletionCustomRoleMessageParam

            converted = [NeMoGymChatCompletionCustomRoleMessageParam(role=m["role"], content=content)]

            state.messages.extend(converted)
            return

        # Delegate standard roles to parent VLLMConverter
        super()._format_message(m, state)


class GenRMModel(VLLMModel):
    """GenRM Response API Model.

    Specialized Response API Model for GenRM (Generative Reward Model) inference.
    Inherits from VLLMModel for code reuse while specializing message formatting
    to support GenRM's custom roles for pairwise comparison.

    Use this model for:
    - Pairwise response comparison with GenRM models
    - Principle-based reward modeling
    - Any task requiring response_1/response_2 custom roles

    Configuration is handled via GenRMModelConfig, which extends VLLMModelConfig.
    """

    config: GenRMModelConfig

    def model_post_init(self, context):
        """Initialize with GenRMConverter instead of VLLMConverter."""
        from nemo_gym.openai_utils import NeMoGymAsyncOpenAI

        # Initialize clients (same as parent VLLMModel)
        self._clients = [
            NeMoGymAsyncOpenAI(
                base_url=base_url,
                api_key=self.config.api_key,
            )
            for base_url in self.config.base_url
        ]

        self._session_id_to_client: Dict[str, NeMoGymAsyncOpenAI] = dict()

        # Use GenRMConverter instead of VLLMConverter
        self._converter = GenRMConverter(
            return_token_id_information=self.config.return_token_id_information,
            supports_principle_role=self.config.supports_principle_role,
        )

        # Call grandparent's model_post_init (SimpleResponsesAPIModel)
        from pydantic import BaseModel as PydanticBaseModel

        return PydanticBaseModel.model_post_init(self, context)


if __name__ == "__main__":
    GenRMModel.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = GenRMModel.run_webserver()  # noqa: F401
