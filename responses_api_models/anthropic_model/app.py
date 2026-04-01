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
Anthropic model server for NeMo-Gym.

Stateless proxy that receives ready-to-send Anthropic API parameters,
forwards them to the Anthropic Messages API, and returns the raw response.
All context management (conversation history, trimming) is handled by the
agent/adapter layer — this server is a pure API relay.
"""

import logging
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
)


logger = logging.getLogger(__name__)


class AnthropicModelServerConfig(BaseResponsesAPIModelConfig):
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_timeout: float = 300.0


class AnthropicProxyRequest(BaseModel):
    """Request body for the Anthropic proxy endpoint.

    Contains Anthropic-native API parameters, prepared by the adapter.
    """

    model_config = ConfigDict(extra="allow")

    messages: List[Dict[str, Any]]
    system: str = ""
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    betas: List[str] = Field(default_factory=list)
    max_tokens: int = 4096
    output_config: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class AnthropicModelServer(SimpleResponsesAPIModel):
    config: AnthropicModelServerConfig

    def model_post_init(self, context):
        self._client = AsyncAnthropic(
            api_key=self.config.anthropic_api_key,
            max_retries=4,
            timeout=self.config.anthropic_timeout,
        )
        return super().model_post_init(context)

    async def responses(self, body: AnthropicProxyRequest = Body()):  # type: ignore[override]
        api_params = {
            "model": body.model or self.config.anthropic_model,
            "max_tokens": body.max_tokens,
            "system": body.system,
            "tools": body.tools,
            "betas": body.betas,
            "messages": body.messages,
        }

        if body.output_config:
            api_params["output_config"] = body.output_config

        logger.info(
            "Anthropic proxy: model=%s, num_messages=%d, betas=%s",
            api_params["model"],
            len(api_params["messages"]),
            api_params["betas"],
        )

        try:
            response = await self._client.beta.messages.create(**api_params)
        except Exception as e:
            logger.error("Anthropic API call failed: %s", repr(e))
            raise

        logger.info("Anthropic proxy response: stop_reason=%s, blocks=%d", response.stop_reason, len(response.content))

        return response.model_dump()

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError("Anthropic model server does not support /v1/chat/completions")


if __name__ == "__main__":
    AnthropicModelServer.run_webserver()
