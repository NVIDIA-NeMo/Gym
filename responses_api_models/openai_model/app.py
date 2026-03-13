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
import logging
import re
from typing import Any, Dict, Optional

from pydantic import Field


logger = logging.getLogger(__name__)

_SENSITIVE_HEADER_RE = re.compile(r"('Authorization': ')[^']*(')", re.IGNORECASE)
_SENSITIVE_COOKIE_RE = re.compile(r"('(?:Set-)?Cookie': ')[^']*(')", re.IGNORECASE)


def _sanitize_error(e: Exception) -> str:
    """Strip sensitive headers (API keys, cookies) from error repr for safe logging."""
    msg = repr(e)
    msg = _SENSITIVE_HEADER_RE.sub(r"\1[REDACTED]\2", msg)
    msg = _SENSITIVE_COOKIE_RE.sub(r"\1[REDACTED]\2", msg)
    return msg


from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


class SimpleModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    openai_organization: Optional[str] = None

    extra_body: Dict[str, Any] = Field(default_factory=dict)


class SimpleModelServer(SimpleResponsesAPIModel):
    config: SimpleModelServerConfig

    def model_post_init(self, context):
        self._client = NeMoGymAsyncOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
            organization=self.config.openai_organization,
        )

        return super().model_post_init(context)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        try:
            openai_response_dict = await self._client.create_response(**body_dict)
        except Exception as e:
            logger.error("OpenAI API call failed: %s", _sanitize_error(e))
            raise
        try:
            return NeMoGymResponse.model_validate(openai_response_dict)
        except Exception as e:
            logger.error("NeMoGymResponse validation failed: %s", repr(e))
            raise

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        openai_response_dict = await self._client.create_chat_completion(**body_dict)
        return NeMoGymChatCompletion.model_validate(openai_response_dict)


if __name__ == "__main__":
    SimpleModelServer.run_webserver()
