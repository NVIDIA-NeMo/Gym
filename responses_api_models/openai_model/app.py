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
import uuid
from typing import Any, Dict, List, Optional

from pydantic import Field

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


def _chat_completion_to_response(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Chat Completions response dict to a Responses API response dict.

    Some providers (e.g. NVIDIA NIM for Claude) return Chat Completions format
    even when the /v1/responses endpoint is called.  NeMoGymResponse requires the
    Responses API shape, so we normalise here before validation.
    """
    output: List[Dict[str, Any]] = []
    for choice in d.get("choices", []):
        msg = choice.get("message", {})
        role = msg.get("role", "assistant")
        raw_content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        # Reasoning / thinking blocks (some providers surface these separately)
        for block in msg.get("reasoning_content") or []:
            output.append(
                {
                    "type": "reasoning",
                    "id": f"rs_{uuid.uuid4().hex[:24]}",
                    "summary": [{"type": "summary_text", "text": block.get("text", "")}],
                }
            )

        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                output.append(
                    {
                        "type": "function_call",
                        "id": tc.get("id", f"fc_{uuid.uuid4().hex[:24]}"),
                        "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", ""),
                    }
                )
        else:
            content_list = [{"type": "output_text", "text": raw_content, "annotations": []}]
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "role": role,
                    "content": content_list,
                    "status": "completed",
                }
            )

    raw_usage: Optional[Dict] = d.get("usage")
    usage = None
    if raw_usage:
        usage = {
            "input_tokens": raw_usage.get("prompt_tokens", 0),
            "output_tokens": raw_usage.get("completion_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

    return {
        "id": d.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
        "created_at": float(d.get("created", 0)),
        "model": d.get("model", ""),
        "object": "response",
        "status": "completed",
        "output": output,
        "usage": usage,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "text": {"format": {"type": "text"}},
    }


class SimpleModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model: str

    extra_body: Dict[str, Any] = Field(default_factory=dict)


class SimpleModelServer(SimpleResponsesAPIModel):
    config: SimpleModelServerConfig

    def model_post_init(self, context):
        self._client = NeMoGymAsyncOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
        )

        return super().model_post_init(context)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        openai_response_dict = await self._client.create_response(**body_dict)
        if openai_response_dict.get("object") == "chat.completion":
            openai_response_dict = _chat_completion_to_response(openai_response_dict)
        return NeMoGymResponse.model_validate(openai_response_dict)

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        openai_response_dict = await self._client.create_chat_completion(**body_dict)
        return NeMoGymChatCompletion.model_validate(openai_response_dict)


if __name__ == "__main__":
    SimpleModelServer.run_webserver()
