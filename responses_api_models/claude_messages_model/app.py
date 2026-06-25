# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Anthropic Messages (`/v1/messages`) front-end for a vLLM-served open model.

Lets a closed-source Anthropic-protocol harness (e.g. the ``claude`` CLI) drive an open
model served by vLLM. This server is purely the protocol translator: it converts Anthropic
Messages <-> Chat Completions and streams SSE the CLI accepts. Token-ID capture is inherited
from ``VLLMModel`` — every call funnels through ``VLLMModel.chat_completions``, which buffers
token IDs whenever the request is run-scoped (``/runs/<token>/...``); the agent reconciles
them later (see ``nemo_gym.base_responses_api_model.TokenIDBufferingMixin`` and
``nemo_gym.base_responses_api_agent.reconcile_token_ids``).
"""

import json
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.claude_messages_model.converter import (
    anthropic_messages_to_chat,
    anthropic_tools_to_chat,
    build_anthropic_message,
    sse_events_for_message,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class ClaudeMessagesModelConfig(VLLMModelConfig):
    # Hard cap on output tokens. The claude CLI hardcodes max_tokens=32000 in its Anthropic
    # request, which for an open model whose whole context is ~32k leaves no room for the
    # prompt (vLLM 400s, surfacing as a spurious "max output tokens exceeded"). We clamp the
    # requested value to this so a turn always fits.
    max_output_tokens: int = 4096


class ClaudeMessagesModel(VLLMModel):
    config: ClaudeMessagesModelConfig

    def setup_webserver(self) -> FastAPI:
        # Inherit VLLMModel's routes + /runs/<token> buffering capability, then add the
        # Anthropic Messages endpoint and a liveness route (GET "/" also answers the CLI's
        # HEAD "/" probe). Buffering rides along automatically via chat_completions.
        app = super().setup_webserver()
        self.setup_liveness(app)
        app.post("/v1/messages")(self._handle_messages)
        return app

    async def _handle_messages(self, request: Request):
        body = await request.json()

        messages = anthropic_messages_to_chat(body.get("system"), body.get("messages", []))
        tools = anthropic_tools_to_chat(body.get("tools"))

        cc_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            # Clamp the client's (often maxed-out) request so prompt + output fit the context.
            "max_tokens": min(body.get("max_tokens") or self.config.max_output_tokens, self.config.max_output_tokens),
        }
        if tools:
            cc_kwargs["tools"] = tools
        if body.get("temperature") is not None:
            cc_kwargs["temperature"] = body["temperature"]

        cc_params = NeMoGymChatCompletionCreateParamsNonStreaming(**cc_kwargs)
        # Token IDs are buffered inside chat_completions when this request is run-scoped.
        completion = await self.chat_completions(request, cc_params)

        choice = completion.choices[0]
        message_dict = choice.message.model_dump()

        input_tokens = completion.usage.prompt_tokens if completion.usage else 0
        output_tokens = completion.usage.completion_tokens if completion.usage else 0
        anthropic_message = build_anthropic_message(
            message=message_dict,
            finish_reason=choice.finish_reason,
            model=body.get("model") or self.config.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            message_id=f"msg_{uuid4().hex}",
        )

        if not body.get("stream", False):
            return JSONResponse(anthropic_message)

        async def event_stream():
            for event_name, data in sse_events_for_message(anthropic_message):
                yield f"event: {event_name}\ndata: {json.dumps(data)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    ClaudeMessagesModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ClaudeMessagesModel.run_webserver()  # noqa: F401
