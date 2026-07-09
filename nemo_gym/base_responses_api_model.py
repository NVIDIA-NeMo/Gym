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
import inspect
import json
from abc import abstractmethod
from typing import Iterator, Optional

from fastapi import Body, FastAPI, Request
from fastapi.responses import StreamingResponse

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.observability.capture_gate import ObservabilityConfig, install_ingress_gate
from nemo_gym.observability.token_sink import capture_streamed_tokens
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


# Stateless; shared by every model server's default /v1/messages handler.
_ANTHROPIC_CONVERTER = AnthropicConverter()


class BaseResponsesAPIModelConfig(BaseRunServerInstanceConfig):
    # Opt-in ingress gate (correlation, capture, token-flag injection, and
    # stripping token ids from sandbox-facing responses). When set and enabled,
    # install_ingress_gate() wraps this model server's app so blackbox rollouts
    # are captured on the server that serves them, with no separate process.
    # See nemo_gym.observability.
    observability: Optional[ObservabilityConfig] = None


class BaseResponsesAPIModel(BaseServer):
    config: BaseResponsesAPIModelConfig


class SimpleResponsesAPIModel(BaseResponsesAPIModel, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        # The route is a thin wrapper so the server also answers streaming chat
        # requests; the abstract chat_completions() stays non-streaming.
        app.post("/v1/chat/completions")(self.chat_completions_route)

        app.post("/v1/responses")(self.responses)

        # Every Gym model server speaks the Anthropic Messages API by default, mapping
        # Messages <-> Responses around its own responses() implementation. This lets blackbox
        # harnesses that require an Anthropic endpoint (e.g. the Claude Code CLI) target any
        # model server directly.
        app.post("/v1/messages")(self.messages)

        # Install the ingress gate (added after session middleware so it is the
        # outermost layer: it strips the /ng-rollout/<id> prefix and correlates
        # before routing, records the served call, and strips token ids from
        # sandbox-facing responses).
        obs: Optional[ObservabilityConfig] = getattr(self.config, "observability", None)
        if obs is not None and obs.enabled:
            model_name = getattr(self.config, "model", "") or ""
            install_ingress_gate(app, obs, model_name=model_name)

        return app

    @abstractmethod
    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        pass

    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    async def messages(self, request: Request, body: dict = Body()):
        """Default Anthropic Messages <-> Responses mapping shared by every Gym model server.

        Translates the inbound Anthropic Messages request to the Responses API, delegates to this
        server's own ``responses()`` (so it reuses whatever backend the server has), and maps the
        result back to an Anthropic Messages response. When the client requested ``stream: true``
        (the Claude Code CLI always does), the complete response is re-emitted as a synthesized
        Anthropic SSE event stream. Servers may override this for native Messages handling.
        """
        params = _ANTHROPIC_CONVERTER.anthropic_request_to_responses(body)
        response = await self._invoke_responses(request, params)
        model_name = body.get("model") or response.model
        if body.get("stream"):
            # The SSE stream drops token ids; record them here (Responses form,
            # which carries them) before synthesizing the Anthropic stream.
            capture_streamed_tokens(response.model_dump(mode="json"))
            anthropic_response = _ANTHROPIC_CONVERTER.responses_to_anthropic_response(response, model=model_name)
            return StreamingResponse(
                _ANTHROPIC_CONVERTER.anthropic_response_to_sse(anthropic_response),
                media_type="text/event-stream",
            )
        return _ANTHROPIC_CONVERTER.responses_to_anthropic_response(response, model=model_name)

    async def _invoke_responses(
        self, request: Request, params: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        # responses() signatures vary across servers: some take a leading `request`, some only
        # `body`. Dispatch on whichever this server declares so the default messages() works for
        # all of them.
        if "request" in inspect.signature(self.responses).parameters:
            return await self.responses(request=request, body=params)
        return await self.responses(body=params)

    async def chat_completions_route(self, request: Request, body: dict = Body()):
        """Chat-completions route that also serves streaming clients.

        The backend chat_completions() is non-streaming. Real OpenAI-compatible
        harnesses (e.g. the Pi CLI) always send ``stream: true``. When they do,
        we run the non-streaming completion and re-emit it as a synthesized
        OpenAI chat.completion SSE stream — the chat-dialect sibling of the
        Anthropic SSE synthesis on /v1/messages. Non-streaming requests keep the
        original behavior. Servers may override this for native streaming.
        """
        streaming = bool(body.get("stream"))
        # The backend call never needs the streaming-only fields.
        served = {k: v for k, v in body.items() if k not in ("stream", "stream_options", "store")}
        params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(served)
        completion = await self._invoke_chat_completions(request, params)
        if not streaming:
            return completion
        payload = completion.model_dump(mode="json") if hasattr(completion, "model_dump") else dict(completion)
        # The SSE stream drops token ids; record them here before synthesizing it.
        capture_streamed_tokens(payload)
        return StreamingResponse(_chat_completion_to_sse(payload), media_type="text/event-stream")

    async def _invoke_chat_completions(
        self, request: Request, params: NeMoGymChatCompletionCreateParamsNonStreaming
    ) -> NeMoGymChatCompletion:
        # As with responses(), some servers' chat_completions() take a leading `request`.
        if "request" in inspect.signature(self.chat_completions).parameters:
            return await self.chat_completions(request=request, body=params)
        return await self.chat_completions(body=params)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def _chat_completion_to_sse(completion: dict) -> Iterator[str]:
    """Re-emit a complete chat.completion as an OpenAI chat.completion.chunk SSE
    stream: a role delta, a content delta, one delta per tool call, a final
    delta carrying finish_reason (+ usage), then [DONE]. Faithful enough for an
    OpenAI-compatible client to reconstruct text and tool calls."""
    base = {
        "id": completion.get("id") or "chatcmpl-synth",
        "object": "chat.completion.chunk",
        "created": completion.get("created") or 0,
        "model": completion.get("model") or "",
    }
    choice = (completion.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason") or "stop"

    def chunk(delta: dict, finish=None) -> str:
        return _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    yield chunk({"role": message.get("role") or "assistant"})
    if message.get("content"):
        yield chunk({"content": message["content"]})
    for index, tool_call in enumerate(message.get("tool_calls") or []):
        fn = tool_call.get("function") or {}
        yield chunk(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": tool_call.get("id"),
                        "type": tool_call.get("type") or "function",
                        "function": {"name": fn.get("name"), "arguments": fn.get("arguments") or ""},
                    }
                ]
            }
        )
    final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
    if completion.get("usage"):
        final["usage"] = completion["usage"]
    yield _sse(final)
    yield "data: [DONE]\n\n"
