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
"""Model server base classes and per-rollout model-call capture."""

import logging
from abc import abstractmethod
from time import perf_counter
from traceback import format_exc
from typing import AsyncGenerator, Awaitable, Callable

from fastapi import Body, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from starlette.background import BackgroundTask

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.capture_records import ModelCallRecord
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.responses_streaming import (
    sanitize_streaming_responses_body,
    synthesize_responses_failure_sse,
    synthesize_responses_sse,
    validate_streaming_responses_params,
)
from nemo_gym.server_utils import (
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
)


logger = logging.getLogger(__name__)


# Stateless; shared by every model server's default /v1/messages handler.
_ANTHROPIC_CONVERTER = AnthropicConverter()
# Stateless; shared by every model server's default /v1/chat/completions handler.
_CHAT_COMPLETIONS_CONVERTER = ResponsesConverter(return_token_id_information=True)


class BaseResponsesAPIModelConfig(BaseRunServerInstanceConfig):
    pass


class BaseResponsesAPIModel(BaseServer):
    config: BaseResponsesAPIModelConfig


class SimpleResponsesAPIModel(BaseResponsesAPIModel, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        # Setup the capture middleware for model calls.
        self.setup_call_capture_middleware(app)
        if self._capture_config.should_capture_calls:
            # We allow both /v1/chat/completions/... and /v1/.../chat/completions since blackbox agents will be passed a base_url e.g. http://.../v1/ and then add their final route
            # whereas most internal calls will specify the route rather than the base_url e.g. /v1/responses
            app.post(f"/v1/chat/completions/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}")(
                self.chat_completions_with_call_capture
            )
            app.post(f"/v1/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/chat/completions")(
                self.chat_completions_with_call_capture
            )

            app.post(f"/v1/responses/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}")(self.responses_with_call_capture)
            app.post(f"/v1/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/responses")(self.responses_with_call_capture)

            app.post(f"/v1/messages/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}")(self.messages_with_call_capture)
            app.post(f"/v1/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/messages")(self.messages_with_call_capture)

        # Setup the canonical routes.
        app.post("/v1/chat/completions")(self.chat_completions)

        app.post("/v1/responses")(self.responses_dispatch)

        # Every Gym model server speaks the Anthropic Messages API by default, mapping
        # Messages <-> Responses around its own responses() implementation. This lets blackbox
        # harnesses that require an Anthropic endpoint (e.g. the Claude Code CLI) target any
        # model server directly.
        app.post("/v1/messages")(self.messages)

        return app

    async def call_capture_middleware(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.model_call_record_dict = {
            "timestamp_start": perf_counter(),
            "model_ref": ModelServerRef(type="responses_api_models", name=self.config.name),
        }

        response = await call_next(request)

        rollout_id = self._get_rollout_id(request)
        if not rollout_id:
            return response

        request.state.model_call_record_dict["timestamp_end"] = perf_counter()
        request.state.model_call_record_dict["status_code"] = response.status_code

        # Record in the background to not block the response
        # The background task only runs after streaming has finished
        task = BackgroundTask(
            self._store.record, rollout_id, [ModelCallRecord.model_validate(request.state.model_call_record_dict)]
        )

        # TODO @bxyu-nvidia: Later on we can handle cases where there are existing background tasks
        assert not response.background
        response.background = task

        return response

    @abstractmethod
    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        pass

    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    async def responses_dispatch(self, request: Request, body: dict = Body()):
        """Default ``/v1/responses`` entrypoint shared by every Gym model server.

        A plain JSON request validates strictly against
        ``NeMoGymResponseCreateParamsNonStreaming`` and delegates to this server's own
        ``responses()``, preserving the historical non-streaming behavior. When the client
        requests ``stream: true`` (blackbox Responses-over-SSE harnesses like the Codex CLI
        always do), the request is first sanitized from the streaming wire dialect (extra
        bookkeeping fields, ``namespace`` tool specs — see ``nemo_gym.responses_streaming``),
        delegated to the same ``responses()``, and the complete response is re-emitted as a
        synthesized Responses SSE event stream. A ``responses()`` failure on this path is turned
        into a terminal ``response.failed`` event rather than an HTTP 500 (bad-request validation
        still fails eagerly, before the stream is committed).
        """
        if not body.get("stream"):
            params = _validate_responses_params(body)
            return await self._invoke_responses(request, params)

        cleaned, ns_map = sanitize_streaming_responses_body(body)
        try:
            params = validate_streaming_responses_params(cleaned)
        except ValidationError as exc:
            raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in exc.errors()])

        try:
            response = await self._invoke_responses(request, params)
            response_json = response.model_dump(mode="json") if isinstance(response, BaseModel) else dict(response)
        except Exception as exc:
            # The streaming contract is already the response's shape, so a backend failure must be a
            # terminal response.failed event, not an HTTP 500 the client would see as a broken stream.
            logger.exception("responses() failed while serving a streaming /v1/responses request")
            return StreamingResponse(
                synthesize_responses_failure_sse(str(exc)),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            synthesize_responses_sse(response_json, ns_map),
            media_type="text/event-stream",
        )

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
        anthropic_response = _ANTHROPIC_CONVERTER.responses_to_anthropic_response(response, model=model_name)
        if body.get("stream"):
            return StreamingResponse(
                _ANTHROPIC_CONVERTER.anthropic_response_to_sse(anthropic_response),
                media_type="text/event-stream",
            )
        return anthropic_response

    async def _invoke_responses(
        self, request: Request, params: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        params = {"body": params}
        self._maybe_inject_request(self.responses, request, params)
        return await self.responses(**params)

    # Model call capture methods
    def _wrap_async_body_iterator_with_response_capture(
        self, request: Request, streaming_response: StreamingResponse
    ) -> None:
        request.state.model_call_record_dict["raw_response"] = []
        original_body_iterator = streaming_response.body_iterator

        async def wrapper() -> AsyncGenerator[None, str]:
            async for item in original_body_iterator:
                request.state.model_call_record_dict["raw_response"].append(item)
                yield item

        # Modifies the streaming_response in-place
        streaming_response.body_iterator = wrapper()

    async def chat_completions_with_call_capture(
        self, rollout_id: str, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump()

        request.state.model_call_record_dict["route"] = "/v1/chat/completions"

        request.state.model_call_record_dict["request"] = (
            _CHAT_COMPLETIONS_CONVERTER.chat_completion_to_responses_create_params(body_dict)
        )
        request.state.model_call_record_dict["raw_request"] = body_dict

        assert not request.state.model_call_record_dict["request"].stream, (
            "Model call capture for /v1/chat/completions with streaming is currently not supported!"
        )

        # Application-level exception catching before it's caught by FastAPI exception middleware
        try:
            response = await self.chat_completions(body)
            request.state.model_call_record_dict["response"] = _CHAT_COMPLETIONS_CONVERTER.chat_completion_to_response(
                body, response
            )
            request.state.model_call_record_dict["raw_response"] = response.model_dump()
            request.state.model_call_record_dict["error_response"] = None
        except Exception as e:
            request.state.model_call_record_dict["response"] = None
            request.state.model_call_record_dict["raw_response"] = None
            request.state.model_call_record_dict["error_response"] = format_exc()

            raise e

        return response

    async def responses_with_call_capture(
        self, rollout_id: str, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        request.state.model_call_record_dict["route"] = "/v1/responses"

        # Directly use the input body since it's already in Responses format
        request.state.model_call_record_dict["request"] = body
        # The raw request is identical to the response, so we dedupe
        request.state.model_call_record_dict["raw_request"] = None

        assert not request.state.model_call_record_dict["request"].stream, (
            "Model call capture for /v1/responses with streaming is currently not supported!"
        )

        # Application-level exception catching before it's caught by FastAPI exception middleware
        try:
            response = await self._invoke_responses(request, body)
            request.state.model_call_record_dict["response"] = response
            # The raw response is identical to the response, so we dedupe
            request.state.model_call_record_dict["raw_response"] = None
            request.state.model_call_record_dict["error_response"] = None
        except Exception as e:
            request.state.model_call_record_dict["response"] = None
            request.state.model_call_record_dict["raw_response"] = None
            request.state.model_call_record_dict["error_response"] = format_exc()

            raise e

        return response

    async def messages_with_call_capture(self, rollout_id: str, request: Request, body: dict = Body()):
        # TODO @bxyu-nvidia: This function may be round tripping with the self.messages(...) implementation
        request.state.model_call_record_dict["route"] = "/v1/messages"

        request.state.model_call_record_dict["request"] = _ANTHROPIC_CONVERTER.anthropic_request_to_responses(body)
        request.state.model_call_record_dict["raw_request"] = body

        # Application-level exception catching before it's caught by FastAPI exception middleware
        try:
            response = await self.messages(request, body)

            if request.state.model_call_record_dict["request"].stream:
                assert isinstance(response, StreamingResponse)

                # TODO @bxyu-nvidia: Need to convert from Anthropic message stream to Response object to populate this
                request.state.model_call_record_dict["response"] = None
                self._wrap_async_body_iterator_with_response_capture(request, response)
            else:
                request.state.model_call_record_dict["response"] = _ANTHROPIC_CONVERTER.anthropic_to_responses(
                    response,
                    request.state.model_call_record_dict["request"],
                    model=request.state.model_call_record_dict["request"].model,
                )
                request.state.model_call_record_dict["raw_response"] = response

            request.state.model_call_record_dict["error_response"] = None
        except Exception as e:
            request.state.model_call_record_dict["response"] = None
            request.state.model_call_record_dict["raw_response"] = None
            request.state.model_call_record_dict["error_response"] = format_exc()

            raise e

        return response


def _validate_responses_params(body: dict) -> NeMoGymResponseCreateParamsNonStreaming:
    """Validate a /v1/responses body dict, surfacing failures as FastAPI's standard 422."""
    try:
        return NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
    except ValidationError as exc:
        raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in exc.errors()])
