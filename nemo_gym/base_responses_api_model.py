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

import fcntl
import inspect
import logging
import os
import threading
from abc import abstractmethod
from pathlib import Path
from shutil import rmtree
from time import perf_counter
from traceback import format_exc
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Mapping, Optional, Union

import orjson
from fastapi import Body, FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator
from starlette.background import BackgroundTask

from nemo_gym import RESULTS_DIR
from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, ModelServerRef
from nemo_gym.global_config import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_converter import ResponsesConverter
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


class ModelCallCaptureConfig(BaseModel):
    """Run-wide model-call capture settings from Gym's global config."""

    should_capture_model_calls: bool = False
    model_call_capture_dir: Optional[Path] = None

    @model_validator(mode="after")
    def validate_capture_dir(self) -> "ModelCallCaptureConfig":
        if not self.should_capture_model_calls:
            return self

        if self.model_call_capture_dir is None:
            raise ValueError("model_call_capture_dir is required when should_capture_model_calls=true")

        if not self.model_call_capture_dir.is_absolute():
            self.model_call_capture_dir = RESULTS_DIR / self.model_call_capture_dir

        return self


class ModelCallRecord(BaseModel):
    """Observability record derived from one captured model-server exchange."""

    # Rollout ID
    rollout_id: str

    # HTTP information
    status_code: int
    route: str

    # Timing information
    timestamp_start: float
    timestamp_end: float

    # Gym information
    model_ref: ModelServerRef

    # Model-call record
    request: NeMoGymResponseCreateParamsNonStreaming
    response: Optional[NeMoGymResponse]  # Only present if the call succeeded
    error_response: Optional[str]  # Only present if the call failed

    # Raw information that is only logged if it differs from the standard request and response types
    # e.g. if it is the /v1/responses route, this will be None
    raw_request: Optional[Dict[str, Any]]
    # List[str] for streaming responses
    raw_response: Optional[Union[Dict[str, Any], List[str]]]


class AggregateModelCallRecords(BaseModel):
    # Any other typically useful aggregate information can be added here
    records: List[ModelCallRecord]

    @classmethod
    def from_records(cls, records: List[ModelCallRecord]) -> "AggregateModelCallRecords":
        return cls(records=records)


class CaptureStore:
    """Append-only, rollout-keyed JSONL sink for model exchanges."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, rollout_id: str) -> Path:
        return self._root / f"{rollout_id}.capture.jsonl"

    def record(self, model_call_record: ModelCallRecord) -> None:
        """Append one exchange and fsync (durable across a killed box).

        ``flock`` serializes appends across worker processes (a model server may run with
        ``num_workers > 1``, where the in-process lock can't coordinate); the in-process lock
        serializes threads. This does blocking file IO + fsync, so callers run it off the event
        loop (the capture middleware offloads it via ``asyncio.to_thread``).
        """
        line = orjson.dumps(model_call_record.model_dump(), default=str, option=orjson.OPT_APPEND_NEWLINE)
        path = self.path_for(model_call_record.rollout_id)
        with self._lock:
            with path.open("ab") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self, rollout_id: str) -> List[ModelCallRecord]:
        path = self.path_for(rollout_id)
        if not path.exists():
            return []

        exchanges: List[ModelCallRecord] = []
        # Stream line-by-line; a capture can be large (token-ids / logprobs).
        with self._lock:
            with path.open("rb") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                try:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        exchanges.append(ModelCallRecord.model_validate(orjson.loads(stripped)))
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return exchanges

    def clear(self) -> None:
        rmtree(self.root, ignore_errors=True)

    def aggregate(self, rollout_id: str) -> AggregateModelCallRecords:
        records = self.read(rollout_id)
        return AggregateModelCallRecords.from_records(records)


def maybe_rollout_id_from_run_body(body: BaseModel | Mapping[str, Any] | None) -> Optional[str]:
    """Per-rollout model-call capture id from a run-request's task/rollout indices.

    Reads the canonical row keys (``_ng_task_index`` / ``_ng_rollout_index``) that
    rollout_collection ships to an agent's ``/run``. When a resume re-dispatch attempt is present
    (``_ng_attempt_index`` > 0), an ``-a<n>`` suffix is appended so a retry's captured model calls
    stay separable from the prior attempt; the first attempt (0) keeps the bare ``<task>-<rollout>``
    key for backward compatibility.
    """
    if isinstance(body, BaseModel):
        data = body.model_dump()
    elif isinstance(body, Mapping):
        data = body
    else:
        return None

    task = data.get(TASK_INDEX_KEY_NAME)
    rollout = data.get(ROLLOUT_INDEX_KEY_NAME)
    if task is None or rollout is None:
        return None

    rollout_id = f"{task}-{rollout}"
    attempt = data.get(ATTEMPT_INDEX_KEY_NAME)
    if attempt is not None:
        attempt_index = int(attempt)
        if attempt_index > 0:
            rollout_id = f"{rollout_id}-a{attempt_index}"

    return rollout_id


class SimpleResponsesAPIModel(BaseResponsesAPIModel, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        self._capture_config = ModelCallCaptureConfig.model_validate(self.server_client.global_config_dict)
        if self._capture_config.should_capture_model_calls:
            # Model call capture middleware must be the final middleware added so
            # 1. It is run first on request, the closest to the original request sent to the endpoint
            # 2. It is run last on response, so it can capture the response closest to what is sent back.
            # Here, we setup the exception middleware first so that we guarantee the ordering
            self._is_exception_middleware_setup = False
            self.setup_exception_middleware(app)
            self.setup_model_call_capture_middleware(app)

        app.post("/v1/chat/completions")(self.chat_completions)

        app.post("/v1/responses")(self.responses)

        # Every Gym model server speaks the Anthropic Messages API by default, mapping
        # Messages <-> Responses around its own responses() implementation. This lets blackbox
        # harnesses that require an Anthropic endpoint (e.g. the Claude Code CLI) target any
        # model server directly.
        app.post("/v1/messages")(self.messages)

        return app

    def setup_exception_middleware(self, app: FastAPI) -> None:
        if self._is_exception_middleware_setup:
            return

        return super().setup_exception_middleware(app)

    def setup_model_call_capture_middleware(self, app: FastAPI) -> None:
        server = self
        self._store = CaptureStore(self._capture_config.model_call_capture_dir)

        # This function is within this closure so it has access to `self._store`
        @app.middleware("http")
        async def model_call_capture_middleware(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            request.state.model_call_record_dict = {
                "timestamp_start": perf_counter(),
                "model_ref": ModelServerRef(type="responses_api_models", name=server.config.name),
            }

            response = await call_next(request)

            # Grab the rollout_id here after the route handler has run to populate the path_params
            rollout_id = request.path_params.get("rollout_id")

            if not rollout_id:
                return response

            request.state.model_call_record_dict["rollout_id"] = rollout_id
            request.state.model_call_record_dict["timestamp_end"] = perf_counter()
            request.state.model_call_record_dict["status_code"] = response.status_code

            # Record in the background to not block the response
            # The background task only runs after streaming has finished
            task = BackgroundTask(
                server._store.record, ModelCallRecord.model_validate(request.state.model_call_record_dict)
            )

            # TODO @bxyu-nvidia: Later on we can handle cases where there are existing background tasks
            assert not response.background
            response.background = task

            return response

        # We allow both /v1/chat/completions/... and /v1/.../chat/completions since blackbox agents will be passed a base_url e.g. http://.../v1/ and then add their final route
        # whereas most internal calls will specify the route rather than the base_url e.g. /v1/responses
        app.post(f"/v1/chat/completions/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}")(self.chat_completions_with_call_capture)
        app.post(f"/v1/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/chat/completions")(self.chat_completions_with_call_capture)

        app.post(f"/v1/responses/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}")(self.responses_with_call_capture)
        app.post(f"/v1/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/responses")(self.responses_with_call_capture)

        app.post(f"/v1/messages/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}")(self.messages_with_call_capture)
        app.post(f"/v1/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/messages")(self.messages_with_call_capture)

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
        # responses() signatures vary across servers: some take a leading `request`, some only
        # `body`. Dispatch on whichever this server declares so the default messages() works for
        # all of them.
        if "request" in inspect.signature(self.responses).parameters:
            return await self.responses(request=request, body=params)
        return await self.responses(body=params)

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
