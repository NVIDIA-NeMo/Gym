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
"""Model server base classes and per-rollout model-call capture.

Every Gym model server derives from ``SimpleResponsesAPIModel``, which wires the three model
dialects (/v1/responses, /v1/chat/completions, /v1/messages) and installs the model-call capture
middleware.

Capture is opt-in, off by default. A pure-ASGI middleware records correlated /v1/responses,
/v1/chat/completions, and /v1/messages exchanges -- including failed calls -- into a
per-rollout CaptureStore, forwarding bytes downstream unchanged so it composes with
streaming (SSE) responses. Best-effort; never alters the response. Correlation is
carried by a /ng-rollout/<rollout_id>/v1/... base_url prefix, which is stripped before
routing.
"""

import asyncio
import fcntl
import inspect
import json
import logging
import os
import re
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal, NotRequired, Optional, TypedDict
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import orjson
from aiohttp import ClientResponseError
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.chat_streaming import SSE_KEEPALIVE, sanitize_streaming_chat_body, synthesize_chat_completion_sse
from nemo_gym.config.config_types import ROLLOUT_PATH_PREFIX, TOKEN_CAPTURE_PATH_SEGMENT
from nemo_gym.core.episode import VERIFIER_INVOCATION_HEADER, rollout_context
from nemo_gym.core.responses_api_types import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.models._checkpoint import (
    CheckpointableLedger,
    PolicyAdmissionMiddleware,
    PolicyModelParticipant,
    attach_capture_context,
    generation_cut_requester,
    ledger_removal_refusal,
    reject_ungated_ledger_writers,
    worker_control_root,
)
from nemo_gym.models._checkpoint_workers import (
    COORDINATOR_SOCKET_ENV,
    PolicyCoordinator,
    PolicyWorkerLink,
    coordinator_socket_path,
)
from nemo_gym.responses_converter import CustomToolStrategy
from nemo_gym.responses_streaming import (
    NamespaceMap,
    is_responses_lifecycle_chunk,
    restore_namespace_tool_calls,
    sanitize_streaming_responses_body,
    synthesize_responses_failure_sse,
    synthesize_responses_keepalive_sse,
    synthesize_responses_sse,
    validate_streaming_responses_params,
)
from nemo_gym.server_runtime._checkpoint.control import install_control_routes, install_participant
from nemo_gym.server_runtime._checkpoint.settings import checkpoint_settings
from nemo_gym.server_runtime.request_handling.rollout_context import scope_with_routed_path
from nemo_gym.server_runtime.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer
from nemo_gym.telemetry.endpoints import traced_endpoint
from nemo_gym.telemetry.span_groups import GymSpanGroup
from nemo_gym.token_id_capture import (
    CaptureContext,
    capture_tokens,
    current_capture_context,
    installed_lineage_store,
    installed_token_sink,
    register_call_intent,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)

# The store factory needs Gym's server stack.
# The leaf package does not re-export it.
from nemo_gym.token_id_capture.config import NonGeneratingRequest, token_id_capture_config
from nemo_gym.token_id_capture.control_routes import install_rollout_control_routes
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore
from nemo_gym.token_id_capture.protocols import (
    CaptureAdmissionError,
    CaptureInputError,
    CaptureLedger,
    LineageResolver,
    ParentResolutionMode,
)
from nemo_gym.token_id_capture.records import CONTEXT_LENGTH_EXCEEDED_ERROR_CODE, UNCOMMITTED_CALL_REASON
from nemo_gym.token_id_capture.store import make_token_store
from nemo_gym.utils.server_process import is_nemo_gym_fastapi_worker


logger = logging.getLogger(__name__)

_CHAT_KEEPALIVE_SECONDS = 15.0
_SSE_KEEPALIVE = SSE_KEEPALIVE
# A model server sets this attribute on an engine refusal it propagates on purpose (a length
# refusal under its propagation setting). Its value is the envelope's string code:
# ``CONTEXT_LENGTH_EXCEEDED_ERROR_CODE`` when the engine's message names the context window,
# ``None`` for a refusal of the request's own output-token parameter. The mark is the attribute's
# presence (``is_propagated_refusal``), so a ``None`` code stays distinct from an unmarked error.
# The shared error handler lets a marked error through to the server's own middleware, which
# answers with OpenAI's error envelope, and the chat-SSE error event reports it with the same
# message and code.
PROPAGATED_REFUSAL_CODE_ATTRIBUTE = "nemo_gym_propagated_refusal_code"


def is_propagated_refusal(error: BaseException) -> bool:
    """Whether a model server marked ``error`` as a refusal it propagates on purpose."""
    return hasattr(error, PROPAGATED_REFUSAL_CODE_ATTRIBUTE)


def context_overflow_error_fields(error: BaseException) -> tuple[str, str | None, str | None]:
    """Return ``(message, code, param)`` for a marked engine refusal.

    The engine's error body travels on the exception as ``response_content`` (attached by
    ``nemo_gym.client.http_client.raise_for_status``). vLLM 0.10.1 and later format it as
    ``{"error": {"message": ..., "type": ..., "param": ..., "code": 400}}``; older engines used a
    flat object with a top-level ``message``. The message is the engine's own sentence from
    either shape, verbatim; any other body is returned as raw text so no information is lost.
    The param is the engine's ``param`` when its body names one. The code is whatever the server
    stored under ``PROPAGATED_REFUSAL_CODE_ATTRIBUTE`` when it marked the error.
    """
    body = getattr(error, "response_content", b"") or b""
    text = body.decode(errors="replace") if isinstance(body, (bytes, bytearray)) else str(body)
    message, param = text, None
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        nested = parsed.get("error")
        if isinstance(nested, dict) and isinstance(nested.get("message"), str):
            message = nested["message"]
            param = nested.get("param") if isinstance(nested.get("param"), str) else None
        elif isinstance(parsed.get("message"), str):
            message = parsed["message"]
            param = parsed.get("param") if isinstance(parsed.get("param"), str) else None
    code = getattr(error, PROPAGATED_REFUSAL_CODE_ATTRIBUTE, CONTEXT_LENGTH_EXCEEDED_ERROR_CODE)
    return message, code, param


# Stateless; shared by every model server's default /v1/messages handler.
_ANTHROPIC_CONVERTER = AnthropicConverter()


class ModelExecutionOutcome(TypedDict):
    error_category: NotRequired[str]
    upstream_attempted: bool
    response_source: Literal["upstream", "local"] | None
    upstream_status_code: int | None
    local_response_reason: str | None


def start_model_execution(request: Request, *, upstream_attempted: bool) -> ModelExecutionOutcome:
    """Keep adapter-owned execution facts separate from the served response and capture settings."""
    outcome = ModelExecutionOutcome(
        upstream_attempted=upstream_attempted,
        response_source=None,
        upstream_status_code=None,
        local_response_reason=None,
    )
    request.state.nemo_gym_model_execution = outcome
    return outcome


def _request_messages(body: Any) -> list[dict]:
    """Return the conversation carried by any supported dialect.

    Lineage uses model-authored turns to identify the parent call.
    Chat and Anthropic use ``messages``.
    Responses uses ``input``.
    The request envelope is prepended as a pseudo-turn because it also shapes the prompt.
    """
    if body is None:
        return []
    getter = body.get if isinstance(body, dict) else lambda key, default=None: getattr(body, key, default)
    messages = getter("messages", None)
    if isinstance(messages, list):
        turns = [m if isinstance(m, dict) else m.model_dump() for m in messages if m is not None]
    else:
        items = getter("input", None)
        if isinstance(items, str):
            turns = [{"role": "user", "content": items}]
        elif isinstance(items, list):
            turns = [i if isinstance(i, dict) else i.model_dump() for i in items if i is not None]
        else:
            turns = []
    return _request_envelope(getter) + turns


# This role cannot collide with a dialect role.
# It is not assistant-authored and does not affect the lookup fingerprint.
_ENVELOPE_ROLE = "_ng_request_envelope"


def _request_envelope(getter: Any) -> list[dict]:
    """Return prompt-shaping request fields that are not turns.

    Instructions and tools can be siblings of the message list.
    The chat template renders both into the prompt.
    Including them prevents prefix reuse across different request envelopes.
    """
    instructions = getter("instructions", None)
    tools = getter("tools", None)
    if not instructions and not tools:
        return []
    envelope = {"instructions": _plain(instructions), "tools": _plain(tools)}
    return [{"role": _ENVELOPE_ROLE, "content": json.dumps(envelope, sort_keys=True, default=str)}]


def _plain(value: Any) -> Any:
    """Reduce a request field to plain data so equal schemas serialize equally.

    Handlers can expose tools as dictionaries or Pydantic models.
    Normalization keeps an unchanged schema stable across handlers.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _orjson_dispatch_response(content: Any) -> Any:
    """Serialize a completed non-streaming model response with orjson.

    FastAPI serializes a bare dictionary or Pydantic model with ``jsonable_encoder``.
    That function recursively visits every token ID and log probability before standard-library ``json.dumps`` runs.

    Convert Pydantic models to JSON-compatible values before encoding the result.
    Return the encoded bytes as a ``Response`` so FastAPI does not encode them again.
    Preserve ``Response`` objects created by model-server overrides.
    """
    if isinstance(content, Response):
        return content
    if isinstance(content, BaseModel):
        content = content.model_dump(mode="json", by_alias=True)
    return Response(content=orjson.dumps(content), media_type="application/json")


class ModelCheckpointConfig(BaseModel):
    """A model server instance's ``checkpoint`` block."""

    model_config = ConfigDict(extra="forbid")

    # Only the policy instance drains during a partial-rollout checkpoint.
    # Judge and simulator instances keep serving so accepted tool and verifier calls can finish.
    policy: bool = False
    # Ask inference workers to stage the prefix of each in-flight call at a checkpoint,
    # so a restored call continues it instead of regenerating.
    # Requires external token staging on workers that implement the generation-cut endpoint.
    generation_cuts: bool = False


class BaseResponsesAPIModelConfig(BaseRunServerInstanceConfig):
    # Exact successful routes whose responses cannot contain policy-generated content.
    token_id_capture_non_generating_requests: list[NonGeneratingRequest] = Field(default_factory=list)
    drop_hosted_tools: bool = Field(
        default=False,
        description=(
            "Drop every tool type other than ``function`` and ``custom`` from a streaming Responses "
            "request instead of passing it to the backend. That covers the provider-hosted tools (web "
            "search, file search, code interpreter, image generation, computer use, remote MCP), which a "
            "backend that serves none of them cannot honor, and also client-executed built-in types such "
            "as ``local_shell``. Off by default, which passes every valid tool spec through."
        ),
    )
    drop_custom_tools: bool = Field(
        default=False,
        description=(
            "Drop Responses ``custom`` (free-form) tools from streaming requests instead of converting "
            "them. A backend that expresses function tools only (a vLLM chat route, for example) answers "
            "a request carrying one with 422, and the client retries the whole request. Off by default "
            "so a backend with free-form tool support keeps them."
        ),
    )
    stream_keepalive_s: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Hold streaming /v1/responses requests open with keepalive events while the backend computes. "
            "Responses streaming is buffered, so a client sees no event until the whole generation returns, "
            "and a client that bounds the silence between events (the Codex CLI's stream_idle_timeout_ms, "
            "300 s by default) abandons a long turn and re-sends it. SSE comments do not help: such a client's "
            "event parser drops them. When set, a call still running after this many seconds opens the stream "
            "with response.created and then sends response.in_progress every this many seconds; the item and "
            "terminal events that follow are unchanged. These events name the response before it exists, so "
            "the response the stream serves (and capture records) takes the id they announced; servers whose "
            "response ids come from an upstream Responses API refuse the setting at startup. A call that fails "
            "after the stream opened ends with response.failed instead of an HTTP error status. Off (null) by "
            "default, which sends nothing until the response is complete."
        ),
    )
    custom_tool_strategy: CustomToolStrategy = Field(
        default="passthrough",
        description=(
            "How a Responses ``custom`` (free-form) tool reaches a backend whose chat schema admits "
            "function tools only. ``passthrough`` sends the converted request unchanged and the backend "
            "decides (``drop_custom_tools`` remains the kill switch). ``function_bridge`` lowers each "
            "custom tool to a function tool with one required plain-string parameter, replays "
            "``custom_tool_call``/``custom_tool_call_output`` history in the lowered form, lifts a "
            "returned call whose name the request declared as custom back into a ``custom_tool_call`` "
            "item, and hoists custom entries from a streaming harness's ``additional_tools`` carrier "
            "instead of dropping them. The Responses-to-Chat conversion implements it, so the setting "
            "takes effect on the servers that convert: ``vllm_model`` and its subclasses (the compaction "
            "and GenRM servers included), ``inference_provider``, and ``azure_openai_model``. A native "
            "Responses path never runs the conversion, so the setting has no effect there; ``vllm_model`` "
            "rejects ``function_bridge`` together with ``is_responses_native``, whose route renders "
            "function tools only. The bridged payload's wire form depends on the backend's tool parser: "
            "an XML-style parser (``qwen3_coder``, ``nemotron_v3``) carries it as raw text, while a "
            "JSON-format parser (``hermes``, ``llama3_json``, ``mistral``) requires the model to "
            "JSON-escape it inside the arguments object."
        ),
    )
    # This instance's part in partial-rollout checkpoints, which the global ``checkpoint`` block turns on.
    checkpoint: ModelCheckpointConfig = Field(default_factory=ModelCheckpointConfig)


class BaseResponsesAPIModel(BaseServer):
    config: BaseResponsesAPIModelConfig


class SimpleResponsesAPIModel(BaseResponsesAPIModel, SimpleServer):
    # Subclasses can declare successful metadata or health routes here.
    # Unknown successful routes fail closed during training-token capture.
    non_generating_model_routes: ClassVar[frozenset[tuple[str, str]]] = frozenset()
    _capture_lineage: LineageResolver | None = PrivateAttr(default=None)
    # Main process of a multi-worker policy server under checkpointing: coordinates the workers.
    _policy_coordinator: Optional[PolicyCoordinator] = PrivateAttr(default=None)

    async def _finalize_served_response(self, response: Any) -> None:
        """Finalize capture after conversion to the response returned to the client."""

    def _validate_capture_request(
        self, body: NeMoGymResponseCreateParamsNonStreaming | NeMoGymChatCompletionCreateParamsNonStreaming
    ) -> None:
        """Validate server-specific request options before recording capture intent."""

    async def _stream_served_response(self, response: Any, events: Iterable[str | bytes]) -> StreamingResponse:
        """Serialize buffered SSE before committing externally staged capture."""
        context = current_capture_context()
        if context is not None and context.external_staging:
            # SSE generators serialize lazily. Finish that work before recording success.
            events = [event.encode("utf-8") if isinstance(event, str) else event for event in events]
            await self._finalize_served_response(response)
        return StreamingResponse(
            iter(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _should_propagate_upstream_http_error(self, error: ClientResponseError) -> bool:
        """Allow adapters to retain an explicit upstream-status compatibility policy."""
        return True

    def _mints_response_ids(self) -> bool:
        """Whether the ids of the responses ``responses()`` returns are this server's own.

        A keepalive stream (``stream_keepalive_s``) names its response before ``responses()`` returns,
        so that response takes the announced id. An id an upstream Responses API issued must reach the
        client unchanged, so a server that serves those cannot keep a stream alive.
        """
        return False

    def setup_webserver(self) -> FastAPI:
        if self.config.stream_keepalive_s is not None and not self._mints_response_ids():
            raise ValueError(
                f"stream_keepalive_s is not supported on {type(self).__name__}: its responses carry the ids "
                "an upstream Responses API issued, which a keepalive stream cannot announce before they exist"
            )
        app = FastAPI()

        @app.exception_handler(ClientResponseError)
        async def upstream_error(request: Request, error: ClientResponseError) -> Response:
            if not self._should_propagate_upstream_http_error(error):
                raise error
            # Preserve the provider's structured error while the buffered response
            # is still uncommitted, including context limits and rate limits.
            return Response(
                content=getattr(error, "response_content", error.message),
                status_code=error.status,
                media_type=(error.headers or {}).get("Content-Type", "application/json"),
            )

        self.setup_session_middleware(app)
        capture_config = ModelCallCaptureConfig.model_validate(self.server_client.global_config_dict)
        self._capture_lineage = capture_ledger = install_model_call_capture(
            app,
            capture_config,
            model_server_name=self.config.name,
            assistant_message_header=self.server_client.assistant_message_header(self.config.name),
            global_config_dict=self.server_client.global_config_dict,
            num_workers=self.config.num_workers,
            # Embedding vectors never contain policy-generated text, so an embeddings call under a
            # rollout's token-capture prefix must not mark that rollout incomplete.
            non_generating_requests=self.non_generating_model_routes
            | {("POST", "/v1/embeddings")}
            | frozenset(
                (request.method, request.path) for request in self.config.token_id_capture_non_generating_requests
            ),
        )

        model_attributes = {"nemo.gym.server.name": self.config.name}
        app.post("/v1/chat/completions")(
            traced_endpoint(
                GymSpanGroup.MODEL_CALL, "gym.model.chat_completions", self.chat_completions_dispatch, model_attributes
            )
        )

        app.post("/v1/responses")(
            traced_endpoint(GymSpanGroup.MODEL_CALL, "gym.model.responses", self.responses_dispatch, model_attributes)
        )

        # Every Gym model server speaks the Anthropic Messages API by default, mapping
        # Messages <-> Responses around its own responses() implementation. This lets blackbox
        # harnesses that require an Anthropic endpoint (e.g. the Claude Code CLI) target any
        # model server directly.
        app.post("/v1/messages")(
            traced_endpoint(GymSpanGroup.MODEL_CALL, "gym.model.messages", self.messages, model_attributes)
        )

        # OpenAI-compatible embeddings. Servers backed by an embeddings-capable provider
        # (e.g. openai_model, inference_provider) override embeddings(); the base default
        # returns 501 so a policy-only server advertises the route but rejects the call.
        app.post("/v1/embeddings")(self.embeddings)

        self.setup_policy_checkpoint(app, capture_ledger)
        return app

    def setup_policy_checkpoint(self, app: FastAPI, capture_ledger: CaptureLedger | None) -> None:
        global_config_dict = getattr(self.server_client, "global_config_dict", None)
        settings = checkpoint_settings(global_config_dict)
        if settings is not None and capture_ledger is not None:
            # Every model server sharing the run's capture ledger checks, so one outside the gate refuses to start.
            reject_ungated_ledger_writers(global_config_dict)
        if settings is None or not self.config.checkpoint.policy:
            return
        capture_settings = token_id_capture_config(global_config_dict)
        if capture_settings is not None and capture_settings.enabled and capture_ledger is None:
            raise ValueError(
                "checkpoint.policy with token_id_capture requires external_staging: only the token-free "
                "capture ledger is checkpointed, and the training framework owns the staged tokens"
            )
        if capture_ledger is not None and not isinstance(capture_ledger, CheckpointableLedger):
            raise ValueError(f"checkpoint.policy requires a checkpointable capture ledger, got {type(capture_ledger)}")
        cut_requester = None
        if self.config.checkpoint.generation_cuts:
            if capture_ledger is None:
                raise ValueError("checkpoint.generation_cuts requires token_id_capture with external_staging")
            cut_requester = generation_cut_requester(
                capture_settings.token_id_capture.resolve_control_auth_token(),
                control_root=self.generation_cut_control_root,
            )
        num_workers = self.config.num_workers or 1
        if num_workers == 1:
            participant = PolicyModelParticipant(
                capture_ledger, server_name=self.config.name, cut_requester=cut_requester
            )
            install_participant(
                app,
                participant,
                auth_token=settings.control_auth_token,
                lease_grace_seconds=settings.lease_grace_seconds,
                instance_name=self.config.name,
            )
            gate = participant.gate
        elif not is_nemo_gym_fastapi_worker():
            # The main process of a multi-worker server serves no requests.
            # It coordinates the workers uvicorn is about to spawn;
            # they inherit the socket path through the environment.
            socket_path = coordinator_socket_path()
            self._policy_coordinator = PolicyCoordinator(
                capture_ledger,
                expected_workers=num_workers,
                instance_name=self.config.name,
                lease_grace_seconds=settings.lease_grace_seconds,
                socket_path=socket_path,
            )
            self._policy_coordinator.start_in_background()
            os.environ[COORDINATOR_SOCKET_ENV] = socket_path
            return
        else:
            link = PolicyWorkerLink(
                socket_path=os.environ[COORDINATOR_SOCKET_ENV],
                server_name=self.config.name,
                cut_requester=cut_requester,
                ledger=capture_ledger,
            )
            install_control_routes(app, link.dispatch, auth_token=settings.control_auth_token)
            original_lifespan = app.router.lifespan_context

            @asynccontextmanager
            async def lifespan_with_coordinator(application: FastAPI) -> AsyncIterator[Any]:
                await link.connect()
                try:
                    async with original_lifespan(application) as state:
                        yield state
                finally:
                    await link.disconnect()

            app.router.lifespan_context = lifespan_with_coordinator
            gate = link.gate
        # Outermost of this app's middleware: a refused call must not register capture intent.
        app.add_middleware(PolicyAdmissionMiddleware, gate=gate)
        app.state.nemo_gym_policy_gate = gate

    def generation_cut_control_root(self, base_url: str) -> str:
        """The URL root where the inference worker serving ``base_url`` takes generation cuts."""
        return worker_control_root(base_url)

    @abstractmethod
    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        pass

    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    def _sanitize_streaming_responses_body(self, body: dict) -> tuple[dict, NamespaceMap]:
        """Apply this server's tool policy while mapping a streaming body to the strict shape.

        Every ``/v1/responses`` dispatch that accepts the streaming wire dialect sanitizes
        through here, so the config's ``drop_custom_tools``, ``drop_hosted_tools`` and
        ``custom_tool_strategy`` settings reach the sanitizer on each of them.
        """
        return sanitize_streaming_responses_body(
            body,
            drop_custom_tools=self.config.drop_custom_tools,
            drop_hosted_tools=self.config.drop_hosted_tools,
            hoist_carrier_custom_tools=self.config.custom_tool_strategy == "function_bridge",
        )

    async def responses_dispatch(self, request: Request, body: dict = Body()):
        """Default ``/v1/responses`` entrypoint shared by every Gym model server.

        A plain JSON request validates strictly against
        ``NeMoGymResponseCreateParamsNonStreaming`` and delegates to this server's own
        ``responses()``, preserving the historical non-streaming behavior. When the client
        requests ``stream: true`` (blackbox Responses-over-SSE harnesses like the Codex CLI
        always do), the request is first sanitized from the streaming wire dialect (extra
        bookkeeping fields, ``namespace`` tool specs — see ``nemo_gym.responses_streaming``),
        delegated to the same ``responses()``, and the complete response is re-emitted as a
        synthesized Responses SSE event stream. Upstream HTTP errors and ``HTTPException``
        retain their status before the stream starts. Other ``responses()`` failures become terminal
        ``response.failed`` events; bad-request validation still fails eagerly. With
        ``stream_keepalive_s`` set, a call that outlasts it opens the stream early and keeps it alive
        with ``response.in_progress`` events, and any later failure becomes ``response.failed``.
        """
        context = current_capture_context()
        if context is not None and context.parent_resolution_mode == "source_history":
            # ``stream: true`` is allowed: the streaming branch below makes exactly one
            # non-streaming ``responses()`` call and re-emits the completed response as
            # synthesized SSE, so the engine-bound request stays non-streaming (external
            # capture refuses a streamed backend request) and ``_stream_served_response``
            # finalizes the capture before the stream is committed. SSE-only harnesses
            # such as the Codex CLI thereby run under source-history capture. Server-held
            # history is still refused: it hides part of the context from capture.
            if body.get("previous_response_id") or body.get("conversation"):
                raise HTTPException(status_code=422, detail="Framework context requires explicit history")
        if not body.get("stream"):
            params = _validate_responses_params(body)
            response = await self._invoke_responses(request, params)
            dispatched = _orjson_dispatch_response(response)
            await self._finalize_served_response(response)
            return dispatched

        cleaned, ns_map = self._sanitize_streaming_responses_body(body)
        try:
            params = validate_streaming_responses_params(cleaned)
        except ValidationError as exc:
            raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in exc.errors()])

        return await self._stream_responses(request, params, ns_map)

    async def _stream_responses(
        self, request: Request, params: NeMoGymResponseCreateParamsNonStreaming, ns_map: NamespaceMap
    ) -> StreamingResponse:
        """Serve the same converted response to capture and Responses SSE clients."""
        keepalive_s = self.config.stream_keepalive_s
        if keepalive_s is None:
            return await self._stream_buffered_responses(request, self._invoke_responses(request, params), ns_map)

        # Keepalive events name the response before it exists, so the response takes this id.
        response_id = f"resp_{uuid4().hex}"
        pending = asyncio.create_task(self._invoke_responses(request, params, response_id=response_id))
        try:
            # StreamingResponse commits HTTP headers before iterating its body. Wait one interval
            # first, so a call that finishes sooner keeps the buffered events and HTTP error statuses.
            done, _ = await asyncio.wait({pending}, timeout=keepalive_s)
        except BaseException:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            raise
        if done:
            return await self._stream_buffered_responses(request, pending, ns_map)
        return StreamingResponse(
            self._responses_keepalive_events(request, pending, ns_map, keepalive_s, response_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _stream_buffered_responses(
        self, request: Request, invocation: Awaitable[Any], ns_map: NamespaceMap
    ) -> StreamingResponse:
        """Stream a completed response; until it completes nothing is sent to the client."""
        try:
            return await self._serve_responses_sse(request, await invocation, ns_map)
        except (ClientResponseError, HTTPException):
            # Generation is buffered: no headers or SSE bytes have been sent yet.
            # Let the shared HTTP handler retain the upstream error code and body.
            raise
        except Exception as exc:
            # Keep the existing terminal-SSE fallback for failures without an
            # upstream HTTP response to preserve.
            logger.exception("responses() failed while serving a streaming /v1/responses request")
            return StreamingResponse(
                synthesize_responses_failure_sse(str(exc)),
                media_type="text/event-stream",
            )

    async def _responses_keepalive_events(
        self,
        request: Request,
        pending: "asyncio.Task[Any]",
        ns_map: NamespaceMap,
        keepalive_s: float,
        response_id: str,
    ) -> AsyncIterator[str | bytes]:
        """Keep an opened Responses stream alive until ``pending``, which serves ``response_id``, completes."""
        try:
            for event in synthesize_responses_keepalive_sse(response_id, include_created=True):
                yield event
            while True:
                done, _ = await asyncio.wait({pending}, timeout=keepalive_s)
                if done:
                    break
                for event in synthesize_responses_keepalive_sse(response_id, include_created=False):
                    yield event
            try:
                served = await self._serve_responses_sse(request, pending.result(), ns_map, include_created=False)
            except Exception as exc:
                # The stream is already open, so only a terminal event can report the failure.
                code, message = self._streamed_responses_failure(exc)
                for event in synthesize_responses_failure_sse(
                    message, code=code, response_id=response_id, include_created=False
                ):
                    yield event
                return
            async for event in served.body_iterator:
                yield event
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    def _streamed_responses_failure(self, exc: Exception) -> tuple[Optional[str], str]:
        """The ``(code, message)`` of the ``response.failed`` event that reports a failure after keepalives.

        Mirrors what the client would have seen before the stream opened: a refusal the server
        propagates keeps its OpenAI error code (``context_length_exceeded`` tells a harness to compact
        and retry), and an upstream HTTP error follows the server's propagation policy. Upstream
        errors never report their request URL.
        """
        if is_propagated_refusal(exc):
            message, code, _ = context_overflow_error_fields(exc)
            logger.warning("responses() refused a streaming request after keepalive events with %s: %s", code, message)
            return code, message
        logger.exception("responses() failed after streaming keepalive events were sent")
        if isinstance(exc, ClientResponseError):
            if not self._should_propagate_upstream_http_error(exc):
                return "server_error", "HTTP 500: Model request failed"
            detail = getattr(exc, "response_content", None) or exc.message
            if isinstance(detail, bytes):
                detail = detail.decode(errors="replace")
            return "server_error", f"Upstream HTTP {exc.status}: {detail}"
        if isinstance(exc, HTTPException):
            return "server_error", f"HTTP {exc.status_code}: {exc.detail}"
        return "server_error", str(exc)

    async def _serve_responses_sse(
        self, request: Request, response: Any, ns_map: NamespaceMap, *, include_created: bool = True
    ) -> StreamingResponse:
        response_json = (
            response.model_dump(mode="json", by_alias=True) if isinstance(response, BaseModel) else dict(response)
        )
        response_json["output"] = restore_namespace_tool_calls(response_json.get("output") or [], ns_map)
        streamed = await self._stream_served_response(
            response_json, synthesize_responses_sse(response_json, include_created=include_created)
        )
        if "usage" in response_json:
            # The SSE usage reports an unknown detail count as 0; capture records Gym's own usage instead.
            request.state.nemo_gym_internal_usage = response_json["usage"]
        return streamed

    async def chat_completions_dispatch(self, request: Request, body: dict = Body()):
        """Default ``/v1/chat/completions`` entrypoint shared by every Gym model server.

        A non-streaming request validates strictly against
        ``NeMoGymChatCompletionCreateParamsNonStreaming`` and delegates to this server's own
        ``chat_completions()``, preserving the historical non-streaming behavior (including the
        standard 422 shape). When the client sends ``stream: true`` (blackbox
        Chat-Completions-over-SSE harnesses like the OpenClaw agent always do), the request is
        sanitized onto that same strict model (drop ``stream``/``stream_options``; see
        ``nemo_gym.chat_streaming``), validated identically, delegated to the same
        ``chat_completions()``, and the complete response is buffered and re-emitted as a
        synthesized ``chat.completion.chunk`` SSE stream. Keepalive comments maintain the
        connection while the backend computes; model output is still buffer-then-replay.

        Only a genuine boolean ``stream: true`` takes the streaming path; any other value
        (e.g. ``"false"`` or ``1``) stays on the strict non-streaming path, which rejects the
        malformed ``stream`` with the same 422 as before.
        """
        context = current_capture_context()
        if (
            context is not None
            and context.parent_resolution_mode == "source_history"
            and body.get("n") not in (None, 1)
        ):
            raise HTTPException(status_code=422, detail="Framework context requires one Chat completion")
        if body.get("stream") is not True:
            params = _validate_chat_params(body)
            response = await self._invoke_chat_completions(request, params)
            dispatched = _orjson_dispatch_response(response)
            await self._finalize_served_response(response)
            return dispatched

        cleaned, include_usage = sanitize_streaming_chat_body(body)
        params = _validate_chat_params(cleaned)

        async def completed_stream(completion: Any) -> StreamingResponse:
            completion_json = (
                completion.model_dump(mode="json") if isinstance(completion, BaseModel) else dict(completion)
            )
            return await self._stream_served_response(
                completion_json,
                synthesize_chat_completion_sse(completion_json, include_usage=include_usage),
            )

        pending = asyncio.create_task(self._invoke_chat_completions(request, params))
        try:
            # StreamingResponse commits HTTP headers before iterating its body.
            # Wait here so fast backend failures retain normal HTTP error handling.
            done, _ = await asyncio.wait({pending}, timeout=_CHAT_KEEPALIVE_SECONDS)
            if done:
                return await completed_stream(await pending)
        except BaseException:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            raise

        async def events() -> AsyncIterator[str | bytes]:
            try:
                # A long silent request can outlive the network's idle timeout even
                # when both the client and backend allow hours for generation.
                yield _SSE_KEEPALIVE
                while not pending.done():
                    done, _ = await asyncio.wait({pending}, timeout=_CHAT_KEEPALIVE_SECONDS)
                    if not done:
                        yield _SSE_KEEPALIVE
                try:
                    completion = await pending
                except Exception as exc:
                    if is_propagated_refusal(exc):
                        # Headers are committed, so the refusal the server chose to propagate is
                        # reported as an SSE error carrying the same OpenAI envelope the HTTP path
                        # answers with; a traceback adds nothing to a refusal.
                        message, code, param = context_overflow_error_fields(exc)
                        logger.warning(
                            "chat_completions() refused a streaming request after the keepalive with %s (%s): %s",
                            getattr(exc, "status", 400),
                            code,
                            message,
                        )
                        error = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
                        yield f"event: error\ndata: {json.dumps({'error': error})}\n\n"
                        return
                    logger.exception("chat_completions() failed after streaming headers were sent")
                    status = getattr(exc, "status_code", None) or getattr(exc, "status", None) or 500
                    detail = exc.detail if isinstance(exc, HTTPException) else "Model request failed"
                    if isinstance(exc, ClientResponseError):
                        # Headers are committed; apply the HTTP propagation policy to the SSE error instead.
                        if self._should_propagate_upstream_http_error(exc):
                            detail = getattr(exc, "response_content", None) or exc.message
                            if isinstance(detail, bytes):
                                detail = detail.decode(errors="replace")
                        else:
                            status = 500
                    error = {
                        "message": f"HTTP {status}: {detail}",
                        "type": "server_error" if status >= 500 else "invalid_request_error",
                        "code": status,
                    }
                    yield f"event: error\ndata: {json.dumps({'error': error})}\n\n"
                    return
                # Capture finalization and serialization failures must propagate:
                # treating them as backend SSE errors would hide an invalid capture.
                response = await completed_stream(completion)
                async for event in response.body_iterator:
                    yield event
            finally:
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _invoke_chat_completions(
        self, request: Request, params: NeMoGymChatCompletionCreateParamsNonStreaming
    ) -> NeMoGymChatCompletion:
        # chat_completions() signatures vary across servers: some take a leading `request`, some
        # only `body`. Dispatch on whichever this server declares so the shared dispatch works for
        # all of them.
        # Resolve the parent from the received request before dispatch.
        # Exact prefix supply and capture share this decision.
        if current_capture_context() is not None:
            self._validate_capture_request(params)
            request_messages = _request_messages(params)
            try:
                await resolve_parent(request_messages)
                await register_call_intent()
            except CaptureAdmissionError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            except CaptureInputError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        else:
            request_messages = None
        if "request" in inspect.signature(self.chat_completions).parameters:
            completion = await self.chat_completions(request=request, body=params)
        else:
            completion = await self.chat_completions(body=params)
        await capture_tokens(
            completion,
            request_messages=request_messages,
        )
        return completion

    async def embeddings(self, body: dict = Body()) -> dict:
        """OpenAI-compatible ``POST /v1/embeddings`` default: 501 unless a server overrides it.

        Servers backed by an embeddings-capable provider (e.g. ``openai_model``,
        ``inference_provider``) override this to pass ``body`` through to the provider and
        return the provider's JSON unchanged.
        """
        raise HTTPException(
            status_code=501,
            detail="This model server does not support the embeddings endpoint.",
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
            return await self._stream_served_response(
                anthropic_response,
                _ANTHROPIC_CONVERTER.anthropic_response_to_sse(anthropic_response),
            )
        dispatched = _orjson_dispatch_response(anthropic_response)
        await self._finalize_served_response(anthropic_response)
        return dispatched

    async def _invoke_responses(
        self, request: Request, params: NeMoGymResponseCreateParamsNonStreaming, *, response_id: Optional[str] = None
    ) -> NeMoGymResponse:
        # responses() signatures vary across servers: some take a leading `request`, some only
        # `body`. Dispatch on whichever this server declares so the default messages() works for
        # all of them.
        # Resolve the parent from the received request before dispatch.
        # Exact prefix supply and capture share this decision.
        context = current_capture_context()
        if context is not None:
            self._validate_capture_request(params)
            request_messages = _request_messages(params)
            try:
                await resolve_parent(request_messages)
                await register_call_intent()
            except CaptureAdmissionError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            except CaptureInputError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        else:
            request_messages = None
        if "request" in inspect.signature(self.responses).parameters:
            response = await self.responses(request=request, body=params)
        else:
            response = await self.responses(body=params)
        if response_id is not None:
            # A keepalive stream announced this id; capture records it and the client keeps it.
            response = (
                response.model_copy(update={"id": response_id})
                if isinstance(response, BaseModel)
                else {**response, "id": response_id}
            )
        # Capture before streaming dispatch wraps the response.
        # Anthropic mapping drops the token fields.
        # The assembled response still carries them here for every dialect.
        await capture_tokens(
            response,
            request_messages=request_messages,
        )
        return response


def _validate_responses_params(body: dict) -> NeMoGymResponseCreateParamsNonStreaming:
    """Validate a /v1/responses body dict, surfacing failures as FastAPI's standard 422."""
    try:
        return NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
    except ValidationError as exc:
        raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in exc.errors()])


def _validate_chat_params(body: dict) -> NeMoGymChatCompletionCreateParamsNonStreaming:
    """Validate a /v1/chat/completions body dict, surfacing failures as FastAPI's standard 422.

    ``include_url=False`` mirrors FastAPI's native body validation, which strips the
    ``errors.pydantic.dev`` url from each detail entry, so the 422 body stays byte-for-byte
    identical to the previous typed-``Body()`` binding.
    """
    try:
        return NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(body)
    except ValidationError as exc:
        raise RequestValidationError(
            [{**error, "loc": ("body", *error["loc"])} for error in exc.errors(include_url=False)]
        )


# --- Capture configuration + rollout-keyed storage ---


class ModelCallCaptureConfig(BaseModel):
    """Run-wide model-call capture settings from Gym's global config."""

    observability_enabled: bool = False
    model_call_capture_dir: Optional[Path] = None

    @model_validator(mode="after")
    def validate_capture_dir(self) -> "ModelCallCaptureConfig":
        if not self.observability_enabled:
            return self
        if self.model_call_capture_dir is None:
            raise ValueError("model_call_capture_dir is required when observability_enabled=true")
        if not self.model_call_capture_dir.is_absolute():
            raise ValueError("model_call_capture_dir must be an absolute path")
        return self


_PENDING_SUFFIX = ".capture.pending"


def _pending_token(model_call_id: str) -> str:
    """A filename-safe, dot-free token; the dot is the separator `has_pending` globs on."""
    token = "".join(char if (char.isascii() and (char.isalnum() or char in "_-")) else "_" for char in model_call_id)
    return token or "unknown"


def _validate_rollout_id(rollout_id: str) -> str:
    if not rollout_id or any(not (char.isascii() and (char.isalnum() or char in "._-")) for char in rollout_id):
        raise ValueError(f"Invalid rollout id: {rollout_id!r}")
    return rollout_id


class CaptureStore:
    """Append-only, rollout-keyed JSONL sink for model exchanges."""

    def __init__(self, root: str | Path, *, create: bool = True) -> None:
        self._root = Path(root)
        if create:
            self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, rollout_id: str) -> Path:
        return self._root / f"{_validate_rollout_id(rollout_id)}.capture.jsonl"

    def incomplete_path_for(self, rollout_id: str) -> Path:
        return self._root / f"{_validate_rollout_id(rollout_id)}.capture.incomplete"

    def mark_incomplete(self, rollout_id: str) -> None:
        self.incomplete_path_for(rollout_id).touch(exist_ok=True)

    def is_incomplete(self, rollout_id: str) -> bool:
        return self.incomplete_path_for(rollout_id).exists()

    def pending_path_for(self, rollout_id: str, model_call_id: str) -> Path:
        return self._root / f"{_validate_rollout_id(rollout_id)}.{_pending_token(model_call_id)}{_PENDING_SUFFIX}"

    def mark_pending(self, rollout_id: str, model_call_id: str) -> None:
        """Claim a write before it is queued, so a reader can tell 'not yet' from 'never'."""
        try:
            self.pending_path_for(rollout_id, model_call_id).touch(exist_ok=True)
        except OSError:
            logger.warning("Could not mark model-call capture pending for rollout %s.", rollout_id, exc_info=True)

    def clear_pending(self, rollout_id: str, model_call_id: str) -> None:
        try:
            self.pending_path_for(rollout_id, model_call_id).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not clear model-call capture pending for rollout %s.", rollout_id, exc_info=True)

    def has_pending(self, rollout_id: str) -> bool:
        """True while any claimed write for this rollout has not finished.

        The literal dot after the rollout id keeps `1-0` from matching `1-0-retry1`.
        """
        pattern = f"{_validate_rollout_id(rollout_id)}.*{_PENDING_SUFFIX}"
        return any(self._root.glob(pattern))

    def clear_all_pending(self, rollout_id: str) -> None:
        pattern = f"{_validate_rollout_id(rollout_id)}.*{_PENDING_SUFFIX}"
        for marker in self._root.glob(pattern):
            marker.unlink(missing_ok=True)

    def record(self, rollout_id: str, exchange: dict[str, Any]) -> None:
        """Append one exchange and fsync (durable across a killed box).

        ``flock`` serializes appends to the same rollout across worker processes and threads while
        allowing independent rollouts to write concurrently. This does blocking file IO + fsync,
        so callers run it off the event loop (the capture middleware uses ``asyncio.to_thread``).
        """
        line = orjson.dumps(exchange, default=str, option=orjson.OPT_APPEND_NEWLINE)
        path = self.path_for(rollout_id)
        with path.open("ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self, rollout_id: str) -> list[dict[str, Any]]:
        path = self.path_for(rollout_id)
        if not path.exists():
            return []
        exchanges: list[dict[str, Any]] = []
        # Stream line-by-line; a capture can be large (token-ids / logprobs).
        with path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    exchanges.append(orjson.loads(stripped))
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return exchanges

    def read_available(self, rollout_id: str) -> tuple[list[tuple[int, dict[str, Any]]], int]:
        """Read valid exchanges without letting one damaged line hide the rest."""
        path = self.path_for(rollout_id)
        if not path.exists():
            return [], 0
        exchanges: list[tuple[int, dict[str, Any]]] = []
        invalid_count = 0
        capture_index = 0
        with path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        exchange = orjson.loads(stripped)
                    except orjson.JSONDecodeError:
                        invalid_count += 1
                    else:
                        if isinstance(exchange, dict):
                            exchanges.append((capture_index, exchange))
                        else:
                            invalid_count += 1
                    capture_index += 1
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return exchanges, invalid_count


# --- Capture middleware ---


_OBSERVED_PATHS = {
    "/v1/responses": "responses",
    "/v1/chat/completions": "chat",
    "/v1/messages": "messages",
}

# A client may declare its persisted session ID for exact correlation with an AgentInvocation.
# It is correlation evidence, not authentication; absence is fail-safe.
_CLIENT_SESSION_HEADER = b"x-session-id"

_TERMINAL_SSE_LINES: dict[str, dict[bytes, str]] = {
    "responses": {
        b"event: response.completed": "complete",
        b"event: response.incomplete": "incomplete",
        b"event: response.failed": "error",
        b"event: error": "error",
    },
    "chat": {b"data: [DONE]": "complete", b"event: error": "error"},
    "messages": {b"event: message_stop": "complete", b"event: error": "error"},
}


def _headers_content_type(headers: list) -> bytes:
    for key, value in headers:
        if key.lower() == b"content-type":
            return value
    return b""


def _unique_request_header(headers: list, name: bytes) -> Optional[str]:
    values = {value.decode("latin-1") for key, value in headers if key.lower() == name and value}
    return values.pop() if len(values) == 1 else None


def _preserve_capture_prefix_on_redirect(
    message: dict[str, Any],
    *,
    capture_prefix: str,
    request_headers: list[tuple[bytes, bytes]],
) -> dict[str, Any]:
    """Keep rollout correlation on root-relative and same-origin redirects."""
    status = int(message.get("status") or 0)
    if not 300 <= status < 400:
        return message

    request_host = next(
        (value.decode("latin-1") for key, value in request_headers if key.lower() == b"host"),
        "",
    )
    headers = list(message.get("headers") or [])
    changed = False
    for index, (key, value) in enumerate(headers):
        if key.lower() != b"location":
            continue
        location = value.decode("latin-1")
        parts = urlsplit(location)
        if parts.netloc and parts.netloc != request_host:
            continue
        if not parts.path.startswith("/") or parts.path.startswith(f"{capture_prefix}/"):
            continue
        headers[index] = (
            key,
            urlunsplit(parts._replace(path=f"{capture_prefix}{parts.path}")).encode("latin-1"),
        )
        changed = True

    if not changed:
        return message
    return {**message, "headers": headers}


def _consume_terminal_sse_event(buffer: bytearray, dialect: str) -> Optional[str]:
    blocks = re.split(rb"\r?\n\r?\n", bytes(buffer))
    buffer[:] = blocks.pop()
    terminal_lines = _TERMINAL_SSE_LINES[dialect]
    for block in blocks:
        lines = block.splitlines()
        for line in lines:
            field, separator, value = line.partition(b":")
            normalized = field + b": " + value.lstrip() if separator else line
            if normalized in terminal_lines:
                return terminal_lines[normalized]
        if dialect == "chat":
            for line in lines:
                if not line.startswith(b"data:"):
                    continue
                try:
                    payload = json.loads(line[5:].lstrip())
                except Exception:
                    continue
                if isinstance(payload, dict) and payload.get("error") is not None:
                    return "error"
    return None


# Consumer side of the URL-prefix protocol: strip /ng-rollout/<id> before routing, key capture by
# <id>. The constant + producer (apply_rollout_prefix) are in server_utils.
_ROLLOUT_PATH_RE = re.compile(
    rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>[^/]+)"
    rf"(?:/(?P<token_capture>{re.escape(TOKEN_CAPTURE_PATH_SEGMENT)}))?"
    rf"(?P<rest>/.*)$"
)


def make_capture_store(config: ModelCallCaptureConfig) -> Optional[CaptureStore]:
    """Build a CaptureStore when observability is enabled; otherwise None."""
    if not config.observability_enabled:
        return None
    root = config.model_call_capture_dir
    assert root is not None  # enforced by ModelCallCaptureConfig
    try:
        return CaptureStore(root)
    except Exception:
        logger.warning("Could not initialize model-call capture at %s; disabling it.", root, exc_info=True)
        return None


def _classify_status(status_code: int) -> Optional[str]:
    """Normalized error_category from an HTTP status (None when < 400)."""
    if status_code < 400:
        return None
    if status_code in (408, 504):
        return "timeout"
    if status_code == 429:
        return "rate_limit"
    if status_code in (401, 403):
        return "auth"
    if status_code == 404:
        return "not_found"
    if status_code < 500:
        return "client_error"
    return "upstream_error"


def _classify_exception(exc: BaseException) -> str:
    """Normalized error_category for an exception raised while calling the model."""
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "conn" in name:
        return "connection"
    return "exception"


def _exception_http_details(exc: BaseException) -> tuple[Optional[int], bytes]:
    def read_attr(value: Any, name: str) -> Any:
        try:
            return getattr(value, name, None)
        except Exception:
            return None

    response = read_attr(exc, "response")
    status = read_attr(exc, "status")
    if not isinstance(status, int):
        status = read_attr(exc, "status_code")
    if not isinstance(status, int) and response is not None:
        status = read_attr(response, "status_code")

    body = read_attr(exc, "response_content")
    if body is None and response is not None:
        body = read_attr(response, "content")
        if body is None:
            body = read_attr(response, "text")
    if isinstance(body, str):
        body = body.encode()
    return (status if isinstance(status, int) else None, bytes(body) if isinstance(body, (bytes, bytearray)) else b"")


# --- SSE reconstruction: rebuild a final response object from a streamed body ---
def _parse_sse_events(raw: bytes) -> list[dict[str, Any]]:
    """Parse an SSE byte stream into its JSON ``data:`` payloads (best-effort; non-JSON skipped)."""
    events: list[dict[str, Any]] = []
    for block in re.split(r"\r?\n\r?\n", raw.decode("utf-8", errors="replace")):
        data_lines = [line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")]
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _reconstruct_anthropic_sse(events: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Rebuild a complete Anthropic Messages response from its streamed events."""
    message: Optional[dict[str, Any]] = None
    blocks: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    tool_json: dict[int, str] = {}
    for event in events:
        etype = event.get("type")
        if etype == "message_start":
            msg = event.get("message") or {}
            message = {k: msg.get(k) for k in ("id", "type", "role", "model", "stop_reason") if msg.get(k) is not None}
            usage.update(msg.get("usage") or {})
        elif etype == "content_block_start":
            blocks[event.get("index", len(blocks))] = dict(event.get("content_block") or {})
        elif etype == "content_block_delta":
            idx = event.get("index", 0)
            block = blocks.setdefault(idx, {})
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                block["type"] = block.get("type") or "text"
                block["text"] = (block.get("text") or "") + (delta.get("text") or "")
            elif dtype == "thinking_delta":
                block["type"] = block.get("type") or "thinking"
                block["thinking"] = (block.get("thinking") or "") + (delta.get("thinking") or "")
            elif dtype == "input_json_delta":
                tool_json[idx] = tool_json.get(idx, "") + (delta.get("partial_json") or "")
        elif etype == "message_delta":
            usage.update(event.get("usage") or {})
            stop = (event.get("delta") or {}).get("stop_reason")
            if message is not None and stop:
                message["stop_reason"] = stop
    if message is None and not blocks:
        return None
    content = []
    for idx in sorted(blocks):
        block = blocks[idx]
        if block.get("type") == "tool_use" and idx in tool_json and not block.get("input"):
            try:
                block["input"] = json.loads(tool_json[idx]) if tool_json[idx] else {}
            except Exception:
                block["input"] = {"_raw": tool_json[idx]}
        content.append(block)
    result: dict[str, Any] = {**(message or {}), "type": "message", "content": content}
    if usage:
        result["usage"] = usage
    return result


def _reconstruct_chat_sse(events: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Rebuild a Chat Completions response from streamed chunks."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    refusal_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: Optional[dict[str, Any]] = None
    model: Optional[str] = None
    response_id: Optional[str] = None
    role = "assistant"
    finish_reason: Optional[str] = None
    saw_choice = False
    for chunk in events:
        model = chunk.get("model") or model
        if isinstance(chunk.get("id"), str):
            response_id = chunk["id"]
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            saw_choice = True
            delta = choice.get("delta") or {}
            role = delta.get("role") or role
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("refusal"):
                refusal_parts.append(delta["refusal"])
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                reasoning_parts.append(reasoning)
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(
                    tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    if not saw_choice:
        return None
    message: dict[str, Any] = {"role": role, "content": "".join(content_parts) or None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if refusal_parts:
        message["refusal"] = "".join(refusal_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    result: dict[str, Any] = {
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if response_id is not None:
        result["id"] = response_id
    if usage:
        result["usage"] = usage
    return result


def _reconstruct_responses_sse(events: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Rebuild a Responses API response: the terminal envelope carries the full response object."""
    for event in reversed(events):
        if event.get("type") in ("response.completed", "response.incomplete", "response.failed") and isinstance(
            event.get("response"), dict
        ):
            return event["response"]
    for event in reversed(events):
        if isinstance(event.get("response"), dict):
            return event["response"]
    return None


def _reconstruct_streamed_response(raw: bytes, dialect: str) -> Optional[dict[str, Any]]:
    """Best-effort: reassemble a final response object from a streamed (SSE) body, by dialect."""
    events = _parse_sse_events(raw)
    if not events:
        return None
    if dialect == "messages":
        return _reconstruct_anthropic_sse(events)
    if dialect == "responses":
        return _reconstruct_responses_sse(events)
    return _reconstruct_chat_sse(events)


def _record(
    store: CaptureStore,
    dialect: str,
    model_server_name: Optional[str],
    request_bytes: bytes,
    *,
    verifier_invocation_id: Optional[str] = None,
    client_session_id: Optional[str] = None,
    client_assistant_message_id: Optional[str] = None,
    rollout_id: str,
    model_call_id: str,
    started_at: float,
    completed_at: float,
    response_body: Any,
    status_code: Optional[int],
    error_category: Optional[str],
    latency_ms: float,
    ttft_ms: Optional[float] = None,
    response_raw: Optional[str] = None,
    execution: Optional[ModelExecutionOutcome] = None,
) -> None:
    """Append one exchange (success or failure). Best-effort: never raises."""
    request_body = None
    request_raw = None
    if request_bytes:
        try:
            parsed_request = json.loads(request_bytes)
            if isinstance(parsed_request, dict):
                request_body = parsed_request
            else:
                request_raw = request_bytes.decode("utf-8", errors="replace")
        except Exception:
            request_raw = request_bytes.decode("utf-8", errors="replace")

    try:
        exchange = {
            "model_call_id": model_call_id,
            "dialect": dialect,
            "model_ref": {"type": "responses_api_models", "name": model_server_name} if model_server_name else None,
            "started_at": started_at,
            "completed_at": completed_at,
            "latency_ms": round(latency_ms, 2),
            "latency_ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "status_code": status_code,
            "error_category": error_category,
            "request": request_body,
            "response": response_body,
        }
        if verifier_invocation_id is not None:
            exchange["verifier_invocation_id"] = verifier_invocation_id
        if client_session_id is not None:
            exchange["client_session_id"] = client_session_id
        if execution is not None:
            exchange.update(execution)
            if execution["local_response_reason"] == "context_length_exceeded":
                exchange["error_category"] = "context_length_exceeded"
        if client_assistant_message_id is not None:
            exchange["client_assistant_message_id"] = client_assistant_message_id
        if request_raw is not None:
            exchange["request_raw"] = request_raw
        if response_raw is not None:
            exchange["response_raw"] = response_raw
        store.record(rollout_id, exchange)
    except Exception:
        logger.warning("Model-call capture failed for one %s call.", dialect, exc_info=True)
        try:
            store.mark_incomplete(rollout_id)
        except Exception:
            logger.warning("Could not mark rollout %s capture as incomplete.", rollout_id, exc_info=True)


async def _fail_uncommitted_external_call(context: CaptureContext | None) -> None:
    """Record a failure when an admitted worker call returns without commit coordinates.

    A call the engine refused is skipped: ``capture_engine_refusal`` already wrote its definite reason.
    """
    if (
        context is None
        or not context.external_staging
        or context.capture_admission is None
        or context.committed
        or context.refused
        or (context.parent_resolution_mode == "source_history" and not context.intent_registered)
        # A checkpoint gate owns a deferred row: it publishes it when the response starts,
        # and a call whose response never starts made no answer to record, so it is not poisoned.
        or context.deferred_commit is not None
        or context.engine_refusal_recorded
        or context.lineage_store is None
    ):
        return
    ledger = context.lineage_store
    if not isinstance(ledger, CaptureLedger):
        logger.error(
            "Cannot poison uncommitted call %s for rollout %s: the lineage resolver is not a CaptureLedger.",
            context.model_call_id,
            context.rollout_id,
        )
        return
    try:
        # A disconnected caller must not cancel the write before it reaches the ledger.
        await asyncio.shield(ledger.record_failure(context.rollout_id, context.model_call_id, UNCOMMITTED_CALL_REASON))
    except Exception:
        logger.exception(
            "Could not poison uncommitted call %s for rollout %s.",
            context.model_call_id,
            context.rollout_id,
        )


class _CaptureMiddleware:
    """Pure-ASGI per-rollout capture.

    Always strips an optional ``/ng-rollout/<id>`` path prefix before routing (used as the capture
    key) so the prefix is a stable routing feature independent of capture.
    When ``store`` is set it buffers the request body and a copy of the response while forwarding both
    downstream unchanged, so it composes with streaming (SSE) responses -- it never consumes or rewraps
    the stream. SSE chunks are forwarded immediately except for the terminal event, which is released
    after the capture is durable. Every chunk is also buffered for post-hoc reassembly, so a very long
    stream is held in memory until it completes. When ``store`` is None it avoids buffering evaluation
    records, while still persisting external capture failures before terminal events are forwarded.
    """

    def __init__(
        self,
        app: Any,
        *,
        store: CaptureStore | None,
        model_server_name: str | None,
        token_store: Any = None,
        configured_sink: Any = None,
        lineage_store: LineageResolver | None = None,
        delta_records: bool = False,
        external_staging: bool = False,
        parent_resolution_mode: ParentResolutionMode = "fingerprint",
        token_capture_enabled: bool = False,
        assistant_message_header: bytes | None = None,
        non_generating_requests: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        self._app = app
        self._store = store
        self._model_server_name = model_server_name
        self._assistant_message_header: bytes | None = (
            assistant_message_header.lower() if assistant_message_header else None
        )
        # This store records training tokens for correlated training-capture calls.
        self._token_store = token_store
        # Built from token_id_capture.sink, once, in this process.
        self._configured_sink = configured_sink
        self._lineage_store: LineageResolver | None = lineage_store
        self._delta_records = delta_records
        # A framework inference worker owns record staging; the lineage store
        # doubles as the per-rollout capture ledger.
        self._external_staging = external_staging
        self._parent_resolution_mode = parent_resolution_mode
        self._capture_ledger: CaptureLedger | None = (
            lineage_store if external_staging and isinstance(lineage_store, CaptureLedger) else None
        )
        # Capture may have no destination in this process.
        # A framework may stage records from its inference worker.
        # This process still resolves the capture identity.
        self._token_capture_enabled = token_capture_enabled
        self._non_generating_requests = non_generating_requests

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        rollout_from_path: Optional[str] = None
        token_capture_requested = False
        capture_prefix = ""
        prefix_match = _ROLLOUT_PATH_RE.match(path)
        if prefix_match:
            rollout_from_path = prefix_match.group("rollout_id")
            token_capture_requested = prefix_match.group("token_capture") is not None
            capture_prefix = path[: prefix_match.start("rest")]
            path = prefix_match.group("rest")
            scope = scope_with_routed_path(scope, path)

        method = str(scope.get("method") or "").upper()
        dialect = _OBSERVED_PATHS.get(path)
        known_non_generating = (method, path) in self._non_generating_requests

        # State is shared with the handler even through middleware scope copies.
        request_state = scope.setdefault("state", {})

        # Forward when no active store needs this correlated endpoint.
        # The prefix is already stripped.
        # An unprefixed call is forwarded rather than mixed with unrelated calls under a shared key.
        # Prefer the configured sink.
        # Then use the installed sink.
        # Finally use the file store.
        # Configured sinks are built in each server process.
        # Launcher-installed sinks do not reach spawned workers.
        # Installed sinks are resolved for each request.
        token_sink = self._configured_sink or installed_token_sink() or self._token_store
        capture_wanted = token_capture_requested and (token_sink is not None or self._token_capture_enabled)
        if (
            capture_wanted
            and self._parent_resolution_mode == "source_history"
            and dialect is not None
            and dialect not in ("responses", "chat")
        ):
            await Response(
                content='{"detail":"Source-history capture currently requires Responses or Chat"}',
                status_code=422,
                media_type="application/json",
            )(scope, receive, send)
            return
        if token_capture_requested and dialect is None and token_sink is not None:
            marked_incomplete = False
            response_started = False

            async def _mark_unobserved_incomplete(reason: str) -> None:
                nonlocal marked_incomplete
                if marked_incomplete:
                    return
                marked_incomplete = True
                try:
                    await token_sink.mark_incomplete(rollout_from_path, "")
                    logger.warning(
                        f"Marked rollout {rollout_from_path} incomplete because unclassified model request "
                        f"{method} {path} {reason}. Declare the exact request on the model server only if its "
                        "response cannot contain policy-generated content."
                    )
                except Exception:
                    logger.warning(
                        f"Could not mark rollout {rollout_from_path} incomplete for unobserved path {path}.",
                        exc_info=True,
                    )

            async def _send_unobserved(message: dict[str, Any]) -> None:
                nonlocal response_started
                if message.get("type") == "http.response.start":
                    response_started = True
                    status = int(message.get("status") or 0)
                    safe_without_capture = status in {404, 405}
                    if not known_non_generating and not safe_without_capture:
                        await _mark_unobserved_incomplete(f"returned HTTP {status}")
                    message = _preserve_capture_prefix_on_redirect(
                        message,
                        capture_prefix=capture_prefix,
                        request_headers=list(scope.get("headers") or []),
                    )
                await send(message)

            try:
                with rollout_context(rollout_from_path):
                    await self._app(scope, receive, _send_unobserved)
            except asyncio.CancelledError:
                if not known_non_generating and not response_started:
                    await _mark_unobserved_incomplete("was cancelled before starting a response")
                raise
            except Exception:
                if not known_non_generating and not response_started:
                    await _mark_unobserved_incomplete("failed before starting a response")
                raise
            if not known_non_generating and not response_started:
                await _mark_unobserved_incomplete("completed without starting a response")
            return
        if (self._store is None and not capture_wanted) or rollout_from_path is None or dialect is None:
            # Publish the id to this handler's current_rollout_id() even on the plain
            # forward path -- this is the common case for a model server with capture
            # disabled, and the id is already known here from the prefix above.
            with rollout_context(rollout_from_path):
                await self._app(scope, receive, send)
            return

        rollout_id = rollout_from_path
        model_call_id = uuid4().hex
        verifier_invocation_id = _unique_request_header(
            scope.get("headers") or [], VERIFIER_INVOCATION_HEADER.encode("ascii")
        )
        client_session_id = _unique_request_header(scope.get("headers") or [], _CLIENT_SESSION_HEADER)
        client_assistant_message_id = (
            _unique_request_header(scope.get("headers") or [], self._assistant_message_header)
            if self._assistant_message_header is not None
            else None
        )

        # Give the model server a token sink keyed to this call.
        # The sink records token ids from the complete response.
        # Middleware cannot recover token ids from SSE.
        # The context exists even without a local destination.
        # External staging uses the identity resolved here.
        sink_token = None
        capture_context = None
        if capture_wanted:
            capture_context = CaptureContext(
                rollout_id=rollout_id,
                model_call_id=model_call_id,
                token_sink=token_sink,
                lineage_store=self._capture_ledger if self._external_staging else self._lineage_store,
                delta_records=self._delta_records,
                external_staging=self._external_staging,
                parent_resolution_mode=self._parent_resolution_mode,
                admitted_at=time.time(),
            )
            sink_token = set_token_sink(capture_context)
            attach_capture_context(capture_context)

        # Training-only capture has no evaluation record.
        # Persist capture failure before forwarding a terminal event or finishing a JSON response.
        if self._store is None:
            streaming = False
            sse_event_buffer = bytearray()

            async def _send_training_only(message: dict[str, Any]) -> None:
                nonlocal streaming
                if message.get("type") == "http.response.start":
                    streaming = _headers_content_type(message.get("headers") or []).startswith(b"text/event-stream")
                elif message.get("type") == "http.response.body":
                    terminal = None
                    if streaming:
                        sse_event_buffer.extend(message.get("body", b"") or b"")
                        terminal = _consume_terminal_sse_event(sse_event_buffer, dialect)
                    if terminal is not None or not message.get("more_body", False):
                        await _fail_uncommitted_external_call(capture_context)
                await send(message)

            try:
                with rollout_context(rollout_id):
                    await self._app(scope, receive, _send_training_only)
            finally:
                await _fail_uncommitted_external_call(capture_context)
                if sink_token is not None:
                    reset_token_sink(sink_token)
            return

        request_body = bytearray()

        async def _receive() -> dict[str, Any]:
            message = await receive()
            if message.get("type") == "http.request":
                request_body.extend(message.get("body", b"") or b"")
            return message

        state: dict[str, Any] = {
            "status": None,
            "streaming": False,
            "body": bytearray(),
            "ttft_ms": None,
            "stream_terminal": None,
        }
        started_at = time.time()
        start = time.perf_counter()
        deferred_response_messages: list[dict[str, Any]] = []
        sse_event_buffer = bytearray()
        defer_response = False

        async def _send(message: dict[str, Any]) -> None:
            nonlocal defer_response
            message_type = message.get("type")
            if message_type == "http.response.start":
                state["status"] = message.get("status")
                content_type = _headers_content_type(message.get("headers") or [])
                state["streaming"] = content_type.startswith(b"text/event-stream")
            elif message_type == "http.response.body":
                chunk = message.get("body", b"") or b""
                if (
                    chunk
                    and chunk != _SSE_KEEPALIVE
                    # Lifecycle events open a `stream_keepalive_s` stream long before the first output item.
                    and not is_responses_lifecycle_chunk(chunk)
                    and state["ttft_ms"] is None
                ):
                    state["ttft_ms"] = (time.perf_counter() - start) * 1000.0
                state["body"].extend(chunk)  # buffered for both shapes; SSE is reassembled below
                if state["streaming"] and chunk and not defer_response:
                    sse_event_buffer.extend(chunk)
                    terminal = _consume_terminal_sse_event(sse_event_buffer, dialect)
                    defer_response = terminal is not None
                    if terminal is not None:
                        state["stream_terminal"] = terminal
                if defer_response or not message.get("more_body", False):
                    deferred_response_messages.append(dict(message))
                    return
            await send(message)  # forward unchanged -> streaming is preserved

        async def _flush_deferred_response() -> None:
            for message in deferred_response_messages:
                await send(message)

        # Claimed before the call is forwarded, for its whole lifetime. A claim taken only at
        # write time does not exist while the call is in flight, so a reader can look, see nothing
        # outstanding, and snapshot the rollout before the row lands.
        self._store.mark_pending(rollout_id, model_call_id)
        try:
            with rollout_context(rollout_id):
                await self._app(scope, _receive, _send)
        except (Exception, asyncio.CancelledError) as exc:
            completed_at = time.time()
            exception_status, exception_body = _exception_http_details(exc)
            upstream_status = state["status"] or exception_status
            upstream_body = bytes(state["body"]) or exception_body
            error_category = _classify_status(upstream_status) if isinstance(upstream_status, int) else None
            error_category = error_category or _classify_exception(exc)
            # Offload the blocking write+fsync so it never stalls the event loop.
            try:
                await asyncio.to_thread(
                    _record,
                    self._store,
                    dialect,
                    self._model_server_name,
                    bytes(request_body),
                    verifier_invocation_id=verifier_invocation_id,
                    client_session_id=client_session_id,
                    client_assistant_message_id=client_assistant_message_id,
                    rollout_id=rollout_id,
                    model_call_id=model_call_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    response_body=None,
                    status_code=upstream_status,
                    error_category=error_category,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    ttft_ms=state["ttft_ms"],
                    response_raw=upstream_body.decode("utf-8", errors="replace") if upstream_body else None,
                    execution=request_state.get("nemo_gym_model_execution"),
                )
            except Exception:
                logger.warning("Model-call capture finalization failed.", exc_info=True)
            finally:
                self._store.clear_pending(rollout_id, model_call_id)
                await _fail_uncommitted_external_call(capture_context)
                await _flush_deferred_response()
            raise
        finally:
            # The sink is only needed while the model server produces the response.
            await _fail_uncommitted_external_call(capture_context)
            if sink_token is not None:
                reset_token_sink(sink_token)

        completed_at = time.time()
        latency_ms = (time.perf_counter() - start) * 1000.0
        status = state["status"]
        body_bytes = bytes(state["body"])
        streaming = state["streaming"]
        stream_terminal = state["stream_terminal"]
        ttft_ms = state["ttft_ms"]
        request_bytes = bytes(request_body)
        store, model_server_name = self._store, self._model_server_name

        def _parse_and_record() -> None:
            # Off the event loop: body parse + SSE reassembly is best-effort and fully guarded, so a
            # malformed body can never surface as an ASGI error after the response was already sent.
            response_body = None
            if body_bytes:
                try:
                    response_body = (
                        _reconstruct_streamed_response(body_bytes, dialect) if streaming else json.loads(body_bytes)
                    )
                    if not isinstance(response_body, dict):
                        response_body = None
                    elif streaming and "nemo_gym_internal_usage" in request_state:
                        # Keep unknown detail counts as None, as non-streaming capture does.
                        response_body["usage"] = request_state["nemo_gym_internal_usage"]
                except Exception:
                    response_body = None
            error_category = _classify_status(status) if status is not None else None
            if error_category is None:
                error_category = {
                    "error": "upstream_error",
                    "incomplete": "incomplete",
                }.get(stream_terminal)
            if error_category is None and streaming and stream_terminal is None:
                error_category = "stream_truncated"
            # A 2xx whose body we couldn't parse/reassemble isn't a clean success -- flag it so it
            # doesn't silently count as a success with null tokens in reliability/cost sums.
            if error_category is None and body_bytes and response_body is None:
                error_category = "capture_parse_error"
            response_raw = (
                body_bytes.decode("utf-8", errors="replace")
                if body_bytes and (streaming or response_body is None)
                else None
            )
            try:
                _record(
                    store,
                    dialect,
                    model_server_name,
                    request_bytes,
                    verifier_invocation_id=verifier_invocation_id,
                    client_session_id=client_session_id,
                    client_assistant_message_id=client_assistant_message_id,
                    rollout_id=rollout_id,
                    model_call_id=model_call_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    response_body=response_body,
                    status_code=status,
                    error_category=error_category,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    response_raw=response_raw,
                    execution=request_state.get("nemo_gym_model_execution"),
                )
            finally:
                store.clear_pending(rollout_id, model_call_id)

        try:
            # Shielded: cancellation here strands the queued write and the row is
            # lost silently -- no log line, no incomplete marker. The error path
            # above already treats CancelledError as recoverable; match it.
            await asyncio.shield(asyncio.to_thread(_parse_and_record))
        except asyncio.CancelledError:
            logger.warning("Model-call capture finalization was cancelled; the write continues detached.")
            raise
        except Exception:
            logger.warning("Model-call capture finalization failed.", exc_info=True)
        finally:
            await _flush_deferred_response()


def install_model_call_capture(
    app: Any,
    config: ModelCallCaptureConfig,
    *,
    model_server_name: str | None = None,
    assistant_message_header: bytes | None = None,
    global_config_dict: Any = None,
    num_workers: int | None = None,
    non_generating_requests: frozenset[tuple[str, str]] = frozenset(),
) -> CaptureLedger | None:
    """Install model-call capture middleware and return the capture ledger when staging is external.

    Always strip ``/ng-rollout/<id>/...`` before routing.
    Evaluation capture records requests and responses for that path.
    Non-terminal SSE chunks continue immediately.
    The terminal event follows the durable evaluation write.
    Training capture uses ``/ng-rollout/<id>/training-token-capture/...``.
    That path provides a request-scoped token sink.
    The model server records token ids from its complete response.
    Consumers access records through ``TokenSource.freeze``.
    """
    capture_settings = token_id_capture_config(global_config_dict) if global_config_dict is not None else None
    token_store = make_token_store(global_config_dict) if global_config_dict is not None else None
    # Build this sink at app startup.
    # Each uvicorn worker constructs its own sink.
    # Spawned workers do not inherit a launcher-installed sink.
    configured_sink = capture_settings.build_sink() if capture_settings is not None else None
    configured_lineage = capture_settings.build_lineage_store() if capture_settings is not None else None
    lineage_store = configured_lineage or installed_lineage_store()
    default_lineage = None
    if lineage_store is None and token_store is not None:
        default_lineage = FileLineageStore(token_store.root)
        lineage_store = default_lineage
    if capture_settings is not None and capture_settings.enabled and (num_workers or 1) > 1:
        if lineage_store is None or not lineage_store.is_process_shared():
            raise ValueError(
                "token_id_capture with num_workers > 1 requires a process-shared lineage resolver "
                "over the same backend used by the token sink"
            )
    external_staging = capture_settings is not None and capture_settings.token_id_capture.external_staging
    if external_staging:
        if lineage_store is None:
            raise ValueError(
                "token_id_capture.external_staging requires a lineage resolver that implements CaptureLedger"
            )
        if isinstance(lineage_store, InMemoryLineageStore):
            # The in-memory index evicts rollouts under memory bounds, which is
            # acceptable for a resolution cache but not for a completeness
            # ledger. External staging requires a non-evicting store.
            raise ValueError(
                "token_id_capture.external_staging requires a non-evicting lineage store "
                "(e.g. FileLineageStore); InMemoryLineageStore cannot serve as the capture ledger"
            )
        if not isinstance(lineage_store, CaptureLedger):
            raise ValueError(
                "token_id_capture.external_staging requires the lineage store to implement "
                "the CaptureLedger protocol (record, record_failure, manifest, has_rows, retire, delete)"
            )
        capture_ledger: CaptureLedger = lineage_store
        install_rollout_control_routes(
            app,
            capture_ledger,
            auth_token=capture_settings.token_id_capture.resolve_control_auth_token(),
            refuse_removal=lambda: ledger_removal_refusal(app),
        )

    owned_endpoints = [
        endpoint
        for endpoint in (configured_sink, token_store, configured_lineage, default_lineage)
        if endpoint is not None
    ]

    async def _close_capture_endpoints() -> None:
        for endpoint in owned_endpoints:
            await endpoint.close()

    if owned_endpoints:
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def _capture_lifespan(application):
            try:
                async with original_lifespan(application) as state:
                    yield state
            finally:
                await _close_capture_endpoints()

        app.router.lifespan_context = _capture_lifespan
    app.add_middleware(
        _CaptureMiddleware,
        store=make_capture_store(config),
        assistant_message_header=assistant_message_header,
        model_server_name=model_server_name,
        token_store=token_store,
        configured_sink=configured_sink,
        lineage_store=lineage_store,
        delta_records=(capture_settings.token_id_capture.delta_records if capture_settings is not None else False),
        external_staging=external_staging,
        parent_resolution_mode=(
            capture_settings.token_id_capture.parent_resolution if capture_settings is not None else "fingerprint"
        ),
        token_capture_enabled=capture_settings.enabled if capture_settings is not None else False,
        non_generating_requests=non_generating_requests,
    )
    return capture_ledger if external_staging else None


# Preserve old reader imports during deprecation without eagerly importing the reader back here.
from nemo_gym.utils.compat import moved_attr_getter  # noqa: E402


__getattr__ = moved_attr_getter(
    __name__,
    {
        **dict.fromkeys(
            (
                "_token_count",
                "_usage_detail_token",
                "extract_token_stats",
                "_cache_signal",
                "_as_arguments",
                "_tool_calls_and_reasoning",
                "ModelCallRecord",
                "build_model_call_record",
                "read_model_call_records",
                "read_available_model_call_records",
                "aggregate_model_call_records",
                "aggregate_model_call_metrics",
                "model_call_capture_dirs_from_config",
                "observability_enabled_from_config",
                "CAPTURE_DRAIN_TIMEOUT_S",
                "_CAPTURE_DRAIN_POLL_S",
                "_store_for_rollout",
                "await_model_call_capture_quiescence",
                "clear_model_call_captures_for_rollouts",
                "merge_model_call_capture_into_record",
            ),
            "nemo_gym.observability.model_calls",
        ),
        "maybe_rollout_id_from_run_body": "nemo_gym.core.episode",
    },
)
