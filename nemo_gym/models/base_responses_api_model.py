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
import inspect
import json
import logging
import os
from abc import abstractmethod
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Iterable, Optional
from uuid import uuid4

import orjson
from aiohttp import ClientResponseError
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

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
    generation_cut_requester,
    reject_ungated_ledger_writers,
    worker_control_root,
)
from nemo_gym.models._checkpoint_workers import (
    COORDINATOR_SOCKET_ENV,
    PolicyCoordinator,
    PolicyWorkerLink,
    coordinator_socket_path,
)
from nemo_gym.models.anthropic_converter import AnthropicConverter
from nemo_gym.models.chat_completions_converter import CustomToolStrategy
from nemo_gym.models.chat_streaming import sanitize_streaming_chat_body, synthesize_chat_completion_sse
from nemo_gym.models.model_call_capture import (
    _SSE_KEEPALIVE,
    ModelCallCaptureConfig,
    install_model_call_capture,
)
from nemo_gym.models.responses_streaming import (
    NamespaceMap,
    restore_namespace_tool_calls,
    sanitize_streaming_responses_body,
    synthesize_responses_failure_sse,
    synthesize_responses_keepalive_sse,
    synthesize_responses_sse,
    validate_streaming_responses_params,
)
from nemo_gym.models.token_capture import (
    capture_tokens,
    current_capture_context,
    register_call_intent,
    resolve_parent,
)

# The store factory needs Gym's server stack.
# The leaf package does not re-export it.
from nemo_gym.models.token_capture.config import NonGeneratingRequest, token_id_capture_config
from nemo_gym.models.token_capture.protocols import (
    CaptureAdmissionError,
    CaptureInputError,
    CaptureLedger,
    LineageResolver,
)
from nemo_gym.models.token_capture.records import CONTEXT_LENGTH_EXCEEDED_ERROR_CODE
from nemo_gym.server_runtime._checkpoint.control import install_control_routes, install_participant
from nemo_gym.server_runtime._checkpoint.settings import checkpoint_settings
from nemo_gym.server_runtime.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer
from nemo_gym.telemetry.endpoints import traced_endpoint
from nemo_gym.telemetry.span_groups import GymSpanGroup
from nemo_gym.utils.server_process import is_nemo_gym_fastapi_worker


logger = logging.getLogger(__name__)

_CHAT_KEEPALIVE_SECONDS = 15.0
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
        bookkeeping fields, ``namespace`` tool specs — see ``nemo_gym.models.responses_streaming``),
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
        ``nemo_gym.models.chat_streaming``), validated identically, delegated to the same
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


# --- Capture middleware ---


# Preserve old reader imports during deprecation without eagerly importing the reader back here.
from nemo_gym._moved_paths import moved_names_getattr  # noqa: E402
from nemo_gym.utils.compat import moved_attr_getter  # noqa: E402


_moved_lazy_names = moved_attr_getter(
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
# Names that moved to other modules stay importable from here until their removal release; see nemo_gym/_moved_paths.py.
_moved_names = moved_names_getattr(__name__)


def __getattr__(name: str) -> Any:
    try:
        return _moved_names(name)
    except AttributeError:
        return _moved_lazy_names(name)
