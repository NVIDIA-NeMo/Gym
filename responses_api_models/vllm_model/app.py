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
import asyncio
import base64
import hashlib
import json
import logging
import os
from copy import deepcopy
from threading import Lock
from time import monotonic, time, time_ns
from typing import Any, ClassVar, Dict, List, Optional, Union

from aiohttp.client_exceptions import ClientResponseError
from fastapi import Request, Response
from openai.types.responses.response import IncompleteDetails
from pydantic import Field, PrivateAttr, model_validator

from nemo_gym._checkpoint.model_control_contracts import (
    GenerationCutBackend,
    GenerationCutInventory,
    GenerationCutPrefix,
    GenerationCutPrefixAck,
    GenerationCutReceipt,
)
from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    REQUIRED_TOKEN_METADATA_FIELDS,
    TOKEN_METADATA_FIELDS,
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChatCompletionMessageForTraining,
    NeMoGymChoice,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_converter import (
    VLLMConverter,
    VLLMConverterResponsesToChatCompletionsState,  # noqa: F401
    split_responses_input_output_items,  # noqa: F401
)
from nemo_gym.rollout_correlation import capture_key_for
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)
from nemo_gym.server_utils import (
    request as http_request,
)
from nemo_gym.token_id_capture import (
    current_capture_context,
    mark_current_generation_started,
    mark_no_generation,
)
from nemo_gym.token_id_capture.config import token_id_capture_config
from nemo_gym.token_id_capture.external_capture import (
    ExternalCaptureHandler,
    make_external_capture_handler,
)
from nemo_gym.token_id_capture.staging.records import (
    GenerationCutContinuation,
)


LOG = logging.getLogger("nemo_gym.vllm_model")
_PROPAGATE_CONTEXT_ERROR_ATTRIBUTE = "nemo_gym_vllm_propagate_context_error"
_GENERATION_CUT_RPC_MAX_CONCURRENCY = 32
_STRUCTURED_GENERATION_FIELDS = (
    "guided_choice",
    "guided_grammar",
    "guided_json",
    "guided_regex",
    "guided_whitespace_pattern",
    "structural_tag",
    "structured_outputs",
)

_TRANSPORT_LOG_CONTEXT_HEADERS = {
    "run_id": "x-nemo-gym-log-run-id",
    "adapter": "x-nemo-gym-log-adapter",
    "task_id": "x-nemo-gym-log-task-id",
    "domain": "x-nemo-gym-log-domain",
    "task_attempt": "x-nemo-gym-log-task-attempt",
    "step": "x-nemo-gym-log-step",
    "parse_attempt": "x-nemo-gym-log-parse-attempt",
}


def _transport_log_context(request: Request) -> Dict[str, Any]:
    """Read opt-in Gym trace headers without changing the model body."""

    context: Dict[str, Any] = {}
    for field, header in _TRANSPORT_LOG_CONTEXT_HEADERS.items():
        value = request.headers.get(header)
        if not value:
            continue
        if field in {"task_attempt", "step", "parse_attempt"}:
            try:
                context[field] = int(value)
            except ValueError:
                continue
        else:
            context[field] = value
    return context


def _jsonable(value: Any) -> Any:
    """Return a JSON-compatible representation for transport logs."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _transport_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Index embedded images while retaining the complete request payload."""

    images: List[Dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str):
                continue
            encoded = url.split(",", 1)[1] if url.startswith("data:") and "," in url else ""
            try:
                decoded = base64.b64decode(encoded, validate=False) if encoded else b""
            except Exception:  # noqa: BLE001 - logging must not break a request.
                decoded = b""
            images.append(
                {
                    "message_index": message_index,
                    "part_index": part_index,
                    "data_url_chars": len(url),
                    "encoded_sha256": hashlib.sha256(encoded.encode("ascii", errors="ignore")).hexdigest(),
                    "decoded_bytes": len(decoded),
                    "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
                }
            )
    return images


def _append_transport_io(event: Dict[str, Any]) -> None:
    """Append exact vLLM request/response data when explicitly enabled."""

    path = os.environ.get("NEMO_GYM_VLLM_TRANSPORT_LOG", "").strip()
    if not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(event), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        LOG.exception("Failed to append vLLM transport log to %s", path)


class VLLMModelConfig(BaseResponsesAPIModelConfig):
    base_url: Union[str, List[str]]
    api_key: str
    model: str
    return_token_id_information: bool
    # Request inline prompt and generation token IDs from compatible vLLM endpoints.
    request_prompt_and_generation_token_ids: bool = False
    propagate_context_overflow_errors: bool = False

    uses_reasoning_parser: bool
    uses_interleaved_reasoning: bool = True
    # Keep reconstructed assistant history byte-for-byte in ``content`` for
    # models whose validated direct-vLLM contract includes <think> tags.
    # Response parsing remains controlled independently by
    # ``uses_reasoning_parser``.
    preserve_reasoning_in_assistant_content: bool = False
    replace_developer_role_with_system: bool = False

    # Whether or not the model can generate a reasoning output, and called again to produce additional reasoning output.
    sequential_reasoning_allowed: bool = True

    # Opt in to supplying a verified parent's exact tokens to the engine.
    # Prefix supply requires generation-time prompt_token_ids as proof.
    # Stock vLLM does not support the required_prefix_token_ids extension.
    # Prefix supply is incompatible with use_completions_api=true.
    supply_prefix_token_ids: bool = False

    # As of Feb 2026, we default this to False since majority of open source models aren't responses native with the exception of GPT-OSS
    is_responses_native: bool = False

    chat_template_kwargs: Optional[Dict[str, Any]] = None

    # When True, if the last input message is an assistant message, forward it to vLLM as a
    # prefix to continue (continue_final_message=True, add_generation_prompt=False) instead of
    # starting a fresh assistant turn. Off by default so default model-server behavior is
    # unchanged; benchmarks that seed an assistant "answer prefix" (e.g. RULER) opt in via config.
    continue_final_assistant_message: bool = False

    # Sampling params this server puts on every request it sends to the engine, replacing what the caller sent.
    # On-policy training requires generation to use the sampling distribution the policy is optimized under,
    # and a caller outside the training loop has no way to know it.
    #
    # The common case is an absent parameter rather than a conflicting one.
    # A caller need not send sampling params at all.
    # Converters forward a field only when it was set, so the outbound body can carry no temperature or top_p,
    # and the engine applies a default of its own that has no relation to the configured one.
    # Replacing rather than filling in covers the other case, a caller that sends values it chose itself.
    #
    # Read from config only, never from a request, so the server and not the caller decides them.
    # Applied at every site that builds a request for the engine,
    # since a pin that covers some endpoints and not others is off-policy while reporting that sampling is pinned.
    #
    # Unset means no pin.
    sampling_overrides: Optional[Dict[str, Any]] = None

    # Corresponds to the extra_body of OpenAI Client.
    extra_body: Optional[Dict[str, Any]] = None

    default_headers: Dict[str, str] = Field(default_factory=dict)

    # Optional path to a file that publishes the current backend base_url.
    # Used for shared serving jobs that move hosts when they restart.
    endpoint_file: Optional[str] = None

    # How long a missing endpoint file allows for the use of the last-known-good clients.
    endpoint_stale_grace_s: float = 300.0

    # Connection-error retry bound applied to clients when endpoint_file is set.
    endpoint_connection_retries: Optional[int] = 8

    # How often endpoint_file may be stat'd; otherwise the `os.stat` results is cached and reused.
    endpoint_check_interval_s: float = 10.0
    # Optional prefix for resolving relative ``metadata.audio_path`` (or
    # entries in ``metadata.audio_paths``) against. Absolute paths are used
    # as-is. When unset, relative paths raise. Audio is always inlined as a
    # ``data:audio/<fmt>;base64,...`` URI at request time — keeps the JSONL
    # small without depending on vLLM's ``--allowed-local-media-path``.
    audio_root: Optional[str] = None

    @model_validator(mode="after")
    def _validate_prefix_supply(self) -> "VLLMModelConfig":
        if self.supply_prefix_token_ids and not self.return_token_id_information:
            raise ValueError("supply_prefix_token_ids requires return_token_id_information=true")
        if self.supply_prefix_token_ids and self.use_completions_api:
            raise ValueError("supply_prefix_token_ids is not supported with use_completions_api=true")
        if self.supply_prefix_token_ids and self.is_responses_native:
            raise ValueError("supply_prefix_token_ids is not supported with is_responses_native=true")
        return self

    # When True, outbound calls go to vLLM's /v1/completions endpoint instead
    # of /v1/chat/completions. The Gym /v1/responses and /v1/chat/completions
    # external endpoints continue to work; only the upstream call swaps.
    #
    # In raw mode (render_chat_template=False, the default) the messages list
    # must be a single user message (optionally preceded by a single system
    # message); tools, multi-turn turns, audio, and non-text blocks are
    # rejected. With render_chat_template=True the messages are rendered into
    # a prompt string client-side via HF AutoTokenizer.apply_chat_template,
    # which lifts the multi-turn restriction.
    use_completions_api: bool = False

    # Only consulted when ``use_completions_api`` is True. When True, render
    # the messages list to a prompt string via HF AutoTokenizer.apply_chat_template
    # (tokenize=False, add_generation_prompt=True) before forwarding to
    # /v1/completions. The HF tokenizer is loaded once at startup from
    # ``tokenizer`` (or ``model`` if unset). Fails at startup if the loaded
    # tokenizer has no chat_template.
    render_chat_template: bool = False

    # HF identifier or local path passed to AutoTokenizer.from_pretrained.
    # When None, falls back to ``model``.
    tokenizer: Optional[str] = None

    def model_post_init(self, context):
        if isinstance(self.base_url, str):
            self.base_url = [self.base_url]
        return super().model_post_init(context)


class VLLMModel(SimpleResponsesAPIModel):
    def _defer_generation_started_to_backend_dispatch(self) -> bool:
        # Prefix recovery may need an asynchronous coordinator claim before a
        # physical backend request exists, and several validation branches return
        # locally. Keep those calls pre-generation so checkpoint prepare can park
        # them instead of trying to cut a generation that was never dispatched.
        return True

    config: VLLMModelConfig

    _TOKENIZE_CHAT_FIELDS: ClassVar[tuple[str, ...]] = (
        "model",
        "messages",
        "tools",
        "chat_template_kwargs",
        "mm_processor_kwargs",
        "required_prefix_token_ids",
    )
    _external_capture_handler: ExternalCaptureHandler | None = PrivateAttr(default=None)
    _generation_prefix_cuts_enabled: bool = PrivateAttr(default=False)
    _generation_cut_control_token: str | None = PrivateAttr(default=None)
    _generation_cut_clients: Dict[str, NeMoGymAsyncOpenAI] = PrivateAttr(default_factory=dict)
    _restored_generation_cuts: Dict[str, GenerationCutPrefixAck] = PrivateAttr(default_factory=dict)
    _generation_cut_restore_lock: Any = PrivateAttr(default_factory=Lock)

    def setup_exception_middleware(self, app) -> None:
        @app.middleware("http")
        async def context_error_middleware(request: Request, call_next):
            try:
                return await call_next(request)
            except ClientResponseError as error:
                if getattr(error, _PROPAGATE_CONTEXT_ERROR_ATTRIBUTE, False):
                    return Response(
                        content=error.response_content,
                        status_code=error.status,
                        media_type="application/json",
                    )
                raise

        super().setup_exception_middleware(app)

    def get_converter(self) -> "VLLMConverter":
        """Return the converter used for Responses API <-> Chat Completions mapping.

        Override in subclasses (e.g. GenRMModel) to use a specialized converter.
        """
        return VLLMConverter(
            return_token_id_information=self.config.return_token_id_information,
            uses_reasoning_parser=self.config.uses_reasoning_parser,
        )

    def model_post_init(self, context):
        self._post_init()
        return super().model_post_init(context)

    def _post_init(self) -> None:
        if self.config.sampling_overrides:
            LOG.info(
                "`%s` pins sampling on every request to the engine: %s",
                self.config.name,
                self.config.sampling_overrides,
            )

        self._clients = [
            NeMoGymAsyncOpenAI(
                base_url=base_url,
                api_key=self.config.api_key,
                default_headers=self.config.default_headers,
                max_connection_retries=(
                    self.config.endpoint_connection_retries if self.config.endpoint_file else None
                ),
            )
            for base_url in self.config.base_url
        ]

        self._session_id_to_client: Dict[str, NeMoGymAsyncOpenAI] = dict()
        self._endpoint_file_mtime: Optional[float] = None
        self._endpoint_missing_since: Optional[float] = None
        self._endpoint_last_check_at: Optional[float] = None

        self._converter = self.get_converter()
        self._transport_call_index = 0

        global_config = getattr(self.server_client, "global_config_dict", None)
        capture_config = token_id_capture_config(global_config) if global_config is not None else None
        self._external_capture_handler = None
        external_capture_enabled = bool(
            capture_config is not None and capture_config.token_id_capture.external_staging
        )
        self._generation_prefix_cuts_enabled = bool(
            capture_config is not None and capture_config.token_id_capture.generation_prefix_cuts_enabled
        )
        if self._generation_prefix_cuts_enabled:
            assert capture_config is not None
            self._generation_cut_control_token = capture_config.token_id_capture.resolve_control_auth_token()
        if external_capture_enabled:
            overrides = (self.config.extra_body or {}) | (self.config.sampling_overrides or {})
            if overrides.get("stream"):
                raise ValueError("external staging requires non-streaming backend requests")
            if self.config.use_completions_api:
                raise ValueError("token_id_capture.external_staging does not support use_completions_api=true")
            if self.config.is_responses_native:
                raise ValueError("token_id_capture.external_staging requires the chat-backed Responses API path")
            if self.config.return_token_id_information:
                raise ValueError(
                    "token_id_capture.external_staging requires return_token_id_information=false; "
                    "worker custody replaces the token echo"
                )
            self._external_capture_handler = make_external_capture_handler(
                capture_config.token_id_capture.external_staging_backend
            )

        self._chat_template_tokenizer = None
        if self.config.use_completions_api and self.config.render_chat_template:
            self._chat_template_tokenizer = self._load_chat_template_tokenizer()

    def generation_cut_backend(self) -> GenerationCutBackend | None:
        """Bridge Gym's frozen call inventory to the owning RL vLLM workers."""
        return self if self._generation_prefix_cuts_enabled else None

    def _remember_generation_cut_client(self, model_call_id: str, client: NeMoGymAsyncOpenAI) -> None:
        """Retain active call routing until the model request exits."""
        existing = self._generation_cut_clients.get(model_call_id)
        if existing is not None and existing is not client:
            raise RuntimeError(f"model call {model_call_id!r} changed generation worker while active")
        self._generation_cut_clients[model_call_id] = client

    def _forget_generation_cut_client(self, model_call_id: str, client: NeMoGymAsyncOpenAI) -> None:
        """Retire one active route without removing a newer owner."""
        if self._generation_cut_clients.get(model_call_id) is client:
            self._generation_cut_clients.pop(model_call_id, None)

    async def checkpoint_generation_cut(self, inventory: GenerationCutInventory) -> GenerationCutReceipt:
        """Ask each owning RL worker to stage its current generated prefix."""
        if not self._generation_prefix_cuts_enabled or self._generation_cut_control_token is None:
            raise RuntimeError("generation-prefix cuts are not enabled for this model server")

        by_client: dict[str, tuple[NeMoGymAsyncOpenAI, list[GenerationCutPrefix]]] = {}
        acknowledgements: list[GenerationCutPrefixAck] = []
        for prefix in inventory.active_prefixes:
            client = self._generation_cut_clients.get(prefix.model_call_id)
            if client is None:
                # The request was admitted but had not reached backend dispatch
                # when admission closed. No generated state exists to preserve.
                acknowledgements.append(
                    GenerationCutPrefixAck(
                        **prefix.model_dump(mode="json"),
                        disposition="durable_failure",
                    )
                )
                continue
            by_client.setdefault(client.base_url, (client, []))[1].append(prefix)

        semaphore = asyncio.Semaphore(_GENERATION_CUT_RPC_MAX_CONCURRENCY)

        async def checkpoint_client(
            client: NeMoGymAsyncOpenAI,
            prefixes: list[GenerationCutPrefix],
        ) -> GenerationCutReceipt:
            worker_inventory = GenerationCutInventory.build(
                checkpoint_id=inventory.checkpoint_id,
                server_name=inventory.server_name,
                active_prefixes=prefixes,
            )
            async with semaphore:
                response = await http_request(
                    method="POST",
                    url=f"{client.base_url.removesuffix('/v1')}/ng-control/v1/generation-cut",
                    headers={"Authorization": f"Bearer {self._generation_cut_control_token}"},
                    json=worker_inventory.model_dump(mode="json"),
                    _internal=True,
                )
                await raise_for_status(response)
                receipt = GenerationCutReceipt.model_validate(await get_response_json(response))
            receipt.validate_for(worker_inventory)
            return receipt

        tasks: list[asyncio.Task[GenerationCutReceipt]] = []
        async with asyncio.TaskGroup() as task_group:
            for base_url in sorted(by_client):
                client, prefixes = by_client[base_url]
                tasks.append(task_group.create_task(checkpoint_client(client, prefixes)))
        backend_receipts = [task.result() for task in tasks]
        for receipt in backend_receipts:
            acknowledgements.extend(receipt.prefixes)

        evidence = {
            "inventory_digest": inventory.inventory_digest,
            "backend_receipts": [receipt.model_dump(mode="json") for receipt in backend_receipts],
        }
        aggregate_digest = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        receipt = GenerationCutReceipt(
            checkpoint_id=inventory.checkpoint_id,
            cut_id=f"gym-{aggregate_digest}",
            inventory_digest=inventory.inventory_digest,
            inventory=inventory,
            backend_snapshot_id=f"tq-{aggregate_digest}",
            prefixes=tuple(sorted(acknowledgements, key=lambda ack: ack.ticket_id)),
        )
        receipt.validate_for(inventory)
        return receipt

    async def restore_generation_cut(
        self,
        receipt: GenerationCutReceipt,
        *,
        excluded_replacements: frozenset[tuple[str, int]] = frozenset(),
    ) -> GenerationCutReceipt:
        """Install durable cuts for the replacement attempts that will consume them."""
        if not self._generation_prefix_cuts_enabled:
            raise RuntimeError("generation-prefix cuts are not enabled for this model server")
        if (self.config.num_workers or 1) > 1:
            raise RuntimeError(
                "generation-prefix restore currently requires a single Gym model-server worker; "
                "multi-worker restore needs a process-shared continuation registry"
            )
        if receipt.inventory.server_name != self.config.name:
            raise ValueError(
                "generation-cut receipt belongs to a different model server: "
                f"expected={self.config.name!r}, actual={receipt.inventory.server_name!r}"
            )
        with self._generation_cut_restore_lock:
            for prefix in receipt.prefixes:
                if prefix.disposition != "durable_prefix" or prefix.cut_kind != "active_prefix":
                    continue
                replacement = (prefix.rollout_id, prefix.attempt_index + 1)
                if replacement in excluded_replacements:
                    continue
                previous = self._restored_generation_cuts.get(prefix.rollout_id)
                if previous is not None and previous != prefix:
                    raise RuntimeError(
                        "multiple durable generation cuts target the same replacement attempt: "
                        f"rollout_id={prefix.rollout_id!r}, "
                        f"attempt_index={prefix.attempt_index + 1}"
                    )
                self._restored_generation_cuts[prefix.rollout_id] = prefix
        return receipt

    def _generation_cut_for_context(self) -> GenerationCutPrefixAck | None:
        context = current_capture_context()
        if context is None or context.logical_rollout_id is None or context.attempt_index is None:
            return None
        with self._generation_cut_restore_lock:
            prefix = self._restored_generation_cuts.get(context.logical_rollout_id)
        if prefix is None or context.attempt_index < prefix.attempt_index + 1:
            return None
        return prefix

    def _retire_generation_cut_for_context(self) -> None:
        context = current_capture_context()
        if context is None or context.generation_cut_key is None:
            return
        with self._generation_cut_restore_lock:
            self._restored_generation_cuts.pop(context.generation_cut_key[0], None)

    def _decline_generation_cut_for_context(self, prefix: GenerationCutPrefixAck) -> None:
        """Retire one restored cut that this replacement call cannot safely use."""
        context = current_capture_context()
        if context is None or context.logical_rollout_id is None:
            raise RuntimeError("cannot decline a generation cut without a logical rollout context")
        with self._generation_cut_restore_lock:
            current = self._restored_generation_cuts.get(context.logical_rollout_id)
            if current is None:
                return
            if current != prefix:
                raise RuntimeError(
                    "restored generation cut changed while declining prefix recovery: "
                    f"rollout_id={context.logical_rollout_id!r}"
                )
            self._restored_generation_cuts.pop(context.logical_rollout_id)

    @staticmethod
    def _generation_cut_restart_reason(body_dict: Dict[str, Any]) -> str | None:
        """Return why a restored prefix cannot safely resume structured decoding."""
        tool_choice = body_dict.get("tool_choice")
        if isinstance(tool_choice, str):
            if tool_choice not in {"auto", "none"}:
                return f"tool_choice:{tool_choice}"
        elif tool_choice is not None:
            # Named function/custom choices and allowed-tool constraints are
            # represented as objects. vLLM may compile them into a structured
            # decoder whose state is not reconstructed by token-prefix restore.
            return "tool_choice:constrained"

        response_format = body_dict.get("response_format")
        if response_format is not None:
            if not isinstance(response_format, dict) or response_format.get("type") != "text":
                format_type = (
                    response_format.get("type", "unknown") if isinstance(response_format, dict) else "unknown"
                )
                return f"response_format:{format_type}"

        for field in _STRUCTURED_GENERATION_FIELDS:
            if field in body_dict and body_dict[field] is not None:
                return field
        return None

    def _load_chat_template_tokenizer(self):
        """Load an HF AutoTokenizer for client-side chat-template rendering.

        Imported lazily so that VLLMModel users who don't set
        ``render_chat_template=True`` aren't forced to have ``transformers``
        imported at server start (it's a heavy import).

        Fails loudly at startup if (a) the tokenizer can't be loaded, or
        (b) it loaded but has no ``chat_template`` configured.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"NeMo Gym server `{self.config.name}` is configured with "
                "use_completions_api=true and render_chat_template=true, which requires "
                "the `transformers` package to load an HF tokenizer for chat-template "
                "rendering. Install it (`pip install transformers`) or set "
                "render_chat_template=false to use raw rendering instead."
            ) from e

        tokenizer_id = self.config.tokenizer or self.config.model
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"NeMo Gym server `{self.config.name}`: AutoTokenizer.from_pretrained({tokenizer_id!r}) "
                "failed. Set `tokenizer:` in the server config to an HF identifier or local "
                "path, or set render_chat_template=false."
            ) from e

        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError(
                f"NeMo Gym server `{self.config.name}`: tokenizer loaded from {tokenizer_id!r} "
                "has no chat_template configured. Point `tokenizer:` at a model whose "
                "tokenizer ships a chat template, or set render_chat_template=false."
            )

        return tokenizer

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        if self.config.is_responses_native:
            return await self._responses_native(request, body)

        # Response Create Params -> Chat Completion Create Params
        chat_completion_create_params = self._converter.responses_to_chat_completion_create_params(body)
        body.model = self.config.model

        # Chat Completion Create Params -> Chat Completion
        chat_completion_response = await self.chat_completions(request, chat_completion_create_params)

        response = self._converter.chat_completion_to_response(
            responses_create_params=body,
            chat_completion=chat_completion_response,
            # Keep the backend envelope id only for captured requests. Terminal
            # attribution matches it to the ledger row.
            preserve_envelope_id=self._preserve_envelope_id(),
        )
        if (
            not self.config.sequential_reasoning_allowed
            and response.output
            and all(item.type == "reasoning" for item in response.output)
            and response.incomplete_details is None
        ):
            # The real call is durably captured, but this model is not allowed
            # another reasoning-only turn. End the episode on this call rather
            # than waiting for a second request and synthesizing a phantom call.
            response = response.model_copy(
                update={
                    "status": "incomplete",
                    "incomplete_details": IncompleteDetails(reason="content_filter"),
                }
            )
        return response

    def _apply_sampling_overrides(self, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Force ``config.sampling_overrides`` onto an outbound body, in place.

        Applied last at every site that builds a request for the engine, so the pinned values win
        over both what the client sent and anything ``extra_body`` merged in, and are present when
        the client sent nothing. Every path has to call this: a harness picks its own endpoint, and
        a pin that covers only one of them yields off-policy generation while reporting that
        sampling is pinned.
        """
        if self.config.sampling_overrides:
            body_dict.update(self.config.sampling_overrides)
        return body_dict

    async def _responses_native(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        """
        The following config parameters are effectively no-ops with Responses native models:
        - uses_reasoning_parser: bool (Not applicable)
        """
        # The following parameters could be supported, but have not been supported yet for Responses-native models:
        if self.config.return_token_id_information:
            raise NotImplementedError
        if self.config.replace_developer_role_with_system:
            raise NotImplementedError
        if not self.config.sequential_reasoning_allowed:
            raise NotImplementedError

        body_dict = body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.model
        if self.config.chat_template_kwargs:
            body_dict["chat_template_kwargs"] = deepcopy(self.config.chat_template_kwargs)
        if self.config.extra_body:
            body_dict = self.config.extra_body | body_dict
        self._apply_sampling_overrides(body_dict)

        client = self._resolve_client(request)
        mark_current_generation_started()
        response_dict = await client.create_response(**body_dict)

        return NeMoGymResponse.model_validate(response_dict)

    # Mapping from common audio file extensions to MIME subtypes used in the
    # ``data:audio/<subtype>;base64,...`` URI. vLLM-side decoders inspect the
    # subtype to pick a backend (libsndfile, ffmpeg, …); guessing wrong would
    # silently mis-decode, so we keep the table conservative and raise on
    # unknown extensions instead of falling back to ``wav``.
    _AUDIO_EXT_TO_MIME: ClassVar[Dict[str, str]] = {
        ".wav": "wav",
        ".flac": "flac",
        ".mp3": "mpeg",
        ".m4a": "mp4",
        ".ogg": "ogg",
        ".opus": "opus",
    }

    def _resolve_audio_path_to_url(self, audio_path: str) -> str:
        """Turn an ``audio_path`` reference into a ``data:audio/...;base64`` URI.

        Reads the file and inlines it as a base64 data URI at request time
        — same strategy NeMo Skills' ``VLLMMultimodalModel.content_text_to_list``
        uses (read once per request, hand vLLM a self-contained content
        block). Keeps the on-disk JSONL small without requiring any vLLM
        server-side flag.

        Relative paths are resolved against ``config.audio_root``; without
        it, relative paths raise so the failure mode is loud rather than
        silently reading from the server CWD.
        """
        if os.path.isabs(audio_path):
            resolved = audio_path
        elif self.config.audio_root:
            resolved = os.path.join(self.config.audio_root, audio_path)
        else:
            raise ValueError(
                f"metadata.audio_path={audio_path!r} is relative but VLLMModelConfig.audio_root "
                "is unset. Set audio_root in the model config or use absolute paths."
            )

        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"metadata.audio_path resolved to {resolved!r}, which does not exist.")

        ext = os.path.splitext(resolved)[1].lower()
        mime = self._AUDIO_EXT_TO_MIME.get(ext)
        if mime is None:
            raise ValueError(
                f"Unsupported audio extension {ext!r} for {resolved!r}. Supported: {sorted(self._AUDIO_EXT_TO_MIME)}."
            )
        with open(resolved, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:audio/{mime};base64,{encoded}"

    @staticmethod
    def _strip_hosted_only_tool_fields(body_dict: Dict[str, Any]) -> None:
        """Remove OpenAI-hosted-only fields from function tool definitions.

        ``strict`` is an OpenAI-hosted schema-enforcement flag that vLLM does
        not implement: its FunctionDefinition keeps the unknown key, and chat
        templates that render unknown function-definition keys (e.g.
        Nemotron's ``render_extra_keys``) inject it into the prompt,
        perturbing every tool-bearing request. Strip it at the vLLM boundary
        so hosted providers keep their ``strict`` semantics.
        """
        for tool_dict in body_dict.get("tools") or []:
            if tool_dict.get("type") == "function":
                (tool_dict.get("function") or {}).pop("strict", None)

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the body dict before issuing a chat completion request.

        Subclasses can override this to apply model-specific transformations
        (e.g. role remapping, extra sampling params).  The base implementation
        handles the features driven by ``VLLMModelConfig``.

        Args:
            request: The originating FastAPI request (available for session /
                client resolution if needed by subclasses).
            body_dict: Mutable dict produced by ``body.model_dump(exclude_unset=True)``.

        Returns:
            The (possibly mutated) ``body_dict`` that will be forwarded to
            ``client.create_chat_completion``.
        """
        self._strip_hosted_only_tool_fields(body_dict)

        if self.config.replace_developer_role_with_system:
            for message_dict in body_dict["messages"]:
                if message_dict.get("role") == "developer":
                    message_dict["role"] = "system"

        body_dict["model"] = self.config.model

        chat_template_kwargs = {}
        if self.config.chat_template_kwargs:
            chat_template_kwargs = deepcopy(self.config.chat_template_kwargs)

        metadata = body_dict.get("metadata") or {}

        # Merge global config chat_template_kwargs with per-request overrides in metadata (e.g. per-sample reasoning on/off)
        metadata_chat_template_kwargs_str = metadata.get("chat_template_kwargs") or "{}"
        chat_template_kwargs.update(json.loads(metadata_chat_template_kwargs_str))

        if chat_template_kwargs:
            body_dict["chat_template_kwargs"] = chat_template_kwargs

        # Merge global config extra_body with per-request overrides from metadata
        extra_body = {}
        if self.config.extra_body:
            extra_body = deepcopy(self.config.extra_body)

        metadata_extra_body_str = metadata.get("extra_body") or "{}"
        extra_body.update(json.loads(metadata_extra_body_str))

        if (
            self.config.continue_final_assistant_message
            and body_dict.get("messages")
            and body_dict["messages"][-1].get("role") == "assistant"
        ):
            body_dict["continue_final_message"] = True
            body_dict["add_generation_prompt"] = False

        if self.config.return_token_id_information:
            body_dict |= dict(
                logprobs=True,
                # Pin top_logprobs=0: capture only needs the chosen token's logprob and id.
                # vLLM computes `logprobs = top_logprobs if logprobs else None`.
                # So an inbound top_logprobs=null yields no logprobs and empties the token ids.
                # Overriding it here makes capture independent of the request.
                top_logprobs=0,
                # Typically passed via OpenAI client extra_body.
                return_tokens_as_token_ids=True,
            )
            if self.config.request_prompt_and_generation_token_ids:
                body_dict["return_token_ids"] = True

        if self.config.uses_reasoning_parser and not self.config.preserve_reasoning_in_assistant_content:
            for message_dict in body_dict["messages"]:
                if message_dict.get("role") != "assistant" or "content" not in message_dict:
                    continue

                content = message_dict["content"]
                if isinstance(content, str):
                    reasoning_matches, remaining_content = self._converter._extract_reasoning_from_content(content)
                    message_dict["content"] = remaining_content
                    if reasoning_matches and self.config.uses_interleaved_reasoning:
                        message_dict["reasoning_content"] = reasoning_matches[0]

                        # TODO when NeMo RL migrates to vLLM>=0.16.0, remove the reasoning_content support above.
                        # Starting with vLLM 0.16.0, the `reasoning_content` field has been deprecated in favor of just `reasoning`
                        message_dict["reasoning"] = reasoning_matches[0]
                elif isinstance(content, list):
                    reasoning_content = None
                    for content_item_dict in content:
                        reasoning_matches, remaining_content = self._converter._extract_reasoning_from_content(
                            content_item_dict["text"]
                        )
                        assert reasoning_content is None or not reasoning_matches, (
                            f"Found multiple reasoning matches in a single assistant message content item list!\nMessage: {message_dict}"
                        )

                        # Even though we set the reasoning content already here, we still loop through all the content item dicts for the assert above.
                        content_item_dict["text"] = remaining_content
                        if reasoning_matches and self.config.uses_interleaved_reasoning:
                            message_dict["reasoning_content"] = reasoning_matches[0]
                            # See the TODO wrt reasoning_content above
                            message_dict["reasoning"] = reasoning_matches[0]
                elif not content:
                    # No content or content None is a no-op
                    pass
                else:
                    raise NotImplementedError

        # Drop a null top_logprobs on the non-capture path (caller-supplied logprobs=True).
        # vLLM treats null as "no logprobs" but a missing field as its default (0), so forwarding null is never useful.
        # The capture path above already set it to 0 and is unaffected.
        if body_dict.get("top_logprobs") is None:
            body_dict.pop("top_logprobs", None)

        if extra_body:
            body_dict = extra_body | body_dict

        # Audio sidechannel: rows can carry audio on
        # ``responses_create_params.metadata`` via three mutually exclusive
        # keys, all spliced as ``audio_url`` content blocks into the most
        # recent user message before forwarding to vLLM Chat Completions:
        #
        #   * ``audio_data``  — a single pre-built ``data:audio/...;base64,``
        #                       URI inlined into the JSONL. Self-contained;
        #                       no audio root needed at request time.
        #   * ``audio_path``  — a single file path; resolved against
        #                       ``config.audio_root`` and encoded to a data
        #                       URI at request time.
        #   * ``audio_paths`` — list of file paths; each encoded and spliced
        #                       in order. Mirrors NeMo Skills' ``audios``
        #                       multi-clip schema.
        #
        # OpenAI's Responses API content union has no audio variant (audio
        # types exist as orphans in the SDK but aren't members of
        # ``ResponseInputContentParam``), so audio rows can't ride in
        # ``input.content`` directly — the metadata-sidechannel hop lets
        # audio benchmarks carry audio without a Gym schema change.
        #
        # Audio is placed BEFORE text in the content list (some audio
        # models care). No-op when none of the three keys are present, so
        # non-audio benchmarks are unaffected.
        audio_keys_present = [k for k in ("audio_data", "audio_path", "audio_paths") if metadata.get(k)]
        if len(audio_keys_present) > 1:
            raise ValueError(
                f"metadata audio keys are mutually exclusive — got {audio_keys_present}. "
                "Set exactly one of audio_data / audio_path / audio_paths per row."
            )

        audio_urls: List[str] = []
        if metadata.get("audio_data"):
            audio_urls.append(metadata["audio_data"])
            metadata.pop("audio_data", None)
        elif metadata.get("audio_path"):
            audio_urls.append(self._resolve_audio_path_to_url(metadata["audio_path"]))
            metadata.pop("audio_path", None)
        elif metadata.get("audio_paths"):
            paths = metadata["audio_paths"]
            if not isinstance(paths, list):
                raise ValueError(f"metadata.audio_paths must be a list, got {type(paths).__name__}.")
            audio_urls.extend(self._resolve_audio_path_to_url(p) for p in paths)
            metadata.pop("audio_paths", None)

        if audio_urls:
            if not metadata and "metadata" in body_dict:
                body_dict.pop("metadata", None)

            audio_blocks = [{"type": "audio_url", "audio_url": {"url": url}} for url in audio_urls]
            messages = body_dict.get("messages", []) or []
            for msg in reversed(messages):
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = audio_blocks + [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    msg["content"] = audio_blocks + list(content)
                else:
                    # ``None`` / unexpected shape — replace with a fresh content list
                    msg["content"] = list(audio_blocks)
                break
            else:
                # No user message found — create one with just the audio blocks.
                body_dict.setdefault("messages", []).append({"role": "user", "content": list(audio_blocks)})

        self._apply_sampling_overrides(body_dict)
        self._validate_single_choice_token_request(body_dict)
        if self._external_capture_handler is not None:
            self._apply_restored_generation_cut(body_dict)
            body_dict = self._external_capture_handler.prepare_request(body_dict)
        else:
            body_dict = self._apply_prefix_supply(body_dict)

        return body_dict

    def _preserve_envelope_id(self) -> bool:
        """Keep the backend envelope id only for requests with an active external-capture context."""
        context = current_capture_context()
        return context is not None and context.external_staging

    def _apply_restored_generation_cut(self, body_dict: Dict[str, Any]) -> None:
        """Attach a restored cut to the request-scoped capture admission."""
        context = current_capture_context()
        if context is None or not context.external_staging:
            return
        admission = context.capture_admission
        if admission is None:
            return
        restored_cut = self._generation_cut_for_context()
        if restored_cut is not None:
            restart_reason = self._generation_cut_restart_reason(body_dict)
            if restart_reason is not None:
                self._decline_generation_cut_for_context(restored_cut)
                LOG.info(
                    "generation prefix restart: rollout_id=%s attempt_index=%s "
                    "model_call_id=%s source_model_call_id=%s reason=%s",
                    context.logical_rollout_id,
                    context.attempt_index,
                    context.model_call_id,
                    restored_cut.model_call_id,
                    restart_reason,
                )
                restored_cut = None
        if restored_cut is not None:
            if (
                not restored_cut.staging_keys
                or restored_cut.prefix_token_count is None
                or restored_cut.prefix_digest is None
            ):
                raise RuntimeError("durable generation cut is missing recovery coordinates")
            admission = admission.model_copy(
                update={
                    "generation_cut": GenerationCutContinuation(
                        source_capture_key=capture_key_for(
                            restored_cut.rollout_id,
                            restored_cut.attempt_index,
                        ),
                        source_model_call_id=restored_cut.model_call_id,
                        staging_keys=restored_cut.staging_keys,
                        generation_token_count=restored_cut.prefix_token_count,
                        digest=restored_cut.prefix_digest,
                        effective_output_limit=restored_cut.effective_output_limit,
                        terminal_finish_reason=restored_cut.terminal_finish_reason,
                        terminal_stop_reason=restored_cut.terminal_stop_reason,
                    )
                }
            )
            context.capture_admission = admission
            context.generation_cut_key = (
                restored_cut.rollout_id,
                restored_cut.attempt_index + 1,
            )

    # Protect the ``[supplied, eligible, total]`` diagnostic counts.
    # Eligible calls have a resolved parent.
    _prefix_supply_counts: List[int] = PrivateAttr(default_factory=lambda: [0, 0, 0])
    _prefix_supply_lock: Any = PrivateAttr(default_factory=Lock)

    def _apply_prefix_supply(self, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add a verified parent's exact tokens to a compatible engine request.

        Prefix supply is opt-in.
        A unique parent match provides the cumulative token prefix.
        The backend must implement the ``required_prefix_token_ids`` extension.
        Stock vLLM does not implement this extension.
        ``prefix_requested`` records that the request included the prefix.
        Only generation-time ``prompt_token_ids`` can prove that the backend applied it.
        That proof sets ``prefix_supplied``.
        A missing or ambiguous parent leaves the request unchanged.
        """
        if not self.config.supply_prefix_token_ids:
            return body_dict
        context = current_capture_context()
        # The parent was resolved before dispatch from the request as received.
        # Conversion and preprocessing may have reshaped this body.
        # Re-resolving here could select against a representation never indexed.
        parent_tokens = context.parent_tokens if context is not None else []
        with self._prefix_supply_lock:
            self._prefix_supply_counts[2] += 1
            if parent_tokens:
                self._prefix_supply_counts[1] += 1
        if context is None:
            # An uncorrelated rollout call has no verified parent.
            return body_dict
        if not parent_tokens:
            return body_dict
        if context.external_staging:
            # External path: worker fetches prefix from TQ via staging_chain in ng_capture.
            # Do not put the large token array in the request body.
            return body_dict
        body_dict["required_prefix_token_ids"] = parent_tokens
        # This records intent only.
        # ``prefix_supplied`` remains false until generation-time prompt_token_ids prove application.
        context.prefix_requested = True
        return body_dict

    @staticmethod
    def _generation_prompt_token_ids(response: dict) -> Any:
        """Return the prompt token IDs reported by generation.

        Prefer the message-level token bundle over top-level transport fields.
        Token capture uses the same source order.
        """
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message")
        if isinstance(message, dict) and message.get("prompt_token_ids") is not None:
            return message["prompt_token_ids"]
        return response.get("prompt_token_ids")

    def _verify_generation_prefix(self, body_dict: dict, response: dict) -> None:
        """Require generation-time proof that the engine applied the requested prefix."""
        context = current_capture_context()
        if context is None or not context.prefix_requested:
            return
        required = body_dict.get("required_prefix_token_ids")
        if not required:
            raise RuntimeError("A requested token prefix was removed before generation.")
        tokens = self._generation_prompt_token_ids(response)
        if not isinstance(tokens, list):
            raise RuntimeError(
                f"`{self.config.name}` (base_url={self.config.base_url}) requested "
                "required_prefix_token_ids, but the generation response did not include prompt_token_ids "
                "proving which prompt the engine used. The backend must implement the "
                "required_prefix_token_ids extension and return generation-time prompt token ids. "
                "Disabling supply_prefix_token_ids is the fallback."
            )
        tokens = [int(token) for token in tokens]
        if tokens[: len(required)] != list(required):
            raise RuntimeError(
                f"`{self.config.name}` (base_url={self.config.base_url}) returned generation "
                "prompt_token_ids that do not start with required_prefix_token_ids. The backend must "
                "implement the required_prefix_token_ids extension and return generation-time prompt "
                "token ids that extend the supplied prefix. Disabling supply_prefix_token_ids is the fallback."
            )
        # This proves that the served prompt extended the exact parent tokens.
        # It does not prove how the backend produced that prompt.
        # A prefix-stable re-render still satisfies the training invariant.
        context.prefix_supplied = True
        with self._prefix_supply_lock:
            self._prefix_supply_counts[0] += 1
            supplied, eligible, total = self._prefix_supply_counts
        if supplied % 10 == 0:
            LOG.info(
                "prefix supply: %d/%d eligible calls supplied (%.0f%%; %d enabled calls total)",
                supplied,
                eligible,
                100.0 * supplied / eligible,
                total,
            )

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        capture_context = current_capture_context()
        generation_cut_client = None
        if self._generation_prefix_cuts_enabled and capture_context is not None:
            generation_cut_client = self._resolve_client(request)
            self._remember_generation_cut_client(capture_context.model_call_id, generation_cut_client)
        try:
            return await self._chat_completions(
                request,
                body,
                resolved_client=generation_cut_client,
            )
        finally:
            if generation_cut_client is not None and capture_context is not None:
                self._forget_generation_cut_client(capture_context.model_call_id, generation_cut_client)

    async def _chat_completions(
        self,
        request: Request,
        body: NeMoGymChatCompletionCreateParamsNonStreaming,
        *,
        resolved_client: NeMoGymAsyncOpenAI | None,
    ) -> NeMoGymChatCompletion:
        if self.config.use_completions_api:
            return await self._chat_completions_via_completions_api(request, body)

        body_dict = body.model_dump(exclude_unset=True)
        body_dict = self._preprocess_chat_completion_create_params(request, body_dict)

        client = resolved_client or self._resolve_client(request)
        if not self.config.sequential_reasoning_allowed:
            last_message = body_dict["messages"][-1]
            if last_message["role"] == "assistant" and not (last_message["content"] or last_message.get("tool_calls")):
                await mark_no_generation()
                res = self._create_empty_chat_completion()
                res.choices[0].finish_reason = "content_filter"
                return res

        transport_io_enabled = bool(os.environ.get("NEMO_GYM_VLLM_TRANSPORT_LOG", "").strip())
        log_context = _transport_log_context(request)
        call_index = 0
        started_ns = 0
        if transport_io_enabled:
            self._transport_call_index += 1
            call_index = self._transport_call_index
            request_value = _jsonable(body_dict)
            request_json = json.dumps(request_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            started_ns = time_ns()
            _append_transport_io(
                {
                    **log_context,
                    "schema_version": 2,
                    "event": "transport_request",
                    "call_index": call_index,
                    "timestamp_unix_ns": started_ns,
                    "pid": os.getpid(),
                    "configured_base_urls": self.config.base_url,
                    "request_payload": request_value,
                    "request_payload_sha256": hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
                    "embedded_images": _transport_images(body_dict.get("messages", [])),
                }
            )

        mark_current_generation_started()
        try:
            chat_completion_dict = await client.create_chat_completion(**body_dict)
        except ClientResponseError as e:
            if transport_io_enabled:
                finished_ns = time_ns()
                _append_transport_io(
                    {
                        **log_context,
                        "schema_version": 2,
                        "event": "transport_error_response",
                        "call_index": call_index,
                        "timestamp_unix_ns": finished_ns,
                        "elapsed_ns": finished_ns - started_ns,
                        "pid": os.getpid(),
                        "http_status": e.status,
                        "raw_response_body": e.response_content.decode(errors="replace"),
                        "error": repr(e),
                    }
                )
            """
            Example messages for out of context length:

            1. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L914
            ```json
            {"object":"error","message":"This model\'s maximum context length is 32768 tokens. However, you requested 32818 tokens in the messages, Please reduce the length of the messages. None","type":"BadRequestError","param":null,"code":400}
            ```
            2. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L940
            3. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L948
            4. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/sampling_params.py#L463
            """
            result_content_str = e.response_content.decode()

            is_out_of_context_length = e.status == 400 and (
                "context length" in result_content_str or "max_tokens" in result_content_str
            )
            if is_out_of_context_length:
                if self.config.propagate_context_overflow_errors:
                    setattr(e, _PROPAGATE_CONTEXT_ERROR_ATTRIBUTE, True)
                    raise
                await mark_no_generation()
                res = self._create_empty_chat_completion()
                res.choices[0].finish_reason = "length"
                self._retire_generation_cut_for_context()
                return res
            else:
                raise e
        except Exception as e:
            if transport_io_enabled:
                finished_ns = time_ns()
                _append_transport_io(
                    {
                        **log_context,
                        "schema_version": 2,
                        "event": "transport_error",
                        "call_index": call_index,
                        "timestamp_unix_ns": finished_ns,
                        "elapsed_ns": finished_ns - started_ns,
                        "pid": os.getpid(),
                        "error_type": type(e).__name__,
                        "error": repr(e),
                    }
                )
            raise

        if transport_io_enabled:
            finished_ns = time_ns()
            _append_transport_io(
                {
                    **log_context,
                    "schema_version": 2,
                    "event": "transport_response",
                    "call_index": call_index,
                    "timestamp_unix_ns": finished_ns,
                    "elapsed_ns": finished_ns - started_ns,
                    "pid": os.getpid(),
                    "raw_response": deepcopy(chat_completion_dict),
                }
            )

        choice_dict = chat_completion_dict["choices"][0]
        self._verify_generation_prefix(body_dict, chat_completion_dict)
        if self.config.uses_reasoning_parser:
            # See the TODO wrt reasoning_content above
            reasoning_content = choice_dict["message"].get("reasoning_content") or choice_dict["message"].get(
                "reasoning"
            )
            if reasoning_content:
                choice_dict["message"].pop("reasoning_content", None)
                # See the TODO wrt reasoning_content above
                choice_dict["message"].pop("reasoning", None)

                if body_dict.get("continue_final_message", False):
                    # by default, the response of continue_final_message will split into reasoning.
                    choice_dict["message"]["content"] = reasoning_content + (choice_dict["message"]["content"] or "")
                else:
                    # We wrap this here in think tags for Gym's sake and to return a valid OpenAI Chat Completions response.
                    choice_dict["message"]["content"] = self._converter._wrap_reasoning_in_think_tags(
                        [reasoning_content]
                    ) + (choice_dict["message"].get("content") or "")

        else:
            # See the TODO wrt reasoning_content above
            assert not (choice_dict["message"].get("reasoning_content") or choice_dict["message"].get("reasoning")), (
                f"NeMo Gym server `{self.config.name}` config has explicitly been set to not use a reasoning parser i.e. `uses_reasoning_parser: false`. Please do not use a reasoning parser in your vLLM endpoint, or fix the `{self.config.name}` server config!"
            )

        if self._external_capture_handler is not None:
            self._external_capture_handler.prepare_response(chat_completion_dict)

        if self.config.return_token_id_information:
            message_dict = choice_dict["message"]

            # Token metadata uses this source order:
            # 1. A complete bundle on the assistant message.
            # 2. Prompt IDs at the response top level and generation IDs on the choice.
            # 3. Generation data from choice logprobs and prompt IDs from `/tokenize`.
            #
            # An earlier source supplies the normalized bundle.
            # Later inline sources are still checked when present.
            # A partially present source is invalid.
            # Duplicate token IDs must agree.
            message_bundle = self._extract_message_token_bundle(message_dict)
            response_token_ids = self._extract_vllm_response_token_ids(chat_completion_dict, choice_dict)

            if message_bundle is not None:
                if response_token_ids is not None:
                    response_prompt_token_ids, response_generation_token_ids = response_token_ids
                    if (
                        message_bundle["prompt_token_ids"] != response_prompt_token_ids
                        or message_bundle["generation_token_ids"] != response_generation_token_ids
                    ):
                        raise RuntimeError("Message-level token metadata disagrees with vLLM response token IDs.")
                message_dict.update(message_bundle)
            else:
                logprob_token_ids, generation_log_probs = self._extract_choice_logprobs(choice_dict)
                if response_token_ids is not None:
                    prompt_token_ids, generation_token_ids = response_token_ids
                    if generation_token_ids != logprob_token_ids:
                        raise RuntimeError(
                            "vLLM response generation token IDs disagree with choice logprob token IDs."
                        )
                else:
                    tokenize_response = await client.create_tokenize(**self._get_tokenize_chat_body(body_dict))
                    prompt_token_ids = self._require_token_id_list(
                        tokenize_response.get("tokens"),
                        "tokenize.tokens",
                    )
                    generation_token_ids = logprob_token_ids

                message_dict.update(
                    self._validate_token_bundle(
                        {
                            "prompt_token_ids": prompt_token_ids,
                            "generation_token_ids": generation_token_ids,
                            "generation_log_probs": generation_log_probs,
                            **(
                                {"routed_experts": message_dict["routed_experts"]}
                                if message_dict.get("routed_experts") is not None
                                else {}
                            ),
                        },
                        "vLLM token transport",
                    )
                )

                # The adapter consumed this compatibility payload.
                choice_dict.pop("logprobs", None)

            # Top-level and choice-level token-ID fields are transport details.
            chat_completion_dict.pop("prompt_token_ids", None)
            choice_dict.pop("token_ids", None)
            choice_dict["message"] = NeMoGymChatCompletionMessageForTraining.model_validate(message_dict)

        return NeMoGymChatCompletion.model_validate(chat_completion_dict)

    async def _finalize_served_response(self, response: Any) -> None:
        """Publish lineage using the final Chat, Responses, or Messages representation."""
        if self._external_capture_handler is not None:
            try:
                await self._external_capture_handler.finalize_response(_jsonable(response))
            finally:
                self._retire_generation_cut_for_context()

    @staticmethod
    def _require_token_id_list(value: Any, field_name: str) -> List[Any]:
        """Check the container without scanning or copying token IDs."""
        if not isinstance(value, list):
            raise RuntimeError(f"`{field_name}` must be a list of integer token IDs.")
        return value

    @staticmethod
    def _require_log_prob_list(value: Any, field_name: str) -> List[Any]:
        """Check the container without scanning or copying log probabilities."""
        if not isinstance(value, list):
            raise RuntimeError(f"`{field_name}` must be a list of numeric log probabilities.")
        return value

    @classmethod
    def _validate_token_bundle(cls, bundle: Dict[str, Any], source: str) -> Dict[str, Any]:
        present_fields = TOKEN_METADATA_FIELDS.intersection(bundle)
        missing_fields = REQUIRED_TOKEN_METADATA_FIELDS.difference(present_fields)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise RuntimeError(f"{source} returned partial token metadata; missing: {missing}.")

        normalized = {
            "prompt_token_ids": cls._require_token_id_list(bundle["prompt_token_ids"], f"{source}.prompt_token_ids"),
            "generation_token_ids": cls._require_token_id_list(
                bundle["generation_token_ids"], f"{source}.generation_token_ids"
            ),
            "generation_log_probs": cls._require_log_prob_list(
                bundle["generation_log_probs"], f"{source}.generation_log_probs"
            ),
        }
        if "routed_experts" in bundle:
            normalized["routed_experts"] = bundle["routed_experts"]

        if len(normalized["generation_token_ids"]) != len(normalized["generation_log_probs"]):
            raise RuntimeError(
                f"{source} returned mismatched generation token IDs and log probabilities: "
                f"{len(normalized['generation_token_ids'])} token IDs and "
                f"{len(normalized['generation_log_probs'])} log probabilities."
            )

        return normalized

    @classmethod
    def _extract_message_token_bundle(cls, message_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        required_fields = REQUIRED_TOKEN_METADATA_FIELDS.intersection(message_dict)
        if not required_fields:
            return None
        present_fields = TOKEN_METADATA_FIELDS.intersection(message_dict)
        return cls._validate_token_bundle(
            {field: message_dict[field] for field in present_fields},
            "choice.message",
        )

    @classmethod
    def _extract_vllm_response_token_ids(
        cls,
        chat_completion_dict: Dict[str, Any],
        choice_dict: Dict[str, Any],
    ) -> Optional[tuple[List[Any], List[Any]]]:
        prompt_value = chat_completion_dict.get("prompt_token_ids")
        generation_value = choice_dict.get("token_ids")
        prompt_present = prompt_value is not None
        generation_present = generation_value is not None

        if prompt_present != generation_present:
            missing = "choice.token_ids" if prompt_present else "prompt_token_ids"
            raise RuntimeError(f"vLLM response returned partial token metadata; missing: {missing}.")
        if not prompt_present:
            return None

        return (
            cls._require_token_id_list(prompt_value, "prompt_token_ids"),
            cls._require_token_id_list(generation_value, "choice.token_ids"),
        )

    def _extract_choice_logprobs(self, choice_dict: Dict[str, Any]) -> tuple[List[int], List[float]]:
        logprobs_block = choice_dict.get("logprobs")
        if not isinstance(logprobs_block, dict) or not isinstance(logprobs_block.get("content"), list):
            raise RuntimeError(
                f"`{self.config.name}` requested per-token logprobs from vLLM "
                "(return_token_id_information=True, logprobs=True, top_logprobs=0), "
                f"but the response had none (choice.logprobs={logprobs_block!r}). "
                "Cannot extract token ids or logprobs."
            )

        generation_token_ids: List[int] = []
        generation_log_probs: List[float] = []
        for index, entry in enumerate(logprobs_block["content"]):
            if not isinstance(entry, dict):
                raise RuntimeError(f"choice.logprobs.content[{index}] must be an object.")
            token = entry.get("token")
            if not isinstance(token, str) or not token.startswith("token_id:"):
                raise RuntimeError(f"choice.logprobs.content[{index}].token must use the `token_id:<int>` format.")
            try:
                token_id = int(token.removeprefix("token_id:"))
            except ValueError as e:
                raise RuntimeError(
                    f"choice.logprobs.content[{index}].token must use the `token_id:<int>` format."
                ) from e
            log_prob = entry.get("logprob")
            if not isinstance(log_prob, (int, float)) or isinstance(log_prob, bool):
                raise RuntimeError(f"choice.logprobs.content[{index}].logprob must be numeric.")
            generation_token_ids.append(token_id)
            generation_log_probs.append(float(log_prob))

        return generation_token_ids, generation_log_probs

    @classmethod
    def _get_tokenize_chat_body(cls, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Keep every known prompt-affecting field aligned with generation."""
        return {field: body_dict[field] for field in cls._TOKENIZE_CHAT_FIELDS if field in body_dict}

    def _validate_single_choice_token_request(self, body_dict: Dict[str, Any]) -> None:
        context = current_capture_context()
        external_capture = context is not None and context.external_staging
        if (self.config.return_token_id_information or external_capture) and body_dict.get("n") not in (None, 1):
            raise ValueError(f"NeMo Gym server `{self.config.name}` requires n=1 for token capture.")

    async def _chat_completions_via_completions_api(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming
    ) -> NeMoGymChatCompletion:
        """Drive vLLM's /v1/completions instead of /v1/chat/completions.

        Primary use case: base (non-instruct) models. Returns the same
        NeMoGymChatCompletion shape as the chat-completions path so external
        callers don't need to change.

        Two render modes (selected by ``render_chat_template``):

        - **raw** (default): the messages list must be a single user message
          (optionally preceded by a single system message); their content is
          forwarded verbatim. Tools and multi-turn turns are rejected.
        - **chat_template**: messages are rendered via
          ``HF AutoTokenizer.apply_chat_template(tokenize=False,
          add_generation_prompt=True)`` before being forwarded. Multi-turn
          and tools are allowed; tools are rendered into the prompt by the
          template, but tool-call output text is **not parsed** by Gym since
          /v1/completions doesn't run vLLM's tool-call parser — the caller
          is responsible for parsing tool calls out of the assistant text.

        Audio metadata and non-text content blocks are rejected in both
        modes — /v1/completions is text-only.
        """
        body_dict = body.model_dump(exclude_unset=True)
        # This path never runs _preprocess_chat_completion_create_params, and
        # _render_messages_via_chat_template hands tools to apply_chat_template,
        # so hosted-only fields must be stripped here too.
        self._strip_hosted_only_tool_fields(body_dict)
        messages = body_dict.get("messages", []) or []
        metadata = body_dict.get("metadata", {}) or {}

        if not self.config.render_chat_template and body_dict.get("tools"):
            raise ValueError(
                f"NeMo Gym server `{self.config.name}`: tools are not supported "
                "with use_completions_api=true and render_chat_template=false. "
                "Set render_chat_template=true (so the chat template can render "
                "tool definitions into the prompt) or set use_completions_api=false "
                "for the standard chat-completions tool-call path."
            )

        for audio_key in ("audio_data", "audio_path", "audio_paths"):
            if metadata.get(audio_key):
                raise ValueError(
                    f"NeMo Gym server `{self.config.name}`: audio metadata "
                    f"({audio_key!r}) is not supported with use_completions_api=true. "
                    "/v1/completions is text-only."
                )

        if self.config.render_chat_template:
            prompt = await asyncio.to_thread(self._render_messages_via_chat_template, body_dict)
        else:
            prompt = self._render_messages_to_prompt(messages)

        completion_body = self._build_completion_body_from_chat_body(body_dict, prompt)
        self._validate_single_choice_token_request(completion_body)

        client = self._resolve_client(request)

        mark_current_generation_started()
        try:
            completion_dict = await client.create_completion(**completion_body)
        except ClientResponseError as e:
            result_content_str = e.response_content.decode()
            is_out_of_context_length = e.status == 400 and (
                "context length" in result_content_str or "max_tokens" in result_content_str
            )
            if is_out_of_context_length:
                if self.config.propagate_context_overflow_errors:
                    setattr(e, _PROPAGATE_CONTEXT_ERROR_ATTRIBUTE, True)
                    raise
                await mark_no_generation()
                res = self._create_empty_chat_completion()
                res.choices[0].finish_reason = "length"
                return res
            raise

        if self.config.return_token_id_information:
            choice_dict = completion_dict["choices"][0]
            if choice_dict.get("prompt_token_ids") is None:
                tokenize_body = dict(
                    model=self.config.model,
                    prompt=prompt,
                )
                if "add_special_tokens" in completion_body:
                    tokenize_body["add_special_tokens"] = completion_body["add_special_tokens"]
                tokenize_response = await client.create_tokenize(**tokenize_body)
                choice_dict["prompt_token_ids"] = tokenize_response["tokens"]

        return self._completion_dict_to_chat_completion(completion_dict)

    def _render_messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert a chat-style messages list into a flat prompt string.

        Allows at most one optional system message followed by exactly one
        user message. Their string contents are joined with ``\\n\\n``.
        Rejects anything else (assistant / tool turns, multiple users,
        list-of-blocks content with non-text parts). The caller is expected
        to do prompt templating upstream before sending.
        """
        if not messages:
            raise ValueError("Cannot render an empty messages list to a prompt.")

        if len(messages) > 2:
            raise ValueError(
                f"use_completions_api=true accepts at most one system + one user message; "
                f"got {len(messages)} messages. Render the prompt upstream and submit a "
                "single user message."
            )

        roles = [m.get("role") for m in messages]
        if len(messages) == 2 and roles != ["system", "user"]:
            raise ValueError(
                f"use_completions_api=true requires the two-message form to be [system, user]; got {roles}."
            )
        if len(messages) == 1 and roles[0] != "user":
            raise ValueError(f"use_completions_api=true requires a user message; got role={roles[0]!r}.")

        parts: List[str] = []
        for m in messages:
            parts.append(self._stringify_message_content(m))

        return "\n\n".join(parts)

    @staticmethod
    def _stringify_message_content(message: Dict[str, Any]) -> str:
        """Coerce a single message's content into a flat string.

        Accepts:
          - ``str``: returned as-is.
          - list of text blocks (``[{"type": "text", "text": ...}, ...]``):
            concatenated with no separator. This is the shape produced by
            VLLMConverter when the caller passes a string ``input`` to
            /v1/responses.

        Anything else (image / audio blocks, None content) is rejected — the
        caller is expected to render upstream when use_completions_api is true.
        """
        content = message.get("content")
        role = message.get("role")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
                    raise ValueError(
                        f"use_completions_api=true only accepts text content blocks; "
                        f"got block type {part.get('type') if isinstance(part, dict) else type(part).__name__!r} "
                        f"for role={role!r}."
                    )
                text_parts.append(part.get("text", ""))
            return "".join(text_parts)
        raise ValueError(
            f"use_completions_api=true requires string or text-block-list content; "
            f"got {type(content).__name__} for role={role!r}."
        )

    def _render_messages_via_chat_template(self, body_dict: Dict[str, Any]) -> str:
        """Render the request's messages to a single prompt string using the
        HF tokenizer's chat template.

        Calls ``apply_chat_template`` with ``add_generation_prompt=True`` so
        the rendered string ends where the assistant turn would begin.
        Assistant and tool messages, plus any tool definitions, are passed
        through to the template unchanged.

        ``chat_template_kwargs`` is merged from two sources: the
        server-level value from config, plus an optional per-request
        override JSON-encoded under ``metadata.chat_template_kwargs``. The
        per-request override wins on key conflicts.
        """
        messages = body_dict.get("messages") or []
        tools = body_dict.get("tools") or None
        self._validate_text_only_messages(messages)

        # Mirror the precedence rules in _preprocess_chat_completion_create_params:
        # global config baseline, per-request metadata overrides on top.
        chat_template_kwargs: Dict[str, Any] = {}
        if self.config.chat_template_kwargs:
            chat_template_kwargs.update(deepcopy(self.config.chat_template_kwargs))
        metadata = body_dict.get("metadata") or {}
        metadata_kwargs_str = metadata.get("chat_template_kwargs") or "{}"
        chat_template_kwargs.update(json.loads(metadata_kwargs_str))

        return self._chat_template_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            **chat_template_kwargs,
        )

    def _validate_text_only_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Reject non-text content before forwarding to /v1/completions."""
        for message in messages:
            if message.get("content") is not None:
                self._stringify_message_content(message)

    def _build_completion_body_from_chat_body(self, chat_body_dict: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Translate a chat-completion request body into a /v1/completions body.

        Only forwards fields that vLLM /v1/completions accepts. Sampling knobs
        (top_k, min_p, repetition_penalty, etc.) that have no first-class
        completion field are passed via ``extra_body`` if the operator set
        ``config.extra_body`` — same precedence rule as the chat path.
        """
        out: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
        }

        # Pass-through sampling fields with the same names on /v1/completions.
        for key in (
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "seed",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "response_format",
            "user",
        ):
            if key in chat_body_dict:
                out[key] = chat_body_dict[key]

        # ``max_completion_tokens`` is the chat-API alias; map onto ``max_tokens``.
        if "max_tokens" not in out and "max_completion_tokens" in chat_body_dict:
            out["max_tokens"] = chat_body_dict["max_completion_tokens"]

        # /v1/completions ``logprobs`` is an int (top-N), not a bool. We mainly
        # need ``logprobs=0`` (just the sampled token's logprob) when the
        # operator wants generation-token-id metadata for RL training.
        chat_logprobs = chat_body_dict.get("logprobs")
        chat_top_logprobs = chat_body_dict.get("top_logprobs")
        if chat_top_logprobs is not None:
            out["logprobs"] = chat_top_logprobs
        elif chat_logprobs is True:
            out["logprobs"] = 0
        elif self.config.return_token_id_information and "logprobs" not in out:
            out["logprobs"] = 0

        # Operator-level extra_body merges in (e.g. return_tokens_as_token_ids).
        # Same precedence as the chat path: extra_body fields do NOT override
        # request-level fields.
        if self.config.extra_body:
            extra_body = deepcopy(self.config.extra_body)
            out = extra_body | out

        if self.config.return_token_id_information:
            # Prefer vLLM's inline prompt and generation token IDs. Keep the
            # token-string form available for older engines that omit them.
            out["return_token_ids"] = True
            out["return_tokens_as_token_ids"] = True

        # This path never runs _preprocess_chat_completion_create_params;
        # chat_completions() branches here before preprocessing, so the pin has to be applied again.
        # vLLM accepts the same sampling field names on /v1/completions, and the body is forwarded as raw JSON,
        # so params without a first-class OpenAI completion field (top_k, min_p) pass through.
        return self._apply_sampling_overrides(out)

    def _completion_dict_to_chat_completion(self, completion_dict: Dict[str, Any]) -> NeMoGymChatCompletion:
        """Wrap a /v1/completions response as a NeMoGymChatCompletion.

        vLLM /v1/completions returns ``choices[i].text``; we lift it into a
        single assistant chat message. Reasoning content (``<think>...</think>``
        in the raw text) is left inline — VLLMConverter._extract_reasoning_from_content
        will pull it out downstream when the result is converted back to a
        Response.
        """
        choice_dict = completion_dict["choices"][0]
        text = choice_dict.get("text") or ""

        message_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": text,
            "tool_calls": None,
        }

        if self.config.return_token_id_information:
            logprobs = choice_dict.get("logprobs")
            if not logprobs or logprobs.get("token_logprobs") is None:
                raise RuntimeError(
                    f"`{self.config.name}` requested per-token logprobs from vLLM "
                    "(return_token_id_information=True, logprobs=0), but the response "
                    f"had none (choice.logprobs={logprobs!r}). Cannot extract token IDs or logprobs."
                )

            tokens = logprobs.get("tokens") or []
            token_logprobs = logprobs["token_logprobs"]

            inline_generation_token_ids = choice_dict.get("token_ids")
            if inline_generation_token_ids is None and logprobs.get("tokens") is None:
                raise RuntimeError(
                    f"`{self.config.name}` requested generation token IDs from vLLM, "
                    "but the response contained neither choice.token_ids nor choice.logprobs.tokens."
                )
            generation_token_ids = (
                inline_generation_token_ids
                if inline_generation_token_ids is not None
                else [t.removeprefix("token_id:") for t in tokens]
            )
            generation_log_probs = list(token_logprobs)

            message_dict.update(
                prompt_token_ids=choice_dict["prompt_token_ids"],
                generation_token_ids=generation_token_ids,
                generation_log_probs=generation_log_probs,
            )

        chat_completion_dict = {
            "id": completion_dict.get("id", "chatcmpl-completions"),
            "object": "chat.completion",
            "created": completion_dict.get("created", int(time())),
            "model": completion_dict.get("model", self.config.model),
            "choices": [
                {
                    "index": choice_dict.get("index", 0),
                    "finish_reason": choice_dict.get("finish_reason") or "stop",
                    "message": message_dict,
                }
            ],
        }

        if completion_dict.get("usage") is not None:
            chat_completion_dict["usage"] = completion_dict["usage"]

        return NeMoGymChatCompletion.model_validate(chat_completion_dict)

    def _create_empty_chat_completion(self) -> NeMoGymChatCompletion:
        return NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=int(time()),
            model=self.config.model,
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=None,
                    ),
                )
            ],
        )

    def _maybe_rebind_endpoint(self) -> None:
        """Rebind clients when a shared serving job publishes a new endpoint.

        Raises when the endpoints have stayed unpublished for longer than `endpoint_stale_grace_s`.
        """
        if not self.config.endpoint_file:
            return
        now = monotonic()
        if (
            self._endpoint_last_check_at is not None
            and now - self._endpoint_last_check_at < self.config.endpoint_check_interval_s
        ):
            if self._endpoint_missing_since is not None:
                self._note_endpoint_unpublished()
            return
        self._endpoint_last_check_at = now
        try:
            mtime = os.stat(self.config.endpoint_file).st_mtime
        except FileNotFoundError:
            # Serving jobs remove the endpoint file while rotating;
            # keep the current clients until the successor publishes.
            self._note_endpoint_unpublished()
            return
        except OSError:
            # Transient filesystem trouble is not a backend exit; retry the current clients.
            return
        if mtime == self._endpoint_file_mtime:
            if self._endpoint_missing_since is not None:
                self._note_endpoint_unpublished()
            return
        try:
            with open(self.config.endpoint_file) as endpoint_stream:
                url = endpoint_stream.read().strip()
        except OSError:
            return
        self._endpoint_file_mtime = mtime
        if not url:
            # An empty file is as unpublished as a missing one.
            self._note_endpoint_unpublished()
            return
        self._endpoint_missing_since = None
        if [url] == self.config.base_url:
            return
        print(
            f"vllm_model '{self.config.name}': backend endpoint changed "
            f"{self.config.base_url} -> {[url]}; rebinding clients.",
            flush=True,
        )
        self.config.base_url = [url]
        self._clients = [
            NeMoGymAsyncOpenAI(
                base_url=url,
                api_key=self.config.api_key,
                default_headers=self.config.default_headers,
                max_connection_retries=self.config.endpoint_connection_retries,
            )
        ]
        # Every session re-resolves onto the new host.
        self._session_id_to_client.clear()

    def _note_endpoint_unpublished(self) -> None:
        now = monotonic()
        if self._endpoint_missing_since is None:
            self._endpoint_missing_since = now
        elif now - self._endpoint_missing_since > self.config.endpoint_stale_grace_s:
            raise RuntimeError(
                f"vllm_model endpoint file {self.config.endpoint_file} unpublished (absent "
                f"or empty) for {now - self._endpoint_missing_since:.0f}s (grace "
                f"{self.config.endpoint_stale_grace_s:.0f}s); refusing to keep serving "
                "against a backend that is no longer published."
            )

    def _resolve_client(self, request: Request) -> NeMoGymAsyncOpenAI:
        self._maybe_rebind_endpoint()
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._session_id_to_client:
            # Uvicorn workers do not share this cache. A stable assignment keeps
            # every turn in a session on the same vLLM endpoint across workers.
            digest = hashlib.sha256(session_id.encode("utf-8")).digest()
            client_idx = int.from_bytes(digest[:8], byteorder="big") % len(self._clients)
            client = self._clients[client_idx]
            self._session_id_to_client[session_id] = client
        client = self._session_id_to_client[session_id]

        return client


if __name__ == "__main__":
    VLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VLLMModel.run_webserver()  # noqa: F401
