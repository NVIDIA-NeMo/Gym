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
import base64
import json
import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter, time
from typing import Any, ClassVar, Dict, List, Optional, Union
from uuid import uuid4

from aiohttp.client_exceptions import ClientResponseError
from fastapi import BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.responses_converter import (
    VLLMConverter,
    VLLMConverterResponsesToChatCompletionsState,  # noqa: F401
    split_responses_input_output_items,  # noqa: F401
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint


class VLLMModelConfig(BaseResponsesAPIModelConfig):
    base_url: Union[str, List[str]]
    api_key: str
    model: str
    return_token_id_information: bool

    uses_reasoning_parser: bool
    uses_interleaved_reasoning: bool = True
    replace_developer_role_with_system: bool = False

    # Whether or not the model can generate a reasoning output, and called again to produce additional reasoning output.
    sequential_reasoning_allowed: bool = True

    # As of Feb 2026, we default this to False since majority of open source models aren't responses native with the exception of GPT-OSS
    is_responses_native: bool = False

    chat_template_kwargs: Optional[Dict[str, Any]] = None

    # Corresponds to the extra_body of OpenAI Client.
    extra_body: Optional[Dict[str, Any]] = None

    default_headers: Dict[str, str] = Field(default_factory=dict)
    # Optional prefix for resolving relative ``metadata.audio_path`` (or
    # entries in ``metadata.audio_paths``) against. Absolute paths are used
    # as-is. When unset, relative paths raise. Audio is always inlined as a
    # ``data:audio/<fmt>;base64,...`` URI at request time — keeps the JSONL
    # small without depending on vLLM's ``--allowed-local-media-path``.
    audio_root: Optional[str] = None

    def model_post_init(self, context):
        if isinstance(self.base_url, str):
            self.base_url = [self.base_url]
        return super().model_post_init(context)


class StreamingToolCallPromptRequest(BaseModel):
    session_id: str
    sequence_no: int
    chat_completion: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    compact_context: bool = False
    max_contexts: Optional[int] = Field(default=None, gt=0)
    context_ttl_seconds: Optional[float] = Field(default=None, gt=0)
    final: bool = False
    exact_incremental_tokenizer: bool = False
    prefill: bool = False
    prefill_continuation: bool = False
    prefill_from_required_prefix: bool = False
    finalize_from_required_prefix: bool = False
    max_candidates: Optional[int] = Field(default=None, gt=0)
    candidate_ttl_seconds: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_final_candidate_limits(self):
        if self.compact_context:
            if not self.exact_incremental_tokenizer:
                raise ValueError("compact context requires exact incremental tokenization")
            if (self.chat_completion is None) == (self.tool_output is None):
                raise ValueError("compact requests require exactly one of chat_completion or tool_output")
            if self.tool_output is not None and self.sequence_no <= 0:
                raise ValueError("compact tool-output requests require a positive sequence")
            if self.chat_completion is not None and self.sequence_no != 0:
                raise ValueError("compact full-context requests require sequence zero")
            if (
                self.chat_completion is not None
                and self.sequence_no == 0
                and not self.final
                and (self.max_contexts is None or self.context_ttl_seconds is None)
            ):
                raise ValueError("compact sequence-zero requests require context limits")
        elif self.chat_completion is None or self.tool_output is not None:
            raise ValueError("non-compact requests require chat_completion")
        if self.final and (self.max_candidates is None or self.candidate_ttl_seconds is None):
            raise ValueError("final tokenizer requests require max_candidates and candidate_ttl_seconds")
        if self.prefill and not self.exact_incremental_tokenizer:
            raise ValueError("prefill requires an exact incremental tokenizer request")
        if self.prefill and not self.prefill_continuation and (self.final or self.sequence_no != 0):
            raise ValueError("one-shot prefill requires a non-final sequence-zero request")
        if self.prefill_continuation and not self.prefill:
            raise ValueError("prefill continuation requires prefill")
        if (
            self.prefill_from_required_prefix or self.finalize_from_required_prefix
        ) and not self.exact_incremental_tokenizer:
            raise ValueError("required-prefix optimization requires exact incremental tokenization")
        if self.prefill_from_required_prefix and (
            not self.prefill_continuation or self.sequence_no != 0 or self.final
        ):
            raise ValueError("prefill from required prefix requires a non-final sequence-zero prefill continuation")
        if self.finalize_from_required_prefix and (not self.final or self.sequence_no != 0 or self.prefill):
            raise ValueError(
                "finalization from required prefix requires a final sequence-zero request without prefill"
            )
        return self


class StreamingToolCallCloseRequest(BaseModel):
    session_id: str
    chat_completion: Dict[str, Any]


class StreamingToolCallAbortRequest(BaseModel):
    session_id: str
    exact_incremental_tokenizer: bool = False
    defer: bool = False


@dataclass(frozen=True)
class StreamingPromptTokenCandidate:
    client: NeMoGymAsyncOpenAI
    prompt: Dict[str, Any]
    prompt_token_ids: tuple[int, ...]
    expires_at: float


@dataclass
class StreamingToolCallTokenizationContext:
    client: NeMoGymAsyncOpenAI
    tokenize_body: Dict[str, Any]
    ttl_seconds: float
    expires_at: float


class VLLMModel(SimpleResponsesAPIModel):
    config: VLLMModelConfig

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

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/v1/streaming_tool_call/tokenize")(self.tokenize_streaming_tool_call)
        app.post("/v1/streaming_tool_call/start")(self.start_streaming_tool_call)
        app.post("/v1/streaming_tool_call/append")(self.append_streaming_tool_call)
        app.post("/v1/streaming_tool_call/close")(self.close_streaming_tool_call)
        app.post("/v1/streaming_tool_call/abort")(self.abort_streaming_tool_call)
        return app

    def _post_init(self) -> None:
        self._clients = [
            NeMoGymAsyncOpenAI(
                base_url=base_url,
                api_key=self.config.api_key,
                default_headers=self.config.default_headers,
            )
            for base_url in self.config.base_url
        ]

        self._session_id_to_client: Dict[str, NeMoGymAsyncOpenAI] = dict()
        self._streaming_prompt_token_candidates: OrderedDict[str, StreamingPromptTokenCandidate] = OrderedDict()
        self._streaming_tool_call_tokenization_contexts: OrderedDict[str, StreamingToolCallTokenizationContext] = (
            OrderedDict()
        )

        self._converter = self.get_converter()

    def _expire_streaming_tool_call_tokenization_contexts(self) -> None:
        now = time()
        expired_session_ids = [
            session_id
            for session_id, context in (self._streaming_tool_call_tokenization_contexts.items())
            if context.expires_at <= now
        ]
        for session_id in expired_session_ids:
            self._streaming_tool_call_tokenization_contexts.pop(session_id, None)

    def _store_streaming_tool_call_tokenization_context(
        self,
        *,
        session_id: str,
        client: NeMoGymAsyncOpenAI,
        tokenize_body: Dict[str, Any],
        max_contexts: int,
        context_ttl_seconds: float,
    ) -> None:
        self._expire_streaming_tool_call_tokenization_contexts()
        self._streaming_tool_call_tokenization_contexts.pop(session_id, None)
        self._streaming_tool_call_tokenization_contexts[session_id] = StreamingToolCallTokenizationContext(
            client=client,
            tokenize_body=tokenize_body,
            ttl_seconds=context_ttl_seconds,
            expires_at=time() + context_ttl_seconds,
        )
        while len(self._streaming_tool_call_tokenization_contexts) > max_contexts:
            self._streaming_tool_call_tokenization_contexts.popitem(last=False)

    def _get_streaming_tool_call_tokenization_context(
        self,
        *,
        session_id: str,
        client: NeMoGymAsyncOpenAI,
    ) -> StreamingToolCallTokenizationContext:
        self._expire_streaming_tool_call_tokenization_contexts()
        context = self._streaming_tool_call_tokenization_contexts.get(session_id)
        if context is None:
            raise HTTPException(
                status_code=409,
                detail="compact tokenization context is missing or expired",
            )
        if context.client is not client:
            self._streaming_tool_call_tokenization_contexts.pop(session_id, None)
            raise HTTPException(
                status_code=409,
                detail="compact tokenization context was routed to another client",
            )
        context.expires_at = time() + context.ttl_seconds
        self._streaming_tool_call_tokenization_contexts.move_to_end(session_id)
        return context

    @staticmethod
    def _replace_streaming_tool_output(
        tokenize_body: Dict[str, Any],
        tool_output: str,
    ) -> Dict[str, Any]:
        messages = tokenize_body.get("messages")
        if not messages or messages[-1].get("role") != "tool":
            raise HTTPException(
                status_code=409,
                detail="compact tokenization context has no trailing tool message",
            )
        updated_messages = list(messages)
        updated_messages[-1] = {
            **messages[-1],
            "content": tool_output,
        }
        return {
            **tokenize_body,
            "messages": updated_messages,
        }

    @staticmethod
    def _streaming_prompt_tokenize_body(
        body_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        tokenize_body = {
            key: deepcopy(body_dict[key]) for key in ("model", "tools", "chat_template_kwargs") if key in body_dict
        }
        token_fields = (
            "prompt_token_ids",
            "generation_token_ids",
            "generation_log_probs",
        )
        messages = body_dict.get("messages", [])
        latest_token_fields_index = next(
            (
                index
                for index in reversed(range(len(messages)))
                if all(field in messages[index] for field in token_fields)
            ),
            None,
        )
        tokenize_body["messages"] = [
            {
                key: deepcopy(value)
                for key, value in message.items()
                if index == latest_token_fields_index or key not in token_fields
            }
            for index, message in enumerate(messages)
        ]
        return tokenize_body

    @staticmethod
    def _streaming_prompt_comparison_body(
        tokenize_body: Dict[str, Any],
    ) -> Dict[str, Any]:
        token_fields = (
            "prompt_token_ids",
            "generation_token_ids",
            "generation_log_probs",
        )
        comparison_body = {
            key: deepcopy(tokenize_body[key])
            for key in ("model", "tools", "chat_template_kwargs")
            if key in tokenize_body
        }
        comparison_body["messages"] = [
            {key: deepcopy(value) for key, value in message.items() if key not in token_fields}
            for message in tokenize_body.get("messages", [])
        ]
        return comparison_body

    def _expire_streaming_prompt_token_candidates(self) -> None:
        now = time()
        while self._streaming_prompt_token_candidates:
            _, candidate = next(iter(self._streaming_prompt_token_candidates.items()))
            if candidate.expires_at > now:
                break
            self._streaming_prompt_token_candidates.popitem(last=False)

    def _store_streaming_prompt_token_candidate(
        self,
        *,
        reuse_id: str,
        client: NeMoGymAsyncOpenAI,
        prompt: Dict[str, Any],
        prompt_token_ids: list[int],
        max_candidates: int,
        candidate_ttl_seconds: float,
    ) -> None:
        self._expire_streaming_prompt_token_candidates()
        self._streaming_prompt_token_candidates.pop(reuse_id, None)
        self._streaming_prompt_token_candidates[reuse_id] = StreamingPromptTokenCandidate(
            client=client,
            prompt=deepcopy(prompt),
            prompt_token_ids=tuple(prompt_token_ids),
            expires_at=time() + candidate_ttl_seconds,
        )
        while len(self._streaming_prompt_token_candidates) > max_candidates:
            self._streaming_prompt_token_candidates.popitem(last=False)

    def _consume_streaming_prompt_token_candidate(
        self,
        *,
        reuse_id: str,
        client: NeMoGymAsyncOpenAI,
        prompt: Dict[str, Any],
    ) -> tuple[
        str,
        Optional[list[int]],
        Optional[StreamingPromptTokenCandidate],
    ]:
        self._expire_streaming_prompt_token_candidates()
        candidate = self._streaming_prompt_token_candidates.pop(reuse_id, None)
        if candidate is None:
            return "missing", None, None
        if candidate.client is not client:
            return "mismatch", None, None
        if candidate.prompt != prompt:
            return "mismatch", None, candidate
        return "matched", list(candidate.prompt_token_ids), None

    @classmethod
    def _first_streaming_prompt_difference(
        cls,
        candidate: Any,
        request: Any,
        path: str = "$",
    ) -> str:
        if type(candidate) is not type(request):
            return f"{path}: type {type(candidate).__name__} != {type(request).__name__}"
        if isinstance(candidate, dict):
            for key in sorted(candidate.keys() | request.keys()):
                if key not in candidate:
                    return f"{path}.{key}: missing from candidate"
                if key not in request:
                    return f"{path}.{key}: missing from request"
                difference = cls._first_streaming_prompt_difference(candidate[key], request[key], f"{path}.{key}")
                if difference:
                    return difference
            return ""
        if isinstance(candidate, list):
            if len(candidate) != len(request):
                return f"{path}: list lengths {len(candidate)} != {len(request)}"
            for index, (candidate_item, request_item) in enumerate(zip(candidate, request)):
                difference = cls._first_streaming_prompt_difference(
                    candidate_item,
                    request_item,
                    f"{path}[{index}]",
                )
                if difference:
                    return difference
            return ""
        if candidate == request:
            return ""
        if isinstance(candidate, str):
            return f"{path}: string lengths {len(candidate)} != {len(request)}"
        return f"{path}: values differ"

    async def _tokenize_streaming_tool_call_prompt(
        self,
        request: Request,
        chat_completion: Dict[str, Any],
    ) -> tuple[NeMoGymAsyncOpenAI, list[int], Dict[str, Any]]:
        body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(chat_completion)
        body_dict = body.model_dump(exclude_unset=True)
        body_dict = self._preprocess_chat_completion_create_params(request, body_dict)
        tokenize_body = self._streaming_prompt_tokenize_body(body_dict)
        client = self._resolve_client(request)
        tokenize_response = await client.create_tokenize(**tokenize_body)
        return client, tokenize_response["tokens"], tokenize_body

    async def start_streaming_tool_call(
        self,
        request: Request,
        body: StreamingToolCallPromptRequest = Body(),
    ) -> Dict[str, Any]:
        assert body.chat_completion is not None
        client, prompt_token_ids, _ = await self._tokenize_streaming_tool_call_prompt(request, body.chat_completion)
        return await client.create_streaming_tool_call(
            "start",
            session_id=body.session_id,
            sequence_no=body.sequence_no,
            prompt_token_ids=prompt_token_ids,
        )

    async def tokenize_streaming_tool_call(
        self,
        request: Request,
        body: StreamingToolCallPromptRequest = Body(),
    ) -> Dict[str, Any]:
        """Tokenize a partial tool result without starting a prefill session."""
        request_handler_started_at = perf_counter()
        preprocess_started_at = perf_counter()
        client = self._resolve_client(request)
        compact_context_hit = False
        compact_context_rebuild_seconds = 0.0
        if body.compact_context and body.tool_output is not None:
            compact_context_rebuild_started_at = perf_counter()
            context = self._get_streaming_tool_call_tokenization_context(
                session_id=body.session_id,
                client=client,
            )
            tokenize_body = self._replace_streaming_tool_output(
                context.tokenize_body,
                body.tool_output,
            )
            compact_context_rebuild_seconds = perf_counter() - compact_context_rebuild_started_at
            compact_context_hit = True
        else:
            assert body.chat_completion is not None
            tokenization_body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(body.chat_completion)
            tokenization_body_dict = tokenization_body.model_dump(exclude_unset=True)
            tokenization_body_dict = self._preprocess_chat_completion_create_params(request, tokenization_body_dict)
            tokenize_body = self._streaming_prompt_tokenize_body(tokenization_body_dict)
        preprocess_seconds = perf_counter() - preprocess_started_at
        compact_context_registered = False
        compact_context_registration_seconds = 0.0
        if body.exact_incremental_tokenizer:
            incremental_kwargs = {
                "session_id": body.session_id,
                "sequence_no": body.sequence_no,
                "final": body.final,
                "prefill": body.prefill,
            }
            if body.prefill_continuation:
                incremental_kwargs["prefill_continuation"] = True
            if body.prefill_from_required_prefix:
                incremental_kwargs["prefill_from_required_prefix"] = True
            if body.finalize_from_required_prefix:
                incremental_kwargs["finalize_from_required_prefix"] = True
            vllm_request_started_at = perf_counter()
            if compact_context_hit:
                assert body.tool_output is not None
                tokenization_result = await client.create_incremental_tokenize_compact(
                    tool_output=body.tool_output,
                    **incremental_kwargs,
                )
            else:
                if body.compact_context:
                    incremental_kwargs["compact_context"] = True
                    if not body.final:
                        assert body.max_contexts is not None
                        assert body.context_ttl_seconds is not None
                        incremental_kwargs["max_contexts"] = body.max_contexts
                        incremental_kwargs["context_ttl_seconds"] = body.context_ttl_seconds
                tokenization_result = await client.create_incremental_tokenize(**tokenize_body, **incremental_kwargs)
            vllm_request_seconds = perf_counter() - vllm_request_started_at
            prompt_token_ids = tokenization_result.pop("tokens", None)
            result = dict(tokenization_result)
            if body.compact_context and not body.final and not compact_context_hit:
                assert body.max_contexts is not None
                assert body.context_ttl_seconds is not None
                compact_context_registration_started_at = perf_counter()
                self._store_streaming_tool_call_tokenization_context(
                    session_id=body.session_id,
                    client=client,
                    tokenize_body=tokenize_body,
                    max_contexts=body.max_contexts,
                    context_ttl_seconds=body.context_ttl_seconds,
                )
                compact_context_registration_seconds = perf_counter() - compact_context_registration_started_at
                compact_context_registered = True
        else:
            tokenization_result = await client.create_tokenize(**tokenize_body)
            prompt_token_ids = tokenization_result["tokens"]
            result = {
                "sequence_no": body.sequence_no,
                "token_count": len(prompt_token_ids),
            }
        if body.final:
            assert body.max_candidates is not None
            assert body.candidate_ttl_seconds is not None
            assert prompt_token_ids is not None
            self._store_streaming_prompt_token_candidate(
                reuse_id=body.session_id,
                client=client,
                prompt=self._streaming_prompt_comparison_body(tokenize_body),
                prompt_token_ids=prompt_token_ids,
                max_candidates=body.max_candidates,
                candidate_ttl_seconds=body.candidate_ttl_seconds,
            )
            result["reuse_id"] = body.session_id
            self._streaming_tool_call_tokenization_contexts.pop(body.session_id, None)
        if body.exact_incremental_tokenizer:
            result.update(
                {
                    "gym_compact_context_registrations": int(compact_context_registered),
                    "gym_compact_context_hits": int(compact_context_hit),
                    "gym_compact_context_rebuild_seconds": (compact_context_rebuild_seconds),
                    "gym_compact_context_registration_seconds": (compact_context_registration_seconds),
                    "gym_preprocess_seconds": preprocess_seconds,
                    "gym_vllm_request_seconds": vllm_request_seconds,
                    "gym_request_handler_seconds": (perf_counter() - request_handler_started_at),
                }
            )
        return result

    async def append_streaming_tool_call(
        self,
        request: Request,
        body: StreamingToolCallPromptRequest = Body(),
    ) -> Dict[str, Any]:
        assert body.chat_completion is not None
        client, prompt_token_ids, _ = await self._tokenize_streaming_tool_call_prompt(request, body.chat_completion)
        return await client.create_streaming_tool_call(
            "append",
            session_id=body.session_id,
            sequence_no=body.sequence_no,
            prompt_token_ids=prompt_token_ids,
        )

    async def close_streaming_tool_call(
        self,
        request: Request,
        body: StreamingToolCallCloseRequest = Body(),
    ) -> Dict[str, Any]:
        client, prompt_token_ids, _ = await self._tokenize_streaming_tool_call_prompt(request, body.chat_completion)
        return await client.create_streaming_tool_call(
            "close",
            session_id=body.session_id,
            final_prompt_token_ids=prompt_token_ids,
        )

    async def abort_streaming_tool_call(
        self,
        request: Request,
        background_tasks: BackgroundTasks,
        body: StreamingToolCallAbortRequest = Body(),
    ) -> Dict[str, Any]:
        client = self._resolve_client(request)
        if body.exact_incremental_tokenizer:
            self._streaming_tool_call_tokenization_contexts.pop(body.session_id, None)
            if body.defer:
                background_tasks.add_task(
                    client.abort_incremental_tokenize,
                    session_id=body.session_id,
                )
                return {"aborted": False, "scheduled": True}
            return await client.abort_incremental_tokenize(session_id=body.session_id)
        return await client.create_streaming_tool_call("abort", session_id=body.session_id)

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

        choice = chat_completion_response.choices[0]

        response_output = self._converter.postprocess_chat_response(choice)
        response_output_dicts = [item.model_dump() for item in response_output]

        usage = None
        if chat_completion_response.usage:
            usage = NeMoGymResponseUsage(
                input_tokens=chat_completion_response.usage.prompt_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=chat_completion_response.usage.completion_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=chat_completion_response.usage.prompt_tokens
                + chat_completion_response.usage.completion_tokens,
            )

        incomplete_details = None
        if choice.finish_reason == "length":
            incomplete_details = {"reason": "max_output_tokens"}
        elif choice.finish_reason == "content_filter":
            incomplete_details = {"reason": "content_filter"}

        # Chat Completion -> Response
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model,
            object="response",
            output=response_output_dicts,
            tool_choice=body.tool_choice if body.tool_choice is not None else "auto",
            parallel_tool_calls=body.parallel_tool_calls,
            tools=body.tools,
            temperature=body.temperature,
            top_p=body.top_p,
            background=body.background,
            max_output_tokens=body.max_output_tokens,
            max_tool_calls=body.max_tool_calls,
            previous_response_id=body.previous_response_id,
            prompt=body.prompt,
            reasoning=body.reasoning,
            service_tier=body.service_tier,
            text=body.text,
            top_logprobs=body.top_logprobs,
            truncation=body.truncation,
            metadata=body.metadata,
            instructions=body.instructions,
            user=body.user,
            incomplete_details=incomplete_details,
            usage=usage,
        )

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

        client = self._resolve_client(request)
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
        if self.config.replace_developer_role_with_system:
            for message_dict in body_dict["messages"]:
                if message_dict.get("role") == "developer":
                    message_dict["role"] = "system"

        body_dict["model"] = self.config.model

        chat_template_kwargs = {}
        if self.config.chat_template_kwargs:
            chat_template_kwargs = deepcopy(self.config.chat_template_kwargs)

        metadata = body_dict.get("metadata", dict())

        # Merge global config chat_template_kwargs with per-request overrides in metadata (e.g. per-sample reasoning on/off)
        metadata_chat_template_kwargs_str = metadata.get("chat_template_kwargs", "{}")
        chat_template_kwargs.update(json.loads(metadata_chat_template_kwargs_str))

        if chat_template_kwargs:
            body_dict["chat_template_kwargs"] = chat_template_kwargs

        # Merge global config extra_body with per-request overrides from metadata
        extra_body = {}
        if self.config.extra_body:
            extra_body = deepcopy(self.config.extra_body)

        metadata_extra_body_str = metadata.get("extra_body", "{}")
        extra_body.update(json.loads(metadata_extra_body_str))

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
                # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
                # For prompt and generation token IDs
                # return_token_ids=True,
                # For prompt token IDs
                # prompt_logprobs=0,
            )

        if self.config.uses_reasoning_parser:
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

        return body_dict

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)
        streaming_prompt_reuse_id = body_dict.pop("streaming_prompt_reuse_id", None)
        body_dict = self._preprocess_chat_completion_create_params(request, body_dict)

        client = self._resolve_client(request)
        streaming_prompt_reuse_status = None
        streaming_prompt_reuse_match_kind = None
        reused_prompt_token_ids = None
        if streaming_prompt_reuse_id is not None:
            tokenize_body_dict = self._streaming_prompt_tokenize_body(body_dict)
            comparison_body_dict = self._streaming_prompt_comparison_body(tokenize_body_dict)
            (
                streaming_prompt_reuse_status,
                reused_prompt_token_ids,
                mismatched_candidate,
            ) = self._consume_streaming_prompt_token_candidate(
                reuse_id=streaming_prompt_reuse_id,
                client=client,
                prompt=comparison_body_dict,
            )
            if reused_prompt_token_ids is not None:
                streaming_prompt_reuse_match_kind = "exact"
            elif mismatched_candidate is not None:
                tokenize_response = await client.create_tokenize(**tokenize_body_dict)
                request_prompt_token_ids = tokenize_response["tokens"]
                if tuple(request_prompt_token_ids) == (mismatched_candidate.prompt_token_ids):
                    streaming_prompt_reuse_status = "matched"
                    streaming_prompt_reuse_match_kind = "token_equivalent"
                    reused_prompt_token_ids = request_prompt_token_ids
                else:
                    common_prefix_tokens = 0
                    for candidate_token, request_token in zip(
                        mismatched_candidate.prompt_token_ids,
                        request_prompt_token_ids,
                    ):
                        if candidate_token != request_token:
                            break
                        common_prefix_tokens += 1
                    difference = self._first_streaming_prompt_difference(
                        mismatched_candidate.prompt,
                        comparison_body_dict,
                    )
                    print(
                        "Streaming prompt reuse token mismatch: "
                        f"{difference}; candidate_tokens="
                        f"{len(mismatched_candidate.prompt_token_ids)}, "
                        f"request_tokens={len(request_prompt_token_ids)}, "
                        f"common_prefix_tokens={common_prefix_tokens}",
                        flush=True,
                    )
            if reused_prompt_token_ids is not None:
                body_dict["required_full_prompt_token_ids"] = reused_prompt_token_ids
                body_dict["streaming_tool_call_session_id"] = streaming_prompt_reuse_id

        if not self.config.sequential_reasoning_allowed:
            last_message = body_dict["messages"][-1]
            if last_message["role"] == "assistant" and not (last_message["content"] or last_message.get("tool_calls")):
                res = self._create_empty_chat_completion()
                res.choices[0].finish_reason = "content_filter"
                return res

        try:
            chat_completion_dict = await client.create_chat_completion(**body_dict)
        except ClientResponseError as e:
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
                res = self._create_empty_chat_completion()
                res.choices[0].finish_reason = "length"
                return res
            else:
                raise e

        choice_dict = chat_completion_dict["choices"][0]
        streaming_tool_call_same_request_status = chat_completion_dict.pop(
            "streaming_tool_call_same_request_status",
            None,
        )
        if self.config.uses_reasoning_parser:
            # See the TODO wrt reasoning_content above
            reasoning_content = choice_dict["message"].get("reasoning_content") or choice_dict["message"].get(
                "reasoning"
            )
            if reasoning_content:
                choice_dict["message"].pop("reasoning_content", None)
                # See the TODO wrt reasoning_content above
                choice_dict["message"].pop("reasoning", None)

                # We wrap this here in think tags for Gym's sake and to return a valid OpenAI Chat Completions response.
                choice_dict["message"]["content"] = self._converter._wrap_reasoning_in_think_tags(
                    [reasoning_content]
                ) + (choice_dict["message"].get("content") or "")
        else:
            # See the TODO wrt reasoning_content above
            assert not (choice_dict["message"].get("reasoning_content") or choice_dict["message"].get("reasoning")), (
                f"NeMo Gym server `{self.config.name}` config has explicitly been set to not use a reasoning parser i.e. `uses_reasoning_parser: false`. Please do not use a reasoning parser in your vLLM endpoint, or fix the `{self.config.name}` server config!"
            )

        if self.config.return_token_id_information and "prompt_token_ids" not in choice_dict["message"]:
            # Check vLLM honored the logprobs request.
            # It returns choice.logprobs=None when it computed none.
            # That happens when a null top_logprobs reached it, or the contract changed across versions.
            # Without this check the code below raises a TypeError or emits empty token ids that zero the loss mask.
            # An empty content list is a valid zero-token generation and passes through.
            logprobs_block = choice_dict.get("logprobs")
            if not logprobs_block or logprobs_block.get("content") is None:
                raise RuntimeError(
                    f"`{self.config.name}` requested per-token logprobs from vLLM "
                    f"(return_token_id_information=True, logprobs=True, top_logprobs=0), but the response "
                    f"had none (choice.logprobs={logprobs_block!r}). Cannot extract token ids or logprobs."
                )
            log_probs = logprobs_block["content"]
            generation_log_probs = [log_prob["logprob"] for log_prob in log_probs]

            """
            START TODO remove this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
            """
            # Looks like `"token_id:151667"`
            generation_token_ids = [log_prob["token"].removeprefix("token_id:") for log_prob in log_probs]

            # The tokenize endpoint doesn't accept any sampling parameters
            # The only relevant params are model, messages, and tools.
            #
            # IMPORTANT: pass through chat-template knobs (e.g. enable_thinking)
            # when tokenizing, otherwise `prompt_token_ids` (and therefore logged
            # `prompt_str`) can be built with different chat template settings than
            # the actual generation request.
            if reused_prompt_token_ids is None:
                tokenize_body_dict = self._streaming_prompt_tokenize_body(body_dict)
                # The base url has /v1 at the end but vLLM's tokenize endpoint
                # does not have v1.
                tokenize_response = await client.create_tokenize(**tokenize_body_dict)
                prompt_token_ids = tokenize_response["tokens"]
            else:
                prompt_token_ids = reused_prompt_token_ids
            """
            END
            """

            message_dict = choice_dict["message"]
            message_dict.update(
                dict(
                    # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
                    # prompt_token_ids=chat_completion_dict["prompt_token_ids"],
                    prompt_token_ids=prompt_token_ids,
                    # generation_token_ids=choice_dict["token_ids"],
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                )
            )

            # Clean the duplicated information
            choice_dict.pop("logprobs")
            # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
            # chat_completion_dict.pop("prompt_token_ids")
            # choice_dict.pop("token_ids")

        if streaming_prompt_reuse_status is not None:
            chat_completion_dict["streaming_prompt_reuse_status"] = streaming_prompt_reuse_status
        if streaming_prompt_reuse_match_kind is not None:
            chat_completion_dict["streaming_prompt_reuse_match_kind"] = streaming_prompt_reuse_match_kind
        if streaming_tool_call_same_request_status is not None:
            chat_completion_dict["streaming_tool_call_same_request_status"] = streaming_tool_call_same_request_status

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

    def _resolve_client(self, request: Request) -> NeMoGymAsyncOpenAI:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._session_id_to_client:
            # There is probably a better way to select the endpoint for this request. But this will do for now.
            client_idx = len(self._session_id_to_client) % len(self._clients)
            client = self._clients[client_idx]
            self._session_id_to_client[session_id] = client
        client = self._session_id_to_client[session_id]

        return client


if __name__ == "__main__":
    VLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VLLMModel.run_webserver()  # noqa: F401
