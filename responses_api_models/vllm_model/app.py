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
import json
import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from time import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from aiohttp.client_exceptions import ClientResponseError
from fastapi import Request
from pydantic import BaseModel, Field, model_validator

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    RESPONSES_TO_TRAIN,
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionAssistantMessageForTrainingParam,
    NeMoGymChatCompletionAssistantMessageParam,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionDeveloperMessageParam,
    NeMoGymChatCompletionMessage,
    NeMoGymChatCompletionMessageParam,
    NeMoGymChatCompletionMessageToolCallFunctionParam,
    NeMoGymChatCompletionMessageToolCallParam,
    NeMoGymChatCompletionSystemMessageParam,
    NeMoGymChatCompletionToolMessageParam,
    NeMoGymChatCompletionToolParam,
    NeMoGymChatCompletionUserMessageParam,
    NeMoGymChoice,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymFunctionDefinition,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
    TokenIDLogProbMixin,
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_worker


class VLLMModelConfig(BaseResponsesAPIModelConfig):
    base_url: Union[str, List[str]]
    api_key: str
    model: str
    return_token_id_information: bool

    uses_reasoning_parser: bool
    replace_developer_role_with_system: bool = False

    # Whether or not the model can generate a reasoning output, and called again to produce additional reasoning output.
    sequential_reasoning_allowed: bool = True

    # As of Feb 2026, we default this to False since majority of open source models aren't responses native with the exception of GPT-OSS
    is_responses_native: bool = False

    chat_template_kwargs: Optional[Dict[str, Any]] = None

    # Corresponds to the extra_body of OpenAI Client.
    extra_body: Optional[Dict[str, Any]] = None

    def model_post_init(self, context):
        if isinstance(self.base_url, str):
            self.base_url = [self.base_url]
        return super().model_post_init(context)


class StreamingToolCallPromptRequest(BaseModel):
    session_id: str
    sequence_no: int
    chat_completion: Dict[str, Any]
    final: bool = False
    exact_incremental_tokenizer: bool = False
    max_candidates: Optional[int] = Field(default=None, gt=0)
    candidate_ttl_seconds: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_final_candidate_limits(self):
        if self.final and (self.max_candidates is None or self.candidate_ttl_seconds is None):
            raise ValueError("final tokenizer requests require max_candidates and candidate_ttl_seconds")
        return self


class StreamingToolCallCloseRequest(BaseModel):
    session_id: str
    chat_completion: Dict[str, Any]


class StreamingToolCallAbortRequest(BaseModel):
    session_id: str
    exact_incremental_tokenizer: bool = False


@dataclass(frozen=True)
class StreamingPromptTokenCandidate:
    client: NeMoGymAsyncOpenAI
    prompt: Dict[str, Any]
    prompt_token_ids: tuple[int, ...]
    expires_at: float


class VLLMModel(SimpleResponsesAPIModel):
    config: VLLMModelConfig

    def get_converter(self) -> "VLLMConverter":
        """Return the converter used for Responses API <-> Chat Completions mapping.

        Override in subclasses (e.g. GenRMModel) to use a specialized converter.
        """
        return VLLMConverter(
            return_token_id_information=self.config.return_token_id_information,
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
            )
            for base_url in self.config.base_url
        ]

        self._session_id_to_client: Dict[str, NeMoGymAsyncOpenAI] = dict()
        self._streaming_prompt_token_candidates: OrderedDict[str, StreamingPromptTokenCandidate] = OrderedDict()

        self._converter = self.get_converter()

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
        tokenization_body = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(body.chat_completion)
        tokenization_body_dict = tokenization_body.model_dump(exclude_unset=True)
        tokenization_body_dict = self._preprocess_chat_completion_create_params(request, tokenization_body_dict)
        tokenize_body = self._streaming_prompt_tokenize_body(tokenization_body_dict)
        client = self._resolve_client(request)
        if body.exact_incremental_tokenizer:
            tokenization_result = await client.create_incremental_tokenize(
                **tokenize_body,
                session_id=body.session_id,
                sequence_no=body.sequence_no,
                final=body.final,
            )
            prompt_token_ids = tokenization_result.pop("tokens", None)
            result = dict(tokenization_result)
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
        return result

    async def append_streaming_tool_call(
        self,
        request: Request,
        body: StreamingToolCallPromptRequest = Body(),
    ) -> Dict[str, Any]:
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
        body: StreamingToolCallAbortRequest = Body(),
    ) -> Dict[str, Any]:
        client = self._resolve_client(request)
        if body.exact_incremental_tokenizer:
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

        # Chat Completion -> Response
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model,
            object="response",
            output=response_output_dicts,
            tool_choice=body.tool_choice if "tool_choice" in body else "auto",
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
            incomplete_details={"reason": "max_output_tokens"} if choice.finish_reason == "length" else None,
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
                    if reasoning_matches:
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
                        if reasoning_matches:
                            message_dict["reasoning_content"] = reasoning_matches[0]
                            # See the TODO wrt reasoning_content above
                            message_dict["reasoning"] = reasoning_matches[0]
                elif not content:
                    # No content or content None is a no-op
                    pass
                else:
                    raise NotImplementedError

        if extra_body:
            body_dict = extra_body | body_dict

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

        if not self.config.sequential_reasoning_allowed:
            last_message = body_dict["messages"][-1]
            if last_message["role"] == "assistant" and not (last_message["content"] or last_message.get("tool_calls")):
                return self._create_empty_chat_completion()

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
            else:
                raise e

        choice_dict = chat_completion_dict["choices"][0]
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
                ) + (choice_dict["message"]["content"] or "")
        else:
            # See the TODO wrt reasoning_content above
            assert not (choice_dict["message"].get("reasoning_content") or choice_dict["message"].get("reasoning")), (
                f"NeMo Gym server `{self.config.name}` config has explicitly been set to not use a reasoning parser i.e. `uses_reasoning_parser: false`. Please do not use a reasoning parser in your vLLM endpoint, or fix the `{self.config.name}` server config!"
            )

        if self.config.return_token_id_information:
            log_probs = choice_dict["logprobs"]["content"]
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

        return NeMoGymChatCompletion.model_validate(chat_completion_dict)

    def _resolve_client(self, request: Request) -> NeMoGymAsyncOpenAI:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._session_id_to_client:
            # There is probably a better way to select the endpoint for this request. But this will do for now.
            client_idx = len(self._session_id_to_client) % len(self._clients)
            client = self._clients[client_idx]
            self._session_id_to_client[session_id] = client
        client = self._session_id_to_client[session_id]

        return client


class VLLMConverterResponsesToChatCompletionsState(BaseModel):
    return_token_id_information: bool

    messages: List[NeMoGymChatCompletionMessageParam] = Field(default_factory=list)

    # We are mapping from Response input items to chat completions messages, which is many to one.
    # Our state will accumulate the reasoning, chat, and tool calls for assistant messages.
    content_buffer: str = ""  # Buffer for reasoning and chat
    tool_calls_buffer: List[NeMoGymChatCompletionMessageToolCallParam] = Field(default_factory=list)

    # Will only be populated if return_token_id_information is True.
    token_information: Optional[TokenIDLogProbMixin] = None

    def flush_assistant(self) -> None:
        if not (self.content_buffer or self.tool_calls_buffer):
            return

        shared_params = dict(
            content=self.content_buffer or None,
            role="assistant",
            tool_calls=self.tool_calls_buffer,
        )

        # We check here that self.token_information is non-empty since it's possible that some assistant messages are entirely inputs and are not generated by the model in this trajectory.
        if self.return_token_id_information and self.token_information:
            message = NeMoGymChatCompletionAssistantMessageForTrainingParam(
                **shared_params,
                **self.token_information.model_dump(),
            )
        else:
            message = NeMoGymChatCompletionAssistantMessageParam(**shared_params)

        self.messages.append(message)

        self.content_buffer = ""
        self.tool_calls_buffer = []


class VLLMConverter(BaseModel):
    return_token_id_information: bool

    # =======================================================
    # Reasoning handling. This may change across models and model families
    # =======================================================

    THINK_TAG_PATTERN: ClassVar = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    @staticmethod
    def _wrap_reasoning_in_think_tags(texts: List[str]) -> str:
        return "".join(f"<think>{t}</think>" for t in texts if t)

    @classmethod
    def _parse_think_tags(cls, content: str) -> Tuple[List[str], str]:
        # Extract reasoning content from between <think></think> tags.
        matches = cls.THINK_TAG_PATTERN.findall(content)
        # Remove reasoning from main content
        cleaned = cls.THINK_TAG_PATTERN.sub("", content)
        return matches, cleaned

    # =======================================================
    # Response create params to Chat Completion create params
    # =======================================================

    def responses_to_chat_completion_create_params(
        self,
        responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        responses_create_params = responses_create_params.model_dump(exclude_unset=True)

        # Tracks messages including reasoning for each respective message type helper function
        state = VLLMConverterResponsesToChatCompletionsState(
            return_token_id_information=self.return_token_id_information
        )

        # Input can be a string. Wrap in a ResponseInput-like
        response_input = responses_create_params["input"]
        if isinstance(response_input, str):
            wrapped_input = {
                "content": [
                    {
                        "text": response_input,
                        "type": "input_text",
                    }
                ],
                "role": "user",
                "type": "message",
            }
            input_messages = [wrapped_input]
        else:
            input_messages = responses_create_params.pop("input", [])

        for m in input_messages:
            if not m.get("type") and m.get("role"):
                m["type"] = "message"

            match m["type"]:
                case "message":
                    self._format_message(m, state)
                case "reasoning":
                    self._format_reasoning(m, state)
                case "function_call":
                    self._format_function_call(m, state)
                case "function_call_output":
                    self._format_function_call_output(m, state)
                case _:  # pragma: no cover
                    raise NotImplementedError(f"Unsupported message type: {m}")

            if self.return_token_id_information and m.get("prompt_token_ids"):
                state.token_information = TokenIDLogProbMixin(
                    prompt_token_ids=m["prompt_token_ids"],
                    generation_token_ids=m["generation_token_ids"],
                    generation_log_probs=m["generation_log_probs"],
                )

        state.flush_assistant()

        model = responses_create_params.pop("model", None)
        if model is not None:
            responses_create_params["model"] = model

        # The corresponding parameter to `max_output_tokens`` is `max_tokens`
        max_output_tokens = responses_create_params.pop("max_output_tokens", None)
        if max_output_tokens is not None:
            responses_create_params["max_tokens"] = max_output_tokens

        tools = responses_create_params.pop("tools", None)
        if tools is not None:
            responses_create_params["tools"] = []
            for tool_dict in tools:
                tool_dict = tool_dict.copy()
                tool_dict.pop("type", None)

                # As of vLLM 0.17.1, vLLM Chat Completions does not accept this `strict` parameter on tool definitions that OpenAI accepts.
                tool_dict.pop("strict", None)
                responses_create_params["tools"].append(
                    NeMoGymChatCompletionToolParam(type="function", function=NeMoGymFunctionDefinition(**tool_dict))
                )

        chat_completion_create_params = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=state.messages,
            **responses_create_params,
        )

        return chat_completion_create_params

    def _format_function_call_output(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        state.flush_assistant()

        assert "call_id" in m
        converted = NeMoGymChatCompletionToolMessageParam(
            content=m["output"],
            role="tool",
            tool_call_id=m["call_id"],
        )
        state.messages.append(converted)

    def _format_message(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        content = m["content"]

        if isinstance(content, list) and m["role"] != "assistant":
            converted_parts = []
            for part_param in content:
                match part_param["type"]:
                    case "input_text":
                        converted_parts.append({"type": "text", "text": part_param["text"]})
                    case "input_image":
                        image_url = part_param.get("image_url", "")
                        detail = part_param.get("detail", "auto")
                        converted_parts.append(
                            {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
                        )
                    case _:
                        raise NotImplementedError(f"Unsupported part param type: {part_param['type']}")
            content = converted_parts
            m["content"] = content

        match m["role"]:
            case "assistant":
                # Handle reasoning
                final_content = ""
                if isinstance(m["content"], list):
                    content_str = "".join([part.get("text", "") for part in m["content"]])
                    final_content += content_str
                elif isinstance(m["content"], str):
                    final_content += m["content"]
                else:
                    raise NotImplementedError(
                        f"Expected m['content'] to be str or list[dict], but got {type(m['content']).__name__!r}: {m['content']!r}"
                    )

                converted = []
                state.content_buffer += final_content
            case "user":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionUserMessageParam(
                        content=content,
                        role="user",
                    )
                ]
            # TODO: Revisit this in case we need separate handling. Not all chat templates may support the 'developer' role.
            case "system":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionSystemMessageParam(
                        content=content,
                        role="system",
                    )
                ]
            case "developer":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionDeveloperMessageParam(
                        content=content,
                        role="developer",
                    )
                ]
            case _:  # pragma: no cover
                raise NotImplementedError(f"Unrecognized role for message: `{m['role']}`")

        state.messages.extend(converted)

    def _format_reasoning(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        """
        Collects text from 'reasoning' messages in responses api and appends it to a buffer.

        This is done to group together one (or multiple) reasoning message(s) into a single,
        cohesive block, later prepending it to a subsequent assistant message.
        See: https://github.com/NVIDIA-NeMo/Gym/blob/main/docs/how-to-faq.md#faq-openai-responses-vs-chat-completions-api for an example of reasoning in responses api.
        """
        if "summary" in m and m["summary"]:
            texts = [s["text"] for s in m["summary"]]
            state.content_buffer += self._wrap_reasoning_in_think_tags(texts)

    def _format_function_call(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        assert "call_id" in m
        tool_call = NeMoGymChatCompletionMessageToolCallParam(
            id=m["call_id"],
            function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                arguments=m["arguments"],
                name=m["name"],
            ),
            type="function",
        )
        state.tool_calls_buffer.append(tool_call)

    # =======================================================
    # Chat Completion to Response
    # =======================================================

    def postprocess_chat_response(self, choice: NeMoGymChoice) -> List[NeMoGymResponseOutputItem]:
        return self.postprocess_assistant_message_dict(choice.message.model_dump())

    def postprocess_assistant_message_dict(self, message_dict: Dict[str, Any]) -> List[NeMoGymResponseOutputItem]:
        response_output = []

        content = message_dict.get("content") or ""
        reasoning_matches, content = self._extract_reasoning_from_content(content)
        if reasoning_matches:
            reasoning_item = NeMoGymResponseReasoningItem(
                id=f"rs_{uuid4().hex}",
                type="reasoning",
                summary=[
                    NeMoGymSummary(text=reasoning_text, type="summary_text") for reasoning_text in reasoning_matches
                ],
                status="completed",
            )
            response_output.append(reasoning_item)

        tool_calls_raw = message_dict.get("tool_calls", []) or []
        # We need to return at least one output item. When the model decides to just stop with no chat or tool calls
        # We just add an output item with empty or null content here. This is prevalent e.g. in the case of base models that may not be the most reliable since they have not been instruction tuned.
        has_empty_output = not (response_output or tool_calls_raw)

        if content or has_empty_output:
            response_output.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    role=message_dict.get("role"),
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=content,
                            annotations=[],
                        )
                    ],
                    status="completed",
                    type="message",
                )
            )

        for tc in tool_calls_raw:
            assert "id" in tc
            response_output.append(
                NeMoGymResponseFunctionToolCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                    call_id=tc["id"],
                    type="function_call",
                    status="completed",
                    id=tc["id"],
                )
            )

        # `"prompt_token_ids" in raw_message`: sometimes the model endpoint may go out of context length, in which case we return an empty response
        # In these cases, there are no token id information provided.
        if self.return_token_id_information and "prompt_token_ids" in message_dict:
            last_response_output_item = response_output[-1]
            train_cls = RESPONSES_TO_TRAIN[last_response_output_item.__class__]
            response_output[-1] = train_cls(
                **last_response_output_item.model_dump(),
                prompt_token_ids=message_dict["prompt_token_ids"],
                generation_token_ids=message_dict["generation_token_ids"],
                generation_log_probs=message_dict["generation_log_probs"],
            )

        return response_output

    def _extract_reasoning_from_content(self, content: str) -> Tuple[List[str], str]:
        # TODO: Currently only parses reasoning wrapped in <think>...</think> tags.
        # Maybe parameterize to support other model formats in the future.
        return self._parse_think_tags(content)

    def chat_completions_messages_to_responses_items(
        self, messages: List[Dict[str, Any]]
    ) -> List[NeMoGymResponseOutputItem]:
        output_items = []

        for message in messages:
            role = message["role"]
            if role in ("user", "system", "developer"):
                output_items.append(NeMoGymEasyInputMessage.model_validate(message))
            elif role == "assistant":
                output_items.extend(self.postprocess_assistant_message_dict(message))
            elif role == "tool":
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=message["tool_call_id"],
                        output=message["content"],
                        status="completed",
                    )
                )
            else:
                raise NotImplementedError(f"Unrecognized role: {role}!")

        return output_items


def split_responses_input_output_items(
    items: List[NeMoGymResponseOutputItem],
) -> Tuple[List[NeMoGymResponseOutputItem], List[NeMoGymResponseOutputItem]]:
    if not items:
        return [], []

    for i, item in enumerate(items):
        if getattr(item, "role", None) == "assistant" or getattr(item, "type", None) in {
            "reasoning",
            "reasoning_item",
        }:
            break

    return items[:i], items[i:]


if __name__ == "__main__":
    VLLMModel.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = VLLMModel.run_webserver()  # noqa: F401
