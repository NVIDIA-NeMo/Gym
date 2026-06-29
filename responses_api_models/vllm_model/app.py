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
import logging
import os
import re
from copy import deepcopy
from time import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from aiohttp.client_exceptions import ClientResponseError
from fastapi import Request
from pydantic import BaseModel, Field

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
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint


LOGGER = logging.getLogger(__name__)

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_LOGPROB_TOKEN_ID_SOURCES = {"logprob", "logprobs", "logprob_tokens"}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_ENV_VALUES


def _metadata_json_dict(value: Any, *, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value.strip() or "{}")
    elif isinstance(value, BaseModel):
        value = value.model_dump(exclude_unset=True)

    if not isinstance(value, dict):
        raise TypeError(f"metadata.{field_name} must be a JSON object")
    return value


def _summarize_forwarded_extra_body(extra_body: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in extra_body.items():
        if key == "required_prompt_token_ids":
            if isinstance(value, list):
                summary[key] = {
                    "count": len(value),
                    "first8": value[:8],
                    "last8": value[-8:],
                    "checksum": sum((idx + 1) * int(token_id) for idx, token_id in enumerate(value))
                    % 1_000_000_007,
                }
            else:
                summary[key] = type(value).__name__
            continue

        if key == "mm_processor_kwargs" and isinstance(value, dict):
            mm_summary = deepcopy(value)
            precomputed = mm_summary.get("precomputed_imgs_sizes")
            if isinstance(precomputed, list):
                mm_summary["precomputed_imgs_sizes"] = {
                    "count": len(precomputed),
                    "first": precomputed[:1],
                    "last": precomputed[-1:],
                }
            summary[key] = mm_summary
            continue

        summary[key] = value
    return summary


def _vllm_token_id_source() -> str:
    return os.environ.get("NEMO_GYM_VLLM_TOKEN_ID_SOURCE", "native").strip().lower()


def _logprob_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("logprob")
    else:
        value = getattr(value, "logprob", None)
    if value is None:
        return None
    return float(value)


def _prompt_logprob_for_token(entry: Any, token_id: int) -> Optional[float]:
    if not isinstance(entry, dict) or not entry:
        return None

    for key in (token_id, str(token_id), f"token_id:{token_id}"):
        if key in entry:
            return _logprob_value(entry[key])

    for key, value in entry.items():
        try:
            if int(str(key).removeprefix("token_id:")) == token_id:
                return _logprob_value(value)
        except ValueError:
            pass

    if len(entry) == 1:
        return _logprob_value(next(iter(entry.values())))
    return None


def _extract_prompt_logprobs_for_tokens(
    prompt_logprobs: Any,
    token_ids: List[int],
    start_idx: int,
) -> Tuple[List[float], str]:
    if not isinstance(prompt_logprobs, list):
        return [], "missing_prompt_logprobs"
    end_idx = start_idx + len(token_ids)
    if len(prompt_logprobs) < end_idx:
        return [], f"prompt_logprobs_short:{len(prompt_logprobs)}<{end_idx}"

    values: List[float] = []
    for rel_idx, token_id in enumerate(token_ids):
        value = _prompt_logprob_for_token(prompt_logprobs[start_idx + rel_idx], token_id)
        if value is None:
            return values, f"missing_token_logprob_at:{rel_idx}"
        values.append(value)
    return values, "ok"


class VLLMModelConfig(BaseResponsesAPIModelConfig):
    base_url: Union[str, List[str]]
    api_key: str
    model: str
    return_token_id_information: bool
    max_input_tokens: Optional[int] = None

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

    def _post_init(self) -> None:
        self._clients = [
            NeMoGymAsyncOpenAI(
                base_url=base_url,
                api_key=self.config.api_key,
            )
            for base_url in self.config.base_url
        ]

        self._session_id_to_client: Dict[str, NeMoGymAsyncOpenAI] = dict()
        self._session_id_to_client_idx: Dict[str, int] = dict()

        self._converter = self.get_converter()

    def _create_context_length_exceeded_chat_completion(
        self, prompt_token_ids: Optional[List[int]] = None
    ) -> NeMoGymChatCompletion:
        message_kwargs = dict(
            role="assistant",
            content=None,
            tool_calls=None,
        )
        if self.config.return_token_id_information and prompt_token_ids is not None:
            message = NeMoGymChatCompletionMessageForTraining(
                **message_kwargs,
                prompt_token_ids=prompt_token_ids,
                generation_token_ids=[],
                generation_log_probs=[],
            )
        else:
            message = NeMoGymChatCompletionMessage(**message_kwargs)

        return NeMoGymChatCompletion(
            id="chtcmpl-context-length-exceeded",
            object="chat.completion",
            created=int(time()),
            model=self.config.model,
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="context_length_exceeded",
                    message=message,
                )
            ],
            context_length_exceeded=True,
        )

    @staticmethod
    def _get_tokenize_body_dict(body_dict: Dict[str, Any]) -> Dict[str, Any]:
        tokenize_body_dict = {}
        for key in ("model", "messages", "tools", "chat_template_kwargs", "mm_processor_kwargs"):
            if key in body_dict:
                tokenize_body_dict[key] = body_dict[key]
        return tokenize_body_dict

    async def _get_prompt_token_ids(
        self, client: NeMoGymAsyncOpenAI, body_dict: Dict[str, Any]
    ) -> List[int]:
        tokenize_response = await self._get_tokenize_response(client, body_dict)
        return tokenize_response["tokens"]

    async def _get_tokenize_response(
        self, client: NeMoGymAsyncOpenAI, body_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        tokenize_response = await client.create_tokenize(
            **self._get_tokenize_body_dict(body_dict)
        )
        return tokenize_response

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
        selected_base_url = getattr(request.state, "nemo_gym_vllm_base_url", None)
        selected_client_idx = getattr(request.state, "nemo_gym_vllm_client_idx", None)
        response_metadata = body.metadata
        if selected_base_url is not None:
            response_metadata = dict(response_metadata or {})
            response_metadata["nemo_gym_vllm_base_url"] = str(selected_base_url)
            if selected_client_idx is not None:
                response_metadata["nemo_gym_vllm_client_idx"] = str(selected_client_idx)

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
            metadata=response_metadata,
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

        body_dict = {key: value for key, value in body_dict.items() if value is not None}
        body_dict["model"] = self.config.model

        chat_template_kwargs = {}
        if self.config.chat_template_kwargs:
            chat_template_kwargs = deepcopy(self.config.chat_template_kwargs)

        metadata = body_dict.get("metadata") or dict()
        if isinstance(metadata, BaseModel):
            metadata = metadata.model_dump(exclude_unset=True)
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")

        # Merge global config chat_template_kwargs with per-request overrides in metadata (e.g. per-sample reasoning on/off)
        chat_template_kwargs.update(
            _metadata_json_dict(
                metadata.get("chat_template_kwargs"),
                field_name="chat_template_kwargs",
            )
        )

        if chat_template_kwargs:
            body_dict["chat_template_kwargs"] = chat_template_kwargs

        # Merge global config extra_body with per-request overrides from metadata
        extra_body = {}
        if self.config.extra_body:
            extra_body = deepcopy(self.config.extra_body)

        extra_body.update(
            _metadata_json_dict(
                metadata.get("extra_body"),
                field_name="extra_body",
            )
        )

        if self.config.return_token_id_information:
            token_id_source = _vllm_token_id_source()
            body_dict |= dict(
                logprobs=True,
                return_tokens_as_token_ids=True,
            )
            debug_top_logprobs_k = int(
                os.environ.get("NEMO_GYM_VLLM_DEBUG_TOP_LOGPROBS_K", "0") or "0"
            )
            if debug_top_logprobs_k > 0:
                body_dict["top_logprobs"] = debug_top_logprobs_k
            return_native_token_ids = _env_flag(
                "NEMO_GYM_VLLM_RETURN_TOKEN_IDS",
                default=token_id_source not in _LOGPROB_TOKEN_ID_SOURCES,
            )
            if return_native_token_ids:
                # Native sampled token ids are the canonical sequence to score
                # with the policy. logprobs.content is still used for behavior
                # logprobs and can be selected explicitly for diagnostics.
                body_dict["return_token_ids"] = True

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
            body_dict = deepcopy(extra_body) | body_dict
            if _env_flag("NRL_DEBUG") or _env_flag("NEMO_GYM_VLLM_DEBUG_EXTRA_BODY"):
                extra_body_summary = _summarize_forwarded_extra_body(extra_body)
                LOGGER.info(
                    "[VLLM_EXTRA_BODY_FORWARD] keys=%s summary=%s",
                    sorted(str(key) for key in extra_body),
                    extra_body_summary,
                )
                print(
                    "[VLLM_EXTRA_BODY_FORWARD] "
                    f"keys={sorted(str(key) for key in extra_body)} "
                    f"summary={extra_body_summary}",
                    flush=True,
                )

        return body_dict

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)
        body_dict = self._preprocess_chat_completion_create_params(request, body_dict)

        client = self._resolve_client(request)
        prompt_token_ids: Optional[List[int]] = None
        vllm_max_model_len: Optional[int] = None

        should_tokenize_prompt = self.config.max_input_tokens is not None
        if should_tokenize_prompt:
            tokenize_response = await self._get_tokenize_response(client, body_dict)
            prompt_token_ids = tokenize_response["tokens"]
            if tokenize_response.get("max_model_len") is not None:
                vllm_max_model_len = int(tokenize_response["max_model_len"])

        max_input_tokens = self.config.max_input_tokens
        if vllm_max_model_len is not None:
            max_input_tokens = (
                vllm_max_model_len
                if max_input_tokens is None
                else min(max_input_tokens, vllm_max_model_len)
            )

        if max_input_tokens is not None and prompt_token_ids is not None:
            prompt_len = len(prompt_token_ids)
            if prompt_len >= max_input_tokens:
                return self._create_context_length_exceeded_chat_completion(
                    prompt_token_ids
                )

            remaining_budget = max_input_tokens - prompt_len
            requested_max_tokens = body_dict.get("max_tokens")
            body_dict["max_tokens"] = (
                remaining_budget
                if requested_max_tokens is None
                else min(requested_max_tokens, remaining_budget)
            )
            if requested_max_tokens != body_dict["max_tokens"]:
                LOGGER.info(
                    "Clamped vLLM max_tokens from %s to %s for prompt_len=%s max_input_tokens=%s",
                    requested_max_tokens,
                    body_dict["max_tokens"],
                    prompt_len,
                    max_input_tokens,
                )

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
                if prompt_token_ids is None:
                    try:
                        prompt_token_ids = await self._get_prompt_token_ids(client, body_dict)
                    except Exception:
                        prompt_token_ids = None

                return self._create_context_length_exceeded_chat_completion(prompt_token_ids)
            else:
                LOGGER.error(
                    "vLLM chat-completions request rejected: status=%s body=%s payload_keys=%s",
                    e.status,
                    result_content_str,
                    sorted(body_dict.keys()),
                )
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
            log_probs = (choice_dict.get("logprobs") or {}).get("content") or []
            generation_log_probs = [log_prob["logprob"] for log_prob in log_probs]

            def _token_id_from_logprob_token(token: Any) -> int:
                if isinstance(token, str) and token.startswith("token_id:"):
                    return int(token.removeprefix("token_id:"))
                raise ValueError(
                    "Cannot recover a token id from logprobs.content token "
                    f"{token!r}. Expected vLLM return_tokens_as_token_ids=True "
                    "format like 'token_id:151667'."
                )

            def _extract_top_logprobs_for_debug() -> tuple[List[List[Dict[str, Any]]], str]:
                if not int(
                    os.environ.get("NEMO_GYM_VLLM_DEBUG_TOP_LOGPROBS_K", "0") or "0"
                ):
                    return [], "disabled"
                per_token: List[List[Dict[str, Any]]] = []
                try:
                    for log_prob in log_probs:
                        entries: List[Dict[str, Any]] = []
                        for top_item in log_prob.get("top_logprobs") or []:
                            try:
                                token_id = _token_id_from_logprob_token(top_item.get("token"))
                            except Exception:
                                continue
                            entries.append(
                                {
                                    "token_id": int(token_id),
                                    "logprob": float(top_item["logprob"]),
                                }
                            )
                        per_token.append(entries)
                except Exception as exc:
                    LOGGER.warning("vLLM top-logprob diagnostic parse failed", exc_info=True)
                    return [], f"error:{type(exc).__name__}"
                return per_token, "ok"

            token_id_source = _vllm_token_id_source()
            use_logprob_token_ids = token_id_source in _LOGPROB_TOKEN_ID_SOURCES
            debug_generation_top_logprobs, debug_generation_top_logprobs_status = (
                _extract_top_logprobs_for_debug()
            )

            logprob_token_ids = None
            mismatch_positions: List[int] = []
            direct_generation_token_ids = choice_dict.get("token_ids")
            if direct_generation_token_ids is not None:
                generation_token_ids = [int(token_id) for token_id in direct_generation_token_ids]
                try:
                    logprob_token_ids = [
                        _token_id_from_logprob_token(log_prob["token"])
                        for log_prob in log_probs
                    ]
                except ValueError:
                    logprob_token_ids = None
                if logprob_token_ids is not None and logprob_token_ids != generation_token_ids:
                    mismatch_positions = [
                        idx
                        for idx, (direct_token_id, logprob_token_id) in enumerate(
                            zip(generation_token_ids, logprob_token_ids)
                        )
                        if direct_token_id != logprob_token_id
                    ]
                    LOGGER.warning(
                        "vLLM native token_ids differ from logprobs token strings "
                        "at %d/%d positions; token source=%s. First mismatches: %s",
                        len(mismatch_positions),
                        len(generation_token_ids),
                        token_id_source,
                        mismatch_positions[:10],
                    )
                if use_logprob_token_ids and logprob_token_ids is not None:
                    generation_token_ids = logprob_token_ids
            else:
                if not use_logprob_token_ids:
                    LOGGER.warning(
                        "vLLM response did not include native token_ids while "
                        "NEMO_GYM_VLLM_TOKEN_ID_SOURCE=%s; falling back to "
                        "logprobs.content token strings. Set "
                        "NEMO_GYM_VLLM_RETURN_TOKEN_IDS=1 or use a vLLM build "
                        "that supports return_token_ids.",
                        token_id_source,
                    )
                generation_token_ids = [
                    _token_id_from_logprob_token(log_prob["token"])
                    for log_prob in log_probs
                ]

            if len(generation_token_ids) != len(generation_log_probs):
                raise ValueError(
                    "vLLM returned mismatched generation token/logprob lengths: "
                    f"len(generation_token_ids)={len(generation_token_ids)}, "
                    f"len(generation_log_probs)={len(generation_log_probs)}"
                )

            native_prompt_token_ids = chat_completion_dict.get("prompt_token_ids")
            if native_prompt_token_ids is not None:
                prompt_token_ids = [int(token_id) for token_id in native_prompt_token_ids]
            elif prompt_token_ids is None:
                prompt_token_ids = await self._get_prompt_token_ids(client, body_dict)

            debug_prefill_logprob_info: Dict[str, Any] = {}
            prefill_log_probs_for_generation: Optional[List[float]] = None
            refit_generation_logprobs = _env_flag("NEMO_GYM_VLLM_REFIT_GENERATION_LOGPROBS", default=False)
            debug_prefill_logprobs = _env_flag("NEMO_GYM_VLLM_DEBUG_PREFILL_LOGPROBS", default=False)
            if debug_prefill_logprobs or refit_generation_logprobs:
                debug_status = "skipped_empty_generation"
                if prompt_token_ids and generation_token_ids:
                    sequence_token_ids = prompt_token_ids + generation_token_ids
                    max_context = vllm_max_model_len or max_input_tokens
                    if max_context is not None and len(sequence_token_ids) >= int(max_context):
                        debug_status = (
                            f"skipped_context_full:{len(sequence_token_ids)}>={int(max_context)}"
                        )
                    else:
                        rescore_body_dict = deepcopy(body_dict)
                        rescore_body_dict["required_prompt_token_ids"] = sequence_token_ids
                        rescore_body_dict["prompt_logprobs"] = 0
                        rescore_body_dict["max_tokens"] = 1
                        rescore_body_dict.pop("max_completion_tokens", None)
                        rescore_body_dict["logprobs"] = False
                        rescore_body_dict["top_logprobs"] = 0
                        rescore_body_dict["return_token_ids"] = False
                        try:
                            rescore_dict = await client.create_chat_completion(
                                **rescore_body_dict
                            )
                            prefill_log_probs, debug_status = (
                                _extract_prompt_logprobs_for_tokens(
                                    rescore_dict.get("prompt_logprobs"),
                                    generation_token_ids,
                                    start_idx=len(prompt_token_ids),
                                )
                            )
                            if debug_status == "ok":
                                prefill_log_probs_for_generation = prefill_log_probs
                                diffs = [
                                    abs(float(decode_lp) - float(prefill_lp))
                                    for decode_lp, prefill_lp in zip(
                                        generation_log_probs, prefill_log_probs
                                    )
                                ]
                                debug_prefill_logprob_info.update(
                                    {
                                        "debug_vllm_prefill_generation_log_probs": prefill_log_probs,
                                        "debug_vllm_prefill_generation_logprob_count": len(prefill_log_probs),
                                        "debug_vllm_prefill_generation_logprob_error_mean": sum(diffs) / len(diffs)
                                        if diffs
                                        else 0.0,
                                        "debug_vllm_prefill_generation_logprob_error_max": max(diffs)
                                        if diffs
                                        else 0.0,
                                    }
                                )
                        except Exception as exc:
                            debug_status = f"error:{type(exc).__name__}"
                            LOGGER.warning("vLLM prefill-logprob diagnostic failed", exc_info=True)
                debug_prefill_logprob_info[
                    "debug_vllm_prefill_generation_logprob_status"
                ] = debug_status

            if refit_generation_logprobs:
                refit_strict = _env_flag("NEMO_GYM_VLLM_REFIT_GENERATION_LOGPROBS_STRICT", default=True)
                if not generation_log_probs:
                    debug_prefill_logprob_info["debug_vllm_generation_logprob_source"] = "decode_empty_generation"
                elif (
                    prefill_log_probs_for_generation is not None
                    and len(prefill_log_probs_for_generation) == len(generation_log_probs)
                ):
                    debug_prefill_logprob_info[
                        "debug_vllm_decode_generation_log_probs"
                    ] = generation_log_probs
                    generation_log_probs = [float(lp) for lp in prefill_log_probs_for_generation]
                    debug_prefill_logprob_info["debug_vllm_generation_logprob_source"] = "prefill_refit"
                else:
                    source = f"decode_refit_unavailable:{debug_prefill_logprob_info.get('debug_vllm_prefill_generation_logprob_status', 'unknown')}"
                    debug_prefill_logprob_info["debug_vllm_generation_logprob_source"] = source
                    if refit_strict:
                        raise RuntimeError(
                            "NEMO_GYM_VLLM_REFIT_GENERATION_LOGPROBS=1 but vLLM "
                            f"prefill/rescore logprobs were unavailable: {source}"
                        )

            message_dict = choice_dict["message"]
            message_dict.update(
                dict(
                    prompt_token_ids=prompt_token_ids,
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                    generation_token_id_source=token_id_source,
                    native_generation_token_ids_count=len(direct_generation_token_ids)
                    if direct_generation_token_ids is not None
                    else None,
                    logprob_generation_token_ids_count=len(logprob_token_ids)
                    if logprob_token_ids is not None
                    else None,
                    native_logprob_token_id_mismatch_count=len(mismatch_positions)
                    if direct_generation_token_ids is not None and logprob_token_ids is not None
                    else None,
                    native_logprob_token_id_first_mismatches=mismatch_positions[:10]
                    if mismatch_positions
                    else [],
                    finish_reason=choice_dict.get("finish_reason"),
                    debug_vllm_generation_top_logprobs=debug_generation_top_logprobs,
                    debug_vllm_generation_top_logprobs_status=debug_generation_top_logprobs_status,
                    **debug_prefill_logprob_info,
                )
            )

            # Clean the duplicated information
            choice_dict.pop("logprobs", None)
            chat_completion_dict.pop("prompt_token_ids", None)
            choice_dict.pop("token_ids", None)

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
            self._session_id_to_client_idx[session_id] = client_idx
        client = self._session_id_to_client[session_id]
        client_idx = self._session_id_to_client_idx.get(session_id)
        request.state.nemo_gym_vllm_base_url = client.base_url
        request.state.nemo_gym_vllm_client_idx = client_idx

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
                    case "image_url":
                        image_url = part_param.get("image_url", "")
                        if isinstance(image_url, dict):
                            image_url = image_url.get("url", "")
                        detail = part_param.get("detail", "auto")
                        converted_parts.append(
                            {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
                        )
                    case "input_video":
                        video_url = part_param.get("video_url", part_param.get("video", ""))
                        if isinstance(video_url, dict):
                            video_url = video_url.get("url", "")
                        converted_parts.append(
                            {"type": "video_url", "video_url": {"url": video_url}}
                        )
                    case "video_url":
                        video_url = part_param.get("video_url", "")
                        if isinstance(video_url, dict):
                            video_url = video_url.get("url", "")
                        converted_parts.append(
                            {"type": "video_url", "video_url": {"url": video_url}}
                        )
                    case "input_audio":
                        if "input_audio" in part_param:
                            converted_parts.append(
                                {"type": "input_audio", "input_audio": part_param["input_audio"]}
                            )
                        else:
                            audio_url = part_param.get("audio_url", part_param.get("audio", ""))
                            if isinstance(audio_url, dict):
                                audio_url = audio_url.get("url", "")
                            converted_parts.append(
                                {"type": "audio_url", "audio_url": {"url": audio_url}}
                            )
                    case "audio_url":
                        audio_url = part_param.get("audio_url", "")
                        if isinstance(audio_url, dict):
                            audio_url = audio_url.get("url", "")
                        converted_parts.append(
                            {"type": "audio_url", "audio_url": {"url": audio_url}}
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
            token_info_keys = (
                "prompt_token_ids",
                "generation_token_ids",
                "generation_log_probs",
                "generation_token_id_source",
                "native_generation_token_ids_count",
                "logprob_generation_token_ids_count",
                "native_logprob_token_id_mismatch_count",
                "native_logprob_token_id_first_mismatches",
                "finish_reason",
                "debug_vllm_prefill_generation_log_probs",
                "debug_vllm_prefill_generation_logprob_count",
                "debug_vllm_prefill_generation_logprob_error_mean",
                "debug_vllm_prefill_generation_logprob_error_max",
                "debug_vllm_prefill_generation_logprob_status",
                "debug_vllm_decode_generation_log_probs",
                "debug_vllm_generation_logprob_source",
                "debug_vllm_generation_top_logprobs",
                "debug_vllm_generation_top_logprobs_status",
            )
            response_output[-1] = train_cls(
                **last_response_output_item.model_dump(),
                **{
                    key: message_dict[key]
                    for key in token_info_keys
                    if key in message_dict
                },
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
                # vLLM may return None content
                if message["content"] is None:
                    message["content"] = ""
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
        if (
            getattr(item, "role", None) == "assistant"
            or getattr(item, "type", None)
            in {
                "reasoning",
                "reasoning_item",
            }
            or getattr(item, "type", None) in ("function_call",)
        ):
            break

    return items[:i], items[i:]


if __name__ == "__main__":
    VLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VLLMModel.run_webserver()  # noqa: F401
