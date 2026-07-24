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
"""Shared Responses API ↔ Chat Completions converter.

This module contains the translation logic between OpenAI's Responses API format
and the Chat Completions API format. It is used by model servers that need to
convert between the two formats (e.g. vllm_model, inference_provider).
"""

import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from nemo_gym.openai_utils import (
    RESPONSES_TO_TRAIN,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionAssistantMessageForTrainingParam,
    NeMoGymChatCompletionAssistantMessageParam,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionDeveloperMessageParam,
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
    NeMoGymFunctionToolParam,
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
    Reasoning,
    TokenIDLogProbMixin,
)


def _message_content_to_text(content: Any) -> str:
    """Plain text of a chat message content (a string, or a list of text parts)."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


class ResponsesConverterState(BaseModel):
    return_token_id_information: bool

    messages: List[NeMoGymChatCompletionMessageParam] = Field(default_factory=list)

    content_buffer: str = ""
    tool_calls_buffer: List[NeMoGymChatCompletionMessageToolCallParam] = Field(default_factory=list)

    token_information: Optional[TokenIDLogProbMixin] = None

    def flush_assistant(self) -> None:
        if not (self.content_buffer or self.tool_calls_buffer):
            return

        shared_params = dict(
            content=self.content_buffer or None,
            role="assistant",
            tool_calls=self.tool_calls_buffer,
        )

        if self.return_token_id_information and self.token_information:
            message = NeMoGymChatCompletionAssistantMessageForTrainingParam(
                **shared_params,
                **self.token_information.model_dump(exclude_none=True),
            )
        else:
            message = NeMoGymChatCompletionAssistantMessageParam(**shared_params)

        self.messages.append(message)

        self.content_buffer = ""
        self.tool_calls_buffer = []


class ResponsesConverter(BaseModel):
    """Converts between OpenAI Responses API and Chat Completions API formats."""

    return_token_id_information: bool
    uses_reasoning_parser: bool = True

    THINK_TAG_PATTERN: ClassVar = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    @staticmethod
    def _wrap_reasoning_in_think_tags(texts: List[str]) -> str:
        return "".join(f"<think>{t}</think>" for t in texts if t)

    @classmethod
    def _parse_think_tags(cls, content: str) -> Tuple[List[str], str]:
        matches = cls.THINK_TAG_PATTERN.findall(content)
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

        state = ResponsesConverterState(return_token_id_information=self.return_token_id_information)

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
                    routed_experts=m.get("routed_experts"),
                )

        state.flush_assistant()

        # The Responses API inserts `instructions` as a system message at the start of the model's
        # context. Chat Completions has no such parameter, so map it explicitly — otherwise it is
        # silently dropped when the remaining params are passed through (extra fields are ignored).
        # The leading run of system/developer messages is folded into the same single system
        # message: chat backends commonly admit only one system message, at position 0 (harnesses
        # like the Codex CLI send instructions plus a leading developer message).
        instructions = responses_create_params.pop("instructions", None)
        if instructions:
            leading_parts = [instructions]
            while state.messages and state.messages[0]["role"] in ("system", "developer"):
                leading_parts.append(_message_content_to_text(state.messages.pop(0)["content"]))
            state.messages.insert(
                0,
                NeMoGymChatCompletionSystemMessageParam(content="\n\n".join(leading_parts), role="system"),
            )

        model = responses_create_params.pop("model", None)
        if model is not None:
            responses_create_params["model"] = model

        max_output_tokens = responses_create_params.pop("max_output_tokens", None)
        if max_output_tokens is not None:
            responses_create_params["max_tokens"] = max_output_tokens

        tools = responses_create_params.pop("tools", None)
        if tools:
            responses_create_params["tools"] = []
            for tool_dict in tools:
                tool_dict = tool_dict.copy()
                tool_dict.pop("type", None)
                tool_dict.pop("strict", None)
                responses_create_params["tools"].append(
                    NeMoGymChatCompletionToolParam(type="function", function=NeMoGymFunctionDefinition(**tool_dict))
                )
        else:
            if responses_create_params.get("tool_choice") == "required":
                raise ValueError("tool_choice='required' requires at least one tool")

            responses_create_params.pop("tool_choice", None)
            responses_create_params.pop("parallel_tool_calls", None)

        chat_completion_create_params = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=state.messages,
            **responses_create_params,
        )

        return chat_completion_create_params

    def _format_function_call_output(
        self,
        m: dict,
        state: ResponsesConverterState,
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
        state: ResponsesConverterState,
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
        state: ResponsesConverterState,
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
        state: ResponsesConverterState,
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
    # Chat Completion create params to Response create params
    # =======================================================

    def _chat_completion_to_responses_tools(
        self, chat_completions_tools: List[NeMoGymChatCompletionToolParam]
    ) -> List[NeMoGymFunctionToolParam]:
        return [tool["function"] | {"type": "function"} for tool in chat_completions_tools]

    def chat_completion_to_responses_create_params(
        self,
        chat_completion_create_params: NeMoGymChatCompletionCreateParamsNonStreaming,
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming(
            input=self.chat_completions_messages_to_responses_items(chat_completion_create_params.messages),
            max_output_tokens=chat_completion_create_params.max_completion_tokens,
            metadata=chat_completion_create_params.metadata,
            model=chat_completion_create_params.model,
            parallel_tool_calls=chat_completion_create_params.parallel_tool_calls,
            reasoning=Reasoning(reasoning_effort=chat_completion_create_params.reasoning_effort),
            service_tier=chat_completion_create_params.service_tier,
            store=chat_completion_create_params.store,
            temperature=chat_completion_create_params.temperature,
            tool_choice=chat_completion_create_params.tool_choice
            if chat_completion_create_params.tool_choice is not None
            else "auto",
            tools=self._chat_completion_to_responses_tools(chat_completion_create_params.tools),
            top_logprobs=chat_completion_create_params.top_logprobs,
            top_p=chat_completion_create_params.top_p,
            user=chat_completion_create_params.user,
            stream=chat_completion_create_params.stream,
        )

    # =======================================================
    # Chat Completion to Response
    # =======================================================

    def postprocess_chat_response(self, choice: NeMoGymChoice) -> List[NeMoGymResponseOutputItem]:
        return self.postprocess_assistant_message_dict(choice.message.model_dump(exclude_none=True))

    def postprocess_assistant_message_dict(self, message_dict: Dict[str, Any]) -> List[NeMoGymResponseOutputItem]:
        response_output = []

        content = message_dict.get("content") or ""
        if self.uses_reasoning_parser:
            reasoning_matches, content = self._extract_reasoning_from_content(content)
        else:
            reasoning_matches = []
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

        if self.return_token_id_information and "prompt_token_ids" in message_dict:
            last_response_output_item = response_output[-1]
            train_cls = RESPONSES_TO_TRAIN[last_response_output_item.__class__]
            extra_training_fields = {}
            if "routed_experts" in message_dict and message_dict["routed_experts"] is not None:
                extra_training_fields["routed_experts"] = message_dict["routed_experts"]
            response_output[-1] = train_cls(
                **last_response_output_item.model_dump(),
                prompt_token_ids=message_dict["prompt_token_ids"],
                generation_token_ids=message_dict["generation_token_ids"],
                generation_log_probs=message_dict["generation_log_probs"],
                **extra_training_fields,
            )

        return response_output

    def _extract_reasoning_from_content(self, content: str) -> Tuple[List[str], str]:
        return self._parse_think_tags(content)

    def chat_completions_messages_to_responses_items(
        self, messages: List[Dict[str, Any]]
    ) -> List[NeMoGymResponseOutputItem]:
        output_items = []

        for message in messages:
            role = message["role"]
            if role in ("user", "system", "developer"):
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

    def chat_completion_to_response(
        self,
        responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
        chat_completion: NeMoGymChatCompletion,
    ) -> NeMoGymResponse:
        choice = chat_completion.choices[0]

        response_output = self.postprocess_chat_response(choice)
        response_output_dicts = [item.model_dump() for item in response_output]

        usage = None
        if chat_completion.usage:
            usage = NeMoGymResponseUsage(
                input_tokens=chat_completion.usage.prompt_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=chat_completion.usage.completion_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=chat_completion.usage.prompt_tokens + chat_completion.usage.completion_tokens,
            )

        incomplete_details = None
        if choice.finish_reason == "length":
            incomplete_details = {"reason": "max_output_tokens"}
        elif choice.finish_reason == "content_filter":
            incomplete_details = {"reason": "content_filter"}

        # Chat Completion -> Response
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=chat_completion.created,
            model=responses_create_params.model,
            object="response",
            output=response_output_dicts,
            tool_choice=responses_create_params.tool_choice
            if responses_create_params.tool_choice is not None
            else "auto",
            parallel_tool_calls=responses_create_params.parallel_tool_calls,
            tools=responses_create_params.tools,
            temperature=responses_create_params.temperature,
            top_p=responses_create_params.top_p,
            background=responses_create_params.background,
            max_output_tokens=responses_create_params.max_output_tokens,
            max_tool_calls=responses_create_params.max_tool_calls,
            previous_response_id=responses_create_params.previous_response_id,
            prompt=responses_create_params.prompt,
            reasoning=responses_create_params.reasoning,
            service_tier=responses_create_params.service_tier,
            text=responses_create_params.text,
            top_logprobs=responses_create_params.top_logprobs,
            truncation=responses_create_params.truncation,
            metadata=responses_create_params.metadata,
            instructions=responses_create_params.instructions,
            user=responses_create_params.user,
            incomplete_details=incomplete_details,
            usage=usage,
        )


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


# Backward-compatible aliases
VLLMConverter = ResponsesConverter
VLLMConverterResponsesToChatCompletionsState = ResponsesConverterState
