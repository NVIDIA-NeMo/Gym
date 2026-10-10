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

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple
from uuid import uuid4

from openai.types.responses.response_create_params import ToolParam
from pydantic import BaseModel, Field

from nemo_gym.openai_utils import (
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
    NeMoGymChatCompletionToolUnionParam,
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
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
    Reasoning,
    TokenIDLogProbMixin,
    _validate_atomic_token_metadata,
    training_variant_of,
)


logger = logging.getLogger(__name__)

# Chat tool messages carry text only; with `drop_unrepresentable_request_fields` an image a tool
# returned becomes this text instead of failing the conversion. The wording is neutral on purpose:
# it replaces part of the prompt, and it must not teach the model anything about its own inputs.
_IMAGE_OUTPUT_PLACEHOLDER = "[image omitted from tool output]"

# Responses reasoning fields that only select what the provider reports back (a reasoning
# summary, under its current and its deprecated name), so a Chat Completions backend answers the
# request faithfully without them. With `drop_unrepresentable_request_fields` these are removed
# from the request rather than refused. `reasoning.context` is not among them: it tells the
# provider which earlier reasoning items to render back to the model, which the conversion cannot
# honor (it renders every item and leaves the rest to the chat template), so it stays refused.
_REPORTING_ONLY_REASONING_FIELDS = ("summary", "generate_summary")


def _message_content_to_text(content: Any) -> str:
    """Plain text of a chat message content (a string, or a list of text parts)."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def _optional_token_count(value: Any) -> Optional[int]:
    """Return a provider-reported token count without coercing missing/invalid values to zero."""
    return value if type(value) is int and value >= 0 else None


def _usage_detail(usage: Any, detail_group: str, detail_name: str, *top_level_aliases: str) -> Optional[int]:
    """Read one canonical nested token detail, then named provider aliases."""
    details = usage.get(detail_group) if isinstance(usage, dict) else getattr(usage, detail_group, None)
    value = details.get(detail_name) if isinstance(details, dict) else getattr(details, detail_name, None)
    value = _optional_token_count(value)
    if value is not None:
        return value
    for name in top_level_aliases:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        value = _optional_token_count(value)
        if value is not None:
            return value
    return None


class ResponsesConverterState(BaseModel):
    return_token_id_information: bool

    messages: List[NeMoGymChatCompletionMessageParam] = Field(default_factory=list)

    content_buffer: str = ""
    refusal_buffer: str = ""
    tool_calls_buffer: List[NeMoGymChatCompletionMessageToolCallParam] = Field(default_factory=list)
    assistant_item_buffered: bool = False

    token_information: Optional[TokenIDLogProbMixin] = None

    def flush_assistant(self) -> None:
        if not (self.assistant_item_buffered or self.content_buffer or self.refusal_buffer or self.tool_calls_buffer):
            self.token_information = None
            return

        shared_params = dict(
            content=self.content_buffer or None,
            role="assistant",
        )
        # Omit rather than send `tool_calls: []` — OpenAI rejects empty arrays.
        if self.tool_calls_buffer:
            shared_params["tool_calls"] = self.tool_calls_buffer
        if self.refusal_buffer:
            shared_params["refusal"] = self.refusal_buffer

        if self.return_token_id_information and self.token_information is not None:
            message = NeMoGymChatCompletionAssistantMessageForTrainingParam(
                **shared_params,
                **self.token_information.model_dump(exclude_none=True),
            )
        else:
            message = NeMoGymChatCompletionAssistantMessageParam(**shared_params)

        self.messages.append(message)

        self.content_buffer = ""
        self.refusal_buffer = ""
        self.tool_calls_buffer = []
        self.assistant_item_buffered = False
        self.token_information = None


def _token_information_from_mapping(value: Dict[str, Any]) -> Optional[TokenIDLogProbMixin]:
    """Return validated token metadata when the mapping contains it."""
    _validate_atomic_token_metadata(value)
    if "prompt_token_ids" not in value:
        return None
    return TokenIDLogProbMixin.model_validate(value)


def _chat_logprobs_to_responses(logprobs: Any) -> Optional[List[Dict[str, Any]]]:
    """Chat choice logprobs in the shape ``ResponseOutputText.logprobs`` expects.

    Same field names on both sides; the Responses models require ``bytes`` to be
    iterable where chat allows null.
    """
    content = getattr(logprobs, "content", None)
    if not content:
        return None

    def _one(entry: Any) -> Dict[str, Any]:
        out = entry.model_dump()
        out["bytes"] = out.get("bytes") or []
        out["top_logprobs"] = [{**alt, "bytes": alt.get("bytes") or []} for alt in (out.get("top_logprobs") or [])]
        return out

    return [_one(entry) for entry in content]


def _align_logprobs_to_text(
    logprobs: Optional[List[Dict[str, Any]]], original: str, removed: List[Tuple[int, int]]
) -> Optional[List[Dict[str, Any]]]:
    """Keep the token entries that survive removing the ``removed`` character spans.

    Reasoning extraction shortens the output text; its logprobs must shrink with it or
    they describe different text. The kept positions come from the removal itself, never
    from searching for the answer text, which may also occur inside the reasoning. If the
    tokens do not reconstruct ``original`` the alignment is unknowable and the logprobs
    are dropped rather than misattributed.
    """
    if not logprobs or not removed:
        return logprobs
    if "".join(e.get("token") or "" for e in logprobs) != original:
        return None
    kept, offset = [], 0
    for entry in logprobs:
        token_end = offset + len(entry.get("token") or "")
        inside_removed = any(start <= offset and token_end <= end for start, end in removed)
        if token_end > offset and not inside_removed:
            kept.append(entry)
        offset = token_end
    return kept or None


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
        *,
        drop_unrepresentable_request_fields: bool = False,
    ) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        """Convert Responses create params into Chat Completions create params.

        A Responses field with no Chat Completions representation raises ``NotImplementedError``,
        so the caller can route the request to a server that serves Responses natively. With
        ``drop_unrepresentable_request_fields`` the reporting-only fields are removed from the
        request instead and the conversion proceeds: a non-empty ``include`` (every value; an
        empty one is already treated as absent), ``reasoning.summary`` and
        ``reasoning.generate_summary``, and a message ``phase``. An image part in a tool's output
        becomes the text placeholder ``_IMAGE_OUTPUT_PLACEHOLDER``, which does replace that part
        of the prompt. Every other unrepresentable field, ``reasoning.context`` included, is still
        refused.
        """
        responses_create_params = responses_create_params.model_dump(exclude_none=True, exclude_unset=True)
        if responses_create_params.get("include") == []:
            del responses_create_params["include"]

        # Codex serializes include=[] even when no additional fields are requested.
        # Only the empty list is equivalent to omission; nonempty includes still require Responses.
        if responses_create_params.get("include") == []:
            responses_create_params.pop("include")

        # `include` selects extra fields on the response object (for example
        # `reasoning.encrypted_content`); it does not change what the model generates.
        if drop_unrepresentable_request_fields:
            dropped_include = responses_create_params.pop("include", None)
            if dropped_include:
                logger.debug(
                    "Dropping Responses `include` %r: a chat backend returns no such fields.", dropped_include
                )

        unsupported_fields = sorted(
            {
                "background",
                "context_management",
                "conversation",
                "include",
                "max_tool_calls",
                "previous_response_id",
                "prompt",
                "truncation",
            }
            & responses_create_params.keys()
        )
        if unsupported_fields:
            raise NotImplementedError(
                f"Responses request field(s) {unsupported_fields} have no Chat Completions "
                "representation, so this request cannot be downconverted. Route it to a model "
                "server that passes Responses through."
            )

        state = ResponsesConverterState(return_token_id_information=self.return_token_id_information)

        response_input = responses_create_params.pop("input")
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
            input_messages = response_input

        for m in input_messages:
            if not m.get("type") and m.get("role"):
                m["type"] = "message"

            match m["type"]:
                case "message":
                    self._format_message(
                        m, state, drop_unrepresentable_request_fields=drop_unrepresentable_request_fields
                    )
                case "reasoning":
                    self._format_reasoning(m, state)
                case "function_call":
                    self._format_function_call(m, state)
                case "function_call_output":
                    self._format_function_call_output(
                        m, state, drop_unrepresentable_request_fields=drop_unrepresentable_request_fields
                    )
                case _:
                    # This fires mid-rollout, on whichever model server downconverts.
                    # Most types that reach it are Responses-only by design.
                    # They are not an omission here.
                    # The item itself is left out of the message.
                    # A compaction record carries an opaque blob, and the tag is what identifies the problem.
                    raise NotImplementedError(
                        f"{m['type']!r} has no Chat Completions representation, so this rollout "
                        f"cannot be downconverted. Route it to a model server that passes "
                        f"Responses through (nemo_gym serves these unconverted), or add a "
                        f"`case {m['type']!r}` here if the type does have a chat equivalent."
                    )

            if self.return_token_id_information:
                token_information = _token_information_from_mapping(m)
                if token_information is not None:
                    state.token_information = token_information

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

        reasoning = responses_create_params.pop("reasoning", None)
        if reasoning is not None:
            # A chat completion carries only the effort. The summary fields select what the provider
            # reports back and are dropped when configured; `context` is an instruction about the
            # prompt and is refused like every other unrepresentable field.
            representable_fields = {"effort"}
            if drop_unrepresentable_request_fields:
                representable_fields.update(_REPORTING_ONLY_REASONING_FIELDS)
                dropped_reasoning_fields = [
                    field for field in _REPORTING_ONLY_REASONING_FIELDS if reasoning.get(field) is not None
                ]
                if dropped_reasoning_fields:
                    logger.debug(
                        "Dropping Responses reasoning field(s) %s: chat completions carry only the effort.",
                        dropped_reasoning_fields,
                    )
            unsupported_reasoning_fields = sorted(
                field for field, value in reasoning.items() if field not in representable_fields and value is not None
            )
            if unsupported_reasoning_fields:
                raise NotImplementedError(
                    "Responses reasoning field(s) "
                    f"{unsupported_reasoning_fields} have no Chat Completions representation."
                )
            if reasoning.get("effort") is not None:
                responses_create_params["reasoning_effort"] = reasoning["effort"]

        text = responses_create_params.pop("text", None)
        if text is not None:
            text_format = text.get("format")
            if text_format is not None:
                format_type = text_format.get("type")
                if format_type == "json_schema":
                    json_schema = {"name": text_format["name"], "schema": text_format["schema"]}
                    for key in ("strict", "description"):
                        if text_format.get(key) is not None:
                            json_schema[key] = text_format[key]
                    responses_create_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": json_schema,
                    }
                elif format_type == "json_object":
                    responses_create_params["response_format"] = {"type": "json_object"}
                elif format_type != "text":
                    raise NotImplementedError(f"Unsupported Responses text format type {format_type!r}.")
            if text.get("verbosity") is not None:
                responses_create_params["verbosity"] = text["verbosity"]

        tool_choice = self._responses_to_chat_tool_choice(responses_create_params.pop("tool_choice", None))
        tools = responses_create_params.pop("tools", None)
        if tools:
            responses_create_params["tools"] = []
            for tool_dict in tools:
                tool_type = tool_dict.get("type")
                if tool_type not in {"function", "custom"}:
                    raise NotImplementedError(
                        f"Responses tool type {tool_type!r} has no implemented Chat Completions conversion."
                    )
                if tool_dict.get("defer_loading"):
                    raise NotImplementedError(
                        f"Responses {tool_type} tools with defer_loading enabled "
                        "have no Chat Completions representation."
                    )
                tool_dict = tool_dict.copy()
                tool_dict.pop("type", None)
                tool_dict.pop("defer_loading", None)
                if tool_type == "function":
                    tool_dict = {key: value for key, value in tool_dict.items() if value is not None}
                    converted_tool = NeMoGymChatCompletionToolParam(
                        type="function",
                        function=NeMoGymFunctionDefinition(**tool_dict),
                    )
                else:
                    converted_tool = {
                        "type": "custom",
                        "custom": self._responses_custom_tool_to_chat(tool_dict),
                    }
                responses_create_params["tools"].append(converted_tool)
            if tool_choice is not None:
                responses_create_params["tool_choice"] = tool_choice
        else:
            if tool_choice not in (None, "auto", "none"):
                raise ValueError(f"tool_choice={tool_choice!r} requires at least one tool")

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
        *,
        drop_unrepresentable_request_fields: bool = False,
    ) -> None:
        state.flush_assistant()

        assert "call_id" in m
        output = m["output"]
        if isinstance(output, str):
            content = output
        elif isinstance(output, list):
            # Chat tool messages carry text only, so every other part type fails the conversion.
            # An image part is the one exception `drop_unrepresentable_request_fields` covers: the
            # text placeholder keeps the position of the image in the tool's output.
            representable_types = {"input_text"}
            if drop_unrepresentable_request_fields:
                representable_types.add("input_image")
            unsupported_types = sorted(
                {part.get("type", "<missing>") for part in output if part.get("type") not in representable_types}
            )
            if unsupported_types:
                raise NotImplementedError(
                    "Cannot convert Responses function_call_output "
                    f"for call {m['call_id']!r} to a Chat Completions tool message: "
                    "Chat tool messages cannot represent content part type(s) "
                    f"{', '.join(repr(part_type) for part_type in unsupported_types)}"
                )
            if any(part["type"] == "input_image" for part in output):
                # Debug, like the other drops: the conversion re-converts the whole transcript on
                # every later turn, so one returned image would otherwise warn on every turn after it.
                logger.debug(
                    "Rendering the image output of call %r as a text placeholder: chat tool messages carry text only.",
                    m["call_id"],
                )
            content = [
                {"type": "text", "text": part["text"]}
                if part["type"] == "input_text"
                else {"type": "text", "text": _IMAGE_OUTPUT_PLACEHOLDER}
                for part in output
            ]
        else:  # pragma: no cover - guarded by NeMoGymFunctionCallOutput validation
            raise TypeError(
                "Responses function_call_output must be a string or a list of structured content parts, "
                f"got {type(output).__name__}"
            )

        converted = NeMoGymChatCompletionToolMessageParam(
            content=content,
            role="tool",
            tool_call_id=m["call_id"],
        )
        state.messages.append(converted)

    def _format_message(
        self,
        m: dict,
        state: ResponsesConverterState,
        *,
        drop_unrepresentable_request_fields: bool = False,
    ) -> None:
        # Tool-call-only assistant turns may omit `content` entirely, not just null it.
        content = m.get("content")
        # A message `phase` labels the message rather than changing its content, and Chat
        # Completions messages have no field for it.
        if m.get("phase") is not None:
            if not drop_unrepresentable_request_fields:
                raise NotImplementedError("Responses message phase has no Chat Completions representation.")
            logger.debug("Dropping Responses message phase %r: chat messages carry no such label.", m["phase"])

        if isinstance(content, list) and m["role"] != "assistant":
            converted_parts = []
            for part_param in content:
                match part_param["type"]:
                    case "input_text":
                        converted_parts.append({"type": "text", "text": part_param["text"]})
                    case "input_image":
                        image_url = part_param.get("image_url")
                        if image_url is None:
                            raise NotImplementedError(
                                "Responses input images referenced by file_id have no Chat Completions representation."
                            )
                        if isinstance(image_url, dict):
                            image_url = image_url.get("url", "")
                        if not image_url:
                            raise ValueError(f"{part_param['type']} requires a non-empty image_url")
                        detail = part_param.get("detail", "auto")
                        if detail == "original":
                            raise NotImplementedError(
                                "Responses input image detail 'original' has no Chat Completions representation."
                            )
                        converted_parts.append(
                            {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
                        )
                    case "input_video":
                        source_key = "video_url" if "video_url" in part_param else "video"
                        video_url = part_param[source_key]
                        if isinstance(video_url, dict):
                            video_url = video_url.get("url", "")
                        if not video_url:
                            raise ValueError(f"input_video.{source_key} requires a non-empty URL")
                        converted_parts.append({"type": "video_url", "video_url": {"url": video_url}})
                    case _:
                        raise NotImplementedError(f"Unsupported part param type: {part_param['type']}")
            content = converted_parts
            m["content"] = content

        match m["role"]:
            case "assistant":
                state.assistant_item_buffered = True
                final_content = ""
                if content is None:
                    # Tool-call only turns have "None" according to the official API spec.
                    pass
                elif isinstance(content, list):
                    text_parts: list[str] = []
                    refusal_parts: list[str] = []
                    for part in content:
                        text = part.get("text")
                        refusal = part.get("refusal")
                        if isinstance(text, str):
                            text_parts.append(text)
                        if isinstance(refusal, str):
                            refusal_parts.append(refusal)
                        if not isinstance(text, str) and not isinstance(refusal, str):
                            raise NotImplementedError(f"Unsupported assistant content part: {part!r}")
                    final_content += "".join(text_parts)
                    state.refusal_buffer += "".join(refusal_parts)
                elif isinstance(content, str):
                    final_content += content
                else:
                    raise NotImplementedError(
                        f"Expected m['content'] to be str or list[dict], but got {type(content).__name__!r}: {content!r}"
                    )

                converted = []
                state.content_buffer += final_content
                # Chat-shaped turns carry tool calls inline, not as separate `function_call` items.
                for tool_call in m.get("tool_calls") or []:
                    state.tool_calls_buffer.append(
                        NeMoGymChatCompletionMessageToolCallParam(
                            id=tool_call["id"],
                            function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                                arguments=tool_call["function"]["arguments"],
                                name=tool_call["function"]["name"],
                            ),
                            type="function",
                        )
                    )
            case "tool":
                # Chat-shaped tool result — the `function_call_output` equivalent.
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionToolMessageParam(
                        content=content,
                        role="tool",
                        tool_call_id=m["tool_call_id"],
                    )
                ]
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
        See: https://docs.nvidia.com/nemo/gym/main/infrastructure/engineering-notes/responses-api-evolution
        for background on reasoning in the Responses API.
        """
        if m.get("encrypted_content") or m.get("content"):
            raise NotImplementedError(
                "Responses reasoning content/encrypted_content cannot be preserved by this Chat Completions "
                "conversion; route the request to a model server that passes Responses through."
            )
        state.assistant_item_buffered = True
        if "summary" in m and m["summary"]:
            texts = [s["text"] for s in m["summary"]]
            state.content_buffer += self._wrap_reasoning_in_think_tags(texts)

    def _format_function_call(
        self,
        m: dict,
        state: ResponsesConverterState,
    ) -> None:
        state.assistant_item_buffered = True
        assert "call_id" in m
        if m.get("namespace") is not None:
            raise NotImplementedError("Responses function call namespace has no Chat Completions representation.")
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

    @staticmethod
    def _tool_reference_to_responses(tool: dict) -> dict:
        tool_type = tool.get("type")
        if tool_type not in {"function", "custom"}:
            raise NotImplementedError(f"Chat tool reference type {tool_type!r} has no Responses representation.")
        return {"type": tool_type, **tool[tool_type]}

    @staticmethod
    def _tool_reference_to_chat(tool: dict) -> dict:
        tool_type = tool.get("type")
        if tool_type not in {"function", "custom"}:
            raise NotImplementedError(f"Responses tool reference type {tool_type!r} has no Chat representation.")
        return {"type": tool_type, tool_type: {key: value for key, value in tool.items() if key != "type"}}

    @staticmethod
    def _chat_custom_tool_to_responses(tool: dict) -> dict:
        converted = dict(tool)
        tool_format = converted.get("format")
        if tool_format is not None and tool_format["type"] == "grammar":
            converted["format"] = {"type": "grammar", **tool_format["grammar"]}
        return converted

    @staticmethod
    def _responses_custom_tool_to_chat(tool: dict) -> dict:
        converted = dict(tool)
        tool_format = converted.get("format")
        if tool_format is not None and tool_format["type"] == "grammar":
            converted["format"] = {
                "type": "grammar",
                "grammar": {key: value for key, value in tool_format.items() if key != "type"},
            }
        return converted

    def _chat_to_responses_tool_choice(self, tool_choice: Any) -> Any:
        if tool_choice is None or isinstance(tool_choice, str):
            return tool_choice

        tool_type = tool_choice.get("type")
        if tool_type in {"function", "custom"}:
            return self._tool_reference_to_responses(tool_choice)
        if tool_type == "allowed_tools":
            allowed_tools = tool_choice["allowed_tools"]
            return {
                "type": "allowed_tools",
                "mode": allowed_tools["mode"],
                "tools": [self._tool_reference_to_responses(tool) for tool in allowed_tools["tools"]],
            }
        raise NotImplementedError(f"Chat tool choice type {tool_type!r} has no Responses representation.")

    def _responses_to_chat_tool_choice(self, tool_choice: Any) -> Any:
        if tool_choice is None or isinstance(tool_choice, str):
            return tool_choice

        tool_type = tool_choice.get("type")
        if tool_type in {"function", "custom"}:
            return self._tool_reference_to_chat(tool_choice)
        if tool_type == "allowed_tools":
            return {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": tool_choice["mode"],
                    "tools": [self._tool_reference_to_chat(tool) for tool in tool_choice["tools"]],
                },
            }
        raise NotImplementedError(f"Responses tool choice type {tool_type!r} has no Chat representation.")

    def _chat_completion_to_responses_tools(
        self, chat_completions_tools: Optional[List[NeMoGymChatCompletionToolUnionParam]]
    ) -> List[ToolParam]:
        if chat_completions_tools is None:
            return []

        converted_tools = []
        for tool in chat_completions_tools:
            tool_type = tool["type"]
            tool_definition = dict(tool[tool_type])
            if tool_type == "function":
                tool_definition.setdefault("parameters", None)
                tool_definition.setdefault("strict", None)
            else:
                tool_definition = self._chat_custom_tool_to_responses(tool_definition)
            converted_tools.append({"type": tool_type, **tool_definition})
        return converted_tools

    def chat_completion_to_responses_create_params(
        self,
        chat_completion_create_params: NeMoGymChatCompletionCreateParamsNonStreaming,
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming(
            input=self.chat_completions_messages_to_responses_items(chat_completion_create_params.messages),
            max_output_tokens=chat_completion_create_params.max_completion_tokens,
            metadata=chat_completion_create_params.metadata,
            model=chat_completion_create_params.model,
            moderation=chat_completion_create_params.moderation,
            parallel_tool_calls=chat_completion_create_params.parallel_tool_calls,
            prompt_cache_key=chat_completion_create_params.prompt_cache_key,
            prompt_cache_retention=chat_completion_create_params.prompt_cache_retention,
            reasoning=Reasoning(effort=chat_completion_create_params.reasoning_effort)
            if chat_completion_create_params.reasoning_effort is not None
            else None,
            safety_identifier=chat_completion_create_params.safety_identifier,
            service_tier=chat_completion_create_params.service_tier,
            store=chat_completion_create_params.store,
            temperature=chat_completion_create_params.temperature,
            text={"verbosity": chat_completion_create_params.verbosity}
            if chat_completion_create_params.verbosity is not None
            else None,
            tool_choice=self._chat_to_responses_tool_choice(chat_completion_create_params.tool_choice)
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
        return self.postprocess_assistant_message_dict(
            choice.message.model_dump(exclude_none=True),
            logprobs=_chat_logprobs_to_responses(choice.logprobs),
        )

    def postprocess_assistant_message_dict(
        self, message_dict: Dict[str, Any], *, logprobs: Any = None
    ) -> List[NeMoGymResponseOutputItem]:
        response_output = []

        content = message_dict.get("content") or ""
        refusal = message_dict.get("refusal") or ""
        if self.uses_reasoning_parser:
            reasoning_matches, content = self._extract_reasoning_from_content(content)
            if reasoning_matches:
                original = message_dict.get("content") or ""
                removed = [m.span() for m in self.THINK_TAG_PATTERN.finditer(original)]
                logprobs = _align_logprobs_to_text(logprobs, original, removed)
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
        has_empty_output = not (response_output or tool_calls_raw or refusal)

        if content or refusal or has_empty_output:
            message_content = []
            if content or has_empty_output:
                message_content.append(
                    NeMoGymResponseOutputText(
                        type="output_text",
                        text=content,
                        annotations=[],
                        logprobs=logprobs,
                    )
                )
            if refusal:
                message_content.append(
                    NeMoGymResponseOutputRefusal(
                        type="refusal",
                        refusal=str(refusal),
                    )
                )
            response_output.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    role=message_dict.get("role"),
                    content=message_content,
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

        token_information = _token_information_from_mapping(message_dict) if self.return_token_id_information else None
        if token_information is not None:
            last_response_output_item = response_output[-1]
            train_cls = training_variant_of(last_response_output_item.__class__)
            response_output[-1] = train_cls(
                **last_response_output_item.model_dump(),
                **token_information.model_dump(exclude_none=True),
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
                content = message["content"]
                if isinstance(content, list):
                    content = [{"type": "input_text", "text": part["text"]} for part in content]
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=message["tool_call_id"],
                        output=content,
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
        *,
        preserve_envelope_id: bool = False,
    ) -> NeMoGymResponse:
        choice = chat_completion.choices[0]

        response_output = self.postprocess_chat_response(choice)
        response_output_dicts = [item.model_dump() for item in response_output]

        usage = None
        if chat_completion.usage:
            cached_tokens = _usage_detail(
                chat_completion.usage,
                "prompt_tokens_details",
                "cached_tokens",
                "cached_input_tokens",
                "cache_read_input_tokens",
            )
            reasoning_tokens = _usage_detail(
                chat_completion.usage,
                "completion_tokens_details",
                "reasoning_tokens",
                "reasoning_output_tokens",
            )
            usage = NeMoGymResponseUsage(
                input_tokens=chat_completion.usage.prompt_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cached_tokens=cached_tokens,
                ),
                output_tokens=chat_completion.usage.completion_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
                # Provider totals can use accounting that differs from prompt + completion.
                total_tokens=chat_completion.usage.total_tokens,
            )

        incomplete_details = None
        if choice.finish_reason == "length":
            incomplete_details = {"reason": "max_output_tokens"}
        elif choice.finish_reason == "content_filter":
            incomplete_details = {"reason": "content_filter"}

        native_finish_reason = getattr(choice, "native_finish_reason", None)
        provider_metadata = (
            {"native_finish_reason": str(native_finish_reason)} if native_finish_reason is not None else {}
        )

        # Chat Completion -> Response
        return NeMoGymResponse(
            # Under external token capture the chat completion's envelope id is
            # reused instead of minting one: capture records the id of the served
            # payload, and the id the client keeps must match it for terminal
            # attribution to join the scored response back to its captured call.
            # Everywhere else Gym mints the id, keeping the resp_* format and
            # uniqueness independent of the backend.
            id=(str(getattr(chat_completion, "id", "") or "") if preserve_envelope_id else "")
            or f"resp_{uuid4().hex}",
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
            status="incomplete" if incomplete_details is not None else "completed",
            incomplete_details=incomplete_details,
            usage=usage,
            **provider_metadata,
        )


# Responses output-item types that mark the start of the model's own generation.
# split_responses_input_output_items cuts at the first item of one of these types.
# Everything before it is replayed prompt.
# Everything after is the segment carrying sampled tokens.
#
# "message" is absent because the role == "assistant" check below already covers an assistant message.
# A user, system or developer message must not open the segment.
_RESPONSE_OUTPUT_BOUNDARY_TYPES = frozenset(
    {
        "code_interpreter_call",
        "computer_call",
        "custom_tool_call",
        "file_search_call",
        "function_call",
        "image_generation_call",
        "local_shell_call",
        "mcp_approval_request",
        "mcp_call",
        "mcp_list_tools",
        "reasoning",
        "web_search_call",
        "apply_patch_call",
        "shell_call",
        "tool_search_call",
    }
)

# These item types are not model-generated.
# They include client-supplied tool results, approvals, and context-management data.
# Excluding them from the boundary set prevents tool results from starting the sampled segment.
#
# The SDK unions do not distinguish generated items from client-supplied items.
_RESPONSE_NON_BOUNDARY_TYPES: frozenset[str] = frozenset(
    {
        "additional_tools",
        "apply_patch_call_output",
        "compaction",
        "compaction_trigger",
        "computer_call_output",
        "custom_tool_call_output",
        "function_call_output",
        "local_shell_call_output",
        "mcp_approval_response",
        "shell_call_output",
        "tool_search_output",
    }
)


def split_responses_input_output_items(
    items: List[NeMoGymResponseOutputItem],
) -> Tuple[List[NeMoGymResponseOutputItem], List[NeMoGymResponseOutputItem]]:
    if not items:
        return [], []

    split_at = len(items)
    for i, item in enumerate(items):
        # The role shortcut is for assistant messages only.
        # Applied to any item carrying a role, it outranks the type set below and decides the split
        # on its own, so a non-message item that happens to have role == "assistant" would open the
        # trained segment however it is classified.
        # Every item type with a role is a message at the pinned SDK, so this is the same behaviour
        # here, stated in a way later versions cannot subvert.
        item_type = getattr(item, "type", None)
        is_assistant_message = item_type == "message" and getattr(item, "role", None) == "assistant"
        if is_assistant_message or item_type in _RESPONSE_OUTPUT_BOUNDARY_TYPES:
            split_at = i
            break

    return items[:split_at], items[split_at:]


# Backward-compatible aliases
VLLMConverter = ResponsesConverter
VLLMConverterResponsesToChatCompletionsState = ResponsesConverterState
