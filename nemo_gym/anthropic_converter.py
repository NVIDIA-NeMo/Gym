# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Bidirectional converter between NeMo Gym Responses API objects and Anthropic Messages.

This module is the single source of truth for the Anthropic <-> Responses mapping. It is
shared by two opposite-direction consumers:

* **Egress** (`responses_api_models/anthropic_model`): NeMo Gym is the client and Anthropic is
  the backend. Uses ``responses_to_anthropic`` (request) and ``anthropic_to_responses``
  (response).
* **Ingress** (an Anthropic-Messages proxy, e.g. for the Claude Code CLI): an Anthropic client
  talks to NeMo Gym, which forwards to a downstream Gym model server. Uses
  ``anthropic_request_to_responses`` (request), ``responses_to_anthropic_response`` (response),
  and ``anthropic_response_to_sse`` (synthesize Anthropic SSE from a complete response).

The converter is **transport-free and SDK-free**: pure dict/Pydantic in, pure dict/Pydantic
out. All HTTP stays in the servers via ``nemo_gym.server_utils.request()`` (the ``anthropic``
SDK is avoided because it uses httpx, whose O(n^2) connection pooling hangs at high
concurrency).

Boundary note: a few methods here implement **egress-only policy** (Anthropic-API-as-backend
concerns) rather than structural mapping: ``_validate_sampling_params_for_model``,
``_model_disallows_sampling_params``, and the thinking-config handling in
``_copy_thinking_params``. They are invoked only on the egress ``responses_to_anthropic`` path;
ingress never calls them (an open-model backend has none of those restrictions). Relocating
them into the egress server is a deliberate follow-up, kept out of this refactor to avoid
changing the egress contract.
"""

import base64
import binascii
import json
from time import time
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

# Types only — never the `anthropic` client. The client uses httpx (O(n^2) connection
# pooling at high concurrency); all transport in Gym stays on aiohttp via server_utils.
# MessageCreateParams (request) is a TypedDict used purely as a hint; NeMoGymAnthropicMessage
# (response) is the BaseModel used to validate what we emit.
from anthropic.types.message_create_params import MessageCreateParams

from nemo_gym.anthropic_utils import NeMoGymAnthropicMessage
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)


SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# These classifications define the supported Anthropic request boundary.
# SDK drift tests require every request field to be classified exactly once.
# Mapped fields have an explicit Responses conversion below.
# A mapped field may still reject nested values that have no lossless representation.
MAPPED_ANTHROPIC_REQUEST_FIELDS = frozenset(
    {
        "max_tokens",  # Becomes Responses max_output_tokens.
        "messages",  # Becomes ordered Responses message, reasoning, and function-call items.
        "metadata",  # Maps metadata.user_id to Responses user.
        "model",  # Keeps the requested model name.
        "output_config",  # Maps supported effort values to Responses reasoning.effort.
        "service_tier",  # Preserves the shared auto tier.
        "system",  # Becomes Responses instructions.
        "temperature",  # Keeps the shared sampling value.
        "tool_choice",  # Maps tool selection and disable_parallel_tool_use.
        "tools",  # Maps client function tools to Responses function tools.
        "top_p",  # Keeps the shared sampling value.
    }
)

# Ignored fields are accepted but intentionally omitted from the Responses request.
# cache_control is a prompt-cache transport hint and does not change generated content.
IGNORED_ANTHROPIC_REQUEST_FIELDS = frozenset({"cache_control"})

# Rejected fields have no lossless Responses request representation.
# _validate_anthropic_request raises when any of these fields has a non-null value.
REJECTED_ANTHROPIC_REQUEST_FIELDS = frozenset(
    {
        "container",  # Responses cannot select Anthropic container state.
        "inference_geo",  # Responses has no equivalent inference-routing field.
        "stop_sequences",  # Responses create params have no stop-sequence field.
        "thinking",  # Anthropic thinking budgets cannot be represented by Responses reasoning config.
        "top_k",  # Responses has no equivalent sampling field.
    }
)

# Mapped content blocks become one or more ordered Responses input items.
# Nested attributes are validated before conversion and may reject independently.
MAPPED_ANTHROPIC_CONTENT_BLOCK_TYPES = frozenset(
    {
        "document",  # Becomes an input_file part for supported URL or base64 PDF sources.
        "image",  # Becomes an input_image part for supported URL or base64 image sources.
        "redacted_thinking",  # Becomes opaque encrypted reasoning with no summary.
        "text",  # Becomes an input_text part.
        "thinking",  # Becomes a reasoning item with a summary and optional signature.
        "tool_result",  # Becomes a function_call_output item.
        "tool_use",  # Becomes a function_call item.
    }
)

# Rejected content blocks describe Anthropic-hosted tools or stateful server operations.
# The ingress block dispatcher raises instead of flattening them into lossy text.
REJECTED_ANTHROPIC_CONTENT_BLOCK_TYPES = frozenset(
    {
        "bash_code_execution_tool_result",  # Contains hosted Bash execution state.
        "code_execution_tool_result",  # Contains hosted code execution state.
        "container_upload",  # Refers to Anthropic-managed container state.
        "mid_conv_system",  # Has no equivalent ordered Responses input role.
        "search_result",  # Carries hosted search result metadata.
        "server_tool_use",  # Represents a tool invocation executed by Anthropic.
        "text_editor_code_execution_tool_result",  # Contains hosted editor execution state.
        "tool_search_tool_result",  # Contains Anthropic tool-discovery results.
        "web_fetch_tool_result",  # Contains hosted web-fetch results and metadata.
        "web_search_tool_result",  # Contains hosted web-search results and metadata.
    }
)
MAPPED_ANTHROPIC_TOOL_VARIANTS = frozenset({"ToolParam"})
REJECTED_ANTHROPIC_TOOL_VARIANTS = frozenset(
    {
        "CodeExecutionTool20250522Param",
        "CodeExecutionTool20250825Param",
        "CodeExecutionTool20260120Param",
        "MemoryTool20250818Param",
        "ToolBash20250124Param",
        "ToolSearchToolBm25_20251119Param",
        "ToolSearchToolRegex20251119Param",
        "ToolTextEditor20250124Param",
        "ToolTextEditor20250429Param",
        "ToolTextEditor20250728Param",
        "WebFetchTool20250910Param",
        "WebFetchTool20260209Param",
        "WebFetchTool20260309Param",
        "WebSearchTool20250305Param",
        "WebSearchTool20260209Param",
    }
)
MAPPED_ANTHROPIC_STOP_REASONS = frozenset({"end_turn", "max_tokens", "refusal", "tool_use"})
REJECTED_ANTHROPIC_STOP_REASONS = frozenset({"pause_turn", "stop_sequence"})

MAPPED_RESPONSES_REQUEST_FIELDS = frozenset(
    {
        "input",
        "instructions",
        "max_output_tokens",
        "model",
        "parallel_tool_calls",
        "service_tier",
        "stream",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
        "user",
    }
)
REJECTED_RESPONSES_REQUEST_FIELDS = frozenset(
    {
        "background",
        "include",
        "max_tool_calls",
        "metadata",
        "previous_response_id",
        "prompt",
        "reasoning",
        "store",
        "text",
        "top_logprobs",
        "truncation",
    }
)
MAPPED_RESPONSES_INPUT_ITEM_TYPES = frozenset({"function_call", "function_call_output", "message", "reasoning"})
REJECTED_RESPONSES_INPUT_ITEM_TYPES = frozenset(
    {
        "code_interpreter_call",
        "computer_call",
        "computer_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "file_search_call",
        "image_generation_call",
        "local_shell_call",
        "local_shell_call_output",
        "mcp_approval_request",
        "mcp_approval_response",
        "mcp_call",
        "mcp_list_tools",
        "web_search_call",
    }
)


class AnthropicConverter:
    ############################################################################
    # Egress: NeMo Gym Responses  ->  Anthropic Messages request
    ############################################################################
    def responses_to_anthropic(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        model: str,
        max_tokens: int,
        thinking: Optional[Dict[str, Any]],
        thinking_budget_tokens: Optional[int],
        extra_body: Dict[str, Any],
    ) -> Dict[str, Any]:
        body_dict = body.model_dump(exclude_unset=True)
        self._validate_responses_request(body_dict, model)
        anthropic_body = dict(extra_body)
        max_output_tokens = body_dict.pop("max_output_tokens", None)
        anthropic_body.update(
            {
                "model": model,
                "max_tokens": max_output_tokens if max_output_tokens is not None else max_tokens,
                "messages": [],
            }
        )

        system_parts = []
        if body.instructions:
            system_parts.append(body.instructions)

        response_input = body_dict.pop("input")
        input_items = self._normalize_input(response_input)
        for item in input_items:
            item_type = item.get("type") or "message"
            if item_type == "message":
                self._append_message_item(item, anthropic_body["messages"], system_parts)
            elif item_type == "reasoning":
                self._append_content(
                    anthropic_body["messages"],
                    "assistant",
                    self._reasoning_item_to_anthropic_blocks(item),
                )
            elif item_type == "function_call":
                self._append_content(
                    anthropic_body["messages"],
                    "assistant",
                    [self._function_call_to_tool_use(item)],
                )
            elif item_type == "function_call_output":
                self._validate_function_call_output_item(item)
                self._append_content(
                    anthropic_body["messages"],
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": item["call_id"],
                            "content": self._function_call_output_to_anthropic(item["output"]),
                        }
                    ],
                )
            else:
                raise NotImplementedError(f"Unsupported Responses API item type for Anthropic: {item_type}")

        if system_parts:
            anthropic_body["system"] = self._system_parts_to_anthropic_blocks(system_parts)

        self._copy_sampling_params(body_dict, anthropic_body)
        self._validate_sampling_params_for_model(model, anthropic_body)
        self._copy_tools(body_dict, anthropic_body)
        self._copy_tool_choice(body_dict, anthropic_body)
        self._copy_user_and_service_tier(body_dict, anthropic_body)
        self._copy_thinking_params(
            anthropic_body=anthropic_body,
            thinking=thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )

        return anthropic_body

    # ---- egress-only policy (see module boundary note) ----
    def _copy_thinking_params(
        self,
        anthropic_body: Dict[str, Any],
        thinking: Optional[Dict[str, Any]],
        thinking_budget_tokens: Optional[int],
    ) -> None:
        configured_sources = sum(
            source_is_set
            for source_is_set in (
                "thinking" in anthropic_body,
                thinking is not None,
                thinking_budget_tokens is not None,
            )
        )
        if configured_sources > 1:
            raise ValueError(
                "Configure Anthropic thinking in only one place: thinking, thinking_budget_tokens, or extra_body."
            )

        if thinking is not None:
            anthropic_body["thinking"] = thinking
        elif thinking_budget_tokens is not None:
            anthropic_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }

    def _validate_sampling_params_for_model(self, model: str, anthropic_body: Dict[str, Any]) -> None:
        if not self._model_disallows_sampling_params(model):
            return
        configured_sampling_params = [
            param for param in ("temperature", "top_p", "top_k") if anthropic_body.get(param) is not None
        ]
        if configured_sampling_params:
            raise ValueError(
                f"{model} does not support configurable sampling parameters; omit {configured_sampling_params}."
            )

    def _model_disallows_sampling_params(self, model: str) -> bool:
        return any(model_id in model for model_id in ("claude-opus-4-7", "claude-opus-4-8"))

    ############################################################################
    # Egress: Anthropic Messages response  ->  NeMo Gym Responses
    ############################################################################
    def anthropic_to_responses(
        self,
        anthropic_response: Dict[str, Any],
        request_body: NeMoGymResponseCreateParamsNonStreaming,
        model: str,
    ) -> NeMoGymResponse:
        self._validate_anthropic_response(anthropic_response)
        output = self._anthropic_content_to_output_items(anthropic_response.get("content", []))
        if not output:
            self._flush_text_output([""], output)

        usage = self._usage_to_responses_usage(anthropic_response.get("usage"))
        stop_reason = anthropic_response.get("stop_reason")
        incomplete_details = self._incomplete_details_from_stop_reason(stop_reason)

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model,
            object="response",
            output=[item.model_dump() for item in output],
            tool_choice=request_body.tool_choice,
            parallel_tool_calls=request_body.parallel_tool_calls,
            tools=request_body.tools,
            temperature=request_body.temperature,
            top_p=request_body.top_p,
            background=request_body.background,
            max_output_tokens=request_body.max_output_tokens,
            max_tool_calls=request_body.max_tool_calls,
            previous_response_id=request_body.previous_response_id,
            prompt=request_body.prompt,
            reasoning=request_body.reasoning,
            service_tier=request_body.service_tier,
            text=request_body.text,
            top_logprobs=request_body.top_logprobs,
            truncation=request_body.truncation,
            metadata=request_body.metadata,
            instructions=request_body.instructions,
            user=request_body.user,
            incomplete_details=incomplete_details,
            usage=usage,
        )

    def _anthropic_content_to_output_items(self, content: List[Dict[str, Any]]) -> List[Any]:
        """Convert Anthropic response content blocks to ordered Responses output items."""
        output: List[Any] = []
        pending_text: List[str] = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                self._reject_non_null_fields(block, {"text", "type"}, "Anthropic response text block")
                pending_text.append(block.get("text", ""))
            elif block_type == "thinking":
                self._reject_non_null_fields(
                    block, {"signature", "thinking", "type"}, "Anthropic response thinking block"
                )
                self._flush_text_output(pending_text, output)
                output.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex}",
                        summary=[
                            NeMoGymSummary(
                                text=block.get("thinking") or block.get("text", ""),
                                type="summary_text",
                            )
                        ],
                        encrypted_content=block.get("signature"),
                    )
                )
            elif block_type == "redacted_thinking":
                self._reject_non_null_fields(block, {"data", "type"}, "Anthropic response redacted_thinking block")
                self._flush_text_output(pending_text, output)
                output.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex}",
                        summary=[],
                        encrypted_content=block["data"],
                        status="completed",
                    )
                )
            elif block_type == "tool_use":
                self._reject_non_null_fields(
                    block, {"id", "input", "name", "type"}, "Anthropic response tool_use block"
                )
                self._flush_text_output(pending_text, output)
                output.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=json.dumps(block.get("input", {})),
                        call_id=block["id"],
                        name=block["name"],
                        id=block["id"],
                        status="completed",
                    )
                )
            else:
                raise NotImplementedError(f"Unsupported Anthropic content block type: {block_type}")

        self._flush_text_output(pending_text, output)
        return output

    def _incomplete_details_from_stop_reason(self, stop_reason: Optional[str]) -> Optional[Dict[str, str]]:
        if stop_reason in REJECTED_ANTHROPIC_STOP_REASONS:
            raise NotImplementedError(f"Unsupported Anthropic stop_reason for Responses: {stop_reason}")
        if stop_reason not in MAPPED_ANTHROPIC_STOP_REASONS and stop_reason is not None:
            raise NotImplementedError(f"Unknown Anthropic stop_reason for Responses: {stop_reason}")
        if stop_reason == "max_tokens":
            return {"reason": "max_output_tokens"}
        if stop_reason == "refusal":
            return {"reason": "content_filter"}
        return None

    ############################################################################
    # Ingress: Anthropic Messages request  ->  NeMo Gym Responses
    ############################################################################
    def anthropic_request_to_responses(
        self, anthropic_body: MessageCreateParams
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        """Inverse of ``responses_to_anthropic`` (the request direction).

        Parses an inbound Anthropic Messages request into Responses create params so it can be
        forwarded to a downstream Gym model server's ``/v1/responses``.

        ``anthropic_body`` is hinted with the Anthropic SDK's native ``MessageCreateParams``
        (a TypedDict union, so it accepts ``stream: true``). It's a type hint only — at runtime
        the value is the raw request dict.
        Unknown semantic fields fail explicitly so SDK changes cannot be dropped during conversion.
        Prompt-cache hints are accepted because they do not change the generated response.
        """
        self._validate_anthropic_request(anthropic_body)
        params: Dict[str, Any] = {"input": self._anthropic_messages_to_input_items(anthropic_body)}

        instructions = self._anthropic_system_to_instructions(anthropic_body.get("system"))
        if instructions:
            params["instructions"] = instructions

        if anthropic_body.get("model") is not None:
            params["model"] = anthropic_body["model"]
        if anthropic_body.get("max_tokens") is not None:
            params["max_output_tokens"] = anthropic_body["max_tokens"]
        if anthropic_body.get("temperature") is not None:
            params["temperature"] = anthropic_body["temperature"]
        if anthropic_body.get("top_p") is not None:
            params["top_p"] = anthropic_body["top_p"]
        output_config = anthropic_body.get("output_config")
        if output_config:
            self._reject_non_null_fields(output_config, {"effort"}, "Anthropic output_config")
            effort = output_config.get("effort")
            if effort is not None:
                if effort not in ("low", "medium", "high"):
                    raise NotImplementedError(f"Unsupported Anthropic output_config effort for Responses: {effort}")
                params["reasoning"] = {"effort": effort}
        metadata = anthropic_body.get("metadata")
        if metadata and metadata.get("user_id") is not None:
            params["user"] = metadata["user_id"]
        service_tier = anthropic_body.get("service_tier")
        if service_tier is not None:
            if service_tier != "auto":
                raise NotImplementedError(f"Unsupported Anthropic service_tier for Responses: {service_tier}")
            params["service_tier"] = "auto"

        tools = self._anthropic_tools_to_responses(anthropic_body.get("tools"))
        if tools:
            params["tools"] = tools
        tool_choice = self._anthropic_tool_choice_to_responses(anthropic_body.get("tool_choice"))
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
            params["parallel_tool_calls"] = not anthropic_body["tool_choice"].get("disable_parallel_tool_use", False)

        return NeMoGymResponseCreateParamsNonStreaming(**params)

    def _anthropic_system_to_instructions(self, system: Any) -> str:
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        texts = []
        for block in system:
            if block.get("type") != "text":
                raise NotImplementedError(f"Unsupported Anthropic system block: {block.get('type')}")
            self._reject_non_null_fields(block, {"cache_control", "text", "type"}, "Anthropic system text block")
            if block.get("text"):
                texts.append(block["text"])
        return "\n".join(texts)

    def _anthropic_messages_to_input_items(self, anthropic_body: Dict[str, Any]) -> List[Any]:
        items: List[Any] = []
        for message in anthropic_body.get("messages", []):
            role = message["role"]
            if role not in ("user", "assistant"):
                raise NotImplementedError(f"Unsupported Anthropic message role for Responses: {role}")
            self._reject_non_null_fields(message, {"content", "role"}, "Anthropic message")
            content = message.get("content", "")
            if isinstance(content, str):
                items.append(NeMoGymEasyInputMessage(role=role, content=content, type="message"))
                continue
            self._append_anthropic_blocks_as_items(role, content, items)
        return items

    def _append_anthropic_blocks_as_items(self, role: str, blocks: List[Dict[str, Any]], items: List[Any]) -> None:
        """Translate one Anthropic message's content blocks into ordered Responses items.

        Text/image blocks group into a single message item; tool_use, tool_result, and thinking
        blocks each become their own item, preserving order.
        """
        pending_parts: List[Dict[str, Any]] = []

        def flush_message() -> None:
            if not pending_parts:
                return
            if len(pending_parts) == 1 and pending_parts[0]["type"] == "input_text":
                items.append(NeMoGymEasyInputMessage(role=role, content=pending_parts[0]["text"], type="message"))
            else:
                items.append(NeMoGymEasyInputMessage(role=role, content=list(pending_parts), type="message"))
            pending_parts.clear()

        for block in blocks:
            block_type = block.get("type")
            if block_type == "text":
                self._reject_non_null_fields(block, {"cache_control", "text", "type"}, "Anthropic text block")
                pending_parts.append({"type": "input_text", "text": block.get("text", "")})
            elif block_type == "image":
                pending_parts.append(self._anthropic_image_to_input_part(block))
            elif block_type == "document":
                pending_parts.append(self._anthropic_document_to_input_part(block))
            elif block_type == "tool_use":
                self._reject_non_null_fields(
                    block, {"cache_control", "id", "input", "name", "type"}, "Anthropic tool_use block"
                )
                flush_message()
                items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=json.dumps(block.get("input", {})),
                        call_id=block["id"],
                        name=block["name"],
                        id=block["id"],
                        status="completed",
                        type="function_call",
                    )
                )
            elif block_type == "tool_result":
                self._reject_non_null_fields(
                    block,
                    {"cache_control", "content", "is_error", "tool_use_id", "type"},
                    "Anthropic tool_result block",
                )
                if block.get("is_error") is True:
                    raise NotImplementedError(
                        "Anthropic tool_result is_error=true cannot be represented losslessly in Responses."
                    )
                flush_message()
                items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=block["tool_use_id"],
                        output=self._anthropic_tool_result_content_to_responses(block.get("content", "")),
                        type="function_call_output",
                    )
                )
            elif block_type == "thinking":
                self._reject_non_null_fields(
                    block, {"cache_control", "signature", "thinking", "type"}, "Anthropic thinking block"
                )
                flush_message()
                items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex}",
                        summary=[NeMoGymSummary(text=block.get("thinking", ""), type="summary_text")],
                        encrypted_content=block.get("signature"),
                        type="reasoning",
                    )
                )
            elif block_type == "redacted_thinking":
                self._reject_non_null_fields(
                    block, {"cache_control", "data", "type"}, "Anthropic redacted_thinking block"
                )
                flush_message()
                items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex}",
                        summary=[],
                        encrypted_content=block["data"],
                        status="completed",
                        type="reasoning",
                    )
                )
            else:
                raise NotImplementedError(f"Unsupported Anthropic content block type for ingress: {block_type}")
        flush_message()

    def _anthropic_image_to_input_part(self, block: Dict[str, Any]) -> Dict[str, Any]:
        self._reject_non_null_fields(block, {"cache_control", "source", "type"}, "Anthropic image block")
        source = block.get("source") or {}
        source_type = source.get("type")
        if source_type == "url":
            self._reject_non_null_fields(source, {"type", "url"}, "Anthropic URL image source")
            image_url = source["url"]
        elif source_type == "base64":
            self._reject_non_null_fields(source, {"data", "media_type", "type"}, "Anthropic base64 image source")
            media_type = source["media_type"]
            if media_type not in SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES:
                raise ValueError(
                    f"Unsupported Anthropic image media type. Supported types: "
                    f"{sorted(SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES)}."
                )
            image_url = self._build_data_url(media_type, source["data"])
            self._parse_data_url(image_url, SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES, "image")
        else:
            raise NotImplementedError(f"Unsupported Anthropic image source type: {source_type}")
        return {
            "type": "input_image",
            "image_url": image_url,
            "detail": "auto",
        }

    def _anthropic_document_to_input_part(self, block: Dict[str, Any]) -> Dict[str, Any]:
        self._reject_non_null_fields(block, {"cache_control", "source", "type"}, "Anthropic document block")
        source = block.get("source") or {}
        source_type = source.get("type")
        if source_type == "url":
            self._reject_non_null_fields(source, {"type", "url"}, "Anthropic URL document source")
            return {"type": "input_file", "file_url": source["url"]}
        if source_type == "base64":
            self._reject_non_null_fields(source, {"data", "media_type", "type"}, "Anthropic base64 document source")
            if source.get("media_type") != "application/pdf":
                raise ValueError("Anthropic base64 documents must use application/pdf.")
            file_data = self._build_data_url("application/pdf", source["data"])
            self._parse_data_url(file_data, {"application/pdf"}, "file")
            return {"type": "input_file", "file_data": file_data}
        raise NotImplementedError(f"Unsupported Anthropic document source type: {source_type}")

    def _anthropic_tool_result_content_to_responses(self, content: Any) -> Any:
        if isinstance(content, str):
            return content
        parts = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                self._reject_non_null_fields(
                    block, {"cache_control", "text", "type"}, "Anthropic tool_result text block"
                )
                parts.append({"type": "input_text", "text": block.get("text", "")})
            elif block_type == "image":
                parts.append(self._anthropic_image_to_input_part(block))
            elif block_type == "document":
                parts.append(self._anthropic_document_to_input_part(block))
            else:
                raise NotImplementedError(f"Unsupported Anthropic tool_result content block for ingress: {block_type}")
        return parts

    def _anthropic_tools_to_responses(self, tools: Any) -> List[Dict[str, Any]]:
        if not tools:
            return []
        responses_tools = []
        for tool in tools:
            tool_type = tool.get("type")
            if tool_type not in (None, "custom"):
                raise NotImplementedError(f"Unsupported Anthropic hosted tool type for Responses: {tool_type}")
            self._reject_non_null_fields(
                tool,
                {"cache_control", "description", "input_schema", "name", "strict", "type"},
                "Anthropic function tool",
            )
            responses_tools.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description"),
                    "parameters": (
                        tool["input_schema"]
                        if tool.get("input_schema") is not None
                        else {"type": "object", "properties": {}}
                    ),
                    "strict": tool.get("strict", False),
                }
            )
        return responses_tools

    def _anthropic_tool_choice_to_responses(self, tool_choice: Any) -> Any:
        if tool_choice is None:
            return None
        choice_type = tool_choice.get("type")
        allowed_fields = {"type"}
        if choice_type in ("auto", "any", "tool"):
            allowed_fields.add("disable_parallel_tool_use")
        if choice_type == "tool":
            allowed_fields.add("name")
        self._reject_non_null_fields(tool_choice, allowed_fields, "Anthropic tool_choice")
        if choice_type == "auto":
            return "auto"
        if choice_type == "none":
            return "none"
        if choice_type == "any":
            return "required"
        if choice_type == "tool":
            return {"type": "function", "name": tool_choice["name"]}
        raise NotImplementedError(f"Unsupported Anthropic tool_choice for ingress: {tool_choice}")

    def _build_data_url(self, media_type: str, data: str) -> str:
        return f"data:{media_type};base64,{data}"

    ############################################################################
    # Ingress: NeMo Gym Responses  ->  Anthropic Messages response (+ SSE)
    ############################################################################
    def responses_to_anthropic_response(self, response: NeMoGymResponse, model: str) -> Dict[str, Any]:
        """Inverse of ``anthropic_to_responses`` (the response direction).

        Renders a downstream ``/v1/responses`` result as a complete Anthropic Messages response
        object (non-streaming shape).

        The assembled object is validated by constructing ``NeMoGymAnthropicMessage`` (a thin
        subclass of the Anthropic SDK's ``Message``) — catching malformed blocks / bad
        stop_reason / missing fields at the boundary — then
        returned as a JSON dict for the SSE synthesizer and the non-streaming JSON response.
        ``exclude_none`` keeps the lean Anthropic shape (drops null SDK-only fields).
        """
        content: List[Dict[str, Any]] = []
        has_tool_use = False
        for item in self._iter_output_dicts(response):
            item_type = item.get("type") or "message"
            if item_type == "message":
                content.extend(self._output_message_to_anthropic_blocks(item))
            elif item_type == "reasoning":
                content.extend(self._reasoning_item_to_anthropic_blocks(item))
            elif item_type == "function_call":
                content.append(self._function_call_to_tool_use(item))
                has_tool_use = True
            else:
                raise NotImplementedError(f"Unsupported Responses output item for Anthropic response: {item_type}")

        usage = response.usage.model_dump() if response.usage is not None else None
        cached_tokens = ((usage or {}).get("input_tokens_details") or {}).get("cached_tokens", 0)
        input_tokens = (usage or {}).get("input_tokens", 0)
        if cached_tokens > input_tokens:
            raise ValueError("Responses cached_tokens cannot exceed input_tokens.")
        reasoning_tokens = ((usage or {}).get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        if reasoning_tokens:
            raise NotImplementedError("Anthropic usage cannot represent Responses reasoning_tokens.")
        message = NeMoGymAnthropicMessage.model_validate(
            {
                "id": f"msg_{uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": self._stop_reason_from_response(response, has_tool_use),
                "stop_sequence": None,
                "usage": {
                    "cache_read_input_tokens": cached_tokens,
                    "input_tokens": input_tokens - cached_tokens,
                    "output_tokens": (usage or {}).get("output_tokens", 0),
                },
            }
        )
        return message.model_dump(mode="json", exclude_none=True)

    def _iter_output_dicts(self, response: NeMoGymResponse) -> List[Dict[str, Any]]:
        items = []
        for item in response.output or []:
            items.append(item if isinstance(item, dict) else item.model_dump())
        return items

    def _output_message_to_anthropic_blocks(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        if item.get("status") not in (None, "completed"):
            raise NotImplementedError(f"Unsupported Responses output message status: {item.get('status')}")
        self._reject_atomic_metadata(item, "Responses output message")
        blocks = []
        for part in item.get("content", []):
            part_type = part.get("type")
            if part_type == "output_text":
                if part.get("annotations"):
                    raise NotImplementedError("Anthropic text blocks cannot represent Responses output annotations.")
                if part.get("logprobs") is not None:
                    raise NotImplementedError("Anthropic text blocks cannot represent Responses output logprobs.")
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "refusal":
                blocks.append({"type": "text", "text": part.get("refusal", "")})
            else:
                raise NotImplementedError(f"Unsupported output_text part for Anthropic response: {part_type}")
        return blocks

    def _stop_reason_from_response(self, response: NeMoGymResponse, has_tool_use: bool) -> str:
        incomplete = response.incomplete_details
        reason = incomplete.reason if incomplete is not None else None
        if reason == "max_output_tokens":
            return "max_tokens"
        if reason == "content_filter":
            return "refusal"
        if reason is not None:
            raise NotImplementedError(f"Unsupported Responses incomplete reason for Anthropic: {reason}")
        if has_tool_use:
            return "tool_use"
        return "end_turn"

    def anthropic_response_to_sse(self, anthropic_response: Dict[str, Any]) -> Iterator[str]:
        """Synthesize an Anthropic Messages SSE stream from a complete response object.

        The downstream call is non-streaming; this fakes the event sequence the Claude Code CLI
        expects: ``message_start`` -> per-block (``content_block_start`` ->
        ``content_block_delta`` -> ``content_block_stop``) -> ``message_delta`` -> ``message_stop``.
        """
        content = anthropic_response.get("content", [])
        usage = anthropic_response.get("usage", {})

        message_shell = {k: v for k, v in anthropic_response.items() if k != "content"}
        message_shell["content"] = []
        message_shell.setdefault("usage", {})
        yield self._sse_event("message_start", {"type": "message_start", "message": message_shell})

        for index, block in enumerate(content):
            yield self._sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": self._empty_block_shell(block)},
            )
            for delta in self._block_deltas(block):
                yield self._sse_event(
                    "content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}
                )
            yield self._sse_event("content_block_stop", {"type": "content_block_stop", "index": index})

        yield self._sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": anthropic_response.get("stop_reason"),
                    "stop_sequence": anthropic_response.get("stop_sequence"),
                },
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
        )
        yield self._sse_event("message_stop", {"type": "message_stop"})

    def _empty_block_shell(self, block: Dict[str, Any]) -> Dict[str, Any]:
        block_type = block.get("type")
        if block_type == "text":
            return {"type": "text", "text": ""}
        if block_type == "thinking":
            return {"type": "thinking", "thinking": ""}
        if block_type == "redacted_thinking":
            return {"type": "redacted_thinking", "data": block["data"]}
        if block_type == "tool_use":
            return {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
        raise NotImplementedError(f"Unsupported Anthropic block for SSE synthesis: {block_type}")

    def _block_deltas(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        block_type = block.get("type")
        if block_type == "text":
            return [{"type": "text_delta", "text": block.get("text", "")}]
        if block_type == "thinking":
            deltas = [{"type": "thinking_delta", "thinking": block.get("thinking", "")}]
            if block.get("signature") is not None:
                deltas.append({"type": "signature_delta", "signature": block["signature"]})
            return deltas
        if block_type == "redacted_thinking":
            return []
        if block_type == "tool_use":
            return [{"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))}]
        raise NotImplementedError(f"Unsupported Anthropic block for SSE synthesis: {block_type}")

    def _sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    ############################################################################
    # Shared structural helpers
    ############################################################################
    def _normalize_input(self, response_input: Any) -> List[Dict[str, Any]]:
        if isinstance(response_input, str):
            return [NeMoGymEasyInputMessage(content=response_input, role="user").model_dump(exclude_unset=True)]
        return [
            item.model_dump(exclude_unset=True) if hasattr(item, "model_dump") else item for item in response_input
        ]

    def _validate_responses_request(self, body: Dict[str, Any], model: str) -> None:
        known_fields = MAPPED_RESPONSES_REQUEST_FIELDS | REJECTED_RESPONSES_REQUEST_FIELDS
        for field, value in body.items():
            if value is None:
                continue
            if field not in known_fields:
                raise NotImplementedError(f"Unknown Responses request field for Anthropic: {field}")
            if field in REJECTED_RESPONSES_REQUEST_FIELDS:
                raise NotImplementedError(f"Unsupported Responses request field for Anthropic: {field}")
        if body.get("model") is not None and body["model"] != model:
            raise ValueError("Responses request model must match the configured Anthropic model.")
        service_tier = body.get("service_tier")
        if service_tier not in (None, "auto"):
            raise NotImplementedError(f"Unsupported Responses service_tier for Anthropic: {service_tier}")

    def _validate_anthropic_request(self, body: Dict[str, Any]) -> None:
        known_fields = (
            MAPPED_ANTHROPIC_REQUEST_FIELDS
            | IGNORED_ANTHROPIC_REQUEST_FIELDS
            | REJECTED_ANTHROPIC_REQUEST_FIELDS
            | {"stream"}
        )
        for field, value in body.items():
            if value is None:
                continue
            if field not in known_fields:
                raise NotImplementedError(f"Unknown Anthropic request field for Responses: {field}")
            if field in REJECTED_ANTHROPIC_REQUEST_FIELDS:
                raise NotImplementedError(f"Unsupported Anthropic request field for Responses: {field}")
        metadata = body.get("metadata")
        if metadata:
            self._reject_non_null_fields(metadata, {"user_id"}, "Anthropic request metadata")

    def _validate_anthropic_response(self, response: Dict[str, Any]) -> None:
        self._reject_non_null_fields(
            response,
            {"content", "id", "model", "role", "stop_reason", "type", "usage"},
            "Anthropic response",
        )
        usage = response.get("usage")
        if usage:
            self._reject_non_null_fields(
                usage,
                {"cache_read_input_tokens", "input_tokens", "output_tokens"},
                "Anthropic response usage",
            )

    def _reject_non_null_fields(self, value: Dict[str, Any], allowed: set[str], context: str) -> None:
        unsupported = sorted(
            field for field, field_value in value.items() if field not in allowed and field_value is not None
        )
        if unsupported:
            raise NotImplementedError(f"{context} contains unsupported fields: {unsupported}")

    def _reject_atomic_metadata(self, item: Dict[str, Any], context: str) -> None:
        unsupported = [field for field in ("token_ids", "logprobs") if field in item and item.get(field) is not None]
        if unsupported:
            raise NotImplementedError(f"{context} contains unsupported training metadata: {unsupported}")

    def _append_message_item(
        self,
        item: Dict[str, Any],
        messages: List[Dict[str, Any]],
        system_parts: List[str],
    ) -> None:
        self._reject_atomic_metadata(item, "Responses input message")
        if item.get("status") not in (None, "completed"):
            raise NotImplementedError(f"Unsupported Responses input message status: {item.get('status')}")
        role = item["role"]
        content = item.get("content", "")
        if role in ("system", "developer"):
            system_parts.append(self._content_to_text(content))
            return
        if role not in ("user", "assistant"):
            raise NotImplementedError(f"Unsupported Responses API role for Anthropic: {role}")
        self._append_content(messages, role, self._content_to_anthropic_blocks(content, role))

    def _append_content(
        self,
        messages: List[Dict[str, Any]],
        role: str,
        content_blocks: List[Dict[str, Any]],
    ) -> None:
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(content_blocks)
        else:
            messages.append({"role": role, "content": content_blocks})

    def _content_to_anthropic_blocks(self, content: Any, role: str) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        blocks = []
        for part in content:
            part_type = part.get("type")
            if part_type in ("input_text", "output_text", "text"):
                if part.get("annotations"):
                    raise NotImplementedError("Anthropic text blocks cannot represent Responses annotations.")
                if part.get("logprobs") is not None:
                    raise NotImplementedError("Anthropic text blocks cannot represent Responses logprobs.")
                blocks.append({"type": "text", "text": part["text"]})
            elif part_type == "input_image" and role == "user":
                blocks.append(self._input_image_to_anthropic_block(part))
            elif part_type == "input_file" and role == "user":
                blocks.append(self._input_file_to_anthropic_block(part))
            elif part_type == "refusal" and role == "assistant":
                blocks.append({"type": "text", "text": part["refusal"]})
            else:
                raise NotImplementedError(f"Unsupported content part for Anthropic: {part_type}")
        return blocks

    def _input_image_to_anthropic_block(self, part: Dict[str, Any]) -> Dict[str, Any]:
        if part.get("file_id") is not None:
            raise NotImplementedError("Anthropic image blocks cannot represent Responses file_id.")
        if part.get("detail") not in (None, "auto"):
            raise NotImplementedError(f"Anthropic image blocks cannot represent detail={part.get('detail')}.")
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if not isinstance(image_url, str):
            raise ValueError("Responses input_image.image_url must be a URL string.")
        if not image_url.startswith("data:"):
            return {"type": "image", "source": {"type": "url", "url": image_url}}
        media_type, data = self._parse_image_data_url(image_url)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    def _input_file_to_anthropic_block(self, part: Dict[str, Any]) -> Dict[str, Any]:
        if part.get("file_id") is not None:
            raise NotImplementedError("Anthropic document blocks cannot represent Responses file_id.")
        if part.get("filename") is not None:
            raise NotImplementedError("Anthropic document blocks cannot represent Responses filename.")
        file_url = part.get("file_url")
        file_data = part.get("file_data")
        if file_url is not None and file_data is not None:
            raise ValueError("Responses input_file must provide only one of file_url or file_data.")
        if file_url is not None:
            return {"type": "document", "source": {"type": "url", "url": file_url}}
        if file_data is not None:
            media_type, data = self._parse_data_url(file_data, {"application/pdf"}, "file")
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        raise ValueError("Responses input_file requires file_url or file_data for Anthropic.")

    def _parse_image_data_url(self, image_url: str) -> tuple[str, str]:
        return self._parse_data_url(image_url, SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES, "image")

    def _parse_data_url(self, data_url: str, supported_media_types: set[str], content_kind: str) -> tuple[str, str]:
        if not data_url.startswith("data:"):
            raise ValueError(f"Responses input_{content_kind} data must be a base64 data URL.")

        header, separator, data = data_url.partition(",")
        if not separator or not data:
            raise ValueError(f"Responses input_{content_kind} data URL must include base64 data.")

        metadata = header[len("data:") :].split(";")
        media_type = metadata[0].lower()
        if media_type == "image/jpg":
            media_type = "image/jpeg"
        if "base64" not in metadata[1:]:
            raise ValueError(f"Responses input_{content_kind} data must be base64 encoded.")
        if media_type not in supported_media_types:
            raise ValueError(
                f"Unsupported Anthropic {content_kind} media type. Supported types: {sorted(supported_media_types)}."
            )

        try:
            base64.b64decode(data, validate=True)
        except binascii.Error as exc:
            raise ValueError(f"Responses input_{content_kind} data contains invalid base64 data.") from exc

        return media_type, data

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        texts = []
        for part in content:
            part_type = part.get("type")
            if part_type in ("input_text", "output_text", "text"):
                texts.append(part["text"])
            else:
                raise NotImplementedError(f"Unsupported system content part for Anthropic: {part_type}")
        return "\n".join(texts)

    def _system_parts_to_anthropic_blocks(self, system_parts: List[str]) -> List[Dict[str, str]]:
        return [{"type": "text", "text": text} for text in system_parts if text]

    def _reasoning_item_to_anthropic_blocks(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._reject_atomic_metadata(item, "Responses reasoning item")
        if item.get("content") is not None:
            raise NotImplementedError("Anthropic thinking blocks cannot represent Responses reasoning content.")
        if item.get("status") not in (None, "completed"):
            raise NotImplementedError(f"Unsupported Responses reasoning status: {item.get('status')}")
        summaries = item.get("summary", [])
        if not summaries:
            if item.get("encrypted_content") is None:
                raise ValueError("Responses reasoning without a summary requires encrypted_content.")
            return [{"type": "redacted_thinking", "data": item["encrypted_content"]}]
        if len(summaries) != 1:
            raise NotImplementedError("Anthropic thinking blocks cannot represent multiple Responses summaries.")
        summary = summaries[0]
        if summary.get("type") != "summary_text":
            raise NotImplementedError(f"Unsupported Responses reasoning summary type: {summary.get('type')}")
        return [
            {
                "type": "thinking",
                "thinking": summary["text"],
                "signature": item.get("encrypted_content") or "",
            }
        ]

    def _function_call_to_tool_use(self, item: Dict[str, Any]) -> Dict[str, Any]:
        self._reject_atomic_metadata(item, "Responses function_call")
        if item.get("status") not in (None, "completed"):
            raise NotImplementedError(f"Unsupported Responses function_call status: {item.get('status')}")
        return {
            "type": "tool_use",
            "id": item["call_id"],
            "name": item["name"],
            "input": self._json_object_from_arguments(item["arguments"]),
        }

    def _json_object_from_arguments(self, arguments: str) -> Dict[str, Any]:
        parsed = json.loads(arguments or "{}")
        if not isinstance(parsed, dict):
            raise ValueError(f"Anthropic tool_use input must be a JSON object, got {type(parsed).__name__}")
        return parsed

    def _function_call_output_to_anthropic(self, output: Any) -> Any:
        if isinstance(output, str):
            return output
        blocks = []
        for part in output:
            part_type = part.get("type")
            if part_type == "input_text":
                blocks.append({"type": "text", "text": part["text"]})
            elif part_type == "input_image":
                blocks.append(self._input_image_to_anthropic_block(part))
            elif part_type == "input_file":
                blocks.append(self._input_file_to_anthropic_block(part))
            else:
                raise NotImplementedError(f"Unsupported Responses function output part: {part_type}")
        return blocks

    def _validate_function_call_output_item(self, item: Dict[str, Any]) -> None:
        if item.get("status") not in (None, "completed"):
            raise NotImplementedError(f"Unsupported Responses function_call_output status: {item.get('status')}")

    def _copy_sampling_params(self, body_dict: Dict[str, Any], anthropic_body: Dict[str, Any]) -> None:
        for source, target in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
        ):
            value = body_dict.get(source)
            if value is not None:
                anthropic_body[target] = value

    def _copy_tools(self, body_dict: Dict[str, Any], anthropic_body: Dict[str, Any]) -> None:
        tools = body_dict.get("tools") or []
        if not tools:
            return

        anthropic_tools = []
        for tool in tools:
            if tool.get("type") != "function":
                raise NotImplementedError(f"Unsupported Responses API tool type for Anthropic: {tool.get('type')}")
            anthropic_tool = {
                "name": tool["name"],
                "input_schema": (
                    tool["parameters"] if tool.get("parameters") is not None else {"type": "object", "properties": {}}
                ),
            }
            if tool.get("description") is not None:
                anthropic_tool["description"] = tool["description"]
            if tool.get("strict") is not None:
                anthropic_tool["strict"] = tool["strict"]
            anthropic_tools.append(anthropic_tool)
        anthropic_body["tools"] = anthropic_tools

    def _copy_tool_choice(self, body_dict: Dict[str, Any], anthropic_body: Dict[str, Any]) -> None:
        tool_choice = body_dict.get("tool_choice")
        parallel_tool_calls = body_dict.get("parallel_tool_calls")
        if tool_choice is None and parallel_tool_calls is None:
            return
        choice: Dict[str, Any]
        if isinstance(tool_choice, str):
            if tool_choice == "required":
                choice = {"type": "any"}
            elif tool_choice in ("auto", "none"):
                choice = {"type": tool_choice}
            else:
                raise NotImplementedError(f"Unsupported tool_choice for Anthropic: {tool_choice}")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            choice = {"type": "tool", "name": tool_choice["name"]}
        elif tool_choice is None:
            choice = {"type": "auto"}
        else:
            raise NotImplementedError(f"Unsupported tool_choice for Anthropic: {tool_choice}")
        if parallel_tool_calls is not None:
            if choice["type"] == "none" and not parallel_tool_calls:
                raise NotImplementedError("Anthropic tool_choice none cannot carry a parallel-tool setting.")
            if choice["type"] != "none":
                choice["disable_parallel_tool_use"] = not parallel_tool_calls
        anthropic_body["tool_choice"] = choice

    def _copy_user_and_service_tier(self, body_dict: Dict[str, Any], anthropic_body: Dict[str, Any]) -> None:
        user = body_dict.get("user")
        if user is not None:
            metadata = anthropic_body.get("metadata")
            if metadata is None:
                metadata = {}
                anthropic_body["metadata"] = metadata
            if metadata.get("user_id") not in (None, user):
                raise ValueError("Responses user conflicts with Anthropic metadata.user_id.")
            metadata["user_id"] = user
        if body_dict.get("service_tier") is not None:
            anthropic_body["service_tier"] = "auto"

    def _flush_text_output(self, pending_text: List[str], output: List[Any]) -> None:
        if not pending_text:
            return
        output.append(
            NeMoGymResponseOutputMessage(
                id=f"msg_{uuid4().hex}",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text=text,
                    )
                    for text in pending_text
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        )
        pending_text.clear()

    def _usage_to_responses_usage(self, usage: Optional[Dict[str, Any]]) -> Optional[NeMoGymResponseUsage]:
        if usage is None:
            return None
        uncached_input_tokens = usage.get("input_tokens", 0)
        cached_input_tokens = usage.get("cache_read_input_tokens") or 0
        # Anthropic reports cached and uncached tokens separately:
        # https://platform.claude.com/docs/en/build-with-claude/prompt-caching#tracking-cache-performance
        input_tokens = uncached_input_tokens + cached_input_tokens
        output_tokens = usage.get("output_tokens", 0)
        return NeMoGymResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_input_tokens),
            output_tokens=output_tokens,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=input_tokens + output_tokens,
        )
