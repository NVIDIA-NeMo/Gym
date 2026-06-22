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
        anthropic_body = dict(extra_body)
        anthropic_body.update(
            {
                "model": model,
                "max_tokens": body_dict.pop("max_output_tokens", None) or max_tokens,
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
                self._append_content(
                    anthropic_body["messages"],
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": item["call_id"],
                            "content": item["output"],
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
        """Anthropic assistant content blocks -> ordered Responses output items.

        Shared by egress ``anthropic_to_responses`` and ingress ``anthropic_request_to_responses``
        (for assistant turns in the input trajectory).
        """
        output: List[Any] = []
        pending_text: List[str] = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                pending_text.append(block.get("text", ""))
            elif block_type == "thinking":
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
            elif block_type == "tool_use":
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
        if stop_reason in ("max_tokens", "model_context_window_exceeded"):
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
        the value is the raw request dict; we read fields defensively so the proxy stays
        permissive toward unknown / future-beta fields the Claude Code CLI may send.
        """
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

        tools = self._anthropic_tools_to_responses(anthropic_body.get("tools"))
        if tools:
            params["tools"] = tools
        tool_choice = self._anthropic_tool_choice_to_responses(anthropic_body.get("tool_choice"))
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        return NeMoGymResponseCreateParamsNonStreaming(**params)

    def _anthropic_system_to_instructions(self, system: Any) -> str:
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        return "\n".join(block["text"] for block in system if block.get("type") == "text" and block.get("text"))

    def _anthropic_messages_to_input_items(self, anthropic_body: Dict[str, Any]) -> List[Any]:
        items: List[Any] = []
        for message in anthropic_body.get("messages", []):
            role = message["role"]
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
                pending_parts.append({"type": "input_text", "text": block.get("text", "")})
            elif block_type == "image":
                pending_parts.append(self._anthropic_image_to_input_part(block))
            elif block_type == "tool_use":
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
                flush_message()
                items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=block["tool_use_id"],
                        output=self._anthropic_tool_result_content_to_text(block.get("content", "")),
                        type="function_call_output",
                    )
                )
            elif block_type == "thinking":
                flush_message()
                items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex}",
                        summary=[NeMoGymSummary(text=block.get("thinking", ""), type="summary_text")],
                        encrypted_content=block.get("signature"),
                        type="reasoning",
                    )
                )
            else:
                raise NotImplementedError(f"Unsupported Anthropic content block type for ingress: {block_type}")
        flush_message()

    def _anthropic_image_to_input_part(self, block: Dict[str, Any]) -> Dict[str, Any]:
        source = block.get("source") or {}
        if source.get("type") != "base64":
            raise NotImplementedError("Anthropic ingress supports base64 image sources only.")
        media_type = source["media_type"]
        if media_type not in SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES:
            raise ValueError(
                f"Unsupported Anthropic image media type. Supported types: "
                f"{sorted(SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES)}."
            )
        return {
            "type": "input_image",
            "image_url": self._build_image_data_url(media_type, source["data"]),
            "detail": "auto",
        }

    def _anthropic_tool_result_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        texts = []
        for block in content:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            else:
                raise NotImplementedError(
                    f"Unsupported Anthropic tool_result content block for ingress: {block.get('type')}"
                )
        return "\n".join(texts)

    def _anthropic_tools_to_responses(self, tools: Any) -> List[Dict[str, Any]]:
        if not tools:
            return []
        responses_tools = []
        for tool in tools:
            responses_tools.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description"),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                    "strict": False,
                }
            )
        return responses_tools

    def _anthropic_tool_choice_to_responses(self, tool_choice: Any) -> Any:
        if tool_choice is None:
            return None
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        if choice_type == "none":
            return "none"
        if choice_type == "any":
            return "required"
        if choice_type == "tool":
            return {"type": "function", "name": tool_choice["name"]}
        raise NotImplementedError(f"Unsupported Anthropic tool_choice for ingress: {tool_choice}")

    def _build_image_data_url(self, media_type: str, data: str) -> str:
        return f"data:{media_type};base64,{data}"

    ############################################################################
    # Ingress: NeMo Gym Responses  ->  Anthropic Messages response (+ SSE)
    ############################################################################
    def responses_to_anthropic_response(self, response: NeMoGymResponse, model: str) -> Dict[str, Any]:
        """Inverse of ``anthropic_to_responses`` (the response direction).

        Renders a downstream ``/v1/responses`` result as a complete Anthropic Messages response
        object (non-streaming shape). Token-id / logprob fields are intentionally dropped here;
        they are carried out-of-band by the ingress server's side channel.

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
                    "input_tokens": (usage or {}).get("input_tokens", 0),
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
        blocks = []
        for part in item.get("content", []):
            part_type = part.get("type")
            if part_type == "output_text":
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
        if block_type == "tool_use":
            return {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
        raise NotImplementedError(f"Unsupported Anthropic block for SSE synthesis: {block_type}")

    def _block_deltas(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        block_type = block.get("type")
        if block_type == "text":
            return [{"type": "text_delta", "text": block.get("text", "")}]
        if block_type == "thinking":
            return [{"type": "thinking_delta", "thinking": block.get("thinking", "")}]
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

    def _append_message_item(
        self,
        item: Dict[str, Any],
        messages: List[Dict[str, Any]],
        system_parts: List[str],
    ) -> None:
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
                blocks.append({"type": "text", "text": part["text"]})
            elif part_type == "input_image" and role == "user":
                blocks.append(self._input_image_to_anthropic_block(part))
            elif part_type == "refusal" and role == "assistant":
                blocks.append({"type": "text", "text": part["refusal"]})
            else:
                raise NotImplementedError(f"Unsupported content part for Anthropic: {part_type}")
        return blocks

    def _input_image_to_anthropic_block(self, part: Dict[str, Any]) -> Dict[str, Any]:
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if not isinstance(image_url, str):
            raise ValueError("Responses input_image.image_url must be a base64 data URL string.")

        media_type, data = self._parse_image_data_url(image_url)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    def _parse_image_data_url(self, image_url: str) -> tuple[str, str]:
        if not image_url.startswith("data:"):
            raise ValueError("Anthropic image inputs require base64 data URLs; remote image URLs are not supported.")

        header, separator, data = image_url.partition(",")
        if not separator or not data:
            raise ValueError("Responses input_image.image_url must include base64 image data.")

        metadata = header[len("data:") :].split(";")
        media_type = metadata[0].lower()
        if media_type == "image/jpg":
            media_type = "image/jpeg"
        if "base64" not in metadata[1:]:
            raise ValueError("Responses input_image.image_url must be base64 encoded.")
        if media_type not in SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES:
            raise ValueError(
                "Unsupported Anthropic image media type. Supported types: "
                f"{sorted(SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES)}."
            )

        try:
            base64.b64decode(data, validate=True)
        except binascii.Error as exc:
            raise ValueError("Responses input_image.image_url contains invalid base64 image data.") from exc

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
        blocks = []
        for summary in item.get("summary", []):
            # Anthropic's ThinkingBlock requires a signature; open-model backends don't
            # produce one, so default to "" (the synthesized SSE never emits it anyway).
            block = {
                "type": "thinking",
                "thinking": summary["text"],
                "signature": item.get("encrypted_content") or "",
            }
            blocks.append(block)
        return blocks

    def _function_call_to_tool_use(self, item: Dict[str, Any]) -> Dict[str, Any]:
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
                "input_schema": tool.get("parameters") or {"type": "object", "properties": {}},
            }
            if tool.get("description"):
                anthropic_tool["description"] = tool["description"]
            anthropic_tools.append(anthropic_tool)
        anthropic_body["tools"] = anthropic_tools

    def _copy_tool_choice(self, body_dict: Dict[str, Any], anthropic_body: Dict[str, Any]) -> None:
        tool_choice = body_dict.get("tool_choice")
        if tool_choice is None:
            return
        if isinstance(tool_choice, str):
            if tool_choice == "required":
                anthropic_body["tool_choice"] = {"type": "any"}
            elif tool_choice in ("auto", "none"):
                anthropic_body["tool_choice"] = {"type": tool_choice}
            else:
                raise NotImplementedError(f"Unsupported tool_choice for Anthropic: {tool_choice}")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            anthropic_body["tool_choice"] = {"type": "tool", "name": tool_choice["name"]}
        else:
            raise NotImplementedError(f"Unsupported tool_choice for Anthropic: {tool_choice}")

    def _flush_text_output(self, pending_text: List[str], output: List[Any]) -> None:
        if not pending_text:
            return
        output.append(
            NeMoGymResponseOutputMessage(
                id=f"msg_{uuid4().hex}",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text="".join(pending_text),
                    )
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
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return NeMoGymResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=NeMoGymResponseInputTokensDetails(
                cached_tokens=usage.get("cache_read_input_tokens", 0)
            ),
            output_tokens=output_tokens,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=input_tokens + output_tokens,
        )
