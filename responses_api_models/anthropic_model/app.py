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
"""
Anthropic CUA model server for NeMo-Gym.

Wraps the Anthropic Messages API (with computer-use betas) behind the standard
NeMo-Gym ``/v1/responses`` endpoint. Translates between NeMo-Gym's OpenAI-style
request/response schema and Anthropic's native protocol.

Stateful: maintains per-session conversation history keyed by a generated
session ID (returned as ``response.id``). The agent passes it back via
``previous_response_id`` to continue the conversation.
"""
import base64
import logging
import uuid
from typing import Any, Dict, List, Optional

from pydantic import Field

from anthropic import Anthropic

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)

logger = logging.getLogger(__name__)

SONNET_COMPUTER_USE_BETA = "computer-use-2025-01-24"
SONNET_CONTEXT_MGMT_BETA = "context-management-2025-06-27"
TOKEN_EFFICIENT_BETA = "token-efficient-tools-2025-02-19"

OPUS_COMPUTER_USE_BETA = "computer-use-2025-11-24"
OPUS_EFFORT_BETA = "effort-2025-11-24"

ANTHROPIC_CUA_SYSTEM_PROMPT = """\
You are an expert computer automation agent. Complete tasks reliably using the computer tool.

AVAILABLE TOOL:
- computer: Browser automation including screenshots, clicks, typing, scrolling, and navigation

STRICT TOOL PAIRING:
- Every tool_use MUST be immediately followed by a user tool_result in the next message.
- Do NOT insert any other user content between tool_use and tool_result.
- If a tool fails, still return a tool_result with a concise error summary.

QUALITY STRATEGY:
- Prefer short perceive-act cycles.
- If no progress after several actions, switch strategy (search, direct URL, alternate navigation).
- Keep natural text brief; spend tokens on precise actions and perception.

IMPORTANT: Do not include extra user messages between paired tool_use and tool_result. \
Screenshots may be included inside tool_result content when needed for verification."""


class AnthropicModelServerConfig(BaseResponsesAPIModelConfig):
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_is_opus: bool = False
    anthropic_effort_level: str = "high"
    anthropic_max_tokens: int = 4096
    anthropic_turns_to_keep: int = 8
    anthropic_screenshot_turn_limit: int = 8

    extra_body: Dict[str, Any] = Field(default_factory=dict)


class _Session:
    """Per-conversation state for an Anthropic CUA session."""

    __slots__ = ("messages", "pending_tool_use_ids")

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.pending_tool_use_ids: List[str] = []


class AnthropicModelServer(SimpleResponsesAPIModel):
    config: AnthropicModelServerConfig

    def model_post_init(self, context):
        self._client = Anthropic(api_key=self.config.anthropic_api_key, max_retries=4)
        self._sessions: Dict[str, _Session] = {}
        return super().model_post_init(context)

    # ── Tool / beta helpers ──────────────────────────────────────

    def _get_tools(self, display_width: int = 1280, display_height: int = 720) -> List[Dict[str, Any]]:
        tool_type = "computer_20251124" if self.config.anthropic_is_opus else "computer_20250124"
        tool = {
            "type": tool_type,
            "name": "computer",
            "display_width_px": display_width,
            "display_height_px": display_height,
            "display_number": 1,
        }
        if self.config.anthropic_is_opus:
            tool["enable_zoom"] = True
        return [tool]

    def _get_beta_flags(self) -> List[str]:
        if self.config.anthropic_is_opus:
            betas = [OPUS_COMPUTER_USE_BETA, TOKEN_EFFICIENT_BETA]
            if self.config.anthropic_effort_level != "high":
                betas.append(OPUS_EFFORT_BETA)
            return betas
        return [SONNET_COMPUTER_USE_BETA, SONNET_CONTEXT_MGMT_BETA, TOKEN_EFFICIENT_BETA]

    # ── Context management ───────────────────────────────────────

    def _has_tool_result(self, message: Dict) -> bool:
        if message.get("role") != "user":
            return False
        content = message.get("content", [])
        if not isinstance(content, list):
            return False
        return any(b.get("type") == "tool_result" for b in content if isinstance(b, dict))

    def _is_screenshot_only(self, message: Dict) -> bool:
        if message.get("role") != "user":
            return False
        content = message.get("content", [])
        if not isinstance(content, list):
            return False
        has_image = any(
            isinstance(b, dict)
            and (b.get("type") == "image" or (b.get("type") == "tool_result" and self._has_image_in_content(b)))
            for b in content
        )
        has_text = any(isinstance(b, dict) and b.get("type") == "text" for b in content)
        return has_image and not has_text

    def _has_image_in_content(self, block: Dict) -> bool:
        content = block.get("content", [])
        if isinstance(content, list):
            return any(isinstance(c, dict) and c.get("type") == "image" for c in content)
        return False

    def _trim_screenshot_messages(self, messages: List[Dict]) -> List[Dict]:
        if self.config.anthropic_screenshot_turn_limit <= 0 or not messages:
            return messages
        screenshot_indices = [i for i, m in enumerate(messages) if self._is_screenshot_only(m)]
        if len(screenshot_indices) <= self.config.anthropic_screenshot_turn_limit:
            return messages
        to_remove = set(screenshot_indices[: -self.config.anthropic_screenshot_turn_limit])
        return [m for i, m in enumerate(messages) if i not in to_remove]

    def _trim_conversation_history(self, messages: List[Dict]) -> List[Dict]:
        if not messages or len(messages) <= 2:
            return self._trim_screenshot_messages(messages)

        initial_message = messages[0]
        total_turns = sum(1 for msg in messages if self._has_tool_result(msg))

        if total_turns <= self.config.anthropic_turns_to_keep:
            return self._trim_screenshot_messages(messages)

        needed = self.config.anthropic_turns_to_keep
        idx = len(messages) - 1
        start_index = 1

        while idx >= 1 and needed > 0:
            if self._has_tool_result(messages[idx]):
                j = idx - 1
                while j >= 1 and not any(
                    isinstance(b, dict) and b.get("type") == "tool_use"
                    for b in (messages[j].get("content", []) if isinstance(messages[j].get("content"), list) else [])
                ):
                    j -= 1
                start_index = max(1, j)
                needed -= 1
            idx -= 1

        trimmed = [initial_message] + messages[start_index:]
        return self._trim_screenshot_messages(trimmed)

    def _strip_leading_orphaned_tool_results(self, messages: List[Dict]) -> List[Dict]:
        while messages and self._has_tool_result(messages[0]):
            messages = messages[1:]
        return messages

    # ── Request translation (NeMo-Gym → Anthropic) ──────────────

    def _extract_viewport_from_tools(self, tools: Optional[List]) -> tuple[int, int]:
        """Pull display dimensions from the ``computer_use_preview`` tool definition."""
        if tools:
            for tool in tools:
                tool_dict = tool if isinstance(tool, dict) else (tool.model_dump() if hasattr(tool, "model_dump") else {})
                if tool_dict.get("type") == "computer_use_preview":
                    return (
                        tool_dict.get("display_width", 1280),
                        tool_dict.get("display_height", 720),
                    )
        return 1280, 720

    def _nemo_input_to_anthropic_messages(self, input_items: Any, is_initial: bool = False) -> List[Dict[str, Any]]:
        """Convert NeMo-Gym input items into Anthropic message format."""
        messages: List[Dict[str, Any]] = []

        if isinstance(input_items, str):
            messages.append({"role": "user", "content": [{"type": "text", "text": input_items}]})
            return messages

        if not isinstance(input_items, list):
            return messages

        for item in input_items:
            item_dict = item if isinstance(item, dict) else (item.model_dump(exclude_unset=True) if hasattr(item, "model_dump") else {})

            item_type = item_dict.get("type", "")

            if item_type == "computer_call_output":
                output = item_dict.get("output", {})
                image_url = output.get("image_url", "") if isinstance(output, dict) else ""
                b64_data = self._extract_b64(image_url)

                call_id = item_dict.get("call_id", "")
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64_data},
                        }
                    ],
                }
                messages.append({"role": "user", "content": [tool_result]})

            elif item_dict.get("role") == "user":
                content = item_dict.get("content", "")
                anthropic_content = self._convert_content_blocks(content, is_initial=is_initial)
                if anthropic_content:
                    messages.append({"role": "user", "content": anthropic_content})

        return messages

    def _convert_content_blocks(self, content: Any, is_initial: bool = False) -> List[Dict[str, Any]]:
        """Convert NeMo-Gym content blocks to Anthropic format.

        When ``is_initial`` is True, images are stripped because Anthropic's CUA
        model expects to obtain screenshots via its own ``computer`` tool, not as
        inline images in the user message.
        """
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        if not isinstance(content, list):
            return []

        blocks: List[Dict[str, Any]] = []
        for block in content:
            block_dict = block if isinstance(block, dict) else (block.model_dump(exclude_unset=True) if hasattr(block, "model_dump") else {})
            block_type = block_dict.get("type", "")

            if block_type == "input_text":
                blocks.append({"type": "text", "text": block_dict.get("text", "")})
            elif block_type == "input_image":
                if not is_initial:
                    image_url = block_dict.get("image_url", "")
                    b64_data = self._extract_b64(image_url)
                    blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64_data},
                    })
            elif block_type == "text":
                blocks.append({"type": "text", "text": block_dict.get("text", "")})

        return blocks

    def _extract_b64(self, image_url: str) -> str:
        if image_url.startswith("data:"):
            return image_url.split(",", 1)[-1]
        return image_url

    # ── Response translation (Anthropic → NeMo-Gym) ─────────────

    def _normalize_action_to_openai(self, anthropic_action: Dict[str, Any]) -> Dict[str, Any]:
        """Translate Anthropic action format to OpenAI computer_call action format.

        This ensures the agent's _map_openai_action works uniformly regardless
        of which model server produced the response.
        """
        action_type = anthropic_action.get("action", "")
        coord = anthropic_action.get("coordinate")

        if action_type in ("left_click", "click"):
            return {"type": "click", "x": coord[0] if coord else 0, "y": coord[1] if coord else 0, "button": "left"}
        elif action_type == "right_click":
            return {"type": "click", "x": coord[0] if coord else 0, "y": coord[1] if coord else 0, "button": "right"}
        elif action_type == "middle_click":
            return {"type": "click", "x": coord[0] if coord else 0, "y": coord[1] if coord else 0, "button": "middle"}
        elif action_type == "double_click":
            return {"type": "double_click", "x": coord[0] if coord else 0, "y": coord[1] if coord else 0}
        elif action_type == "triple_click":
            return {"type": "triple_click", "x": coord[0] if coord else 0, "y": coord[1] if coord else 0}
        elif action_type == "type":
            return {"type": "type", "text": anthropic_action.get("text", "")}
        elif action_type == "key":
            keys = anthropic_action.get("text", "").split("+")
            return {"type": "keypress", "keys": keys}
        elif action_type == "mouse_move":
            return {"type": "hover", "x": coord[0] if coord else 0, "y": coord[1] if coord else 0}
        elif action_type == "left_click_drag":
            start = anthropic_action.get("start_coordinate", coord)
            end = anthropic_action.get("coordinate", coord)
            return {
                "type": "drag",
                "start_x": start[0] if start else 0, "start_y": start[1] if start else 0,
                "destination_x": end[0] if end else 0, "destination_y": end[1] if end else 0,
            }
        elif action_type == "scroll":
            direction = anthropic_action.get("direction", "down")
            amount = anthropic_action.get("amount", 3)
            scroll_y = -amount if direction == "up" else amount if direction == "down" else 0
            scroll_x = -amount if direction == "left" else amount if direction == "right" else 0
            return {
                "type": "scroll",
                "x": coord[0] if coord else 0, "y": coord[1] if coord else 0,
                "scroll_x": scroll_x, "scroll_y": scroll_y,
            }
        elif action_type == "screenshot":
            return {"type": "screenshot"}
        elif action_type == "wait":
            return {"type": "wait", "ms": anthropic_action.get("duration", 1000)}
        elif action_type == "zoom":
            return {"type": "zoom", "region": anthropic_action.get("region", [])}
        elif action_type == "hold_key":
            return {"type": "keypress", "keys": [anthropic_action.get("key", "")]}
        else:
            logger.warning("Unknown Anthropic action '%s', passing through as-is", action_type)
            return anthropic_action

    def _anthropic_response_to_nemo(
        self, response, session_id: str
    ) -> NeMoGymResponse:
        """Convert an Anthropic response to NeMoGymResponse format."""
        output_items: List[Dict[str, Any]] = []

        for block in response.content:
            if block.type == "tool_use":
                action = self._normalize_action_to_openai(block.input or {})
                output_items.append({
                    "type": "computer_call",
                    "id": f"cu_{block.id}",
                    "call_id": block.id,
                    "action": action,
                    "pending_safety_checks": [],
                    "status": "completed",
                })
            elif block.type == "text":
                output_items.append({
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:12]}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": block.text, "annotations": []}],
                    "status": "completed",
                })

        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0 if response.usage else 0
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0 if response.usage else 0

        usage_data = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {
                "cached_tokens": cache_read + cache_creation,
            },
            "output_tokens_details": {
                "reasoning_tokens": 0,
            },
        }

        response_dict = {
            "id": session_id,
            "created_at": 0,
            "model": response.model,
            "object": "response",
            "output": output_items,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": usage_data,
        }

        try:
            return NeMoGymResponse.model_validate(response_dict)
        except Exception as e:
            logger.error("NeMoGymResponse validation failed for Anthropic response: %s", repr(e))
            raise

    # ── Core API ─────────────────────────────────────────────────

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = body.model_dump(exclude_unset=True)

        previous_response_id = body_dict.get("previous_response_id")
        display_w, display_h = self._extract_viewport_from_tools(body_dict.get("tools"))

        if previous_response_id and previous_response_id in self._sessions:
            session_id = previous_response_id
            session = self._sessions[session_id]
        else:
            session_id = f"anthropic_session_{uuid.uuid4().hex}"
            session = _Session()
            self._sessions[session_id] = session

        is_new_session = not session.messages
        new_messages = self._nemo_input_to_anthropic_messages(body_dict.get("input", []), is_initial=is_new_session)

        if not session.messages and new_messages:
            session.messages = new_messages
        else:
            session.messages.extend(new_messages)

        trimmed = self._trim_conversation_history(list(session.messages))
        trimmed = self._strip_leading_orphaned_tool_results(trimmed)

        api_params = {
            "model": self.config.anthropic_model,
            "max_tokens": self.config.anthropic_max_tokens,
            "system": ANTHROPIC_CUA_SYSTEM_PROMPT,
            "tools": self._get_tools(display_w, display_h),
            "betas": self._get_beta_flags(),
            "messages": trimmed,
        }

        if self.config.anthropic_is_opus and self.config.anthropic_effort_level != "high":
            api_params["output_config"] = {"effort": self.config.anthropic_effort_level}

        logger.info(
            "Anthropic API request: model=%s, tools=%s, betas=%s, num_messages=%d, system_len=%d",
            api_params["model"],
            [t.get("type") for t in api_params["tools"]],
            api_params["betas"],
            len(api_params["messages"]),
            len(api_params.get("system", "")),
        )
        for idx, msg in enumerate(api_params["messages"]):
            content_summary = []
            for b in (msg.get("content", []) if isinstance(msg.get("content"), list) else []):
                if isinstance(b, dict):
                    content_summary.append(b.get("type", "?"))
            logger.info("  msg[%d] role=%s content_types=%s", idx, msg.get("role"), content_summary)

        try:
            response = self._client.beta.messages.create(**api_params)
        except Exception as e:
            logger.error("Anthropic API call failed: %s", repr(e))
            raise

        logger.info(
            "Anthropic API response: stop_reason=%s, content_blocks=%d",
            response.stop_reason,
            len(response.content),
        )
        for idx, block in enumerate(response.content):
            if block.type == "tool_use":
                logger.info("  block[%d] type=tool_use name=%s input=%s", idx, block.name, block.input)
            elif block.type == "text":
                logger.info("  block[%d] type=text text=%s", idx, block.text[:200])

        assistant_content = [
            {"type": b.type, **(b.model_dump() if hasattr(b, "model_dump") else {})}
            for b in response.content
        ]
        session.messages.append({"role": "assistant", "content": assistant_content})

        session.pending_tool_use_ids = [
            b.id for b in response.content if b.type == "tool_use"
        ]

        nemo_response = self._anthropic_response_to_nemo(response, session_id)

        if response.stop_reason == "end_turn" and not session.pending_tool_use_ids:
            self._sessions.pop(session_id, None)

        return nemo_response

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError("Anthropic model server does not support /v1/chat/completions for CUA")


if __name__ == "__main__":
    AnthropicModelServer.run_webserver()
