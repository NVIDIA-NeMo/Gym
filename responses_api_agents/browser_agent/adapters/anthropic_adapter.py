# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Anthropic CUA adapter (Sonnet + Opus).

Context management: Client-side turn-based trimming.
Full conversation history maintained in messages list, trimmed before each API call:
1. _trim_conversation_history: preserve initial task, keep last N turns, never split tool_use/tool_result pairs
2. _trim_screenshot_messages: cap screenshot-only messages
3. _strip_leading_orphaned_tool_results: ensure valid message ordering
"""

import base64
import logging
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

from resources_servers.browser_gym.schemas import BrowserAction
from responses_api_agents.browser_agent.adapters.base import BaseCUAAdapter, CUAAdapterResponse

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


class AnthropicCUAAdapter(BaseCUAAdapter):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        is_opus: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        max_tokens: int = 4096,
        turns_to_keep: int = 8,
        screenshot_turn_limit: int = 8,
        effort_level: str = "high",
    ):
        self._client = Anthropic(api_key=api_key, max_retries=4)
        self._model = model
        self._is_opus = is_opus
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._max_tokens = max_tokens
        self._turns_to_keep = turns_to_keep
        self._screenshot_turn_limit = screenshot_turn_limit
        self._effort_level = effort_level

        self._messages: List[Dict[str, Any]] = []
        self._system_prompt: str = ""
        self._pending_tool_use_ids: List[str] = []

    def _get_tools(self) -> List[Dict[str, Any]]:
        tool_type = "computer_20251124" if self._is_opus else "computer_20250124"
        return [
            {
                "type": tool_type,
                "name": "computer",
                "display_width_px": self._viewport_width,
                "display_height_px": self._viewport_height,
                "display_number": 1,
            }
        ]

    def _get_beta_flags(self) -> List[str]:
        if self._is_opus:
            betas = [OPUS_COMPUTER_USE_BETA, TOKEN_EFFICIENT_BETA]
            if self._effort_level != "high":
                betas.append(OPUS_EFFORT_BETA)
            return betas
        return [SONNET_COMPUTER_USE_BETA, SONNET_CONTEXT_MGMT_BETA, TOKEN_EFFICIENT_BETA]

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
        """Cap screenshot-only user messages to the configured limit."""
        if self._screenshot_turn_limit <= 0 or not messages:
            return messages

        screenshot_indices = [i for i, m in enumerate(messages) if self._is_screenshot_only(m)]
        if len(screenshot_indices) <= self._screenshot_turn_limit:
            return messages

        to_remove = set(screenshot_indices[: -self._screenshot_turn_limit])
        return [m for i, m in enumerate(messages) if i not in to_remove]

    def _trim_conversation_history(self, messages: List[Dict]) -> List[Dict]:
        """Keep initial task message + last N complete turns. Never split tool_use/tool_result pairs."""
        if not messages or len(messages) <= 2:
            return self._trim_screenshot_messages(messages)

        initial_message = messages[0]
        total_turns = sum(1 for msg in messages if self._has_tool_result(msg))

        if total_turns <= self._turns_to_keep:
            return self._trim_screenshot_messages(messages)

        needed = self._turns_to_keep
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
        """Ensure messages list never starts with a tool_result."""
        while messages and self._has_tool_result(messages[0]):
            messages = messages[1:]
        return messages

    def _parse_response(self, response) -> CUAAdapterResponse:
        """Parse Anthropic response into CUAAdapterResponse."""
        actions: List[BrowserAction] = []
        message: Optional[str] = None
        done = False
        self._pending_tool_use_ids = []

        for block in response.content:
            if block.type == "tool_use":
                self._pending_tool_use_ids.append(block.id)
                action_input = block.input or {}
                browser_action = self._map_anthropic_action(action_input)
                if browser_action:
                    actions.append(browser_action)
            elif block.type == "text":
                message = block.text

        if response.stop_reason == "end_turn" and not actions:
            done = True

        raw = {"id": response.id, "model": response.model, "stop_reason": response.stop_reason}
        return CUAAdapterResponse(actions=actions, message=message, raw_response=raw, done=done)

    def _map_anthropic_action(self, action: Dict[str, Any]) -> Optional[BrowserAction]:
        """Map Anthropic action to unified BrowserAction."""
        action_type = action.get("action", "")

        if action_type in ("left_click", "click"):
            coord = action.get("coordinate")
            return BrowserAction(action_type="click", coordinate=coord, button="left")
        elif action_type == "right_click":
            return BrowserAction(action_type="right_click", coordinate=action.get("coordinate"))
        elif action_type == "middle_click":
            return BrowserAction(action_type="middle_click", coordinate=action.get("coordinate"))
        elif action_type == "double_click":
            return BrowserAction(action_type="double_click", coordinate=action.get("coordinate"))
        elif action_type == "type":
            return BrowserAction(action_type="type", text=action.get("text", ""))
        elif action_type == "key":
            return BrowserAction(action_type="keypress", key=action.get("text", ""))
        elif action_type == "mouse_move":
            return BrowserAction(action_type="hover", coordinate=action.get("coordinate"))
        elif action_type == "left_click_drag":
            return BrowserAction(
                action_type="drag",
                start_coordinate=action.get("start_coordinate"),
                end_coordinate=action.get("coordinate"),
            )
        elif action_type == "scroll":
            return BrowserAction(
                action_type="scroll",
                coordinate=action.get("coordinate"),
                scroll_direction=action.get("direction"),
                scroll_amount=action.get("amount"),
            )
        elif action_type == "screenshot":
            return BrowserAction(action_type="screenshot")
        elif action_type == "wait":
            return BrowserAction(action_type="wait", duration=action.get("duration", 1000))
        elif action_type == "hold_key":
            return BrowserAction(action_type="keypress", key=action.get("key", ""), duration=action.get("duration"))
        elif action_type == "zoom":
            return BrowserAction(action_type="screenshot", region=action.get("region"))
        else:
            logger.warning(f"Unknown Anthropic action: {action_type}")
            return None

    async def initialize(self, task_prompt: str, screenshot_b64: str) -> CUAAdapterResponse:
        self._messages = []
        self._system_prompt = ANTHROPIC_CUA_SYSTEM_PROMPT
        self._pending_tool_use_ids = []

        self._messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                ],
            }
        )

        return await self._call_api()

    async def step(self, screenshot_b64: str, action_result: Optional[str] = None) -> CUAAdapterResponse:
        tool_results = []
        for tool_id in self._pending_tool_use_ids:
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64,
                            },
                        }
                    ],
                }
            )

        if tool_results:
            self._messages.append({"role": "user", "content": tool_results})
        else:
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64},
                        }
                    ],
                }
            )

        return await self._call_api()

    async def _call_api(self) -> CUAAdapterResponse:
        trimmed = self._trim_conversation_history(list(self._messages))
        trimmed = self._strip_leading_orphaned_tool_results(trimmed)

        api_params = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": self._system_prompt,
            "tools": self._get_tools(),
            "betas": self._get_beta_flags(),
            "messages": trimmed,
        }

        if self._is_opus and self._effort_level != "high":
            api_params["output_config"] = {"effort": self._effort_level}

        response = self._client.beta.messages.create(**api_params)

        assistant_content = [{"type": b.type, **(b.model_dump() if hasattr(b, "model_dump") else {})} for b in response.content]
        self._messages.append({"role": "assistant", "content": assistant_content})

        return self._parse_response(response)

    def reset(self):
        self._messages = []
        self._pending_tool_use_ids = []
        self._system_prompt = ""
