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
1. _trim_conversation_history: preserve initial task, keep last N turns, validate tool_use/tool_result pairs
2. _trim_screenshot_messages: cap screenshot-only messages
3. _strip_leading_orphaned_tool_results: ensure valid message ordering
4. _gc_old_screenshots: replace base64 data in messages outside the trim window with a placeholder

All API calls are routed through an injected api_caller (model server proxy).
"""

import copy
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from resources_servers.browser_gym.schemas import BrowserAction
from responses_api_agents.browser_agent.adapters.base import (
    BaseCUAAdapter,
    CUAAdapterResponse,
    CUAAdapterUsage,
    extract_token_ids_from_response,
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

AUTONOMOUS EXECUTION:
- NEVER ask for user confirmation, approval, or permission before taking an action.
- Execute all actions directly and autonomously.
- Do not pause to verify your plan with the user — just execute it.

IMPORTANT: Do not include extra user messages between paired tool_use and tool_result. \
Screenshots may be included inside tool_result content when needed for verification."""

SCREENSHOT_PLACEHOLDER = "[screenshot-trimmed]"

ApiCaller = Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]]


class AnthropicCUAAdapter(BaseCUAAdapter):
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        is_opus: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        max_tokens: int = 4096,
        turns_to_keep: int = 8,
        screenshot_turn_limit: int = 8,
        effort_level: str = "high",
        api_caller: Optional[ApiCaller] = None,
    ):
        self._model = model
        self._is_opus = is_opus
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._max_tokens = max_tokens
        self._turns_to_keep = turns_to_keep
        self._screenshot_turn_limit = screenshot_turn_limit
        self._effort_level = effort_level
        self._api_caller = api_caller
        self._messages: List[Dict[str, Any]] = []
        self._system_prompt: str = ""
        self._pending_tool_use_ids: List[str] = []
        self._last_raw_response: Dict[str, Any] = {}

    # ── Tool / beta helpers ──────────────────────────────────────

    def _get_tools(self) -> List[Dict[str, Any]]:
        tool_type = "computer_20251124" if self._is_opus else "computer_20250124"
        tool = {
            "type": tool_type,
            "name": "computer",
            "display_width_px": self._viewport_width,
            "display_height_px": self._viewport_height,
            "display_number": 1,
        }
        if self._is_opus:
            tool["enable_zoom"] = True
        return [tool]

    def _get_beta_flags(self) -> List[str]:
        if self._is_opus:
            betas = [OPUS_COMPUTER_USE_BETA, TOKEN_EFFICIENT_BETA]
            if self._effort_level != "high":
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
        if self._screenshot_turn_limit <= 0 or not messages:
            return messages

        screenshot_indices = [i for i, m in enumerate(messages) if self._is_screenshot_only(m)]
        if len(screenshot_indices) <= self._screenshot_turn_limit:
            return messages

        to_remove = set(screenshot_indices[: -self._screenshot_turn_limit])
        return [m for i, m in enumerate(messages) if i not in to_remove]

    def _trim_conversation_history(self, messages: List[Dict]) -> List[Dict]:
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
        while messages and self._has_tool_result(messages[0]):
            messages = messages[1:]
        return messages

    def _validate_tool_pairs(self, messages: List[Dict]) -> List[Dict]:
        """Ensure every assistant tool_use has a matching user tool_result and vice versa.

        Walks the message list and collects tool_use IDs from assistant messages.
        If the next user message doesn't provide tool_results for all of them,
        both the assistant and user messages are dropped to maintain validity.
        """
        result = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                tool_use_ids = set()
                if isinstance(content, list):
                    tool_use_ids = {
                        b["id"] for b in content if isinstance(b, dict) and b.get("type") == "tool_use" and "id" in b
                    }

                if not tool_use_ids:
                    result.append(msg)
                    i += 1
                    continue

                if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                    next_content = messages[i + 1].get("content", [])
                    result_ids = set()
                    if isinstance(next_content, list):
                        result_ids = {
                            b.get("tool_use_id")
                            for b in next_content
                            if isinstance(b, dict) and b.get("type") == "tool_result"
                        }

                    if tool_use_ids <= result_ids:
                        result.append(msg)
                        result.append(messages[i + 1])
                        i += 2
                        continue

                logger.warning("Dropping orphaned tool_use assistant message (ids=%s)", tool_use_ids)
                i += 1
            else:
                result.append(msg)
                i += 1

        return result

    # ── Memory management ────────────────────────────────────────

    def _gc_old_screenshots(self, trim_window_size: int) -> None:
        """Replace base64 image data in messages outside the recent trim window with a placeholder.

        Only touches self._messages in-place. The trimming step will discard these
        messages anyway, so replacing the data has no effect on API calls — it just
        frees memory.

        Always skips messages[0] (the initial task message) because
        _trim_conversation_history always preserves it.
        """
        if len(self._messages) <= trim_window_size:
            return

        cutoff = len(self._messages) - trim_window_size
        for msg in self._messages[1:cutoff]:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "image":
                    source = block.get("source", {})
                    if isinstance(source, dict) and source.get("type") == "base64":
                        source["data"] = SCREENSHOT_PLACEHOLDER
                elif block.get("type") == "tool_result":
                    inner = block.get("content", [])
                    if isinstance(inner, list):
                        for inner_block in inner:
                            if isinstance(inner_block, dict) and inner_block.get("type") == "image":
                                source = inner_block.get("source", {})
                                if isinstance(source, dict) and source.get("type") == "base64":
                                    source["data"] = SCREENSHOT_PLACEHOLDER

    # ── API call lifecycle ───────────────────────────────────────

    def _prepare_api_params(self) -> Dict[str, Any]:
        """Build trimmed, ready-to-send Anthropic API params."""
        self._gc_old_screenshots(self._turns_to_keep * 2 + 2)

        trimmed = self._trim_conversation_history(copy.deepcopy(self._messages))
        trimmed = self._strip_leading_orphaned_tool_results(trimmed)
        trimmed = self._validate_tool_pairs(trimmed)

        params: Dict[str, Any] = {
            "max_tokens": self._max_tokens,
            "system": self._system_prompt,
            "tools": self._get_tools(),
            "betas": self._get_beta_flags(),
            "messages": trimmed,
        }

        if self._is_opus and self._effort_level != "high":
            params["output_config"] = {"effort": self._effort_level}

        return params

    async def _execute_api_call(self, api_params: Dict[str, Any]):
        """Route API call through the injected model server proxy."""
        if not self._api_caller:
            raise RuntimeError("AnthropicCUAAdapter requires an api_caller (model server proxy). No direct API calls.")
        raw = await self._api_caller(api_params)
        if isinstance(raw, dict):
            self._last_raw_response = raw
            from anthropic.types.beta import BetaMessage

            return BetaMessage.model_validate(raw)
        self._last_raw_response = {}
        return raw

    def _update_history_from_response(self, response):
        """Append assistant response to conversation history and track pending tool IDs."""
        assistant_content = [
            {"type": b.type, **(b.model_dump() if hasattr(b, "model_dump") else {})} for b in response.content
        ]
        self._messages.append({"role": "assistant", "content": assistant_content})

        self._pending_tool_use_ids = [b.id for b in response.content if b.type == "tool_use"]

    # ── Response parsing ─────────────────────────────────────────

    def _parse_response(self, response) -> CUAAdapterResponse:
        actions: List[BrowserAction] = []
        message: Optional[str] = None
        done = False

        for block in response.content:
            if block.type == "tool_use":
                action_input = block.input or {}
                browser_action = self._map_anthropic_action(action_input)
                if browser_action:
                    actions.append(browser_action)
            elif block.type == "text":
                message = block.text

        if response.stop_reason == "end_turn" and not actions:
            done = True

        raw = {"id": response.id, "model": response.model, "stop_reason": response.stop_reason}

        usage = None
        resp_usage = getattr(response, "usage", None)
        if resp_usage:
            usage = CUAAdapterUsage(
                input_tokens=getattr(resp_usage, "input_tokens", 0),
                output_tokens=getattr(resp_usage, "output_tokens", 0),
                total_tokens=getattr(resp_usage, "input_tokens", 0) + getattr(resp_usage, "output_tokens", 0),
            )

        token_ids = extract_token_ids_from_response(getattr(self, "_last_raw_response", {}))

        return CUAAdapterResponse(
            actions=actions,
            message=message,
            raw_response=raw,
            done=done,
            usage=usage,
            prompt_token_ids=token_ids["prompt_token_ids"],
            generation_token_ids=token_ids["generation_token_ids"],
            generation_log_probs=token_ids["generation_log_probs"],
        )

    def _map_anthropic_action(self, action: Dict[str, Any]) -> Optional[BrowserAction]:
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
        elif action_type == "triple_click":
            return BrowserAction(action_type="triple_click", coordinate=action.get("coordinate"))
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
            return BrowserAction(action_type="zoom", region=action.get("region"))
        else:
            logger.warning(f"Unknown Anthropic action: {action_type}")
            return None

    # ── Public interface ─────────────────────────────────────────

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

        api_params = self._prepare_api_params()

        logger.info(
            "Anthropic API request: model=%s, num_messages=%d, betas=%s",
            api_params.get("model", self._model),
            len(api_params["messages"]),
            api_params["betas"],
        )

        response = await self._execute_api_call(api_params)
        self._update_history_from_response(response)

        logger.info(
            "Anthropic API response: stop_reason=%s, content_blocks=%d",
            response.stop_reason,
            len(response.content),
        )

        return self._parse_response(response)

    async def step(self, screenshot_b64: str, action_result: Optional[str] = None) -> CUAAdapterResponse:
        current_tool_ids = self._pending_tool_use_ids
        self._pending_tool_use_ids = []

        tool_results = []
        for tool_id in current_tool_ids:
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

        api_params = self._prepare_api_params()
        response = await self._execute_api_call(api_params)
        self._update_history_from_response(response)
        return self._parse_response(response)

    def reset(self):
        self._messages = []
        self._pending_tool_use_ids = []
        self._system_prompt = ""
        self._last_raw_response = {}
