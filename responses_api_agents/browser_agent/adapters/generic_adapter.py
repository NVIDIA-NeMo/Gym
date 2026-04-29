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
Generic CUA adapter for providers whose model servers speak OpenAI Responses API format.

Context management: Client-side. Maintains a list of OpenAI input items
(messages, computer_call, computer_call_output) and sends the full (trimmed)
list on each request. The model server handles translation to/from the
provider's native API.

Used with: Anthropic model server, Gemini model server, and any future
provider that conforms to the NeMoGymResponseCreateParamsNonStreaming /
NeMoGymResponse interface.
"""

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

SCREENSHOT_PLACEHOLDER = "[screenshot-trimmed]"

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

GEMINI_CUA_SYSTEM_PROMPT = """\
You are a browser automation agent. Your goal is to complete tasks efficiently using the Computer Use tool.

**Context Window:**
- You only have access to the last 8 conversation turns (due to token limits)
- Once you complete a subtask, MOVE FORWARD to the next one
- DO NOT revisit or verify already-completed subtasks
- If you lose context, check the current page state and continue from where you are

**Sequential Task Execution:**
- Break down the task into sequential steps
- Complete each step ONCE, then move to the next step
- NEVER return to a previous step unless explicitly required by the task
- Track your progress mentally: "I completed X, now doing Y, next is Z"
- If a form is already filled, skip it and move to the next requirement

**Avoiding Loops:**
- If you find yourself on the same page/form twice, you are in a LOOP - move forward immediately
- Do not re-verify or re-check completed work
- If stuck after 2-3 attempts on one step, skip it or try a completely different approach
- Progress forward is more important than perfect execution

**Key Guidelines:**
- Act decisively: Don't overthink simple UI interactions
- Be direct: Click, type, and navigate without excessive analysis
- Stay focused: Complete the current step before planning ahead
- Be efficient: Minimize the number of actions needed
- CRITICAL: Never repeat the same subtask twice

AUTONOMOUS EXECUTION:
- NEVER ask for user confirmation, approval, or permission before taking an action.
- Execute all actions directly and autonomously.
- Do not pause to verify your plan with the user — just execute it.

Remember: Speed and forward progress matter most. Complete the task sequentially \
without revisiting completed subtasks."""

ApiCaller = Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]]


class GenericCUAAdapter(BaseCUAAdapter):
    """Provider-agnostic CUA adapter with client-side context management.

    The adapter builds OpenAI-format input items and sends them to a model
    server that accepts NeMoGymResponseCreateParamsNonStreaming and returns
    NeMoGymResponse. The model server handles all provider-specific
    translation.

    Args:
        model: Model identifier (passed in the request but the server may override it).
        viewport_width: Browser viewport width in pixels.
        viewport_height: Browser viewport height in pixels.
        turns_to_keep: Max number of computer_call/computer_call_output pairs to retain.
        denormalize_coords: If True, convert 0-999 normalized coordinates to pixel values (Gemini).
        system_prompt: System prompt sent as ``instructions`` in every request.
        api_caller: Async callable that posts the request dict to the model server.
    """

    def __init__(
        self,
        model: str = "",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        turns_to_keep: int = 8,
        denormalize_coords: bool = False,
        system_prompt: str = "",
        api_caller: Optional[ApiCaller] = None,
    ):
        self._model = model
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._turns_to_keep = turns_to_keep
        self._denormalize_coords = denormalize_coords
        self._system_prompt = system_prompt
        self._api_caller = api_caller

        self._input_items: List[Dict[str, Any]] = []
        self._pending_call_ids: List[str] = []
        self._pending_safety_checks: List[Dict[str, Any]] = []

    # ── Context management ───────────────────────────────────────

    def _trim_input_items(self) -> None:
        """Trim input items to keep first message + last N call/output pairs."""
        if len(self._input_items) <= 2:
            return

        pair_count = sum(
            1 for item in self._input_items if isinstance(item, dict) and item.get("type") == "computer_call_output"
        )
        if pair_count <= self._turns_to_keep:
            return

        first_item = self._input_items[0]
        remaining = self._input_items[1:]

        keep_count = self._turns_to_keep * 2
        if len(remaining) > keep_count:
            remaining = remaining[-keep_count:]

        self._input_items = [first_item] + remaining

    def _gc_old_screenshots(self) -> None:
        """Replace base64 image data outside the recent window with a placeholder."""
        window = self._turns_to_keep * 2
        if len(self._input_items) <= window + 1:
            return

        cutoff = len(self._input_items) - window
        for item in self._input_items[1:cutoff]:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")

            if item_type == "computer_call_output":
                output = item.get("output", {})
                if isinstance(output, dict) and "image_url" in output:
                    output["image_url"] = SCREENSHOT_PLACEHOLDER

            elif item_type == "message":
                content = item.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "input_image":
                            part["image_url"] = SCREENSHOT_PLACEHOLDER

    # ── Coordinate helpers ───────────────────────────────────────

    def _denorm_x(self, x: int) -> int:
        return int((x / 1000.0) * self._viewport_width)

    def _denorm_y(self, y: int) -> int:
        return int((y / 1000.0) * self._viewport_height)

    # ── Action mapping ───────────────────────────────────────────

    def _map_action(self, action: Dict[str, Any]) -> Optional[BrowserAction]:
        """Map a computer_call action dict to BrowserAction.

        Handles both Anthropic-style actions (keyed by "action") and
        Gemini/OpenAI-style actions (keyed by "type").
        """
        if "action" in action and "type" not in action:
            return self._map_anthropic_action(action)
        return self._map_openai_gemini_action(action)

    def _map_anthropic_action(self, action: Dict[str, Any]) -> Optional[BrowserAction]:
        """Map Anthropic computer-use action format to BrowserAction."""
        action_type = action.get("action", "")

        if action_type in ("left_click", "click"):
            return BrowserAction(action_type="click", coordinate=action.get("coordinate"), button="left")
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
            logger.warning("Unknown Anthropic action: %s", action_type)
            return None

    def _map_openai_gemini_action(self, action: Dict[str, Any]) -> Optional[BrowserAction]:
        """Map OpenAI/Gemini-style action (keyed by 'type') to BrowserAction."""
        action_type = action.get("type", "")

        if action_type == "click":
            coord = self._get_coord(action)
            return BrowserAction(action_type="click", coordinate=coord, button=action.get("button", "left"))
        elif action_type == "click_at":
            coord = self._get_gemini_coord(action)
            return BrowserAction(action_type="click", coordinate=coord)
        elif action_type == "hover_at":
            coord = self._get_gemini_coord(action)
            return BrowserAction(action_type="hover", coordinate=coord)
        elif action_type in ("double_click",):
            return BrowserAction(action_type="double_click", coordinate=self._get_coord(action))
        elif action_type in ("triple_click",):
            return BrowserAction(action_type="triple_click", coordinate=self._get_coord(action))
        elif action_type in ("type", "type_text_at"):
            coord = self._get_gemini_coord(action) if action_type == "type_text_at" else None
            return BrowserAction(
                action_type="type",
                coordinate=coord,
                text=action.get("text", ""),
                press_enter=action.get("press_enter", False),
                clear_before_typing=action.get("clear_before_typing", True),
            )
        elif action_type == "scroll":
            return BrowserAction(
                action_type="scroll",
                coordinate=self._get_coord(action),
                scroll_x=action.get("scroll_x"),
                scroll_y=action.get("scroll_y"),
            )
        elif action_type == "scroll_document":
            direction = action.get("direction", "down")
            vh = self._viewport_height
            vw = self._viewport_width
            amount = int(vh * 0.8) if direction in ("up", "down") else vw // 2
            return BrowserAction(
                action_type="scroll",
                scroll_direction=direction,
                scroll_x=(-amount if direction == "left" else amount if direction == "right" else 0),
                scroll_y=(-amount if direction == "up" else amount if direction == "down" else 0),
            )
        elif action_type == "scroll_at":
            coord = self._get_gemini_coord(action)
            direction = action.get("direction", "down")
            magnitude = int(action.get("magnitude", 800))
            actual_mag = self._denorm_y(magnitude) if direction in ("up", "down") else self._denorm_x(magnitude)
            dx, dy = 0, 0
            if direction == "up":
                dy = -actual_mag
            elif direction == "down":
                dy = actual_mag
            elif direction == "left":
                dx = -actual_mag
            elif direction == "right":
                dx = actual_mag
            return BrowserAction(
                action_type="scroll", coordinate=coord, scroll_direction=direction, scroll_x=dx, scroll_y=dy
            )
        elif action_type in ("keypress", "key_combination"):
            keys = action.get("keys", action.get("key", []))
            if isinstance(keys, str):
                keys = keys.split("+") if "+" in keys else keys.split(",") if "," in keys else [keys]
            return BrowserAction(action_type="keypress", keys=list(keys) if keys else [])
        elif action_type in ("move", "mouse_move", "hover"):
            return BrowserAction(action_type="hover", coordinate=self._get_coord(action))
        elif action_type == "screenshot":
            return BrowserAction(action_type="screenshot")
        elif action_type == "wait":
            duration = action.get("ms") or action.get("duration") or 1000
            return BrowserAction(action_type="wait", duration=int(duration))
        elif action_type == "wait_5_seconds":
            return BrowserAction(action_type="wait", duration=5000)
        elif action_type in ("goto", "navigate"):
            url = action.get("url", "")
            if url and not url.startswith(("http://", "https://")):
                url = "https://" + url
            return BrowserAction(action_type="goto", url=url)
        elif action_type == "go_back":
            return BrowserAction(action_type="go_back")
        elif action_type == "go_forward":
            return BrowserAction(action_type="go_forward")
        elif action_type == "search":
            return BrowserAction(action_type="goto", url="https://www.google.com")
        elif action_type == "new_tab":
            return BrowserAction(action_type="new_tab", url=action.get("url", ""))
        elif action_type == "switch_tab":
            idx = action.get("tab_index", action.get("index", 0))
            return BrowserAction(action_type="switch_tab", tab_index=int(idx))
        elif action_type == "close_tab":
            return BrowserAction(action_type="close_tab")
        elif action_type == "drag":
            start = self._get_coord(action) or (
                [int(action["start_x"]), int(action["start_y"])]
                if "start_x" in action and "start_y" in action
                else action.get("start_coordinate")
            )
            end = (
                [int(action["destination_x"]), int(action["destination_y"])]
                if "destination_x" in action and "destination_y" in action
                else action.get("destination_coordinate") or action.get("target_coordinate")
            )
            return BrowserAction(action_type="drag", start_coordinate=start, end_coordinate=end)
        elif action_type == "drag_and_drop":
            sx = (
                self._denorm_x(int(action.get("x", action.get("start_x", 0))))
                if self._denormalize_coords
                else int(action.get("x", action.get("start_x", 0)))
            )
            sy = (
                self._denorm_y(int(action.get("y", action.get("start_y", 0))))
                if self._denormalize_coords
                else int(action.get("y", action.get("start_y", 0)))
            )
            ex = (
                self._denorm_x(int(action.get("destination_x", action.get("end_x", 0))))
                if self._denormalize_coords
                else int(action.get("destination_x", action.get("end_x", 0)))
            )
            ey = (
                self._denorm_y(int(action.get("destination_y", action.get("end_y", 0))))
                if self._denormalize_coords
                else int(action.get("destination_y", action.get("end_y", 0)))
            )
            return BrowserAction(action_type="drag", start_coordinate=[sx, sy], end_coordinate=[ex, ey])
        elif action_type in ("list_tabs", "open_web_browser", "WebAgentState", "web_agent_state"):
            return BrowserAction(action_type="screenshot")
        else:
            logger.warning("Unknown action type: %s", action_type)
            return None

    def _get_coord(self, action: Dict[str, Any]) -> Optional[List[int]]:
        """Extract pixel coordinates — handles both x/y and coordinate formats."""
        if "x" in action and "y" in action:
            return [int(action["x"]), int(action["y"])]
        if "coordinate" in action:
            return action["coordinate"]
        return None

    def _get_gemini_coord(self, action: Dict[str, Any]) -> Optional[List[int]]:
        """Extract coordinates, applying 0-999 → pixel denormalization if configured."""
        if "x" not in action or "y" not in action:
            return self._get_coord(action)
        x, y = int(action["x"]), int(action["y"])
        if self._denormalize_coords:
            return [self._denorm_x(x), self._denorm_y(y)]
        return [x, y]

    # ── Response parsing ─────────────────────────────────────────

    def _parse_response(self, response: Dict[str, Any]) -> CUAAdapterResponse:
        """Parse a NeMoGymResponse dict into CUAAdapterResponse."""
        output = response.get("output", [])
        actions: List[BrowserAction] = []
        message: Optional[str] = None
        done = False
        pending_call_ids: List[str] = []
        pending_safety_checks: List[Dict[str, Any]] = []

        for item in output:
            item_type = item.get("type")

            if item_type == "computer_call":
                call_id = item.get("call_id", "")
                pending_call_ids.append(call_id)
                safety_checks = item.get("pending_safety_checks", [])
                if safety_checks:
                    pending_safety_checks.extend(safety_checks)
                action_data = item.get("action", {})
                browser_action = self._map_action(action_data)
                if browser_action:
                    actions.append(browser_action)

                self._input_items.append(item)

            elif item_type == "message" and item.get("role") == "assistant":
                content = item.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        message = block.get("text", "")

        if not actions and message is not None:
            done = True

        self._pending_call_ids = pending_call_ids
        self._pending_safety_checks = pending_safety_checks

        usage = None
        resp_usage = response.get("usage")
        if resp_usage and isinstance(resp_usage, dict):
            in_tok = resp_usage.get("input_tokens", 0) or 0
            out_tok = resp_usage.get("output_tokens", 0) or 0
            usage = CUAAdapterUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok)

        token_ids = extract_token_ids_from_response(response)

        return CUAAdapterResponse(
            actions=actions,
            message=message,
            raw_response=response,
            done=done,
            usage=usage,
            prompt_token_ids=token_ids["prompt_token_ids"],
            generation_token_ids=token_ids["generation_token_ids"],
            generation_log_probs=token_ids["generation_log_probs"],
        )

    # ── API call lifecycle ───────────────────────────────────────

    def _build_request(self) -> Dict[str, Any]:
        """Build a NeMoGymResponseCreateParamsNonStreaming-compatible request dict."""
        return {
            "model": self._model,
            "input": self._input_items,
            "instructions": self._system_prompt or None,
            "tools": [
                {
                    "type": "computer_use_preview",
                    "display_width": self._viewport_width,
                    "display_height": self._viewport_height,
                    "environment": "browser",
                }
            ],
            "truncation": "auto",
        }

    async def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._api_caller:
            raise RuntimeError("GenericCUAAdapter requires an api_caller (model server proxy).")
        return await self._api_caller(payload)

    # ── Public interface ─────────────────────────────────────────

    async def initialize(self, task_prompt: str, screenshot_b64: str) -> CUAAdapterResponse:
        self._input_items = []
        self._pending_call_ids = []
        self._pending_safety_checks = []

        initial_message = {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": task_prompt},
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                },
            ],
        }
        self._input_items = [initial_message]

        payload = self._build_request()
        response = await self._call_api(payload)
        return self._parse_response(response)

    async def step(
        self, screenshot_b64: str, action_result: Optional[str] = None, action_error: Optional[str] = None
    ) -> CUAAdapterResponse:
        for call_id in self._pending_call_ids:
            output_dict: Dict[str, Any] = {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:image/png;base64,{screenshot_b64}",
            }
            if action_result:
                output_dict["current_url"] = action_result
            if action_error:
                output_dict["current_url"] = f"error: {action_error}"

            call_output: Dict[str, Any] = {
                "type": "computer_call_output",
                "call_id": call_id,
                "output": output_dict,
            }
            if self._pending_safety_checks:
                call_output["acknowledged_safety_checks"] = self._pending_safety_checks

            self._input_items.append(call_output)

        self._pending_call_ids = []
        self._pending_safety_checks = []

        if not self._input_items:
            return CUAAdapterResponse(done=True)

        self._gc_old_screenshots()
        self._trim_input_items()

        payload = self._build_request()
        response = await self._call_api(payload)
        return self._parse_response(response)

    def reset(self):
        self._input_items = []
        self._pending_call_ids = []
        self._pending_safety_checks = []
