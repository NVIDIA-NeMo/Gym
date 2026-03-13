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
OpenAI CUA adapter.

Context management: Server-side via previous_response_id.
No client-side conversation history is maintained. OpenAI manages full context
server-side with truncation="auto" and reasoning={"summary": "auto"}.

All API calls are routed through an injected api_caller (model server proxy).
"""

import logging
from typing import Any, Dict, List, Optional

from resources_servers.browser_gym.schemas import BrowserAction
from responses_api_agents.browser_agent.adapters.base import BaseCUAAdapter, CUAAdapterResponse, CUAAdapterUsage


logger = logging.getLogger(__name__)


class OpenAICUAAdapter(BaseCUAAdapter):
    def __init__(
        self,
        model: str = "computer-use-preview",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        api_caller=None,
    ):
        self._model = model
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._api_caller = api_caller
        self._last_response_id: Optional[str] = None

    def _get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "computer_use_preview",
                "display_width": self._viewport_width,
                "display_height": self._viewport_height,
                "environment": "browser",
            }
        ]

    async def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Route API call through the injected model server proxy."""
        if not self._api_caller:
            raise RuntimeError("OpenAICUAAdapter requires an api_caller (model server proxy). No direct API calls.")
        return await self._api_caller(payload)

    def _parse_actions(self, response: Dict[str, Any]) -> CUAAdapterResponse:
        """Parse OpenAI response into CUAAdapterResponse."""
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
                browser_action = self._map_openai_action(action_data)
                if browser_action:
                    actions.append(browser_action)

            elif item_type == "message" and item.get("role") == "assistant":
                content = item.get("content", [])
                for block in content:
                    if block.get("type") == "output_text":
                        message = block.get("text", "")
                if not actions:
                    done = True

        self._pending_call_ids = pending_call_ids
        self._pending_safety_checks = pending_safety_checks

        usage = None
        resp_usage = response.get("usage")
        if resp_usage and isinstance(resp_usage, dict):
            in_tok = resp_usage.get("input_tokens", 0) or 0
            out_tok = resp_usage.get("output_tokens", 0) or 0
            usage = CUAAdapterUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok)

        return CUAAdapterResponse(actions=actions, message=message, raw_response=response, done=done, usage=usage)

    @staticmethod
    def _get_coord(action: Dict[str, Any]) -> Optional[List[int]]:
        """Extract coordinates from OpenAI action -- handles both x/y and coordinate formats."""
        if "x" in action and "y" in action:
            return [int(action["x"]), int(action["y"])]
        if "coordinate" in action:
            return action["coordinate"]
        return None

    @staticmethod
    def _normalize_drag_path(path: Any) -> Optional[List[List[int]]]:
        """Convert OpenAI drag path (list of {"x":..,"y":..} dicts) to list of [x,y] arrays."""
        if not path or not isinstance(path, list):
            return None
        result = []
        for point in path:
            if isinstance(point, dict) and "x" in point and "y" in point:
                result.append([int(point["x"]), int(point["y"])])
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                result.append([int(point[0]), int(point[1])])
        return result if result else None

    def _map_openai_action(self, action: Dict[str, Any]) -> Optional[BrowserAction]:
        """Map an OpenAI action dict to a unified BrowserAction."""
        action_type = action.get("type", "")
        coord = self._get_coord(action)

        if action_type == "click":
            return BrowserAction(
                action_type="click",
                coordinate=coord,
                button=action.get("button", "left"),
            )
        elif action_type == "double_click":
            return BrowserAction(action_type="double_click", coordinate=coord)
        elif action_type == "triple_click":
            return BrowserAction(action_type="triple_click", coordinate=coord)
        elif action_type == "drag":
            normalized_path = self._normalize_drag_path(action.get("path"))
            start = None
            end = None
            if normalized_path and len(normalized_path) >= 2:
                start = normalized_path[0]
                end = normalized_path[-1]
            else:
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
            return BrowserAction(
                action_type="drag",
                start_coordinate=start,
                end_coordinate=end,
                path=normalized_path,
            )
        elif action_type == "keypress":
            keys = action.get("keys", [])
            if not keys and "key" in action:
                keys = [action["key"]]
            return BrowserAction(action_type="keypress", keys=keys)
        elif action_type == "type":
            return BrowserAction(action_type="type", text=action.get("text", ""))
        elif action_type == "scroll":
            return BrowserAction(
                action_type="scroll",
                coordinate=coord,
                scroll_x=action.get("scroll_x"),
                scroll_y=action.get("scroll_y"),
            )
        elif action_type in ("move", "mouse_move", "hover"):
            return BrowserAction(action_type="hover", coordinate=coord)
        elif action_type == "screenshot":
            return BrowserAction(action_type="screenshot")
        elif action_type == "wait":
            duration = action.get("ms") or action.get("duration") or 1000
            return BrowserAction(action_type="wait", duration=int(duration))
        elif action_type == "goto":
            return BrowserAction(action_type="goto", url=action.get("url"))
        elif action_type == "new_tab":
            return BrowserAction(action_type="new_tab", url=action.get("url"))
        elif action_type == "close_tab":
            return BrowserAction(action_type="close_tab")
        elif action_type == "switch_tab":
            return BrowserAction(action_type="switch_tab", tab_index=action.get("tab_index"))
        elif action_type == "list_tabs":
            return BrowserAction(action_type="screenshot")
        else:
            logger.warning(f"Unknown OpenAI action type: {action_type}")
            return None

    async def initialize(self, task_prompt: str, screenshot_b64: str) -> CUAAdapterResponse:
        """First call: send full input with user prompt + initial screenshot."""
        self._last_response_id = None
        self._pending_call_ids = []
        self._pending_safety_checks = []

        input_items = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": task_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                        "detail": "auto",
                    },
                ],
            }
        ]

        payload = {
            "model": self._model,
            "input": input_items,
            "tools": self._get_tools(),
            "truncation": "auto",
            "reasoning": {"summary": "auto"},
        }

        response = await self._call_api(payload)
        self._last_response_id = response.get("id")
        return self._parse_actions(response)

    async def step(self, screenshot_b64: str, action_result: Optional[str] = None) -> CUAAdapterResponse:
        """Follow-up call using previous_response_id for context chaining."""
        followups = []
        safety_checks = getattr(self, "_pending_safety_checks", [])

        for call_id in getattr(self, "_pending_call_ids", []):
            item: Dict[str, Any] = {
                "type": "computer_call_output",
                "call_id": call_id,
                "output": {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                },
            }
            if safety_checks:
                item["acknowledged_safety_checks"] = safety_checks
            followups.append(item)

        if not followups:
            logger.warning("No pending call IDs for step — signaling done")
            return CUAAdapterResponse(actions=[], message=None, raw_response={}, done=True)

        payload = {
            "model": self._model,
            "previous_response_id": self._last_response_id,
            "input": followups,
            "tools": self._get_tools(),
            "truncation": "auto",
            "reasoning": {"summary": "auto"},
        }

        response = await self._call_api(payload)
        self._last_response_id = response.get("id")
        return self._parse_actions(response)

    def reset(self):
        """Clear context state."""
        self._last_response_id = None
        self._pending_call_ids = []
        self._pending_safety_checks = []
