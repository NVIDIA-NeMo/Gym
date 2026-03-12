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
Gemini CUA adapter.

Context management: Client-side paired-turn trimming.
Full conversation maintained in contents list, trimmed before each generate_content call.
Two strategies based on model version:
- Gemini 3 Pro: _trim_with_thought_signatures (keeps last N paired turns)
- Gemini 2.5: _trim_paired_turns (keeps last N function_call/function_response pairs)
"""

import base64
import logging
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from google.genai.types import Content, FunctionResponse, Part

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """You are a browser automation agent. Your goal is to complete tasks efficiently using the Computer Use tool.

**Context Window:**
- You only have access to the last 8 conversation turns (due to token limits)
- Once you complete a subtask, MOVE FORWARD to the next one
- DO NOT revisit or verify already-completed subtasks
- If you lose context, check the current page state and continue from where you are

**Guidelines:**
- Think step by step
- Be precise with coordinates
- Wait for pages to load before interacting
- If an action fails, try an alternative approach"""


class GeminiCUAAdapter:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        max_conversation_turns: int = 8,
    ):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._max_conversation_turns = max_conversation_turns

        self._contents: List[Content] = []
        self._is_gemini_3 = "gemini-3" in model.lower()

        self._generate_config = types.GenerateContentConfig(
            tools=[types.Tool(computer_use=types.ToolComputerUse(environment="browser"))],
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,
        )

    def _is_function_call(self, content: Content) -> bool:
        return any(part.function_call for part in content.parts if hasattr(part, "function_call") and part.function_call)

    def _is_function_response(self, content: Content) -> bool:
        return any(
            part.function_response for part in content.parts if hasattr(part, "function_response") and part.function_response
        )

    def _trim_paired_turns(self):
        """Keep last N complete function_call/function_response pairs. Never split pairs."""
        total_turns = sum(1 for c in self._contents if self._is_function_response(c))
        if total_turns <= self._max_conversation_turns:
            return

        needed = self._max_conversation_turns
        idx = len(self._contents) - 1
        start_index = 1

        while idx >= 1 and needed > 0:
            if self._is_function_response(self._contents[idx]):
                j = idx - 1
                while j >= 1 and not self._is_function_call(self._contents[j]):
                    j -= 1
                start_index = max(1, j)
                needed -= 1
            idx -= 1

        if start_index < len(self._contents) - 1 and self._is_function_response(self._contents[start_index]):
            prev_idx = start_index - 1
            if prev_idx >= 1 and self._is_function_call(self._contents[prev_idx]):
                start_index = prev_idx

        self._contents = [self._contents[0]] + self._contents[start_index:]

    def _trim_with_thought_signatures(self):
        """Gemini 3 Pro: same paired-turn logic, thought signatures maintain reasoning context."""
        self._trim_paired_turns()

    def _trim_conversation_history(self):
        if not self._contents or len(self._contents) <= 2:
            return
        if self._is_gemini_3:
            self._trim_with_thought_signatures()
        else:
            self._trim_paired_turns()

    def _parse_response(self, response) -> "CUAAdapterResponse":
        from responses_api_agents.browser_agent.adapters.base import CUAAdapterResponse
        from resources_servers.browser_gym.schemas import BrowserAction

        actions: List[BrowserAction] = []
        message: Optional[str] = None
        done = False

        if not response.candidates:
            return CUAAdapterResponse(done=True)

        candidate = response.candidates[0]

        if candidate.content:
            self._contents.append(candidate.content)

        for part in candidate.content.parts if candidate.content else []:
            if part.function_call:
                fc = part.function_call
                browser_action = self._map_gemini_action(fc.name, fc.args or {})
                if browser_action:
                    actions.append(browser_action)
            elif part.text:
                message = part.text

        if not actions and message:
            done = True

        raw = {"model": self._model}
        return CUAAdapterResponse(actions=actions, message=message, raw_response=raw, done=done)

    def _map_gemini_action(self, action_name: str, args: Dict[str, Any]) -> Optional["BrowserAction"]:
        from resources_servers.browser_gym.schemas import BrowserAction

        if action_name == "click_at":
            return BrowserAction(action_type="click", coordinate=[int(args.get("x", 0)), int(args.get("y", 0))])
        elif action_name == "hover_at":
            return BrowserAction(action_type="hover", coordinate=[int(args.get("x", 0)), int(args.get("y", 0))])
        elif action_name == "type_text_at":
            return BrowserAction(
                action_type="type",
                coordinate=[int(args.get("x", 0)), int(args.get("y", 0))],
                text=args.get("text", ""),
                press_enter=args.get("press_enter", False),
            )
        elif action_name == "scroll_document":
            direction = "down" if args.get("direction", "down") == "down" else "up"
            return BrowserAction(action_type="scroll", scroll_direction=direction, scroll_amount=3)
        elif action_name == "scroll_at":
            return BrowserAction(
                action_type="scroll",
                coordinate=[int(args.get("x", 0)), int(args.get("y", 0))],
                scroll_direction=args.get("direction", "down"),
                scroll_amount=int(args.get("amount", 3)),
            )
        elif action_name == "keypress":
            return BrowserAction(action_type="keypress", key=args.get("key", ""))
        elif action_name == "key_combination":
            keys = args.get("keys", [])
            if isinstance(keys, str):
                keys = [keys]
            return BrowserAction(action_type="keypress", keys=keys)
        elif action_name == "drag_and_drop":
            return BrowserAction(
                action_type="drag",
                start_coordinate=[int(args.get("start_x", 0)), int(args.get("start_y", 0))],
                end_coordinate=[int(args.get("end_x", 0)), int(args.get("end_y", 0))],
            )
        elif action_name == "navigate":
            return BrowserAction(action_type="goto", url=args.get("url", ""))
        elif action_name == "go_back":
            return BrowserAction(action_type="go_back")
        elif action_name == "go_forward":
            return BrowserAction(action_type="go_forward")
        elif action_name == "search":
            return BrowserAction(action_type="goto", url=f"https://www.google.com/search?q={args.get('query', '')}")
        elif action_name == "open_web_browser":
            return BrowserAction(action_type="screenshot")
        else:
            logger.warning(f"Unknown Gemini action: {action_name}")
            return None

    def _build_function_responses(self, screenshot_b64: str, action_names: List[str]) -> Content:
        """Build function response content with screenshot for each pending action."""
        screenshot_bytes = base64.b64decode(screenshot_b64)
        parts = []
        for name in action_names:
            parts.append(
                Part(
                    function_response=FunctionResponse(
                        name=name,
                        response={"screenshot": screenshot_b64, "status": "success"},
                    )
                )
            )
        return Content(role="user", parts=parts)

    async def initialize(self, task_prompt: str, screenshot_b64: str) -> "CUAAdapterResponse":
        from responses_api_agents.browser_agent.adapters.base import CUAAdapterResponse

        self._contents = []
        self._pending_action_names: List[str] = []

        screenshot_bytes = base64.b64decode(screenshot_b64)
        parts = [
            Part(text=task_prompt),
            Part.from_bytes(data=screenshot_bytes, mime_type="image/png"),
        ]
        self._contents = [Content(role="user", parts=parts)]

        self._trim_conversation_history()

        response = self._client.models.generate_content(
            model=self._model,
            contents=self._contents,
            config=self._generate_config,
        )

        result = self._parse_response(response)

        self._pending_action_names = [a.action_type for a in result.actions]
        return result

    async def step(self, screenshot_b64: str, action_result: Optional[str] = None) -> "CUAAdapterResponse":
        from responses_api_agents.browser_agent.adapters.base import CUAAdapterResponse

        action_names = getattr(self, "_pending_action_names", ["click_at"])
        if action_names:
            fn_response_content = self._build_function_responses(screenshot_b64, action_names)
            self._contents.append(fn_response_content)

        self._trim_conversation_history()

        response = self._client.models.generate_content(
            model=self._model,
            contents=self._contents,
            config=self._generate_config,
        )

        result = self._parse_response(response)
        self._pending_action_names = [a.action_type for a in result.actions]
        return result

    def reset(self):
        self._contents = []
        self._pending_action_names = []
