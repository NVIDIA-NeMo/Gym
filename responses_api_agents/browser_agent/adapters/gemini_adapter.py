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

Context management: Client-side paired-turn trimming (same pattern as Anthropic adapter).
Full conversation maintained in _contents list, trimmed before each API call.

All API calls are routed through an injected api_caller (model server proxy).

Handles all 17 predefined Gemini Computer Use functions:
  open_web_browser, click_at, hover_at, type_text_at, scroll_document, scroll_at,
  wait_5_seconds, go_back, go_forward, search, navigate, keypress, key_combination,
  drag_and_drop, new_tab, switch_tab, close_tab
"""

import base64
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


class GeminiCUAAdapter(BaseCUAAdapter):
    def __init__(
        self,
        model: str = "gemini-2.5-computer-use-preview-10-2025",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        max_conversation_turns: int = 8,
        api_caller: Optional[ApiCaller] = None,
        thinking_level: str = "THINKING_LEVEL_MEDIUM",
        include_thoughts: bool = True,
    ):
        self._model = model
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._max_conversation_turns = max_conversation_turns
        self._api_caller = api_caller
        self._is_gemini_3 = "gemini-3" in model.lower()
        self._contents: list = []
        self._pending_action_names: List[str] = []
        self._pending_safety_decisions: Dict[str, bool] = {}
        self._current_url: str = "about:blank"

        self._build_generate_config(thinking_level, include_thoughts)

    def _build_generate_config(self, thinking_level: str, include_thoughts: bool):
        from google.genai import types

        common_kwargs = {
            "system_instruction": GEMINI_CUA_SYSTEM_PROMPT,
            "tools": [
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER,
                    ),
                ),
            ],
        }

        if self._is_gemini_3:
            common_kwargs["temperature"] = 1.0
            common_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level,
                include_thoughts=include_thoughts,
            )
        else:
            common_kwargs["temperature"] = 0.0

        self._generate_config = types.GenerateContentConfig(**common_kwargs)

    # ── Content helpers ──────────────────────────────────────────

    def _is_function_call(self, content) -> bool:
        return any(
            part.function_call
            for part in (content.parts or [])
            if hasattr(part, "function_call") and part.function_call
        )

    def _is_function_response(self, content) -> bool:
        return any(
            part.function_response
            for part in (content.parts or [])
            if hasattr(part, "function_response") and part.function_response
        )

    # ── Context management (mirrors Anthropic adapter patterns) ──

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

        before = len(self._contents)
        self._contents = [self._contents[0]] + self._contents[start_index:]
        logger.debug("Trimmed conversation: %d -> %d items", before, len(self._contents))

    def _trim_conversation_history(self):
        if not self._contents or len(self._contents) <= 2:
            return
        self._trim_paired_turns()

    def _validate_function_pairs(self):
        """Ensure every function_call content has a matching function_response content
        and vice versa, with matching names.

        Walks the _contents list and drops:
        - orphaned function_call (no following function_response)
        - orphaned function_response (not preceded by a function_call)
        - pairs where the response names don't match the call names
        """
        result = []
        i = 0
        while i < len(self._contents):
            content = self._contents[i]
            if self._is_function_call(content):
                if i + 1 < len(self._contents) and self._is_function_response(self._contents[i + 1]):
                    call_names = self._get_function_names(content, "function_call")
                    resp_names = self._get_function_names(self._contents[i + 1], "function_response")
                    if call_names == resp_names:
                        result.append(content)
                        result.append(self._contents[i + 1])
                        i += 2
                        continue
                    else:
                        logger.warning(
                            "Dropping function pair at index %d: call names %s != response names %s",
                            i,
                            call_names,
                            resp_names,
                        )
                        i += 2
                else:
                    logger.warning("Dropping orphaned function_call at index %d", i)
                    i += 1
            elif self._is_function_response(content):
                logger.warning("Dropping orphaned function_response at index %d", i)
                i += 1
            else:
                result.append(content)
                i += 1
        self._contents = result

    def _get_function_names(self, content, part_type: str) -> List[str]:
        """Extract function call or response names from a Content object."""
        names = []
        for part in content.parts or []:
            if part_type == "function_call" and hasattr(part, "function_call") and part.function_call:
                names.append(part.function_call.name)
            elif part_type == "function_response" and hasattr(part, "function_response") and part.function_response:
                names.append(part.function_response.name)
        return names

    def _gc_old_screenshots(self, trim_window_size: int) -> None:
        """Replace inline image data in old messages with a placeholder to free memory."""
        if len(self._contents) <= trim_window_size:
            return

        cutoff = len(self._contents) - trim_window_size

        for content in self._contents[:cutoff]:
            for part in content.parts or []:
                if hasattr(part, "function_response") and part.function_response:
                    resp = part.function_response.response
                    if isinstance(resp, dict) and "screenshot" in resp:
                        resp["screenshot"] = SCREENSHOT_PLACEHOLDER

    # ── Coordinate denormalization ────────────────────────────────

    def _denorm_x(self, x: int) -> int:
        """Convert Gemini normalized X (0-999) to pixel coordinate."""
        return int((x / 1000.0) * self._viewport_width)

    def _denorm_y(self, y: int) -> int:
        """Convert Gemini normalized Y (0-999) to pixel coordinate."""
        return int((y / 1000.0) * self._viewport_height)

    # ── Action mapping (all 17 predefined functions) ─────────────

    def _map_gemini_action(self, action_name: str, args: Dict[str, Any]) -> Optional[BrowserAction]:
        """Map a Gemini Computer Use function call to a unified BrowserAction.

        Gemini CUA outputs coordinates in a 0-999 normalized space.
        All coordinates are denormalized to pixel values before creating
        BrowserActions, matching the harness behavior.
        """
        if action_name == "click_at":
            px = self._denorm_x(int(args.get("x", 0)))
            py = self._denorm_y(int(args.get("y", 0)))
            return BrowserAction(action_type="click", coordinate=[px, py])
        elif action_name == "hover_at":
            px = self._denorm_x(int(args.get("x", 0)))
            py = self._denorm_y(int(args.get("y", 0)))
            return BrowserAction(action_type="hover", coordinate=[px, py])
        elif action_name == "type_text_at":
            px = self._denorm_x(int(args.get("x", 0)))
            py = self._denorm_y(int(args.get("y", 0)))
            return BrowserAction(
                action_type="type",
                coordinate=[px, py],
                text=args.get("text", ""),
                press_enter=args.get("press_enter", False),
                clear_before_typing=args.get("clear_before_typing", True),
            )
        elif action_name == "scroll_document":
            direction = args.get("direction", "down")
            vh = self._viewport_height
            vw = self._viewport_width
            if direction in ("up", "down"):
                amount = int(vh * 0.8)
            else:
                amount = vw // 2
            return BrowserAction(
                action_type="scroll",
                scroll_direction=direction,
                scroll_x=(-amount if direction == "left" else amount if direction == "right" else 0),
                scroll_y=(-amount if direction == "up" else amount if direction == "down" else 0),
            )
        elif action_name == "scroll_at":
            px = self._denorm_x(int(args.get("x", 0)))
            py = self._denorm_y(int(args.get("y", 0)))
            direction = args.get("direction", "down")
            magnitude = int(args.get("magnitude", 800))
            if direction in ("up", "down"):
                actual_mag = self._denorm_y(magnitude)
            else:
                actual_mag = self._denorm_x(magnitude)
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
                action_type="scroll",
                coordinate=[px, py],
                scroll_direction=direction,
                scroll_x=dx,
                scroll_y=dy,
            )
        elif action_name == "wait_5_seconds":
            return BrowserAction(action_type="wait", duration=5000)
        elif action_name == "go_back":
            return BrowserAction(action_type="go_back")
        elif action_name == "go_forward":
            return BrowserAction(action_type="go_forward")
        elif action_name == "search":
            return BrowserAction(action_type="goto", url="https://www.google.com")
        elif action_name == "navigate":
            url = args.get("url", "")
            if url and not url.startswith(("http://", "https://")):
                url = "https://" + url
            return BrowserAction(action_type="goto", url=url)
        elif action_name == "keypress":
            keys_raw = args.get("keys", args.get("key", ""))
            if isinstance(keys_raw, str):
                if "+" in keys_raw:
                    keys = keys_raw.split("+")
                elif "," in keys_raw:
                    keys = keys_raw.split(",")
                else:
                    keys = [keys_raw]
            else:
                keys = list(keys_raw) if keys_raw else []
            return BrowserAction(action_type="keypress", keys=keys)
        elif action_name == "key_combination":
            keys_raw = args.get("keys", [])
            if isinstance(keys_raw, str):
                if "+" in keys_raw:
                    keys = keys_raw.split("+")
                elif "," in keys_raw:
                    keys = keys_raw.split(",")
                else:
                    keys = [keys_raw]
            else:
                keys = list(keys_raw) if keys_raw else []
            return BrowserAction(action_type="keypress", keys=keys)
        elif action_name == "drag_and_drop":
            sx = self._denorm_x(int(args.get("x", args.get("start_x", 0))))
            sy = self._denorm_y(int(args.get("y", args.get("start_y", 0))))
            ex = self._denorm_x(int(args.get("destination_x", args.get("end_x", 0))))
            ey = self._denorm_y(int(args.get("destination_y", args.get("end_y", 0))))
            return BrowserAction(
                action_type="drag",
                start_coordinate=[sx, sy],
                end_coordinate=[ex, ey],
            )
        elif action_name == "new_tab":
            return BrowserAction(action_type="new_tab", url=args.get("url", ""))
        elif action_name == "switch_tab":
            idx = args.get("index", args.get("tab_index", 0))
            return BrowserAction(action_type="switch_tab", tab_index=int(idx))
        elif action_name == "close_tab":
            return BrowserAction(action_type="close_tab")
        elif action_name == "list_tabs":
            return BrowserAction(action_type="screenshot")
        elif action_name == "open_web_browser":
            return BrowserAction(action_type="screenshot")
        elif action_name in ("WebAgentState", "web_agent_state"):
            return BrowserAction(action_type="screenshot")
        else:
            logger.warning("Unknown Gemini action: %s", action_name)
            return None

    # ── Response parsing ─────────────────────────────────────────

    def _parse_response(self, response_data) -> CUAAdapterResponse:
        """Parse a serialized (dict) response from the model server proxy."""
        from google.genai.types import Content, FunctionCall, Part

        actions: List[BrowserAction] = []
        message: Optional[str] = None
        done = False
        safety_decisions: Dict[str, bool] = {}

        candidates = response_data.get("candidates", [])
        if not candidates:
            return CUAAdapterResponse(done=True)

        candidate = candidates[0]
        content_data = candidate.get("content")
        if content_data:
            parts = []
            for p in content_data.get("parts", []):
                if "function_call" in p:
                    fc_data = p["function_call"]
                    args = fc_data.get("args", {})
                    if "safety_decision" in args:
                        safety_decisions[fc_data["name"]] = True
                        logger.info("Safety decision for %s: %s", fc_data["name"], args["safety_decision"])
                    parts.append(Part(function_call=FunctionCall(name=fc_data["name"], args=args)))
                elif "text" in p:
                    if p.get("thought"):
                        parts.append(Part(text=p["text"], thought=True))
                    else:
                        parts.append(Part(text=p["text"]))
            reconstructed = Content(role=content_data.get("role", "model"), parts=parts)
            self._contents.append(reconstructed)

            for p in content_data.get("parts", []):
                if "function_call" in p:
                    fc_data = p["function_call"]
                    browser_action = self._map_gemini_action(fc_data["name"], fc_data.get("args", {}))
                    if browser_action:
                        actions.append(browser_action)
                elif "text" in p and not p.get("thought"):
                    message = p["text"]

        self._pending_safety_decisions = safety_decisions

        if not actions and message:
            done = True

        usage = None
        usage_meta = response_data.get("usage_metadata")
        if usage_meta:
            in_tok = usage_meta.get("prompt_token_count", 0) or 0
            out_tok = usage_meta.get("candidates_token_count", 0) or 0
            usage = CUAAdapterUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok)

        raw = {"model": self._model}
        token_ids = extract_token_ids_from_response(response_data)

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

    # ── Function response building ───────────────────────────────

    def _build_function_responses(
        self, screenshot_b64: str, action_names: List[str], action_error: Optional[str] = None
    ):
        """Build function response content with screenshot for each pending action.

        Gemini CUA requires every function response to include a 'url' or
        'current_url' field in the response dict. When a function call included
        a safety_decision, the corresponding function_response must include
        safety_acknowledgement.
        """
        from google.genai.types import Content, FunctionResponse, Part

        url = self._current_url or "about:blank"
        screenshot_bytes = base64.b64decode(screenshot_b64)
        safety_decisions = getattr(self, "_pending_safety_decisions", {})

        parts = []
        for name in action_names:
            if action_error:
                response_dict: Dict[str, Any] = {"url": url, "status": "error", "error": action_error}
            else:
                response_dict = {"url": url, "status": "success"}
            if safety_decisions.get(name):
                response_dict["safety_acknowledgement"] = "true"
            parts.append(
                Part(
                    function_response=FunctionResponse(
                        name=name,
                        response=response_dict,
                    )
                )
            )
            parts.append(Part.from_bytes(data=screenshot_bytes, mime_type="image/png"))
        return Content(role="user", parts=parts)

    # ── Serialization for model-server routing ───────────────────

    def _serialize_contents(self) -> List[Dict[str, Any]]:
        """Serialize google.genai Content objects to dicts for model-server transport."""
        import base64 as b64

        serialized = []
        for content in self._contents:
            parts_list = []
            for part in content.parts or []:
                if part.text:
                    p: Dict[str, Any] = {"text": part.text}
                    if getattr(part, "thought", False):
                        p["thought"] = True
                    thought_sig = getattr(part, "thoughtSignature", None) or getattr(part, "thought_signature", None)
                    if thought_sig:
                        p["thought_signature"] = thought_sig
                    parts_list.append(p)
                elif part.function_call:
                    parts_list.append(
                        {
                            "function_call": {
                                "name": part.function_call.name,
                                "args": dict(part.function_call.args or {}),
                            }
                        }
                    )
                elif part.function_response:
                    fr_response = dict(part.function_response.response or {})
                    parts_list.append(
                        {
                            "function_response": {
                                "name": part.function_response.name,
                                "response": fr_response,
                            }
                        }
                    )
                elif hasattr(part, "inline_data") and part.inline_data:
                    data = part.inline_data.data
                    if isinstance(data, bytes):
                        data = b64.b64encode(data).decode("utf-8")
                    parts_list.append(
                        {
                            "inline_data": {
                                "mime_type": part.inline_data.mime_type or "image/png",
                                "data": data,
                            }
                        }
                    )
            serialized.append({"role": content.role, "parts": parts_list})
        return serialized

    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize GenerateContentConfig to a dict for model-server transport."""
        cfg = self._generate_config
        result: Dict[str, Any] = {
            "temperature": cfg.temperature,
        }

        if cfg.system_instruction:
            result["system_instruction"] = cfg.system_instruction

        if cfg.tools:
            tools_list = []
            for tool in cfg.tools:
                if hasattr(tool, "computer_use") and tool.computer_use:
                    env = getattr(tool.computer_use, "environment", None)
                    env_name = env.name if hasattr(env, "name") else str(env)
                    tools_list.append({"computer_use": {"environment": env_name}})
            result["tools"] = tools_list

        if hasattr(cfg, "thinking_config") and cfg.thinking_config:
            tc = cfg.thinking_config
            tl = tc.thinking_level
            tl_name = tl.name if hasattr(tl, "name") else str(tl)
            result["thinking_config"] = {
                "thinking_level": tl_name,
                "include_thoughts": tc.include_thoughts,
            }

        for attr in ("top_p", "top_k", "max_output_tokens"):
            val = getattr(cfg, attr, None)
            if val is not None:
                result[attr] = val

        return result

    # ── API call lifecycle ───────────────────────────────────────

    def _prepare_for_api(self):
        """Trim and validate contents, then return a deep copy for the API call."""
        self._gc_old_screenshots(self._max_conversation_turns * 2 + 2)
        self._trim_conversation_history()
        self._validate_function_pairs()

    async def _execute_api_call(self):
        """Route API call through the injected model server proxy."""
        if not self._api_caller:
            raise RuntimeError("GeminiCUAAdapter requires an api_caller (model server proxy). No direct API calls.")
        serialized_contents = self._serialize_contents()
        serialized_config = self._serialize_config()
        api_params = {
            "contents": serialized_contents,
            "config": serialized_config,
        }
        return await self._api_caller(api_params)

    # ── Public interface ─────────────────────────────────────────

    async def initialize(self, task_prompt: str, screenshot_b64: str) -> CUAAdapterResponse:
        from google.genai.types import Content, Part

        self._contents = []
        self._pending_action_names = []

        screenshot_bytes = base64.b64decode(screenshot_b64)
        parts = [
            Part(text=task_prompt),
            Part.from_bytes(data=screenshot_bytes, mime_type="image/png"),
        ]
        self._contents = [Content(role="user", parts=parts)]

        self._prepare_for_api()

        logger.info(
            "Gemini adapter initialize: model=%s, is_gemini_3=%s",
            self._model,
            self._is_gemini_3,
        )

        response = await self._execute_api_call()
        result = self._parse_response(response)

        self._pending_action_names = self._extract_function_call_names()

        return result

    def _extract_function_call_names(self) -> List[str]:
        """Extract all function call names from the last model response in _contents.

        This must return one name per function call the model emitted so that
        ``_build_function_responses`` sends exactly the right number of
        ``FunctionResponse`` items back to the Gemini API.
        """
        names: List[str] = []
        if self._contents:
            last_content = self._contents[-1]
            for part in last_content.parts or []:
                if hasattr(part, "function_call") and part.function_call:
                    names.append(part.function_call.name)
        return names

    async def step(
        self, screenshot_b64: str, action_result: Optional[str] = None, action_error: Optional[str] = None
    ) -> CUAAdapterResponse:
        if action_result:
            self._current_url = action_result

        current_action_names = self._pending_action_names
        self._pending_action_names = []

        if not current_action_names:
            current_action_names = ["click_at"]

        fn_response_content = self._build_function_responses(screenshot_b64, current_action_names, action_error)
        self._contents.append(fn_response_content)

        self._prepare_for_api()

        response = await self._execute_api_call()
        result = self._parse_response(response)

        self._pending_action_names = self._extract_function_call_names()

        return result

    def reset(self):
        self._contents = []
        self._pending_action_names = []
        self._pending_safety_decisions = {}
        self._current_url = "about:blank"
