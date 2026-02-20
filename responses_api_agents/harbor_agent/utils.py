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
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)


@dataclass
class HarborAgentUtils:
    @staticmethod
    def _wrap_reasoning_in_think_tags(texts: List[str]) -> str:
        return "".join(f"<think>{text}</think>" for text in texts if text)

    @staticmethod
    def _merge_message_and_reasoning(message: str, reasoning_content: Optional[str]) -> str:
        """Merge reasoning back into the message text with ``<think>`` tags.

        The model's native output format is ``<think>reasoning</think>content``,
        so we prepend the wrapped reasoning before the message to preserve the
        original token order (important for training data alignment).
        """
        if not reasoning_content:
            return message
        wrapped_reasoning = HarborAgentUtils._wrap_reasoning_in_think_tags([reasoning_content])
        return f"{wrapped_reasoning}{message}"

    @staticmethod
    def get_default_response_object() -> Dict[str, Any]:
        return {
            "id": f"resp_{str(uuid4())}",
            "created_at": int(time.time()),
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "background": False,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": {
                "effort": None,
                "generate_summary": None,
                "summary": None,
            },
            "service_tier": "default",
            "status": "completed",
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "top_logprobs": 0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 0,
            },
            "user": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
            "store": True,
        }

    @staticmethod
    def extract_reward(verifier_result: Optional[Dict[str, Any]]) -> float:
        """Extract reward from Harbor's VerifierResult.rewards dict.

        Harbor rewards are typically {"reward": 0.0 or 1.0} or a dict of named rewards.
        Returns the primary reward value, defaulting to 0.0 on failure.
        """
        if verifier_result is None:
            return 0.0

        rewards = verifier_result.get("rewards")
        if not rewards or not isinstance(rewards, dict):
            return 0.0

        # Return the "reward" key if present, otherwise return the first value
        if "reward" in rewards:
            return float(rewards["reward"])

        # Fallback: return first reward value
        for value in rewards.values():
            return float(value)

        return 0.0

    # ------------------------------------------------------------------ #
    #  Input extraction — populate responses_create_params.input          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_input_from_trajectory(
        trajectory: Optional[Dict[str, Any]],
    ) -> List[NeMoGymEasyInputMessage]:
        """Extract the initial user instruction(s) from an ATIF trajectory.

        Harbor tasks provide the instruction via a file (not through the NeMo Gym
        request body).  The instruction appears as the first step(s) with
        ``source: "user"`` in the ATIF trajectory.  We convert these into
        ``NeMoGymEasyInputMessage`` dicts so they populate
        ``responses_create_params.input`` in the final output.

        Returns an empty list when no trajectory is available.
        """
        if not trajectory:
            return []

        input_messages: List[NeMoGymEasyInputMessage] = []
        for step in trajectory.get("steps", []):
            if step.get("source") == "user":
                input_messages.append(
                    NeMoGymEasyInputMessage(
                        role="user",
                        content=step.get("message", ""),
                        type="message",
                    )
                )
            else:
                # User messages always come first in ATIF; stop once we hit
                # the first non-user step.
                break

        return input_messages

    # ------------------------------------------------------------------ #
    #  Usage extraction                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_usage(
        trial_result: Dict[str, Any],
        trajectory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the ``usage`` dict for the NeMo Gym response.

        Prefers ATIF ``final_metrics`` (exact totals from the trajectory) and
        falls back to ``agent_result`` token counts from ``result.json``.
        """
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0

        # Try trajectory final_metrics first
        if trajectory:
            fm = trajectory.get("final_metrics", {})
            input_tokens = fm.get("total_prompt_tokens", 0)
            output_tokens = fm.get("total_completion_tokens", 0)
            cached_tokens = fm.get("total_cached_tokens", 0)

        # Fall back to trial result agent_result
        if input_tokens == 0 and output_tokens == 0:
            agent_result = trial_result.get("agent_result") or {}
            input_tokens = agent_result.get("n_input_tokens", 0) or 0
            output_tokens = agent_result.get("n_output_tokens", 0) or 0
            cached_tokens = agent_result.get("n_cache_tokens", 0) or 0

        return {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        }

    # ------------------------------------------------------------------ #
    #  Raw content parsing — extract function calls from raw LLM JSON     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Try to extract a JSON object from text that may have surrounding content.

        Handles the common case where terminus-2's raw LLM response is valid
        JSON, as well as responses with extra text before/after the JSON object.
        """
        # Fast path: try direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Slow path: find the first balanced {...} in the text
        if not text or "{" not in text:
            return None

        brace_depth = 0
        start = -1
        in_string = False
        escape_next = False

        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0 and start >= 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    start = -1

        return None

    @staticmethod
    def _parse_raw_content_tool_calls(
        message: str, agent_step_index: int
    ) -> List[Dict[str, Any]]:
        """Parse function calls from a raw terminus-2 JSON response.

        When ``raw_content=true`` in the trajectory config, the step message
        contains the full LLM JSON response (with ``analysis``, ``plan``,
        ``commands``, ``task_complete`` fields).  We extract the ``commands``
        array and ``task_complete`` flag to build ATIF-compatible tool_call
        dicts so that downstream processing can treat them identically to
        steps produced with ``raw_content=false``.
        """
        parsed = HarborAgentUtils._extract_json_object(message)
        if parsed is None:
            return []

        tool_calls: List[Dict[str, Any]] = []

        commands = parsed.get("commands", [])
        if isinstance(commands, list):
            for i, cmd in enumerate(commands):
                if not isinstance(cmd, dict) or "keystrokes" not in cmd:
                    continue
                tool_calls.append(
                    {
                        "tool_call_id": f"call_{agent_step_index}_{i + 1}",
                        "function_name": "bash_command",
                        "arguments": {
                            "keystrokes": cmd["keystrokes"],
                            "duration": cmd.get("duration", 1.0),
                        },
                    }
                )

        task_complete = parsed.get("task_complete", False)
        if isinstance(task_complete, str):
            task_complete = task_complete.lower() in ("true", "1", "yes")
        if task_complete:
            tool_calls.append(
                {
                    "tool_call_id": f"call_{agent_step_index}_task_complete",
                    "function_name": "mark_task_complete",
                    "arguments": {},
                }
            )

        return tool_calls

    # ------------------------------------------------------------------ #
    #  Output conversion — trajectory → NeMo Gym output items             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def trajectory_to_responses(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert ATIF trajectory agent steps to NeMo Gym output items.

        Each agent step in the trajectory is converted to:
        1. An assistant **message** containing the agent's analysis/plan text.
           Uses ``NeMoGymResponseOutputMessageForTraining`` when the step
           carries token IDs / logprobs, otherwise ``NeMoGymResponseOutputMessage``.
        2. One **function_call** item per tool call the agent made.
        3. One **function_call_output** item per observation result.

        When the trajectory was produced with ``raw_content=true`` (no
        ``tool_calls`` on agent steps), function calls are parsed from the
        raw LLM JSON stored in ``step.message``.
        """
        output_items: List[Dict[str, Any]] = []
        agent_step_index = 0

        for step in trajectory.get("steps", []):
            if step.get("source") != "agent":
                continue

            text = HarborAgentUtils._merge_message_and_reasoning(
                step.get("message", "") or "",
                step.get("reasoning_content"),
            )
            content = [
                NeMoGymResponseOutputText(
                    annotations=[],
                    text=text,
                    type="output_text",
                    logprobs=None,
                ),
            ]

            # Use the training variant when token-level details are available
            # in the step metrics (written by NemoGymLLM / LiteLLM).
            metrics = step.get("metrics") or {}
            prompt_token_ids = metrics.get("prompt_token_ids")
            completion_token_ids = metrics.get("completion_token_ids")
            logprobs = metrics.get("logprobs")
            has_token_details = prompt_token_ids or completion_token_ids or logprobs

            if has_token_details:
                message = NeMoGymResponseOutputMessageForTraining(
                    id=f"cht_{uuid4().hex[:12]}",
                    content=content,
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=prompt_token_ids or [],
                    generation_token_ids=completion_token_ids or [],
                    generation_log_probs=logprobs or [],
                )
            else:
                message = NeMoGymResponseOutputMessage(
                    id=f"cht_{uuid4().hex[:12]}",
                    content=content,
                    role="assistant",
                    status="completed",
                    type="message",
                )
            output_items.append(message.model_dump())

            tool_calls = step.get("tool_calls") or []
            # raw_content mode: parse function calls from the raw JSON message
            if not tool_calls:
                tool_calls = HarborAgentUtils._parse_raw_content_tool_calls(
                    step.get("message", "") or "", agent_step_index
                )

            observation = step.get("observation", {})
            results = observation.get("results", [])

            # --- Function calls ---
            for tc in tool_calls:
                arguments = tc.get("arguments", {})
                fc = NeMoGymResponseFunctionToolCall(
                    arguments=json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                    call_id=tc.get("tool_call_id", f"call_{uuid4().hex[:8]}"),
                    name=tc.get("function_name", "unknown"),
                    type="function_call",
                    id=f"fc_{uuid4().hex[:8]}",
                    status="completed",
                )
                output_items.append(fc.model_dump())

            # --- Observation / function call outputs ---
            for i, result in enumerate(results):
                call_id = (
                    tool_calls[i].get("tool_call_id", f"call_{uuid4().hex[:8]}")
                    if i < len(tool_calls)
                    else f"call_{uuid4().hex[:8]}"
                )
                fco = NeMoGymFunctionCallOutput(
                    call_id=call_id,
                    output=result.get("content", ""),
                    type="function_call_output",
                    id=f"fco_{uuid4().hex[:8]}",
                    status="completed",
                )
                output_items.append(fco.model_dump())

            agent_step_index += 1

        return output_items

    # ------------------------------------------------------------------ #
    #  Main entry point — trial result → NeMo Gym output items            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def trial_result_to_responses(
        trial_result: Dict[str, Any],
        trajectory: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert Harbor trial output to NeMo Gym output items.

        All output is derived from the ATIF trajectory.  Token IDs and
        logprobs are read from each step's ``metrics`` (populated by
        ``NemoGymLLM``).  Returns an empty list when no trajectory is
        available.
        """
        if trajectory and trajectory.get("steps"):
            return HarborAgentUtils.trajectory_to_responses(trajectory)
        return []
