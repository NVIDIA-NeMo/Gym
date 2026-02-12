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
    #  Output conversion — trajectory → NeMo Gym output items             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_assistant_texts_from_trajectory(
        trajectory: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Extract assistant message text from ATIF trajectory steps."""
        if not trajectory:
            return []

        assistant_texts: List[str] = []
        for step in trajectory.get("steps", []):
            if step.get("source") == "agent":
                assistant_texts.append(step.get("message", "") or "")
        return assistant_texts

    @staticmethod
    def trajectory_to_responses(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert ATIF trajectory agent steps to NeMo Gym output items.

        Each agent step in the trajectory is converted to:
        1. An assistant **message** containing the agent's analysis/plan text,
           preserving the original assistant content.
        2. One **function_call** item per tool call the agent made.
        3. One **function_call_output** item per observation result.
        """
        output_items: List[Dict[str, Any]] = []

        for step in trajectory.get("steps", []):
            if step.get("source") != "agent":
                continue

            message = NeMoGymResponseOutputMessage(
                id=f"cht_{uuid4().hex[:12]}",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text=step.get("message", ""),
                        type="output_text",
                        logprobs=None,
                    ),
                ],
                role="assistant",
                status="completed",
                type="message",
            )
            output_items.append(message.model_dump())

            tool_calls = step.get("tool_calls", [])
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

        Behavior:
        1. If trajectory is available, always include full rich output
           (assistant messages + function_call + function_call_output).
        2. If rollout_details are available, overlay token IDs/logprobs onto
           assistant message items in order.
        3. If only rollout_details exist, emit assistant messages from those.
        4. Otherwise, return an empty list.
        """
        output_items: List[Dict[str, Any]] = []

        agent_result = trial_result.get("agent_result")
        assistant_turn_texts = HarborAgentUtils.extract_assistant_texts_from_trajectory(
            trajectory
        )
        rollout_details = (
            agent_result.get("rollout_details")
            if agent_result and isinstance(agent_result, dict)
            else None
        )

        # Build a flat list of rollout turns to overlay in order.
        # Each item corresponds to one assistant response turn.
        rollout_turns: List[Dict[str, Any]] = []
        if rollout_details:
            for rollout in rollout_details:
                prompt_token_ids_list = rollout.get("prompt_token_ids") or []
                completion_token_ids_list = rollout.get("completion_token_ids") or []
                logprobs_list = rollout.get("logprobs") or []

                n_turns = max(
                    len(prompt_token_ids_list),
                    len(completion_token_ids_list),
                    len(logprobs_list),
                )
                for turn_idx in range(n_turns):
                    rollout_turns.append(
                        {
                            "prompt_token_ids": (
                                prompt_token_ids_list[turn_idx]
                                if turn_idx < len(prompt_token_ids_list)
                                else []
                            ),
                            "generation_token_ids": (
                                completion_token_ids_list[turn_idx]
                                if turn_idx < len(completion_token_ids_list)
                                else []
                            ),
                            "generation_log_probs": (
                                logprobs_list[turn_idx]
                                if turn_idx < len(logprobs_list)
                                else []
                            ),
                        }
                    )

        # Case 1: trajectory available -> preserve full rich output by default.
        # If rollout token details exist, overlay onto assistant message turns.
        if trajectory and trajectory.get("steps"):
            output_items = HarborAgentUtils.trajectory_to_responses(trajectory)
            if rollout_turns:
                assistant_idx = 0
                for item in output_items:
                    if item.get("type") == "message" and item.get("role") == "assistant":
                        if assistant_idx >= len(rollout_turns):
                            break
                        turn = rollout_turns[assistant_idx]
                        item["prompt_token_ids"] = turn["prompt_token_ids"]
                        item["generation_token_ids"] = turn["generation_token_ids"]
                        item["generation_log_probs"] = turn["generation_log_probs"]
                        assistant_idx += 1
            return output_items

        # Case 2: rollout_details available without trajectory.
        if rollout_turns:
            for turn_idx, turn in enumerate(rollout_turns):
                wrapped_message = NeMoGymResponseOutputMessageForTraining(
                    id=f"cht_{uuid4().hex[:12]}",
                    content=[
                        NeMoGymResponseOutputText(
                            annotations=[],
                            text=(
                                assistant_turn_texts[turn_idx]
                                if turn_idx < len(assistant_turn_texts)
                                else ""
                            ),
                            type="output_text",
                            logprobs=None,
                        ),
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=turn["prompt_token_ids"],
                    generation_token_ids=turn["generation_token_ids"],
                    generation_log_probs=turn["generation_log_probs"],
                )
                output_items.append(wrapped_message.model_dump())

            return output_items

        # Case 3: no trajectory and no rollout_details -> no output items.
        return output_items
