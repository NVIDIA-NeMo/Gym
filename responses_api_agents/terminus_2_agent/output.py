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
import json
from typing import Any
from uuid import uuid4

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)


def trajectory_to_responses(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert each ATIF agent step to a message, calls, and call outputs."""
    output_items: list[dict[str, Any]] = []
    agent_step_index = 0

    for step in trajectory.get("steps", []):
        if step.get("source") != "agent":
            continue

        text = step.get("message", "") or ""
        if reasoning := step.get("reasoning_content"):
            text = f"<think>{reasoning}</think>{text}"
        content = [
            NeMoGymResponseOutputText(
                annotations=[],
                text=text,
                type="output_text",
                logprobs=None,
            ),
        ]

        metrics = step.get("metrics") or {}
        prompt_token_ids = metrics.get("prompt_token_ids")
        completion_token_ids = metrics.get("completion_token_ids")
        logprobs = metrics.get("logprobs")
        metrics_extra = metrics.get("extra") or {}
        if not isinstance(metrics_extra, dict):
            metrics_extra = {}
        routed_experts = metrics.get("routed_experts")
        if routed_experts is None:
            routed_experts = metrics_extra.get("routed_experts")
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
                routed_experts=routed_experts,
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
        if not tool_calls:
            raw_message = step.get("message", "") or ""
            parsed = None
            for start, character in enumerate(raw_message):
                if character != "{":
                    continue
                try:
                    candidate, _ = json.JSONDecoder().raw_decode(raw_message[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    parsed = candidate
                    break

            commands = (parsed or {}).get("commands", [])
            if not isinstance(commands, list):
                commands = []
            for command_index, command in enumerate(commands):
                if (
                    not isinstance(command, dict)
                    or not isinstance(command.get("keystrokes"), str)
                    or not command["keystrokes"]
                ):
                    continue
                try:
                    duration = float(command.get("duration", 1.0))
                except (TypeError, ValueError):
                    duration = 1.0
                tool_calls.append(
                    {
                        "tool_call_id": f"call_{agent_step_index}_{command_index + 1}",
                        "function_name": "bash_command",
                        "arguments": {"keystrokes": command["keystrokes"], "duration": duration},
                    }
                )

        observation = step.get("observation", {})
        results = observation.get("results", [])

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
