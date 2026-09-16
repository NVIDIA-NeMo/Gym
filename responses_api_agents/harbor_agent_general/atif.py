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
import logging
from uuid import uuid4

from harbor.models.trajectories import ContentPart, Step, Trajectory
from pydantic import ValidationError

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)


logger = logging.getLogger(__name__)


def convert_atif_to_gym_responses(trajectory: Trajectory, conversion_warnings: list[str] | None = None) -> list[dict]:
    output_items = []
    warnings = conversion_warnings if conversion_warnings is not None else []

    def warn(message: str) -> None:
        warnings.append(message)
        logger.warning("ATIF conversion: %s", message)

    def convert_input_content(parts: list[ContentPart], context: str) -> str | list[dict]:
        serialized_content = [part.model_dump(mode="json", exclude_none=True) for part in parts]
        local_image_paths = [
            part.source.path
            for part in parts
            if part.type == "image"
            and part.source is not None
            and not part.source.path.startswith(("http://", "https://", "data:"))
        ]
        if local_image_paths:
            warn(
                f"{context}: content serialized as JSON because local image paths are not portable Gym image "
                f"URLs: {local_image_paths}"
            )
            return json.dumps(serialized_content)

        return [
            {"type": "input_text", "text": part.text}
            if part.type == "text"
            else {"type": "input_image", "image_url": part.source.path, "detail": "auto"}
            for part in parts
            if part.type == "text" or part.source is not None
        ]

    def append_observations(step: Step) -> None:
        observation = step.observation
        for result_index, result in enumerate(observation.results if observation is not None else []):
            context = f"step {step.step_id} observation {result_index}"
            if isinstance(result.content, str):
                tool_output = result.content
            elif result.content is None:
                tool_output = ""
                warn(f"{context}: absent content represented as empty text")
            else:
                tool_output = convert_input_content(result.content, context)

            call_id = result.source_call_id
            if call_id is None:
                call_id = f"atif-step-{step.step_id}-observation-{result_index}"
                warn(f"{context}: missing source_call_id represented with synthetic call_id {call_id}")
            if result.subagent_trajectory_ref:
                warn(f"{context}: subagent trajectory references are preserved only in the source ATIF trajectory")
            if result.extra:
                warn(f"{context}: extra metadata is preserved only in the source ATIF trajectory")
            output_items.append(
                NeMoGymFunctionCallOutput(
                    call_id=call_id,
                    output=tool_output,
                    type="function_call_output",
                    id=f"fco_{uuid4().hex[:8]}",
                    status="completed",
                ).model_dump()
            )

    if trajectory.continued_trajectory_ref is not None:
        warn("continued_trajectory_ref is preserved only in the source ATIF trajectory")
    if trajectory.subagent_trajectories:
        warn("embedded subagent trajectories are preserved only in the source ATIF trajectory")
    if trajectory.notes is not None or trajectory.extra:
        warn("trajectory notes or extra metadata are preserved only in the source ATIF trajectory")
    if trajectory.final_metrics is not None:
        warn("ATIF final_metrics are preserved only in the source trajectory; Gym usage comes from Harbor")
    if trajectory.agent.tool_definitions:
        warn("ATIF tool definitions are preserved only in the source trajectory; Gym response tools remain empty")
    if trajectory.agent.extra:
        warn("ATIF agent extra metadata is preserved only in the source trajectory")

    for step in trajectory.steps:
        if step.source != "agent":
            message_content = (
                step.message
                if isinstance(step.message, str)
                else convert_input_content(step.message, f"step {step.step_id} {step.source} message")
            )
            output_items.append(
                NeMoGymEasyInputMessage(
                    role=step.source,
                    content=message_content,
                    type="message",
                ).model_dump()
            )
            append_observations(step)
            continue
        if (
            step.timestamp is not None
            or step.model_name is not None
            or step.reasoning_effort is not None
            or step.extra
        ):
            warn(
                f"step {step.step_id}: timestamp, model, reasoning effort, or extra metadata is preserved only "
                "in the source ATIF trajectory"
            )

        if step.reasoning_content:
            warn(f"step {step.step_id}: reasoning_content represented as a Gym reasoning summary")
            output_items.append(
                NeMoGymResponseReasoningItem(
                    id=f"rs_{uuid4().hex[:12]}",
                    summary=[NeMoGymSummary(text=step.reasoning_content, type="summary_text")],
                    status="completed",
                ).model_dump()
            )

        if isinstance(step.message, str):
            message_text = step.message
        else:
            message_text = json.dumps([part.model_dump(mode="json", exclude_none=True) for part in step.message])
            warn(
                f"step {step.step_id}: multimodal message serialized as JSON because Gym assistant output "
                "messages support only text or refusal content"
            )

        content = [
            NeMoGymResponseOutputText(
                annotations=[],
                text=message_text,
                type="output_text",
                logprobs=None,
            )
        ]
        metrics = step.metrics
        metrics_extra = metrics.extra if metrics is not None and metrics.extra is not None else {}
        routed_experts = metrics_extra.get("routed_experts")
        if metrics is not None and (
            metrics.prompt_tokens is not None
            or metrics.completion_tokens is not None
            or metrics.cached_tokens is not None
            or metrics.cost_usd is not None
            or set(metrics_extra) - {"routed_experts"}
        ):
            warn(
                f"step {step.step_id}: scalar metrics, cost, or unrecognized metric extras are preserved only "
                "in the source ATIF trajectory"
            )
        prompt_token_ids = metrics.prompt_token_ids if metrics is not None else None
        completion_token_ids = metrics.completion_token_ids if metrics is not None else None
        logprobs = metrics.logprobs if metrics is not None else None
        token_metadata_present = any(
            value is not None for value in (prompt_token_ids, completion_token_ids, logprobs, routed_experts)
        )
        token_metadata_issues = []
        if not completion_token_ids:
            token_metadata_issues.append("completion_token_ids are missing or empty")
        if prompt_token_ids is None:
            token_metadata_issues.append("prompt_token_ids are missing")
        if logprobs is None:
            token_metadata_issues.append("logprobs are missing")
        elif completion_token_ids is None or len(logprobs) != len(completion_token_ids):
            token_metadata_issues.append("completion_token_ids and logprobs have different lengths")
        if step.is_copied_context:
            token_metadata_issues.append("step is copied context")
        if step.llm_call_count not in (None, 1):
            token_metadata_issues.append(f"llm_call_count is {step.llm_call_count}")
        if step.reasoning_content or step.tool_calls:
            token_metadata_issues.append("tokens cannot be attributed across reasoning or tool-call items")

        message = None
        if token_metadata_present and not token_metadata_issues:
            try:
                message = NeMoGymResponseOutputMessageForTraining(
                    id=f"msg_{uuid4().hex[:12]}",
                    content=content,
                    role="assistant",
                    status="completed",
                    prompt_token_ids=prompt_token_ids,
                    generation_token_ids=completion_token_ids,
                    generation_log_probs=logprobs,
                    routed_experts=routed_experts,
                )
            except ValidationError as err:
                token_metadata_issues.append(f"Gym rejected token metadata: {err.errors(include_url=False)}")

        if message is None:
            message = NeMoGymResponseOutputMessage(
                id=f"msg_{uuid4().hex[:12]}",
                content=content,
                role="assistant",
                status="completed",
            )
            if token_metadata_present:
                warn(f"step {step.step_id}: training metadata omitted: {'; '.join(token_metadata_issues)}")
        output_items.append(message.model_dump())

        tool_calls = step.tool_calls or []
        if len(tool_calls) > 1:
            warn(
                f"step {step.step_id}: ATIF does not record whether multiple tool calls were parallel; Gym "
                "parallel_tool_calls remains false"
            )
        for tool_call in tool_calls:
            if tool_call.extra:
                warn(
                    f"step {step.step_id} tool call {tool_call.tool_call_id}: extra metadata is preserved only "
                    "in the source ATIF trajectory"
                )
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=json.dumps(tool_call.arguments),
                    call_id=tool_call.tool_call_id,
                    name=tool_call.function_name,
                    type="function_call",
                    id=f"fc_{uuid4().hex[:8]}",
                    status="completed",
                ).model_dump()
            )

        append_observations(step)

    return output_items
