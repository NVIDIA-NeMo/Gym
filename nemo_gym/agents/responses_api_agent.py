# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The built-in Responses API model/tool-loop agent harness."""

import json
from collections.abc import Mapping
from time import perf_counter, time
from typing import Any, List

from fastapi import Request, Response
from pydantic import ValidationError

from nemo_gym.agents.base import (
    BaseResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    Body,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    accumulate_response_usage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"


class ResponsesAPIAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None


class ResponsesAPIAgent(BaseResponsesAPIAgent):
    """A reusable model/tool-loop harness with no rollout endpoint."""

    config: ResponsesAPIAgentConfig

    async def _run_turn(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_url_path: str,
        resources_server_cookies: Any = None,
        task_id: str = "unscoped",
        rollout_id: str = "unscoped",
        collect_trajectory: bool = False,
    ) -> tuple[NeMoGymResponse, TrajectoryRecord | None, Any, Any]:
        invocation_id = "root"
        tool_records: list[TrajectoryToolCall] = []
        model_calls: list[ModelCallRef] = []
        turns: list[TrajectoryTurn] = []
        trajectory_gaps: list[ObservationGap] = []
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        usage = None
        step = 0
        invocation_status = "completed"
        model_server_cookies = None

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})
            if collect_trajectory:
                turn_timestamp = time()

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=model_url_path,
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as error:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from error

            output = model_response.output
            new_outputs.extend(output)
            if collect_trajectory:
                turn_model_calls = []
                if model_response.id:
                    model_call_ref = ModelCallRef(model_ref=self.config.model_server, response_id=model_response.id)
                    model_calls.append(model_call_ref)
                    turn_model_calls.append(model_call_ref)
                else:
                    trajectory_gaps.append(
                        ObservationGap(
                            code="model_call_reference_unavailable",
                            invocation_id=invocation_id,
                            detail=f"turn:{step}",
                        )
                    )
                reasoning = [item.model_dump(mode="json") for item in output if item.type == "reasoning"] or None
                answer = [item for item in output if item.type != "reasoning"]
                turns.append(
                    TrajectoryTurn(
                        invocation_id=invocation_id,
                        task_id=task_id,
                        rollout_id=rollout_id,
                        turn_no=step,
                        timestamp=turn_timestamp,
                        question=new_body.input,
                        answer=answer,
                        reasoning_content=reasoning,
                        step_count=len(tool_records),
                        model_calls=turn_model_calls,
                    )
                )

            usage = accumulate_response_usage(usage, model_response.usage)
            model_response.usage = None
            if model_response.incomplete_details:
                invocation_status = "incomplete"
                break

            function_calls: List[NeMoGymResponseFunctionToolCall] = [
                item for item in output if item.type == "function_call"
            ]
            messages: List[NeMoGymResponseOutputMessage] = [
                item for item in output if item.type == "message" and item.role == "assistant"
            ]
            if not function_calls and messages:
                break

            for function_call in function_calls:
                if collect_trajectory:
                    started_at = time()
                    started_monotonic = perf_counter()
                try:
                    parsed_arguments = json.loads(function_call.arguments)
                except (json.JSONDecodeError, TypeError) as error:
                    tool_output = json.dumps({"error": f"Invalid tool call arguments: {error!r}"})
                    if collect_trajectory:
                        error_type = type(error).__name__
                        tool_status = "failed"
                else:
                    api_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=f"/{function_call.name}",
                        json=parsed_arguments,
                        cookies=resources_server_cookies,
                    )
                    tool_output = (await api_response.content.read()).decode()
                    resources_server_cookies = api_response.cookies
                    if collect_trajectory:
                        completed = 200 <= api_response.status < 400
                        tool_status = "completed" if completed else "failed"
                        error_type = None if completed else f"http_{api_response.status}"

                if collect_trajectory:
                    tool_records.append(
                        TrajectoryToolCall(
                            invocation_id=invocation_id,
                            tool_call_id=function_call.call_id,
                            tool_name=function_call.name,
                            started_at=started_at,
                            completed_at=max(started_at, time()),
                            duration_ms=(perf_counter() - started_monotonic) * 1000,
                            timing_source="executor",
                            status=tool_status,
                            error_type=error_type,
                            output=tool_output,
                        )
                    )
                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=function_call.call_id,
                        output=tool_output,
                    )
                )

            if collect_trajectory and function_calls:
                turns[-1].step_count = len(tool_records)
            if self.config.max_steps and step >= self.config.max_steps:
                invocation_status = "incomplete"
                break

        model_response.output = new_outputs
        model_response.usage = usage
        trajectory = None
        if collect_trajectory:
            trajectory = TrajectoryRecord(
                task_id=task_id,
                rollout_id=rollout_id,
                invocations=[
                    AgentInvocation(
                        invocation_id=invocation_id,
                        status=invocation_status,
                        model_calls=model_calls,
                        conversation=[*body.input, *new_outputs],
                    )
                ],
                turns=turns,
                tool_calls=tool_records,
                gaps=trajectory_gaps,
            )
        return model_response, trajectory, model_server_cookies, resources_server_cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        collect_trajectory = self._model_call_capture_enabled() and isinstance(rollout_id, str)
        model_response, trajectory, model_server_cookies, resources_server_cookies = await self._run_turn(
            body,
            model_url_path=self.url_path_for_request("/v1/responses", request),
            resources_server_cookies=request.cookies,
            rollout_id=rollout_id or "unscoped",
            collect_trajectory=collect_trajectory,
        )
        for key, value in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(key, value)
        if trajectory is not None:
            model_response = model_response.model_copy(
                update={INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")}
            )
        return model_response
