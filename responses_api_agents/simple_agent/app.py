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
import uuid
from collections.abc import Mapping
from time import perf_counter, time
from typing import Any, List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, TypeAdapter, ValidationError

from nemo_gym._checkpoint import (
    EXPECTED_RESOURCE_STATE_REVISION_HEADER,
    RESOURCE_REQUEST_ID_HEADER,
    RESOURCE_STATE_REVISION_HEADER,
    AgentBoundaryKind,
    AgentBoundaryRecord,
    PendingModelPayload,
)
from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.rollout_correlation import (
    MODEL_CALL_ID_HEADER,
    current_attempt_index,
    current_logical_rollout_id,
    current_rollout_id,
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


_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"
_INTERNAL_RESOURCE_REVISIONS_KEY = "_ng_resource_state_revisions"
_INPUT_ITEMS_ADAPTER = TypeAdapter(List[NeMoGymResponseInputItem])


def _cookie_values(cookies: Any) -> dict[str, str]:
    return {
        name: str(getattr(cookie, "value", cookie))
        for name, cookie in (cookies.items() if cookies is not None else ())
    }


def _merge_cookies(current: Any, updates: Any) -> Any:
    if current is None or current is updates:
        return updates
    merged = _cookie_values(current)
    merged.update(_cookie_values(updates))
    return merged


class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None


class SimpleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SimpleAgent(SimpleResponsesAPIAgent):
    config: SimpleAgentConfig
    checkpoint_continuation_supported = True

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_url_path: str,
        resources_server_cookies: Any = None,
        task_id: str = "unscoped",
        rollout_id: str = "unscoped",
        collect_trajectory: bool = False,
        continuation: Optional[AgentBoundaryRecord] = None,
        request: Optional[Request] = None,
        initial_resource_revision: int = 0,
    ) -> tuple[NeMoGymResponse, TrajectoryRecord | None, Any, Any]:
        invocation_id = "root"
        tool_records: list[TrajectoryToolCall] = []
        model_calls: list[ModelCallRef] = []
        turns: list[TrajectoryTurn] = []
        trajectory_gaps: list[ObservationGap] = []
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs: list[NeMoGymResponseInputItem] = []
        usage = None
        turn_index = 0
        boundary_index = 0
        invocation_status = "completed"
        model_server_cookies = None
        resource_revision = initial_resource_revision
        model_response: Optional[NeMoGymResponse] = None
        model_call_id: Optional[str] = None
        pending_cursor = 0
        resource_request_id: Optional[str] = None
        pending_response_usage: Optional[dict[str, Any]] = None
        if continuation is not None:
            new_outputs.extend(_INPUT_ITEMS_ADAPTER.validate_python(continuation.output_items))
            usage = NeMoGymResponseUsage.model_validate(continuation.usage) if continuation.usage is not None else None
            turn_index = continuation.turn_index
            boundary_index = continuation.boundary_index
            model_server_cookies = continuation.agent_state.get("model_server_cookies") or None
            resources_server_cookies = continuation.agent_state.get("resources_server_cookies") or None
            resource_revision = continuation.resource_state_revisions.get(self.config.resources_server.name, 0)
            if continuation.pending_model is not None:
                pending = continuation.pending_model
                model_response = NeMoGymResponse.model_validate(pending.response)
                model_response.usage = None
                model_call_id = pending.model_call_id
                model_server_cookies = pending.model_server_cookies or None
                pending_cursor = pending.pending_action_cursor
                resource_request_id = pending.resource_request_id
                pending_response_usage = pending.usage
            else:
                restored_response = continuation.agent_state.get("last_model_response")
                if restored_response is not None:
                    model_response = NeMoGymResponse.model_validate(restored_response)
                else:
                    model_response = NeMoGymResponse(
                        id=continuation.last_committed_model_call_id or "restored-checkpoint-boundary",
                        created_at=continuation.created_at,
                        model=body.model or self.config.model_server.name,
                        object="response",
                        output=[],
                        parallel_tool_calls=True,
                        tool_choice="auto",
                        tools=[],
                    )

        async def commit_boundary(
            kind: AgentBoundaryKind,
            *,
            pending_model: Optional[PendingModelPayload] = None,
        ) -> None:
            logical_rollout_id = current_logical_rollout_id()
            attempt_index = current_attempt_index()
            execution = self.checkpoint_execution(request)
            if (
                self._checkpoint_participant is None
                or logical_rollout_id is None
                or attempt_index is None
                or execution is None
            ):
                return
            await self.checkpoint_participant().commit_boundary(
                execution,
                AgentBoundaryRecord(
                    rollout_id=logical_rollout_id,
                    attempt_index=attempt_index,
                    boundary_index=boundary_index,
                    turn_index=turn_index,
                    boundary_kind=kind,
                    pending_model=pending_model,
                    output_items=[item.model_dump(mode="json") for item in new_outputs],
                    usage=usage.model_dump(mode="json") if usage is not None else None,
                    last_committed_model_call_id=model_call_id,
                    resource_state_revisions={self.config.resources_server.name: resource_revision},
                    agent_state={
                        "model_server_cookies": _cookie_values(model_server_cookies),
                        "resources_server_cookies": _cookie_values(resources_server_cookies),
                        "last_model_response": (
                            model_response.model_copy(update={"output": new_outputs, "usage": usage}).model_dump(
                                mode="json"
                            )
                            if model_response is not None
                            else None
                        ),
                    },
                ),
            )

        while True:
            if resource_request_id is None:
                if self.config.max_steps and turn_index >= self.config.max_steps:
                    invocation_status = "incomplete"
                    break
                turn_index += 1
                new_body = body.model_copy(update={"input": body.input + new_outputs})
                if collect_trajectory:
                    turn_timestamp = time()

                model_http_response = await self.retry_checkpoint_refusal(
                    lambda: self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path=model_url_path,
                        json=new_body,
                        cookies=model_server_cookies,
                    ),
                    request=request,
                )
                model_call_id = None
                if self._checkpoint_participant is not None:
                    headers = getattr(model_http_response, "headers", None)
                    if isinstance(headers, Mapping):
                        model_call_id = headers.get(MODEL_CALL_ID_HEADER)
                await raise_for_status(model_http_response)
                model_response_json = await get_response_json(model_http_response)
                model_server_cookies = _merge_cookies(model_server_cookies, model_http_response.cookies)
                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as e:
                    raise RuntimeError(
                        f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                    ) from e

                full_model_response = model_response.model_dump(mode="json")
                output = model_response.output
                new_outputs.extend(output)
                if collect_trajectory:
                    turn_model_calls = []
                    if model_response.id:
                        model_call_ref = ModelCallRef(
                            model_ref=self.config.model_server, response_id=model_response.id
                        )
                        model_calls.append(model_call_ref)
                        turn_model_calls.append(model_call_ref)
                    else:
                        trajectory_gaps.append(
                            ObservationGap(
                                code="model_call_reference_unavailable",
                                invocation_id=invocation_id,
                                detail=f"turn:{turn_index}",
                            )
                        )
                    reasoning = [item.model_dump(mode="json") for item in output if item.type == "reasoning"] or None
                    answer = [item for item in output if item.type != "reasoning"]
                    turns.append(
                        TrajectoryTurn(
                            invocation_id=invocation_id,
                            task_id=task_id,
                            rollout_id=rollout_id,
                            turn_no=turn_index,
                            timestamp=turn_timestamp,
                            question=new_body.input,
                            answer=answer,
                            reasoning_content=reasoning,
                            step_count=len(tool_records),
                            model_calls=turn_model_calls,
                        )
                    )

                pending_response_usage = (
                    model_response.usage.model_dump(mode="json") if model_response.usage is not None else None
                )
                usage = accumulate_response_usage(usage, model_response.usage)
                model_response.usage = None
                model_call_id = model_call_id or model_response.id or f"turn-{turn_index}"
                pending_cursor = 0
                resource_request_id = uuid.uuid4().hex
                boundary_index += 1
                pending_model = None
                if self._checkpoint_participant is not None and self.checkpoint_execution(request) is not None:
                    pending_model = PendingModelPayload(
                        model_call_id=model_call_id,
                        response=full_model_response,
                        model_server_cookies=_cookie_values(model_server_cookies),
                        usage=pending_response_usage,
                        pending_action_cursor=pending_cursor,
                        resource_request_id=resource_request_id,
                    )
                await commit_boundary(
                    AgentBoundaryKind.PENDING_MODEL,
                    pending_model=pending_model,
                )

            assert model_response is not None
            output = model_response.output
            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if model_response.incomplete_details or (not all_fn_calls and all_output_messages):
                invocation_status = "incomplete" if model_response.incomplete_details else invocation_status
                resource_request_id = None
                boundary_index += 1
                await commit_boundary(AgentBoundaryKind.TURN_COMPLETE)
                break

            for action_index in range(pending_cursor, len(all_fn_calls)):
                output_function_call = all_fn_calls[action_index]
                assert resource_request_id is not None
                if collect_trajectory:
                    started_at = time()
                    started_monotonic = perf_counter()
                try:
                    parsed_arguments = json.loads(output_function_call.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    tool_output = json.dumps({"error": f"Invalid tool call arguments: {e!r}"})
                    if collect_trajectory:
                        error_type = type(e).__name__
                        tool_status = "failed"
                else:
                    # Resource-server errors are valid model-visible tool outputs.
                    api_response = await self.retry_checkpoint_refusal(
                        lambda: self.server_client.post(
                            server_name=self.config.resources_server.name,
                            url_path=f"/{output_function_call.name}",
                            json=parsed_arguments,
                            cookies=resources_server_cookies,
                            headers={
                                EXPECTED_RESOURCE_STATE_REVISION_HEADER: str(resource_revision),
                                RESOURCE_REQUEST_ID_HEADER: resource_request_id,
                            },
                        ),
                        request=request,
                    )
                    tool_output = (await api_response.content.read()).decode()
                    resources_server_cookies = _merge_cookies(resources_server_cookies, api_response.cookies)
                    headers = getattr(api_response, "headers", None)
                    if isinstance(headers, Mapping) and headers.get(RESOURCE_STATE_REVISION_HEADER) is not None:
                        resource_revision = int(headers[RESOURCE_STATE_REVISION_HEADER])
                    if collect_trajectory:
                        completed = 200 <= api_response.status < 400
                        tool_status = "completed" if completed else "failed"
                        error_type = None if completed else f"http_{api_response.status}"

                if collect_trajectory:
                    tool_records.append(
                        TrajectoryToolCall(
                            invocation_id=invocation_id,
                            tool_call_id=output_function_call.call_id,
                            tool_name=output_function_call.name,
                            started_at=started_at,
                            completed_at=max(started_at, time()),
                            duration_ms=(perf_counter() - started_monotonic) * 1000,
                            timing_source="executor",
                            status=tool_status,
                            error_type=error_type,
                            output=tool_output,
                        )
                    )

                function_output = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=tool_output,
                )
                new_outputs.append(function_output)
                pending_cursor = action_index + 1
                if pending_cursor < len(all_fn_calls):
                    resource_request_id = uuid.uuid4().hex
                boundary_index += 1
                pending_model = None
                if self._checkpoint_participant is not None and self.checkpoint_execution(request) is not None:
                    pending_model = PendingModelPayload(
                        model_call_id=model_call_id or model_response.id or f"turn-{turn_index}",
                        response=model_response.model_copy(
                            update={
                                "usage": (
                                    NeMoGymResponseUsage.model_validate(pending_response_usage)
                                    if pending_response_usage is not None
                                    else None
                                )
                            }
                        ).model_dump(mode="json"),
                        model_server_cookies=_cookie_values(model_server_cookies),
                        usage=pending_response_usage,
                        pending_action_cursor=pending_cursor,
                        resource_request_id=resource_request_id,
                    )
                await commit_boundary(
                    AgentBoundaryKind.PENDING_MODEL,
                    pending_model=pending_model,
                )

            if collect_trajectory and all_fn_calls:
                turns[-1].step_count = len(tool_records)

            resource_request_id = None
            boundary_index += 1
            await commit_boundary(AgentBoundaryKind.TURN_COMPLETE)

        if model_response is None:
            raise RuntimeError("agent episode ended before producing or restoring a model response")
        model_response.output = new_outputs
        model_response.usage = usage
        trajectory = None
        if collect_trajectory:
            invocation = AgentInvocation(
                invocation_id=invocation_id,
                status=invocation_status,
                model_calls=model_calls,
                conversation=[*body.input, *new_outputs],
            )
            trajectory = TrajectoryRecord(
                task_id=task_id,
                rollout_id=rollout_id,
                invocations=[invocation],
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
        rollout_id = current_rollout_id()
        if rollout_id is None and isinstance(path_params, Mapping):
            rollout_id = path_params.get("rollout_id")
        collect_trajectory = self._model_call_capture_enabled() and isinstance(rollout_id, str)
        continuation = self.checkpoint_continuation(body, request)
        initial_resource_revision = int(request.headers.get(RESOURCE_STATE_REVISION_HEADER, "0"))
        model_response, trajectory, model_server_cookies, resources_server_cookies = await self._create_episode(
            body,
            model_url_path=self.url_path_for_request("/v1/responses", request),
            resources_server_cookies=request.cookies,
            rollout_id=rollout_id or "unscoped",
            collect_trajectory=collect_trajectory,
            continuation=continuation,
            request=request,
            initial_resource_revision=initial_resource_revision,
        )
        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)
        if trajectory is not None:
            model_response = model_response.model_copy(
                update={_INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")}
            )
        if self.checkpoint_execution(request) is not None:
            model_response = model_response.model_copy(
                update={
                    _INTERNAL_RESOURCE_REVISIONS_KEY: {
                        self.config.resources_server.name: self.checkpoint_execution(
                            request
                        ).boundary.resource_state_revisions[self.config.resources_server.name]
                    }
                }
            )
        return model_response

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        cookies = request.cookies
        continuation = self.checkpoint_continuation(body)
        resource_revision = 0
        if continuation is None:
            seed_headers = (
                {RESOURCE_REQUEST_ID_HEADER: uuid.uuid4().hex} if self.checkpoint_execution() is not None else None
            )
            seed_session_response = await self.retry_checkpoint_refusal(
                lambda: self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/seed_session",
                    json=body.model_dump(),
                    cookies=cookies,
                    **({"headers": seed_headers} if seed_headers is not None else {}),
                )
            )
            await raise_for_status(seed_session_response)
            cookies = seed_session_response.cookies
            seed_headers = getattr(seed_session_response, "headers", None)
            if isinstance(seed_headers, Mapping) and seed_headers.get(RESOURCE_STATE_REVISION_HEADER) is not None:
                resource_revision = int(seed_headers[RESOURCE_STATE_REVISION_HEADER])
            logical_rollout_id = current_logical_rollout_id()
            attempt_index = current_attempt_index()
            execution = self.checkpoint_execution()
            if logical_rollout_id is not None and attempt_index is not None and execution is not None:
                await self.checkpoint_participant().commit_boundary(
                    execution,
                    AgentBoundaryRecord(
                        rollout_id=logical_rollout_id,
                        attempt_index=attempt_index,
                        boundary_index=0,
                        turn_index=0,
                        boundary_kind=AgentBoundaryKind.TURN_COMPLETE,
                        output_items=[],
                        resource_state_revisions={self.config.resources_server.name: resource_revision},
                        agent_state={"resources_server_cookies": _cookie_values(cookies)},
                    ),
                )
        else:
            cookies = continuation.agent_state.get("resources_server_cookies") or cookies
            resource_revision = continuation.resource_state_revisions.get(self.config.resources_server.name, 0)

        execution_headers = self.checkpoint_execution_headers()
        if execution_headers is not None:
            execution_headers[RESOURCE_STATE_REVISION_HEADER] = str(resource_revision)
        response = await self.retry_checkpoint_refusal(
            lambda: self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
                **({"headers": execution_headers} if execution_headers is not None else {}),
            )
        )
        await raise_for_status(response)
        model_response_json = await get_response_json(response)
        cookies = response.cookies
        resource_revisions = model_response_json.pop(_INTERNAL_RESOURCE_REVISIONS_KEY, {})
        if isinstance(resource_revisions, Mapping):
            resource_revision = int(resource_revisions.get(self.config.resources_server.name, resource_revision))

        trajectory = None
        expected_rollout_id = self.rollout_id_from_run(body)
        raw_trajectory = (
            model_response_json.pop(_INTERNAL_TRAJECTORY_KEY, None) if expected_rollout_id is not None else None
        )
        if isinstance(raw_trajectory, dict):
            trajectory = TrajectoryRecord.model_validate(raw_trajectory)
            extra = body.model_extra or {}
            task_id = next(
                (
                    str(extra[key])
                    for key in ("task_id", "problem_id", "instance_id", "_ng_task_index")
                    if extra.get(key) is not None
                ),
                "unknown",
            )
            rollout_id = expected_rollout_id or trajectory.rollout_id
            trajectory = trajectory.model_copy(
                update={
                    "task_id": task_id,
                    "rollout_id": rollout_id,
                    "turns": [
                        turn.model_copy(update={"task_id": task_id, "rollout_id": rollout_id})
                        for turn in trajectory.turns
                    ],
                }
            )

        if self.config.skip_verification:
            result = body.model_dump() | {
                "response": model_response_json,
                "reward": float(self.config.skip_verification_reward),
                "verification_skipped": True,
            }
        else:
            verify_request = SimpleAgentVerifyRequest.model_validate(
                body.model_dump() | {"response": model_response_json}
            )
            verify_request_id = uuid.uuid4().hex
            verify_response = await self.retry_checkpoint_refusal(
                lambda: self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=verify_request.model_dump(),
                    cookies=cookies,
                    headers={
                        EXPECTED_RESOURCE_STATE_REVISION_HEADER: str(resource_revision),
                        RESOURCE_REQUEST_ID_HEADER: verify_request_id,
                    },
                )
            )
            await raise_for_status(verify_response)
            result = await get_response_json(verify_response)
        if trajectory is not None:
            resolved = result.get("resolved")
            if isinstance(resolved, bool) and trajectory.turns:
                trajectory.turns[-1].resolved = resolved
            else:
                trajectory.gaps.append(ObservationGap(code="resolution_unavailable", invocation_id="root"))
            result["ng_trajectory"] = trajectory.model_dump(mode="json")
        return SimpleAgentVerifyResponse.model_validate(result)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)

        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    SimpleAgent.run_webserver()
