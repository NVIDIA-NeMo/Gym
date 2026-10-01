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
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter, time
from typing import Any, List

from fastapi import Request, Response
from pydantic import ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    AgentToolLoopPolicy,
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
    NeMoGymResponseOutputMessage,
    accumulate_response_usage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from nemo_gym.tool_access import ContextualToolCallRequest, ContextualToolCallResponse, DirectHTTPToolAccess


LOG = logging.getLogger(__name__)

_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"
_INTERNAL_COMPLETED_TURN_KEY = "_ng_completed_turn"
TOOL_CALL_ID_HEADER = "X-NeMo-Gym-Tool-Call-Id"
TURN_ID_HEADER = "X-NeMo-Gym-Turn-Id"


def _merge_cookie_values(current: Mapping[str, str] | None, updates: Mapping[str, Any]) -> dict[str, str]:
    """Merge aiohttp response morsels into a JSON-safe cookie mapping."""
    merged = dict(current or {})
    merged.update({str(name): str(getattr(value, "value", value)) for name, value in updates.items()})
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


@dataclass
class _SimpleAgentSession:
    agent_session_id: str
    episode_id: Any
    task_id: Any
    resources_cookies: dict[str, str]
    tool_batch_path: str | None = None
    tool_headers: dict[str, str] = field(default_factory=dict)
    tool_call_context: dict[str, Any] | None = None
    tool_loop_policy: AgentToolLoopPolicy | None = None
    trajectories: list[TrajectoryRecord] = field(default_factory=list)


class SimpleAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: SimpleAgentConfig
    session_id_to_state: dict[str, _SimpleAgentSession] = Field(default_factory=dict)

    async def seed_agent_session(
        self,
        request: Request,
        body: AgentSeedSessionRequest,
    ) -> AgentSeedSessionResponse:
        """Activate episode-scoped Resources access for this Agent."""
        session_id = request.session[SESSION_ID_KEY]
        if session_id in self.session_id_to_state:
            raise RuntimeError("SimpleAgent session is already active")
        direct_accesses = [
            access for access in self.effective_tool_accesses(body) if isinstance(access, DirectHTTPToolAccess)
        ]
        if len(direct_accesses) > 1:
            raise ValueError("SimpleAgent supports at most one direct HTTP Resources access")
        resources_cookies = dict(direct_accesses[0].cookies) if direct_accesses else {}
        self.session_id_to_state[session_id] = _SimpleAgentSession(
            agent_session_id=body.agent_session_id,
            episode_id=body.episode_id,
            task_id=body.task_id,
            resources_cookies=resources_cookies,
            tool_batch_path=direct_accesses[0].batch_path if direct_accesses else None,
            tool_headers=dict(direct_accesses[0].headers) if direct_accesses else {},
            tool_call_context=deepcopy(direct_accesses[0].tool_call_context) if direct_accesses else None,
            tool_loop_policy=body.tool_loop_policy,
        )
        return AgentSeedSessionResponse(agent_session_id=body.agent_session_id)

    async def close_agent_session(
        self,
        request: Request,
        body: AgentCloseSessionRequest,
    ) -> AgentCloseSessionResponse:
        """Close an Agent session and return its tool/model observations."""
        session_id = request.session[SESSION_ID_KEY]
        state = self.session_id_to_state.get(session_id)
        if state is None:
            raise RuntimeError("SimpleAgent session is not active")
        if state.agent_session_id != body.agent_session_id or state.episode_id != body.episode_id:
            raise ValueError("SimpleAgent session does not match the active episode")
        records = []
        gaps = []
        for trajectory in state.trajectories:
            records.extend(trajectory.invocations)
            records.extend(
                ToolCallObservation.model_validate(tool.model_dump(exclude={"output"}))
                for tool in trajectory.tool_calls
            )
            gaps.extend(trajectory.gaps)
        observations = (
            AgentObservationBundle(source=self.config.name, records=records, gaps=gaps) if records or gaps else None
        )
        del self.session_id_to_state[session_id]
        return AgentCloseSessionResponse(
            agent_session_id=body.agent_session_id,
            agent_observations=observations,
            resources_cookies=state.resources_cookies,
        )

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_url_path: str,
        resources_server_cookies: Any = None,
        task_id: str = "unscoped",
        rollout_id: str = "unscoped",
        collect_trajectory: bool = False,
        invocation_id: str = "root",
        tool_loop_policy: AgentToolLoopPolicy | None = None,
        tool_batch_path: str | None = None,
        tool_headers: Mapping[str, str] | None = None,
        tool_call_context: Mapping[str, Any] | None = None,
    ) -> tuple[NeMoGymResponse, TrajectoryRecord | None, Any, Any]:
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
        executed_call_count = 0
        force_synthesis = False
        completed_transcript: list[dict[str, Any]] = []
        completed_model_calls: list[dict[str, Any]] = []
        semantic_turn_id = (
            str(tool_call_context["turn_id"])
            if tool_call_context is not None and tool_call_context.get("turn_id") is not None
            else (tool_headers or {}).get(TURN_ID_HEADER)
        )
        completed_resources_context: dict[str, Any] | None = None

        while True:
            step += 1
            model_input = _model_replay_input(body.input, new_outputs, tool_loop_policy)
            new_body = body.model_copy(update={"input": model_input})
            final_synthesis_step = (
                tool_loop_policy is not None
                and tool_loop_policy.final_synthesis
                and (force_synthesis or step == tool_loop_policy.max_model_calls)
            )
            if final_synthesis_step:
                new_body = new_body.model_copy(update={"tools": [], "tool_choice": "auto"})
            if collect_trajectory:
                turn_timestamp = time()

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=model_url_path,
                json=new_body,
                cookies=model_server_cookies,
            )
            # We raise for status here since we expect model calls to always work.
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            output = model_response.output
            raw_usage = model_response.usage
            assistant_response = _assistant_response(model_response)
            completed_model_calls.append(
                {
                    "response": assistant_response,
                    "usage": (
                        {
                            "input_tokens": raw_usage.input_tokens,
                            "output_tokens": raw_usage.output_tokens,
                        }
                        if raw_usage is not None
                        else None
                    ),
                }
            )
            completed_transcript.append(assistant_response)
            tool_call_cap_reached = False
            if (
                semantic_turn_id is None
                and tool_loop_policy is not None
                and tool_loop_policy.max_tool_calls is not None
            ):
                remaining_calls = max(tool_loop_policy.max_tool_calls - executed_call_count, 0)
                filtered_output = []
                for item in output:
                    if item.type != "function_call":
                        filtered_output.append(item)
                    elif remaining_calls > 0:
                        filtered_output.append(item)
                        remaining_calls -= 1
                    else:
                        tool_call_cap_reached = True
                output = filtered_output
                model_response.output = output
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
                            code="model_call_reference_unavailable", invocation_id=invocation_id, detail=f"turn:{step}"
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

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if final_synthesis_step and all_fn_calls:
                invocation_status = "incomplete"
                break
            if not all_fn_calls:
                if not all_output_messages:
                    invocation_status = "incomplete"
                    termination_message = (
                        "Ending trajectory: model returned no assistant message or tool calls "
                        "(reasoning-only or empty output) without reported truncation. "
                        "This is the stop-token case (finish_reason='stop'), not length truncation "
                        "(finish_reason='length', handled separately via incomplete_details). "
                        "This indicates either a badly trained model requiring training-level fixes "
                        "or a bug in the inference engine."
                    )
                    termination_reason = "incomplete_reasoning" if output else "empty_output"
                    model_response.status = "incomplete"
                    model_response.metadata = {
                        **(model_response.metadata or {}),
                        "ng_termination_reason": termination_reason,
                        "ng_termination_message": termination_message,
                    }
                    LOG.warning(
                        "%s model_server=%s response_id=%s rollout_id=%s step=%s",
                        termination_message,
                        self.config.model_server.name,
                        model_response.id,
                        rollout_id,
                        step,
                    )
                break

            parsed_calls: list[tuple[NeMoGymResponseFunctionToolCall, dict[str, Any]]] = []
            parse_errors: dict[str, str] = {}
            for output_function_call in all_fn_calls:
                try:
                    arguments = json.loads(output_function_call.arguments)
                    parsed_calls.append(
                        (
                            output_function_call,
                            arguments if isinstance(arguments, dict) or tool_call_context is None else {},
                        )
                    )
                except (json.JSONDecodeError, TypeError) as error:
                    if tool_call_context is not None:
                        parsed_calls.append((output_function_call, {}))
                    else:
                        parse_errors[output_function_call.call_id] = json.dumps(
                            {"error": f"Invalid tool call arguments: {error!r}"}
                        )

            batch_outputs: dict[str, str] = {}
            if tool_batch_path is not None and all_fn_calls and semantic_turn_id is not None:
                batch_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=tool_batch_path,
                    json={
                        "turn_id": semantic_turn_id,
                        "round_id": f"round-{step:06d}",
                        "assistant_response": assistant_response,
                    },
                    cookies=resources_server_cookies,
                    headers=dict(tool_headers or {}),
                )
                receipt = await get_response_json(batch_response)
                if not isinstance(receipt, Mapping) or not isinstance(receipt.get("receipts"), list):
                    raise RuntimeError("Resources semantic tool batch returned invalid receipts")
                batch_outputs = {str(item["tool_call_id"]): str(item["payload"]) for item in receipt["receipts"]}
                tool_call_cap_reached = bool(receipt.get("limit_reached"))
                resources_server_cookies = _merge_cookie_values(resources_server_cookies, batch_response.cookies)
            elif tool_batch_path is not None and len(parsed_calls) > 1:
                batch_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=tool_batch_path,
                    json=[
                        {
                            "tool_call_id": call.call_id,
                            "tool_name": call.name,
                            "arguments": arguments,
                        }
                        for call, arguments in parsed_calls
                    ],
                    cookies=resources_server_cookies,
                )
                payloads = await get_response_json(batch_response)
                if not isinstance(payloads, list) or len(payloads) != len(parsed_calls):
                    raise RuntimeError("Resources tool batch returned an invalid payload list")
                batch_outputs = {
                    call.call_id: str(payload) for (call, _), payload in zip(parsed_calls, payloads, strict=True)
                }
                resources_server_cookies = _merge_cookie_values(resources_server_cookies, batch_response.cookies)

            for output_function_call in all_fn_calls:
                if semantic_turn_id is not None and output_function_call.call_id not in batch_outputs:
                    if tool_call_context is None:
                        continue
                if collect_trajectory:
                    started_at = time()
                    started_monotonic = perf_counter()
                if output_function_call.call_id in batch_outputs:
                    tool_output = batch_outputs[output_function_call.call_id]
                    if collect_trajectory:
                        completed = 200 <= batch_response.status < 400
                        tool_status = "completed" if completed else "failed"
                        error_type = None if completed else f"http_{batch_response.status}"
                elif output_function_call.call_id in parse_errors:
                    tool_output = parse_errors[output_function_call.call_id]
                    if collect_trajectory:
                        error_type = "invalid_arguments"
                        tool_status = "failed"
                else:
                    parsed_arguments = next(
                        arguments for call, arguments in parsed_calls if call.call_id == output_function_call.call_id
                    )
                    # Resource-server errors are valid model-visible tool outputs.
                    request_body: dict[str, Any]
                    if tool_call_context is None:
                        request_body = parsed_arguments
                    else:
                        request_body = ContextualToolCallRequest(
                            arguments=parsed_arguments,
                            tool_call_context=dict(tool_call_context),
                            tool_call_id=output_function_call.call_id,
                            round_id=f"round-{step:06d}",
                            assistant_response=assistant_response,
                        ).model_dump(mode="json")
                    api_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=f"/{output_function_call.name}",
                        json=request_body,
                        cookies=resources_server_cookies,
                        headers={
                            **dict(tool_headers or {}),
                            TOOL_CALL_ID_HEADER: output_function_call.call_id,
                        },
                    )
                    resources_server_cookies = _merge_cookie_values(resources_server_cookies, api_response.cookies)
                    if tool_call_context is None:
                        tool_output = (await api_response.content.read()).decode()
                    else:
                        contextual_result = ContextualToolCallResponse.model_validate(
                            await get_response_json(api_response)
                        )
                        completed_resources_context = contextual_result.tool_call_context
                        tool_call_cap_reached = contextual_result.limit_reached
                        if contextual_result.output is None:
                            continue
                        tool_output = contextual_result.output
                    if collect_trajectory:
                        completed = 200 <= api_response.status < 400
                        tool_status = "completed" if completed else "failed"
                        error_type = None if completed else f"http_{api_response.status}"

                executed_call_count += 1
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

                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=output_function_call.call_id,
                        output=tool_output,
                    )
                )
                completed_transcript.append(
                    {
                        "role": "tool",
                        "content": tool_output,
                        "tool_call_id": output_function_call.call_id,
                    }
                )

            if collect_trajectory and all_fn_calls:
                turns[-1].step_count = len(tool_records)

            if all_fn_calls and tool_loop_policy is not None and tool_loop_policy.mode == "single":
                force_synthesis = tool_loop_policy.final_synthesis
                if not force_synthesis:
                    break

            if (
                semantic_turn_id is None
                and tool_loop_policy is not None
                and tool_loop_policy.max_tool_calls is not None
                and executed_call_count >= tool_loop_policy.max_tool_calls
            ):
                tool_call_cap_reached = True
            if tool_call_cap_reached and tool_loop_policy is not None:
                force_synthesis = tool_loop_policy.final_synthesis

            if tool_loop_policy is not None and step >= tool_loop_policy.max_model_calls:
                invocation_status = "incomplete"
                break

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                invocation_status = "incomplete"
                break

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
        completed_turn = (
            {
                "transcript": completed_transcript,
                "model_calls": completed_model_calls,
                "resources_context": completed_resources_context,
            }
            if tool_loop_policy is not None
            else None
        )
        if completed_turn is not None:
            model_response = model_response.model_copy(update={_INTERNAL_COMPLETED_TURN_KEY: completed_turn})
        return model_response, trajectory, model_server_cookies, resources_server_cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        session_id = request.session[SESSION_ID_KEY]
        session = self.session_id_to_state.get(session_id)
        collect_trajectory = session is not None or (
            self._model_call_capture_enabled() and isinstance(rollout_id, str)
        )
        model_response, trajectory, model_server_cookies, resources_server_cookies = await self._create_episode(
            body,
            model_url_path=self.url_path_for_request("/v1/responses", request),
            resources_server_cookies=session.resources_cookies if session is not None else request.cookies,
            task_id=str(session.task_id) if session is not None else "unscoped",
            rollout_id=rollout_id or "unscoped",
            collect_trajectory=collect_trajectory,
            invocation_id=f"activation-{len(session.trajectories)}" if session is not None else "root",
            tool_loop_policy=session.tool_loop_policy if session is not None else None,
            tool_batch_path=session.tool_batch_path if session is not None else None,
            tool_headers=session.tool_headers if session is not None else None,
            tool_call_context=session.tool_call_context if session is not None else None,
        )
        if session is not None:
            session.resources_cookies = dict(resources_server_cookies)
            if trajectory is not None:
                session.trajectories.append(trajectory)
        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)
        if trajectory is not None:
            model_response = model_response.model_copy(
                update={_INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")}
            )
        return model_response

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(response)
        model_response_json = await get_response_json(response)
        cookies = response.cookies

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
            verify_payload = body.model_dump() | {"response": model_response_json}
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_payload,
                cookies=cookies,
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


def _assistant_response(response: NeMoGymResponse) -> dict[str, Any]:
    reasoning: list[str] = []
    content: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in response.output:
        if item.type == "reasoning":
            reasoning.extend(
                part.text for part in [*item.summary, *(item.content or [])] if getattr(part, "text", None)
            )
        elif item.type == "function_call":
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                }
            )
        elif item.type == "message" and item.role == "assistant":
            content.extend(
                text
                for part in item.content
                if (text := getattr(part, "text", None) or getattr(part, "refusal", None))
            )
    result: dict[str, Any] = {"role": "assistant", "content": "\n".join(content)}
    if reasoning:
        result["reasoning_content"] = "\n".join(reasoning)
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _model_replay_input(
    base_input: list[Any],
    outputs: list[Any],
    policy: AgentToolLoopPolicy | None,
) -> list[Any]:
    combined = deepcopy([*base_input, *outputs])
    if policy is None:
        return combined
    replay = [item for item in combined if policy.replay_reasoning or _replay_item_value(item, "type") != "reasoning"]
    if not policy.project_document_tool_results:
        return replay
    newest: dict[str, tuple[int, int]] = {}
    retrievals: list[tuple[int, int, dict[str, Any]]] = []
    call_names: dict[str, str] = {}
    current_user_turn = 0
    for index, item in enumerate(replay):
        item_type = _replay_item_value(item, "type")
        if item_type == "message" and _replay_item_value(item, "role") == "user":
            current_user_turn += 1
            continue
        if item_type == "function_call":
            call_names[str(_replay_item_value(item, "call_id") or "")] = str(_replay_item_value(item, "name") or "")
            continue
        call_id = str(_replay_item_value(item, "call_id") or "")
        if item_type != "function_call_output" or call_names.get(call_id) != "kb_search":
            continue
        try:
            payload = json.loads(str(_replay_item_value(item, "output") or ""))
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            continue
        retrievals.append((index, current_user_turn, payload))
        for document_index, document in enumerate(payload["results"]):
            if isinstance(document, Mapping) and document.get("id"):
                newest[str(document["id"])] = (index, document_index)
    for index, document_turn, payload in retrievals:
        documents = []
        changed = False
        for document_index, document in enumerate(payload["results"]):
            if not isinstance(document, Mapping) or not document.get("id"):
                documents.append(document)
                continue
            keep_body = (
                newest[str(document["id"])] == (index, document_index) and current_user_turn - document_turn < 8
            )
            if keep_body:
                documents.append(document)
            else:
                changed = True
                documents.append({key: value for key, value in document.items() if key != "body"})
        if changed:
            replay[index] = _replace_replay_item(
                replay[index],
                output=json.dumps({**payload, "results": documents}, ensure_ascii=False),
            )
    return replay


def _replay_item_value(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def _replace_replay_item(item: Any, **updates: Any) -> Any:
    if isinstance(item, Mapping):
        return {**item, **updates}
    return item.model_copy(update=updates)


if __name__ == "__main__":
    SimpleAgent.run_webserver()
