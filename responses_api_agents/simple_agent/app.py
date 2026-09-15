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
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter, time
from typing import Any, List

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

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
from nemo_gym.episode import AgentSeedSessionRequest, DirectResourcesToolAccess
from nemo_gym.episode_sessions import AgentCloseSessionResult, AgentSession
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
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"


@dataclass
class SimpleAgentSessionState:
    resources_cookies: dict[str, Any]
    resources_headers: dict[str, str]
    observations: AgentObservationBundle | None = None


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
    supports_agent_sessions = True

    async def initialize_agent_session_state(
        self,
        agent_session_id: str,
        body: AgentSeedSessionRequest,
    ) -> SimpleAgentSessionState:
        if body.sandbox_access is not None:
            raise ValueError("simple_agent does not run in or control a sandbox")
        if not isinstance(body.resources_access, DirectResourcesToolAccess):
            raise ValueError("simple_agent requires direct HTTP resources access")
        expected_base_url = self.server_client._resolve_base_url(self.config.resources_server.name)
        if body.resources_access.base_url.rstrip("/") != expected_base_url.rstrip("/"):
            raise ValueError("resources access does not match simple_agent's configured resources server")
        return SimpleAgentSessionState(
            resources_cookies=dict(body.resources_access.cookies),
            resources_headers=dict(body.resources_access.headers),
        )

    async def close_agent_session_state(
        self,
        agent_session_id: str,
        session: AgentSession,
    ) -> AgentCloseSessionResult:
        if not isinstance(session.state, SimpleAgentSessionState):
            raise TypeError("Unexpected simple_agent session state")
        cookies = session.state.resources_cookies
        return AgentCloseSessionResult(
            agent_observations=session.state.observations,
            resources_cookies={str(name): str(getattr(value, "value", value)) for name, value in cookies.items()},
        )

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_url_path: str,
        resources_server_cookies: Any = None,
        resources_server_headers: dict[str, str] | None = None,
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
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
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
                    api_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=f"/{output_function_call.name}",
                        json=parsed_arguments,
                        cookies=resources_server_cookies,
                        headers=resources_server_headers,
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

            if collect_trajectory and all_fn_calls:
                turns[-1].step_count = len(tool_records)

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
        return model_response, trajectory, model_server_cookies, resources_server_cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        agent_session_id = self.agent_session_id_from_request(request)
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        collect_trajectory = self._model_call_capture_enabled() and isinstance(rollout_id, str)
        if isinstance(agent_session_id, str):
            if not isinstance(rollout_id, str):
                raise ValueError("Agent sessions require an attempt-qualified rollout path")
            agent_session = self.begin_agent_activation(agent_session_id, rollout_id)
        else:
            agent_session = None
        if agent_session is not None and not isinstance(agent_session.state, SimpleAgentSessionState):
            raise TypeError("Unexpected simple_agent session state")
        resources_server_cookies = (
            agent_session.state.resources_cookies if agent_session is not None else request.cookies
        )
        resources_server_headers = agent_session.state.resources_headers if agent_session is not None else None
        model_response, trajectory, model_server_cookies, resources_server_cookies = await self._create_episode(
            body,
            model_url_path=self.url_path_for_request("/v1/responses", request),
            resources_server_cookies=resources_server_cookies,
            resources_server_headers=resources_server_headers,
            task_id=agent_session.request.task_id.task_id if agent_session is not None else "unscoped",
            rollout_id=rollout_id or "unscoped",
            collect_trajectory=collect_trajectory,
        )
        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)
        if agent_session is not None:
            agent_session.state.resources_cookies = resources_server_cookies
            if trajectory is not None:
                agent_session.state.observations = AgentObservationBundle(
                    source="simple_agent",
                    records=trajectory.invocations,
                    gaps=trajectory.gaps,
                )
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
            verify_request = SimpleAgentVerifyRequest.model_validate(
                body.model_dump() | {"response": model_response_json}
            )
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
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


if __name__ == "__main__":
    SimpleAgent.run_webserver()
