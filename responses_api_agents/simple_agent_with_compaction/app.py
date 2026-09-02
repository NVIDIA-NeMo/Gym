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
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from responses_api_agents.simple_agent_with_compaction.compaction import (
    ContextCompactedResponse,
    ContextCompactedTransportResponse,
    ContextCompactionSession,
    ContextHistoryConfig,
    ContextMeasurements,
    PreparedContextCompactionCall,
    build_generation_contract,
    build_transport_response,
)


_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"
_CONTEXT_COMPACTION_SEED_COUNT_COOKIE = "_nemo_gym_cc_seed_obs_count"
_CONTEXT_COMPACTION_ROLLOUT_ID_COOKIE = "_nemo_gym_cc_rollout_id"


class ContextCompactionResponseCreateParams(NeMoGymResponseCreateParamsNonStreaming):
    """Responses request with the exact-prefix control used by the vLLM adapter."""

    required_prefix_token_ids: list[int] | None = None


class SimpleAgentWithCompactionConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None
    context_history: ContextHistoryConfig = Field(default_factory=ContextHistoryConfig)


class SimpleAgentWithCompactionRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    context_compaction_rollout_id: str | None = None
    context_compaction_group_id: str | None = None
    context_compaction_task_id: str | None = None
    context_compaction_rollout_index: int | None = Field(default=None, ge=0)
    context_compaction_attempt_index: int | None = Field(default=None, ge=0)


class SimpleAgentWithCompactionVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentWithCompactionVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: ContextCompactedTransportResponse | ContextCompactedResponse | NeMoGymResponse


class SimpleAgentWithCompaction(SimpleResponsesAPIAgent):
    config: SimpleAgentWithCompactionConfig

    async def _tool_response_items(
        self,
        output: str,
        call_id: str,
    ) -> list[NeMoGymFunctionCallOutput | NeMoGymEasyInputMessage]:
        """Decode one resources-server response into trajectory observations."""

        return [
            NeMoGymFunctionCallOutput(
                type="function_call_output",
                call_id=call_id,
                output=output,
            )
        ]

    async def _seed_session_response_messages(
        self,
        seed_session_response,
    ) -> list[NeMoGymEasyInputMessage]:
        """Decode optional opening observations from a seed response."""

        return []

    async def _create_episode(
        self,
        body: ContextCompactionResponseCreateParams,
        *,
        model_url_path: str,
        resources_server_cookies: Any = None,
        task_id: str = "unscoped",
        rollout_id: str = "unscoped",
        context_compaction_rollout_id: str | None = None,
        seed_count: int = 0,
        collect_trajectory: bool = False,
    ) -> tuple[ContextCompactedResponse | NeMoGymResponse, TrajectoryRecord | None, Any, Any]:
        invocation_id = "root"
        tool_records: list[TrajectoryToolCall] = []
        model_calls: list[ModelCallRef] = []
        turns: list[TrajectoryTurn] = []
        trajectory_gaps: list[ObservationGap] = []
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        complete_input = list(body.input)
        if seed_count < 0 or seed_count > len(complete_input):
            raise RuntimeError("Internal context-compaction seed count is out of range")
        if seed_count:
            agent_input = complete_input[:-seed_count]
            seed_observations = complete_input[-seed_count:]
        else:
            agent_input = complete_input
            seed_observations = []

        context_session = None
        if self.config.context_history.enabled:
            context_session = ContextCompactionSession(
                config=self.config.context_history,
                rollout_id=context_compaction_rollout_id or rollout_id,
                generation_contract=build_generation_contract(
                    body=body,
                    model_server=self.config.model_server,
                    context_history=self.config.context_history,
                ),
                initial_context=agent_input,
                seed_observations=seed_observations,
            )

        new_outputs = []
        usage = None
        step = 0
        invocation_status = "completed"
        model_server_cookies = None

        while True:
            step += 1
            request_input = body.input + new_outputs
            prepared_call = None
            if context_session is not None:

                async def measure_context(
                    call: PreparedContextCompactionCall,
                ) -> ContextMeasurements:
                    nonlocal model_server_cookies
                    prompt_token_count = 0
                    guard_config = self.config.context_history.guards
                    if guard_config.max_total_tokens is not None:
                        tokenize_body = body.model_copy(
                            update={
                                "input": list(call.request_input),
                                "required_prefix_token_ids": (
                                    list(call.required_prefix_token_ids)
                                    if call.required_prefix_token_ids is not None
                                    else None
                                ),
                            }
                        )
                        tokenize_response = await self.server_client.post(
                            server_name=self.config.model_server.name,
                            url_path=model_url_path.removesuffix("/v1/responses") + "/tokenize",
                            json=tokenize_body,
                            cookies=model_server_cookies,
                        )
                        await raise_for_status(tokenize_response)
                        tokenize_payload = await get_response_json(tokenize_response)
                        tokens = tokenize_payload.get("tokens")
                        if not isinstance(tokens, list) or not all(isinstance(token_id, int) for token_id in tokens):
                            raise RuntimeError("Model tokenize preflight returned invalid tokens")
                        prompt_token_count = len(tokens)
                        if tokenize_response.cookies:
                            model_server_cookies = tokenize_response.cookies

                    active_image_count = len(call.prepared_history.view.media_ids)
                    vision_tokens_per_image = guard_config.projected_vision_tokens_per_image or 0
                    return ContextMeasurements(
                        prompt_token_count=prompt_token_count,
                        active_image_count=active_image_count,
                        vision_token_count=(active_image_count * vision_tokens_per_image),
                    )

                prepared_call = await context_session.prepare_model_call(
                    turn_id=step,
                    measure_context=measure_context,
                )
                request_input = list(prepared_call.request_input)

            new_body = body.model_copy(
                update={
                    "input": request_input,
                    "required_prefix_token_ids": (
                        list(prepared_call.required_prefix_token_ids)
                        if prepared_call is not None and prepared_call.required_prefix_token_ids is not None
                        else None
                    ),
                }
            )
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
            if context_session is not None:
                assert prepared_call is not None
                context_session.record_model_response(
                    call=prepared_call,
                    output_items=output,
                    finish_reason=(
                        model_response.incomplete_details.reason
                        if model_response.incomplete_details is not None
                        else None
                    ),
                )
            else:
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

                try:
                    tool_responses = await self._tool_response_items(
                        tool_output,
                        output_function_call.call_id,
                    )
                except ValidationError as exc:
                    tool_responses = [
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=output_function_call.call_id,
                            output=json.dumps({"error": f"Invalid tool envelope: {exc!r}"}),
                        )
                    ]
                if context_session is not None:
                    context_session.append_observation(
                        tool_responses,
                        turn_id=step,
                        conditions_action_turn=step + 1,
                    )
                else:
                    new_outputs.extend(tool_responses)

            if collect_trajectory and all_fn_calls:
                turns[-1].step_count = len(tool_records)

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                invocation_status = "incomplete"
                break

        if context_session is not None:
            context_session.finalize()
        episode_output = list(context_session.output_items) if context_session is not None else new_outputs
        model_response.output = episode_output
        model_response.usage = usage
        if context_session is not None:
            model_response = context_session.build_response(
                model_response,
                agent_input=agent_input,
                seed_obs=seed_observations,
            )
        trajectory = None
        if collect_trajectory:
            invocation = AgentInvocation(
                invocation_id=invocation_id,
                status=invocation_status,
                model_calls=model_calls,
                conversation=[*body.input, *episode_output],
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
        body: ContextCompactionResponseCreateParams = Body(),
    ) -> ContextCompactedResponse | NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        collect_trajectory = self._model_call_capture_enabled() and isinstance(rollout_id, str)
        seed_count_raw = request.cookies.get(_CONTEXT_COMPACTION_SEED_COUNT_COOKIE, "0")
        try:
            seed_count = int(seed_count_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Invalid internal context-compaction seed count") from exc
        context_compaction_rollout_id = request.cookies.get(_CONTEXT_COMPACTION_ROLLOUT_ID_COOKIE)
        if context_compaction_rollout_id is None:
            context_compaction_rollout_id = rollout_id or str(request.session.get(SESSION_ID_KEY, "request"))
        resources_server_cookies = {
            key: value
            for key, value in request.cookies.items()
            if key
            not in {
                _CONTEXT_COMPACTION_SEED_COUNT_COOKIE,
                _CONTEXT_COMPACTION_ROLLOUT_ID_COOKIE,
            }
        }
        model_response, trajectory, model_server_cookies, resources_server_cookies = await self._create_episode(
            body,
            model_url_path=self.url_path_for_request("/v1/responses", request),
            resources_server_cookies=resources_server_cookies,
            rollout_id=rollout_id or "unscoped",
            context_compaction_rollout_id=context_compaction_rollout_id,
            seed_count=seed_count,
            collect_trajectory=collect_trajectory,
        )
        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)
        if trajectory is not None:
            model_response = model_response.model_copy(
                update={_INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")}
            )
        return model_response

    async def run(
        self,
        request: Request,
        body: SimpleAgentWithCompactionRunRequest,
    ) -> SimpleAgentWithCompactionVerifyResponse:
        cookies = dict(request.cookies)

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies.update(seed_session_response.cookies)
        if body.context_compaction_rollout_id is not None:
            # Session cookies are server-owned and cannot carry a caller-owned
            # logical rollout identity through this agent's self-call.
            cookies[_CONTEXT_COMPACTION_ROLLOUT_ID_COOKIE] = body.context_compaction_rollout_id

        seed_messages = await self._seed_session_response_messages(seed_session_response)
        responses_body = body.responses_create_params
        if seed_messages:
            responses_body = responses_body.model_copy(deep=True)
            if isinstance(responses_body.input, str):
                responses_body.input = [
                    NeMoGymEasyInputMessage(
                        role="user",
                        content=responses_body.input,
                    )
                ]
            responses_body.input = [*responses_body.input, *seed_messages]
            cookies[_CONTEXT_COMPACTION_SEED_COUNT_COOKIE] = str(len(seed_messages))

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=responses_body,
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

        context_compacted_response = None
        if model_response_json.get("context_compaction_contract") is not None:
            context_compacted_response = ContextCompactedResponse.model_validate(model_response_json)
            original_input = body.responses_create_params.input
            if isinstance(original_input, str):
                original_input = [
                    NeMoGymEasyInputMessage(
                        role="user",
                        content=original_input,
                    )
                ]
            context_compacted_response = context_compacted_response.model_copy(
                update={
                    "agent_input": list(original_input),
                    "seed_obs": seed_messages,
                }
            )

        if self.config.skip_verification:
            result = body.model_dump() | {
                "response": model_response_json,
                "reward": float(self.config.skip_verification_reward),
                "verification_skipped": True,
            }
        else:
            verifier_response_payload = model_response_json
            if seed_messages:
                verifier_response_payload = dict(model_response_json)
                verifier_response_payload["output"] = [
                    *(message.model_dump() for message in seed_messages),
                    *model_response_json.get("output", []),
                ]
            verify_request = SimpleAgentWithCompactionVerifyRequest.model_validate(
                body.model_dump() | {"response": verifier_response_payload}
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
        verified = SimpleAgentWithCompactionVerifyResponse.model_validate(result)
        if context_compacted_response is None:
            return verified

        contract = context_compacted_response.context_compaction_contract
        context_compacted_response = context_compacted_response.model_copy(
            update={
                "context_compaction_contract": contract.model_copy(
                    update={
                        "group_id": body.context_compaction_group_id,
                        "task_id": body.context_compaction_task_id,
                        "rollout_index": body.context_compaction_rollout_index,
                        "attempt_index": body.context_compaction_attempt_index,
                    }
                )
            }
        )
        if self.config.skip_verification:
            return verified.model_copy(update={"response": context_compacted_response})
        return verified.model_copy(update={"response": build_transport_response(context_compacted_response)})

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
    SimpleAgentWithCompaction.run_webserver()
