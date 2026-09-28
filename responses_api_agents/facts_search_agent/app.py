# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The complete seven-hop FACTS Search-On agent loop."""

from __future__ import annotations

import json
from time import perf_counter, time
from typing import Any, List

from pydantic import Field

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
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


FINAL_REQUEST = "Please provide a final answer based on the information gathered so far."


class FACTSSearchAgentConfig(SimpleAgentConfig):
    max_hops: int = Field(default=7, ge=1)


class FACTSSearchAgent(SimpleAgent):
    config: FACTSSearchAgentConfig

    async def _model_call(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        new_outputs: list,
        model_url_path: str,
        model_server_cookies: Any,
        *,
        force_final: bool = False,
    ):
        input_items = list(body.input) + list(new_outputs)
        if force_final:
            input_items.append(NeMoGymEasyInputMessage(role="user", content=FINAL_REQUEST))
        updates: dict[str, Any] = {"input": input_items}
        if force_final:
            updates.update({"tools": [], "tool_choice": "none", "parallel_tool_calls": False})
        call_body = body.model_copy(update=updates)
        api_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=model_url_path,
            json=call_body,
            cookies=model_server_cookies,
        )
        await raise_for_status(api_response)
        payload = await get_response_json(api_response)
        return NeMoGymResponse.model_validate(payload), api_response.cookies, call_body

    async def _create_episode(
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
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs: list = []
        tool_records: list[TrajectoryToolCall] = []
        model_calls: list[ModelCallRef] = []
        turns: list[TrajectoryTurn] = []
        gaps: list[ObservationGap] = []
        usage = None
        model_server_cookies = None
        hops = 0
        queries = 0
        forced_final = False
        invocation_status = "completed"
        final_response: NeMoGymResponse | None = None

        while hops < self.config.max_hops:
            turn_timestamp = time()
            response, model_server_cookies, call_body = await self._model_call(
                body, new_outputs, model_url_path, model_server_cookies
            )
            output = response.output
            new_outputs.extend(output)
            usage = accumulate_response_usage(usage, response.usage)
            response.usage = None
            fn_calls: List[NeMoGymResponseFunctionToolCall] = [item for item in output if item.type == "function_call"]
            assistant_messages: List[NeMoGymResponseOutputMessage] = [
                item for item in output if item.type == "message" and item.role == "assistant"
            ]

            turn_model_refs: list[ModelCallRef] = []
            if collect_trajectory:
                if response.id:
                    ref = ModelCallRef(model_ref=self.config.model_server, response_id=response.id)
                    model_calls.append(ref)
                    turn_model_refs.append(ref)
                else:
                    gaps.append(
                        ObservationGap(
                            code="model_call_reference_unavailable", invocation_id=invocation_id, detail=f"hop:{hops}"
                        )
                    )
                turns.append(
                    TrajectoryTurn(
                        invocation_id=invocation_id,
                        task_id=task_id,
                        rollout_id=rollout_id,
                        turn_no=len(turns) + 1,
                        timestamp=turn_timestamp,
                        question=call_body.input,
                        answer=[item for item in output if item.type != "reasoning"],
                        reasoning_content=[item.model_dump(mode="json") for item in output if item.type == "reasoning"]
                        or None,
                        step_count=len(tool_records),
                        model_calls=turn_model_refs,
                    )
                )

            if response.incomplete_details:
                invocation_status = "incomplete"
                final_response = response
                break
            if not fn_calls and assistant_messages:
                final_response = response
                break
            if not fn_calls:
                invocation_status = "incomplete"
                final_response = response
                break

            hops += 1
            queries += len(fn_calls)
            for function_call in fn_calls:
                started_at = time()
                started_monotonic = perf_counter()
                error_type = None
                try:
                    arguments = json.loads(function_call.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    tool_output = json.dumps({"error": f"Invalid tool call arguments: {exc!r}"})
                    status = "failed"
                    error_type = type(exc).__name__
                else:
                    api_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=f"/{function_call.name}",
                        json=arguments,
                        cookies=resources_server_cookies,
                    )
                    tool_output = (await api_response.content.read()).decode(errors="replace")
                    resources_server_cookies = api_response.cookies
                    status = "completed" if 200 <= api_response.status < 400 else "failed"
                    error_type = None if status == "completed" else f"http_{api_response.status}"
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
                            status=status,
                            error_type=error_type,
                            output=tool_output,
                        )
                    )
                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output", call_id=function_call.call_id, output=tool_output
                    )
                )
            if collect_trajectory:
                turns[-1].step_count = len(tool_records)

        if final_response is None:
            forced_final = True
            turn_timestamp = time()
            final_response, model_server_cookies, call_body = await self._model_call(
                body, new_outputs, model_url_path, model_server_cookies, force_final=True
            )
            output = final_response.output
            new_outputs.extend(output)
            usage = accumulate_response_usage(usage, final_response.usage)
            final_response.usage = None
            if collect_trajectory:
                refs: list[ModelCallRef] = []
                if final_response.id:
                    ref = ModelCallRef(model_ref=self.config.model_server, response_id=final_response.id)
                    model_calls.append(ref)
                    refs.append(ref)
                else:
                    gaps.append(
                        ObservationGap(
                            code="model_call_reference_unavailable", invocation_id=invocation_id, detail="forced-final"
                        )
                    )
                turns.append(
                    TrajectoryTurn(
                        invocation_id=invocation_id,
                        task_id=task_id,
                        rollout_id=rollout_id,
                        turn_no=len(turns) + 1,
                        timestamp=turn_timestamp,
                        question=call_body.input,
                        answer=[item for item in output if item.type != "reasoning"],
                        reasoning_content=[item.model_dump(mode="json") for item in output if item.type == "reasoning"]
                        or None,
                        step_count=len(tool_records),
                        model_calls=refs,
                    )
                )
            if final_response.incomplete_details:
                invocation_status = "incomplete"

        assert final_response is not None
        final_response.output = new_outputs
        final_response.usage = usage
        final_response.metadata = dict(final_response.metadata or {}) | {
            "facts_search_hops": str(hops),
            "facts_search_queries": str(queries),
            "facts_search_forced_final": "true" if forced_final else "false",
        }

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
                gaps=gaps,
            )
        return final_response, trajectory, model_server_cookies, resources_server_cookies


if __name__ == "__main__":
    FACTSSearchAgent.run_webserver()
