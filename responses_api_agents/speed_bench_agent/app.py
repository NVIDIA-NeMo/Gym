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
"""SPEED-Bench multi-turn replay agent.

Skills' `SpecdecGenerationTask.process_single_datapoint` plays multi-turn
prompts one user turn at a time: send turn 1 → assistant 1 → send
turn 1 + assistant 1 + turn 2 → assistant 2, and so on. The accumulated
assistant outputs are returned as a list. Each turn is a separate model call,
so the spec-decode counters reflect per-turn prompt distributions, which
matters for parity.

Gym's `simple_agent` does NOT do this. Given a list of user messages with no
interspersed assistant responses, it sends them all in a single model call.

This agent reproduces Skills' behaviour:

1. Read `body.input` as the *entire* dialogue. Split into "user turns" — each
   user message is a turn boundary.
2. For each turn, send the conversation up to and including that turn,
   accumulate the model's assistant reply into the running conversation.
3. Aggregate token usage across all turns and return a single response that
   summarizes the multi-turn run.

System / developer messages at the front of `body.input` are kept as a
preamble across all turns. Existing assistant messages in `body.input` are
preserved (so a partially-completed conversation can be resumed) — only
*trailing* user messages after the last assistant trigger fresh model calls.

For tasks that only have a single user turn, this collapses to a single
model call (same shape as simple_agent).
"""

import json
from time import perf_counter, time
from typing import List

from fastapi import Request, Response
from pydantic import ConfigDict, JsonValue, ValidationError

from nemo_gym.agents import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    accumulate_response_usage,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryTurn,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"


class SpeedBenchAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class SpeedBenchAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SpeedBenchAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SpeedBenchAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def _content_text(content) -> str:
    """Best-effort flatten of a message's `content` field to a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text") or item.get("content")
            if text:
                chunks.append(text)
        return "".join(chunks)
    return ""


class SpeedBenchAgent(SimpleResponsesAPIAgent):
    """Multi-turn fixed-replay agent for SPEED-Bench."""

    config: SpeedBenchAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        generation_started_at = time()
        generation_start = perf_counter()
        request_metrics: list[dict[str, JsonValue] | None] = []
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        # Identify how many *trailing* user turns need fresh model calls.
        # Anything before the first trailing user turn is treated as preamble.
        msgs = list(body.input)
        # Walk from the end backwards. Collect contiguous trailing user msgs
        # and stop at the first non-user.
        trailing_user_indices: List[int] = []
        for idx in range(len(msgs) - 1, -1, -1):
            m = msgs[idx]
            role = getattr(m, "role", None)
            if role is None and isinstance(m, dict):
                role = m.get("role")
            if role == "user":
                trailing_user_indices.append(idx)
                continue
            break
        trailing_user_indices.reverse()

        if not trailing_user_indices:
            # Nothing for the model to respond to. Defer to a single model call
            # to preserve simple_agent's "let the server decide" behaviour.
            trailing_user_indices = [len(msgs) - 1] if msgs else []

        # If only one user turn (trailing), this collapses to a single call.
        # Multiple trailing user turns get replayed one at a time.
        first_user_idx = trailing_user_indices[0]
        running_input = list(msgs[: first_user_idx + 1])  # preamble + first turn
        remaining_user_indices = trailing_user_indices[1:]

        accumulated_outputs = []
        accumulated_text_parts: List[str] = []
        usage = None
        last_response: NeMoGymResponse = None
        model_server_cookies = None
        resources_server_cookies = request.cookies
        rollout_id = request.path_params.get("rollout_id")
        invocation = None
        observations = None
        turns: list[TrajectoryTurn] = []
        if self._model_call_capture_enabled() and isinstance(rollout_id, str):
            invocation = AgentInvocation(invocation_id="root", in_session=False)
            observations = AgentObservationBundle(
                source="speed_bench_agent",
                records=[invocation],
            )
            # The inner route has no task identity; only /run publishes a bound trajectory.
            request.state.nemo_gym_agent_observations = observations

        async def _call_model(turn_input):
            new_body = body.model_copy(update={"input": turn_input})
            turn_timestamp = time()
            try:
                model_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path=self.url_path_for_request("/v1/responses", request),
                    json=new_body,
                    cookies=model_server_cookies,
                    **({"headers": {"x-session-id": "root"}} if invocation is not None else {}),
                )
                await raise_for_status(model_response)
                payload = await get_response_json(model_response)
                try:
                    parsed = NeMoGymResponse.model_validate(payload)
                except ValidationError as e:
                    raise RuntimeError(f"Received an invalid response from model server: {json.dumps(payload)}") from e
            except BaseException as error:
                if invocation is not None:
                    invocation.status = "failed" if isinstance(error, Exception) else "incomplete"
                    invocation.error_type = type(error).__name__
                raise
            if invocation is not None:
                refs = [ModelCallRef(model_ref=self.config.model_server, response_id=parsed.id)] if parsed.id else []
                invocation.model_calls.extend(refs)
                invocation.conversation = [*turn_input, *parsed.output]
                status = parsed.status
                if status == "cancelled":
                    status = "incomplete"
                elif status not in ("completed", "failed", "incomplete"):
                    status = "unknown"
                priority = {"completed": 0, "unknown": 1, "incomplete": 2, "failed": 3}
                if not turns or priority[status] > priority[invocation.status]:
                    invocation.status = status
                turn_no = len(turns) + 1
                if not refs:
                    observations.gaps.append(
                        ObservationGap(
                            code="model_call_reference_unavailable", invocation_id="root", detail=f"turn:{turn_no}"
                        )
                    )
                reasoning = [item.model_dump(mode="json") for item in parsed.output if item.type == "reasoning"]
                turns.append(
                    TrajectoryTurn(
                        invocation_id="root",
                        task_id="unknown",
                        rollout_id=rollout_id,
                        turn_no=turn_no,
                        timestamp=turn_timestamp,
                        question=list(turn_input),
                        answer=[item for item in parsed.output if item.type != "reasoning"],
                        reasoning_content=reasoning or None,
                        step_count=0,
                        model_calls=refs,
                    )
                )
            # Retain every turn, including missing metrics, so partial reporting
            # cannot silently produce an apparently complete acceptance rate.
            request_metrics.append(parsed.metrics)
            return parsed, model_response.cookies

        # First turn (always at least one).
        last_response, model_server_cookies = await _call_model(running_input)

        def _gather_assistant_text(resp: NeMoGymResponse) -> str:
            parts: List[str] = []
            for item in resp.output:
                if isinstance(item, NeMoGymResponseOutputMessage) and item.role == "assistant":
                    for c in item.content:
                        if isinstance(c, NeMoGymResponseOutputText):
                            parts.append(c.text)
                        elif isinstance(c, dict) and c.get("type") == "output_text":
                            parts.append(c.get("text", ""))
            return "".join(parts)

        accumulated_outputs.extend(last_response.output)
        accumulated_text_parts.append(_gather_assistant_text(last_response))
        usage = last_response.usage
        last_response.usage = None

        # Subsequent turns: append the assistant reply and the next user turn,
        # then call again. Accumulate tokens into `usage`.
        for next_user_idx in remaining_user_indices:
            assistant_text = accumulated_text_parts[-1]
            running_input.append(NeMoGymEasyInputMessage(role="assistant", content=assistant_text))
            running_input.append(msgs[next_user_idx])
            last_response, model_server_cookies = await _call_model(running_input)
            accumulated_outputs.extend(last_response.output)
            accumulated_text_parts.append(_gather_assistant_text(last_response))
            usage = accumulate_response_usage(usage, last_response.usage)
            last_response.usage = None

        # Propagate any cookies the resources server set so /verify sees them.
        for k, v in (*resources_server_cookies.items(), *(model_server_cookies or {}).items()):
            response.set_cookie(k, v)

        last_response.output = accumulated_outputs
        last_response.usage = usage
        duration = perf_counter() - generation_start
        last_response.metrics = {
            "requests": request_metrics,
            "generation_started_at": generation_started_at,
            "generation_finished_at": generation_started_at + duration,
            "generation_duration": duration,
        }
        if observations is not None:
            trajectory = TrajectoryRecord(
                task_id="unknown",
                rollout_id=rollout_id,
                invocations=[invocation],
                turns=turns,
                gaps=observations.gaps,
            )
            last_response = last_response.model_copy(
                update={_INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")}
            )
        return last_response

    async def run(self, request: Request, body: SpeedBenchAgentRunRequest) -> SpeedBenchAgentVerifyResponse:
        def bind_trajectory(raw: object) -> TrajectoryRecord:
            trajectory = TrajectoryRecord.model_validate(raw)
            task = body.model_extra or {}
            task_id = next(
                (
                    str(task[key])
                    for key in ("task_id", "problem_id", "instance_id", "_ng_task_index")
                    if task.get(key) is not None
                ),
                "unknown",
            )
            rollout_id = self.rollout_id_from_run(body)
            if rollout_id is None or trajectory.rollout_id != rollout_id:
                raise ValueError("SPEED-Bench trajectory does not match this run's rollout")
            trajectory = trajectory.model_copy(
                update={
                    "task_id": task_id,
                    "turns": [turn.model_copy(update={"task_id": task_id}) for turn in trajectory.turns],
                }
            )
            request.state.nemo_gym_agent_trajectory = trajectory
            return trajectory

        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        api_response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(api_response)
        cookies = api_response.cookies
        model_response_json = await get_response_json(api_response)
        raw_trajectory = model_response_json.pop(_INTERNAL_TRAJECTORY_KEY, None)
        trajectory = bind_trajectory(raw_trajectory) if raw_trajectory is not None else None

        verify_request = SpeedBenchAgentVerifyRequest.model_validate(
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
            for invocation in trajectory.invocations:
                invocation.verification_skipped = False
            result["ng_trajectory"] = trajectory.model_dump(mode="json")
        return SpeedBenchAgentVerifyResponse.model_validate(result)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        api_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(api_response)
        return AggregateMetrics.model_validate(await get_response_json(api_response))


if __name__ == "__main__":
    SpeedBenchAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = SpeedBenchAgent.run_webserver()  # noqa: F401
