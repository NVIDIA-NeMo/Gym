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

"""Discoverable server entrypoint for the core ResponsesAPIAgent."""

from fastapi import Request

from nemo_gym.agents.base import Body
from nemo_gym.agents.responses_api_agent import (
    INTERNAL_TRAJECTORY_KEY,
    ResponsesAPIAgent,
)
from nemo_gym.agents.responses_api_agent import (
    ResponsesAPIAgentConfig as SimpleAgentConfig,
)
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.config_types import (
    AggregateMetrics,
    AggregateMetricsRequest,
)
from nemo_gym.processors.single_agent_turn import (
    SingleAgentTurnRunRequest as SimpleAgentRunRequest,
)
from nemo_gym.processors.single_agent_turn import (
    SingleAgentTurnVerifyRequest as SimpleAgentVerifyRequest,
)
from nemo_gym.processors.single_agent_turn import (
    SingleAgentTurnVerifyResponse as SimpleAgentVerifyResponse,
)
from nemo_gym.rollout_observability import ObservationGap, TrajectoryRecord
from nemo_gym.server_utils import get_response_json, raise_for_status


class SimpleAgent(ResponsesAPIAgent, SimpleResponsesAPIAgent):
    """Deprecated combined agent retained for backward compatibility."""

    config: SimpleAgentConfig

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        cookies = request.cookies
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(response)
        response_json = await get_response_json(response)
        cookies = response.cookies

        trajectory = None
        expected_rollout_id = self.rollout_id_from_run(body)
        raw_trajectory = response_json.pop(INTERNAL_TRAJECTORY_KEY, None) if expected_rollout_id is not None else None
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
                "response": response_json,
                "reward": float(self.config.skip_verification_reward),
                "verification_skipped": True,
            }
        else:
            verify_request = SimpleAgentVerifyRequest.model_validate(body.model_dump() | {"response": response_json})
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
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


__all__ = [
    "ResponsesAPIAgent",
    "SimpleAgent",
    "SimpleAgentConfig",
    "SimpleAgentRunRequest",
    "SimpleAgentVerifyRequest",
    "SimpleAgentVerifyResponse",
]


if __name__ == "__main__":
    ResponsesAPIAgent.run_webserver()
