# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate one outer agent turn containing one or more internal steps."""

from fastapi import Body, Request
from pydantic import ConfigDict

from nemo_gym.agents.responses_api_agent import (
    INTERNAL_TRAJECTORY_KEY,
)
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.config_types import (
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    ResourcesServerRef,
)
from nemo_gym.processors.base import BaseProcessor, BaseProcessorConfig
from nemo_gym.rollout_observability import ObservationGap, TrajectoryRecord
from nemo_gym.server_utils import get_response_json, raise_for_status


class SingleAgentTurnRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SingleAgentTurnVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SingleAgentTurnVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SingleAgentTurnProcessorConfig(BaseProcessorConfig):
    agent_server: AgentServerRef
    resources_server: ResourcesServerRef


class SingleAgentTurnProcessor(BaseProcessor):
    """Run seed → agent turn → verification for a single-agent episode."""

    config: SingleAgentTurnProcessorConfig

    async def run(self, request: Request, body: SingleAgentTurnRunRequest) -> SingleAgentTurnVerifyResponse:
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
            server_name=self.config.agent_server.name,
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
            model_response_json.pop(INTERNAL_TRAJECTORY_KEY, None) if expected_rollout_id is not None else None
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
            verify_request = SingleAgentTurnVerifyRequest.model_validate(
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
        return SingleAgentTurnVerifyResponse.model_validate(result)

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
