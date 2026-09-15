# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed, one harness request, verify: the episode protocol almost every benchmark uses."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

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
from nemo_gym.episode_context import EpisodeContext
from nemo_gym.processors.base import BaseProcessor, BaseProcessorConfig
from nemo_gym.rollout_observability import ObservationGap, TrajectoryRecord
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config
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
    # The runtime is selected per run, not per task: the environment declares the spec through
    # `/sandbox_spec`, the deployment picks what provisions it.
    sandbox_provider: Optional[Union[str, dict[str, Any]]] = None


class SingleAgentTurnProcessor(BaseProcessor):
    """Run seed → agent turn → verification for a single-agent episode."""

    config: SingleAgentTurnProcessorConfig

    async def sandbox_spec(self, body: SingleAgentTurnRunRequest, cookies: Any) -> Optional[Mapping[str, Any]]:
        """Ask the environment what runtime this task needs. 204 (or an empty body) means none."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/sandbox_spec",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(response)
        if response.status == 204:
            return None
        spec = await get_response_json(response)
        return spec if isinstance(spec, Mapping) and spec else None

    @asynccontextmanager
    async def runtime(self, body: SingleAgentTurnRunRequest, cookies: Any) -> AsyncIterator[EpisodeContext]:
        """Own the sandbox for exactly as long as the episode lasts.

        The spec comes from the environment because it is a property of the task; the provider
        comes from this processor's config because it is a property of the run. Teardown is in
        `finally`, so it happens on the failure path too, which is the reason the episode's owner
        is the one that holds the box.
        """
        context = EpisodeContext(
            rollout_id=self.rollout_id_from_run(body),
            env=self.config.resources_server,
        )

        spec = await self.sandbox_spec(body, cookies)
        if spec is None:
            yield context
            return

        if self.config.sandbox_provider is None:
            raise RuntimeError(
                f"`{self.config.resources_server.name}` declares a sandbox spec but processor "
                f"`{self.config.name}` sets no `sandbox_provider`. The spec belongs to the task and "
                "the provider to the run, so the environment cannot supply this one."
            )
        provider = resolve_provider_config(
            self.config.sandbox_provider, getattr(self.server_client, "global_config_dict", None)
        )
        sandbox = await AsyncSandbox(provider, SandboxSpec(**dict(spec))).start()
        try:
            try:
                descriptor = await sandbox.serialize()
            except RuntimeError as error:
                raise RuntimeError(
                    f"{error} A processor-owned sandbox has to be handed to the agent server and the "
                    "verifier, which are separate processes, so the provider must support "
                    "serialize()/connect(). Today that is `opensandbox` and `e2b`; the rest need the "
                    "sandbox server (NVIDIA-NeMo/Gym#2085) to front them."
                ) from error
            yield context.model_copy(update={"sandbox": descriptor})
        finally:
            await sandbox.stop()

    async def run(self, request: Request, body: SingleAgentTurnRunRequest) -> SingleAgentTurnVerifyResponse:
        cookies = request.cookies

        async with self.runtime(body, cookies) as episode_context:
            return await self._run_episode(body, cookies, episode_context)

    async def _run_episode(
        self,
        body: SingleAgentTurnRunRequest,
        cookies: Any,
        episode_context: EpisodeContext,
    ) -> SingleAgentTurnVerifyResponse:
        body = body.model_copy(update={"episode_context": episode_context})

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        responses_create_params = body.responses_create_params.model_copy(update={"episode_context": episode_context})
        response = await self.server_client.post(
            server_name=self.config.agent_server.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=responses_create_params,
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
            # A subclass that never declares the field would drop it under `extra="ignore"`, and
            # the verifier would look like it simply had no runtime. Fail loudly instead.
            if verify_request.episode_context != episode_context:
                raise RuntimeError(
                    f"`episode_context` did not survive into the verify request for "
                    f"`{self.config.resources_server.name}`. Its request model must inherit "
                    "BaseVerifyRequest rather than redeclaring the fields it needs."
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
        # The context is scaffolding for the episode, not a result. It can also carry a lease
        # token, which has no business in the rollouts file.
        result.pop("episode_context", None)
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
