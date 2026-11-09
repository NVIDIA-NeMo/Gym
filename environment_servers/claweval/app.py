# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ClawEval conversation protocol, independent of the assistant harness."""

import asyncio
from typing import Any, Literal
from uuid import uuid4

from fastapi import Body
from pydantic import BaseModel

from nemo_gym.agents import (
    AGENT_ACTIVATION_HEADER,
    AGENT_TIMEOUT_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentContinuationRequirements,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.client import get_response_json, raise_boundary_failure
from nemo_gym.config import (
    TOKEN_CAPTURE_BLOCK,
    TOKEN_CAPTURE_PATH_SEGMENT,
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    ResourcesServerRef,
    get_first_server_config_dict,
)
from nemo_gym.core import BaseEpisodeRequest, BaseEpisodeResponse, NeMoGymResponse, NeMoGymResponseUsage
from nemo_gym.server_runtime import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    DirectHTTPToolAccess,
    MCPStreamableHTTPConnection,
    MCPToolAccess,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    dependency_failure,
)
from nemo_gym.trajectory import AgentObservationBundle, TrajectoryRecord
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.claweval.task_data import TaskData


class ClawEvalEnvironmentConfig(BaseEnvironmentServerConfig):
    resources_server: ResourcesServerRef
    agent_server: AgentServerRef
    tool_transport: Literal["mcp", "direct_http"] = "mcp"
    turn_sessions: Literal["continue", "separate"] = "continue"


class ClawEvalResult(BaseModel):
    reward: float
    evaluation_completed: bool = True
    mask_sample: bool = False
    reward_components: dict[str, float]
    response: NeMoGymResponse
    activation_responses: list[NeMoGymResponse]
    verification: dict[str, Any]
    ng_agent_observations: AgentObservationBundle
    ng_trajectory: TrajectoryRecord | None = None


class ClawEvalRequest(BaseEpisodeRequest[TaskData]):
    """A typed ClawEval task with a caller-assigned rollout identity."""


class ClawEvalResponse(BaseEpisodeResponse[ClawEvalResult]):
    """The native score and harness evidence for one episode."""


def model_turn_count(observations: AgentObservationBundle, trajectory: TrajectoryRecord | None) -> int:
    """Count explicit calls or persisted assistant turns before capture IDs are joined by collection."""
    count = 0
    for invocation in observations.records:
        if invocation.kind != "agent_invocation":
            continue
        if invocation.model_calls:
            count += len(invocation.model_calls)
            continue
        turns = (
            [turn for turn in trajectory.turns if turn.invocation_id == invocation.invocation_id] if trajectory else []
        )
        identities = {turn.source_message_id for turn in turns}
        if not turns or None in identities or len(identities) != len(turns):
            raise ValueError("Harness returned no unambiguous model-turn evidence; native grading is prohibited")
        count += len(turns)
    if not count:
        raise ValueError("Harness returned no model-turn evidence; native grading is prohibited")
    return count


class ClawEvalEnvironmentServer(BaseEnvironmentServer[ClawEvalRequest, ClawEvalResponse]):
    config: ClawEvalEnvironmentConfig
    request_model = ClawEvalRequest
    response_model = ClawEvalResponse

    async def run(self, request: ClawEvalRequest, cleanup: CleanupContext) -> ClawEvalResponse:
        if request.task.task_id.task_id != request.task.task_input.task_id:
            raise ValueError("Task identity does not match the native ClawEval reference")
        resources = self.config.resources_server.name
        agent = self.config.agent_server.name
        resource_id = f"resources-{uuid4().hex}"
        resource_cookies: dict[str, str] = {}
        stage = "seed"

        async def post(
            server: str, path: str, body: object, *, cookies: dict | None = None, headers: dict | None = None
        ) -> tuple[dict, dict[str, str]]:
            reply = await self.server_client.post(
                server_name=server,
                url_path=path,
                json=body,
                cookies=cookies,
                **({"headers": headers} if headers is not None else {}),
            )
            await raise_boundary_failure(reply)
            return await get_response_json(reply), {name: value.value for name, value in reply.cookies.items()}

        async def close_resources() -> None:
            await post(
                resources,
                "/close_session",
                ResourcesCloseSessionRequest(
                    resources_session_id=resource_id, episode_id=request.episode_id
                ).model_dump(mode="json"),
                cookies=resource_cookies,
            )

        resource_cleanup = cleanup.register_cleanup("ClawEval resources", close_resources)
        try:
            seed_json, cookies = await post(
                resources,
                "/seed_session",
                ResourcesSeedSessionRequest(
                    resources_session_id=resource_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    task_data=request.task.task_input.model_dump(mode="json"),
                ).model_dump(mode="json"),
            )
            resource_cookies.update(cookies)
            seed = ResourcesSeedSessionResponse.model_validate(seed_json)
            if seed.resources_session_id != resource_id or not resource_cookies:
                raise ValueError("Resources seed returned a mismatched identity or no session cookie")
            info, _ = await post(resources, "/task_info", {}, cookies=resource_cookies)
            base_url = self.server_client._resolve_base_url(resources).rstrip("/")
            if self.config.tool_transport == "mcp":
                if seed.resources_tools is None:
                    raise ValueError("ClawEval Resources did not return an MCP grant")
                access = MCPToolAccess(
                    name="claweval",
                    required=True,
                    connection=MCPStreamableHTTPConnection(
                        url=base_url + seed.resources_tools.url_path, headers=seed.resources_tools.headers
                    ),
                )
            else:
                access = DirectHTTPToolAccess(
                    name="claweval", required=True, base_url=base_url, cookies=resource_cookies
                )
            history: list[dict] = [{"role": "user", "content": info["input"]}]
            activation_responses: list[NeMoGymResponse] = []
            observations = AgentObservationBundle(source="claweval_agent_sessions")
            trajectory: TrajectoryRecord | None = None
            session_id: str | None = None
            agent_seeded = False
            agent_cookies: dict[str, str] = {}
            agent_cleanup = None
            completed_close: AgentCloseSessionResponse | None = None
            continuing = self.config.turn_sessions == "continue"

            async def close_agent() -> None:
                nonlocal completed_close, trajectory
                close_json, _ = await post(
                    agent,
                    "/v1/agent_sessions/close",
                    AgentCloseSessionRequest(agent_session_id=session_id, episode_id=request.episode_id).model_dump(
                        mode="json"
                    ),
                    cookies=agent_cookies,
                )
                completed_close = AgentCloseSessionResponse.model_validate(close_json)
                if completed_close.agent_session_id != session_id:
                    raise ValueError("Agent close returned a different session identity")
                if completed_close.cleanup_confirmed is False:
                    raise ValueError("Harness termination was not confirmed; native grading is prohibited")
                if completed_close.agent_observations is None and agent_seeded:
                    raise ValueError("Harness close did not return its observations")
                bundle = completed_close.agent_observations
                if bundle is None:
                    return
                if continuing:
                    # Continuing sessions expose globally unique activation IDs used by capture joins.
                    trajectory = completed_close.trajectory
                    if trajectory is not None and (
                        trajectory.task_id != request.task.task_id.task_id
                        or trajectory.rollout_id != request.episode_id.capture_key
                    ):
                        raise ValueError("Harness close returned a trajectory for a different task or episode")
                for record in bundle.records:
                    updates = {}
                    if not continuing:
                        # Independent sessions may each call their root "root"; explicit call IDs remain intact.
                        for key in ("invocation_id", "parent_invocation_id"):
                            if (value := getattr(record, key, None)) is not None:
                                updates[key] = f"{session_id}:{value}"
                    observations.records.append(record.model_copy(update=updates))
                observations.gaps.extend(
                    gap.model_copy(update={"invocation_id": f"{session_id}:{gap.invocation_id}"})
                    if gap.invocation_id and not continuing
                    else gap
                    for gap in bundle.gaps
                )
                if completed_close.resources_cookies:
                    resource_cookies.update(completed_close.resources_cookies)

            async with asyncio.timeout(info["timeout_seconds"]):
                for turn in range(info["max_user_rounds"] + 1):
                    stage = "agent"
                    if session_id is None:
                        session_id = f"agent-{uuid4().hex}"
                        agent_seeded = False
                        agent_cookies = {}
                        agent_cleanup = cleanup.register_cleanup("ClawEval assistant", close_agent)
                        agent_seed, jar = await post(
                            agent,
                            "/v1/agent_sessions",
                            AgentSeedSessionRequest(
                                agent_session_id=session_id,
                                episode_id=request.episode_id,
                                task_id=request.task.task_id,
                                tool_accesses=[access],
                                continuation=AgentContinuationRequirements() if continuing else None,
                                agent_timeout_sec=info["timeout_seconds"],
                            ).model_dump(mode="json"),
                        )
                        agent_cookies.update(jar)
                        if AgentSeedSessionResponse.model_validate(agent_seed).agent_session_id != session_id:
                            raise ValueError("Agent seed returned a different session identity")
                        if not agent_cookies:
                            raise ValueError("Agent seed did not establish its session cookie")
                        agent_seeded = True
                    delta = history[-1]["content"]
                    params = {
                        "input": (delta if isinstance(delta, str) else [history[-1]]) if continuing else list(history),
                        "instructions": info["instructions"],
                        "metadata": {AGENT_TIMEOUT_METADATA_KEY: str(info["timeout_seconds"])},
                    }
                    if self.config.tool_transport == "direct_http":
                        params["tools"] = [
                            {
                                "type": "function",
                                "strict": False,
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": tool["input_schema"],
                            }
                            for tool in info["tools"]
                        ]
                    capture = self.server_client.global_config_dict.get(TOKEN_CAPTURE_BLOCK) or {}
                    agent_settings = get_first_server_config_dict(self.server_client.global_config_dict, agent)
                    token_capture = capture.get("enabled", False) and (
                        capture.get("all_agents", False) or agent_settings.get("token_id_capture", False)
                    )
                    capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
                    response_json, jar = await post(
                        agent,
                        f"/ng-rollout/{request.episode_id.capture_key}{capture_segment}/v1/responses",
                        params,
                        cookies=agent_cookies,
                        headers={AGENT_ACTIVATION_HEADER: str(turn + 1)} if continuing else None,
                    )
                    agent_cookies.update(jar)
                    response = NeMoGymResponse.model_validate(response_json)
                    if response.status != "completed":
                        raise ValueError(f"Harness did not complete: {response.status}; native grading is prohibited")
                    activation_responses.append(response)
                    outputs = [item.model_dump(mode="json") for item in response.output]
                    history.extend(outputs)
                    if not continuing:
                        stage = "cleanup"
                        await agent_cleanup.close()
                    stage = "agent"
                    simulation, _ = await post(
                        resources,
                        "/simulate_user",
                        {
                            "conversation": history,
                            "usage": response.usage.model_dump(mode="json") if response.usage else None,
                        },
                        cookies=resource_cookies,
                    )
                    if simulation["done"]:
                        break
                    history.append({"role": "user", "content": simulation["input"]})
                    if not continuing:
                        session_id = None
            if continuing:
                stage = "cleanup"
                await agent_cleanup.close()
            model_turns = model_turn_count(observations, trajectory)
            if info["max_turns"] is not None and model_turns > info["max_turns"]:
                raise ValueError("Harness exceeded the native task's model-turn budget; result is not benchmark-valid")
            stage = "verification"
            verification, _ = await post(
                resources,
                "/verify",
                {
                    "episode_id": request.episode_id.model_dump(mode="json"),
                    "task_id": request.task.task_id.model_dump(mode="json"),
                    "verification_input": {"conversation": history, "model_turns": model_turns},
                },
                cookies=resource_cookies,
            )
            if verification.get("mask_sample"):
                raise ValueError("Native verification was masked; no benchmark score is available")
            stage = "cleanup"
            await resource_cleanup.close()
            # The top-level response describes the whole episode. Keep original activation receipts separately.
            episode_response = response.model_copy(
                update={
                    "id": f"claweval-{uuid4().hex}",
                    "output": [item for activation in activation_responses for item in activation.output],
                    "usage": NeMoGymResponseUsage.sum_from_list([item.usage for item in activation_responses])
                    if all(item.usage is not None for item in activation_responses)
                    else None,
                }
            )
            return ClawEvalResponse(
                episode_id=request.episode_id,
                task_id=request.task.task_id,
                result=ClawEvalResult(
                    reward=verification["reward"],
                    reward_components=verification["reward_components"],
                    response=episode_response,
                    activation_responses=activation_responses,
                    verification=verification,
                    ng_agent_observations=observations,
                    ng_trajectory=trajectory,
                ),
            )
        except Exception as error:
            raise dependency_failure(error, stage=stage) from error

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return AggregateMetrics()


if __name__ == "__main__":
    ClawEvalEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ClawEvalEnvironmentServer.run_webserver()
