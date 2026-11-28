# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive PinchBench episodes with a borrowed task sandbox and the upstream verifier."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat, SecretStr

from environment_servers.pinchbench.task_input import TaskInput
from nemo_gym import __version__
from nemo_gym.agents import AgentOutcomePolicyConfig, OutcomeAction, agent_outcome_policy
from nemo_gym.agents.session_client import AgentSession, TurnDeadlineExpired, TurnRecord
from nemo_gym.config import AgentServerRef, AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core import BaseEpisodeRequest, BaseEpisodeResponse, NeMoGymResponse
from nemo_gym.metrics import compute_aggregate_metrics
from nemo_gym.sandbox import AsyncSandbox, DirectSandboxConnection, SandboxAccess, SandboxSpec, resolve_provider_config
from nemo_gym.sandbox.transfer import download_dir, upload_dir
from nemo_gym.sandbox.utils import read_text, upload_text
from nemo_gym.server_runtime import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    handled_episode_failure,
)
from nemo_gym.trajectory import AgentObservationBundle, TrajectoryRecord, transcript_to_atif
from nemo_gym.trajectory.atif_v1_7 import AtifContent
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


class Config(BaseEnvironmentServerConfig):
    agent_server: AgentServerRef
    sandbox_provider: str = "sandbox"
    agent_image: str
    fws_image: str = "pinchbench-fws:v2.0.0"
    verifier_image: str
    agent_outcome_policy: AgentOutcomePolicyConfig = Field(default_factory=AgentOutcomePolicyConfig)
    timeout_multiplier: PositiveFloat = 3
    judge_model: str | None = None
    judge_base_url: str | None = None
    judge_api_key: SecretStr | None = None


class Request(BaseEpisodeRequest[TaskInput]):
    pass


class Prompt(BaseModel):
    prompt: str = Field(min_length=1)
    new_session: bool = False


class Plan(BaseModel):
    prompts: list[Prompt | str] = Field(min_length=1)
    timeout: PositiveFloat
    grading_type: Literal["automated", "hybrid", "llm_judge"]
    requires_fws: bool


class Result(BaseModel):
    task_id: str
    reward: float = Field(ge=0, le=1)
    grading_type: str
    grading_breakdown: dict[str, float]
    grading_notes: str
    response: NeMoGymResponse
    turns: list[TurnRecord]
    # Each close covers one conversation. Keep all evidence when the task resets the conversation.
    session_observations: list[AgentObservationBundle]
    session_trajectories: list[TrajectoryRecord]
    ng_agent_observations: AgentObservationBundle | None = None
    ng_trajectory: TrajectoryRecord | None = None


class Response(BaseEpisodeResponse[Result]):
    pass


def conversations(plan: Plan) -> list[list[str]]:
    """Group prompts that share conversation history. Every group uses the same workspace."""
    groups: list[list[str]] = []
    for item in plan.prompts:
        prompt = Prompt(prompt=item) if isinstance(item, str) else item
        if not groups or prompt.new_session:
            groups.append([])
        groups[-1].append(prompt.prompt)
    return groups


def _content(value: AtifContent) -> list[dict]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}] if value else []
    return [part.model_dump(mode="json", exclude_none=True) for part in value]


def transcript(turns: list[TurnRecord], *, agent: str, session_id: str) -> list[dict]:
    """Present Gym's recorded messages, tool calls and results in the pinned grader's event format."""
    events = []
    for turn in turns:
        trajectory = transcript_to_atif(
            turn.input,
            [i.model_dump(mode="json") for i in turn.response.output],
            session_id=session_id,
            agent_name=agent,
            agent_version=__version__,
        )
        for step in trajectory.steps:
            calls = step.tool_calls or []
            content = _content(step.message)
            content.extend(
                {"type": "toolCall", "id": c.tool_call_id, "name": c.function_name, "arguments": c.arguments}
                for c in calls
            )
            role = "assistant" if step.source == "agent" else step.source
            events.append({"type": "message", "message": {"role": role, "content": content}})
            names = {c.tool_call_id: c.function_name for c in calls}
            for result in step.observation.results if step.observation else []:
                events.append(
                    {
                        "type": "message",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": result.source_call_id,
                            "toolName": names.get(result.source_call_id),
                            "content": _content(result.content or ""),
                        },
                    }
                )
    return events


class PinchBenchEnvironmentServer(BaseEnvironmentServer[Request, Response]):
    config: Config
    request_model = Request
    response_model = Response

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        return compute_aggregate_metrics(body.verify_responses)

    async def _start(self, request: Request, cleanup: CleanupContext, image: str, purpose: str) -> AsyncSandbox:
        provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        if "local" in provider:
            raise ValueError("PinchBench needs an isolated sandbox provider")
        sandbox = AsyncSandbox(provider)
        cleanup.register_cleanup(purpose, sandbox.stop)
        await sandbox.start(
            SandboxSpec(
                image=image,
                ttl_s=round(self.config.default_episode_timeout_seconds or 7200)
                + round(self.config.cleanup_timeout_seconds)
                + 120,
                metadata={
                    "benchmark": "pinchbench",
                    "purpose": purpose,
                    "rollout_id": request.episode_id.capture_key[:63],
                },
            )
        )
        return sandbox

    async def run(self, request: Request, cleanup: CleanupContext) -> Response:
        task = request.task.task_input
        task_id = task.verifier_metadata.task_id
        verifier = await self._start(request, cleanup, self.config.verifier_image, "verifier")

        async def worker(action: str, *, env: dict[str, str] | None = None) -> str:
            result = await verifier.exec(
                f"python3 -I /opt/pinchbench-worker.py {action} {task_id}", env=env, timeout_s=900
            )
            if result.error_type or result.return_code != 0:
                detail = (result.stderr or result.error_type or "nonzero exit")[-1000:]
                raise handled_episode_failure(
                    stage="seed" if action == "seed" else "verification",
                    operation="episode",
                    terminal=False,
                    failure_reason=f"PinchBench {action} failed: {detail}",
                )
            return result.stdout

        plan = Plan.model_validate_json(await worker("seed"))
        cfg = self.config
        if plan.grading_type != "automated" and not (cfg.judge_base_url and cfg.judge_api_key and cfg.judge_model):
            raise handled_episode_failure(
                stage="seed",
                operation="episode",
                terminal=True,
                failure_reason="This PinchBench task requires judge_model, judge_base_url and judge_api_key",
            )
        sandbox = await self._start(request, cleanup, cfg.fws_image if plan.requires_fws else cfg.agent_image, "agent")
        with tempfile.TemporaryDirectory() as directory:
            fixtures = Path(directory) / "fixtures"
            await download_dir(verifier, "/sandbox/pinchbench-episode/fixtures", fixtures)
            await upload_dir(sandbox, fixtures, "/sandbox/workspace")
            access = SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=cfg.sandbox_provider, descriptor=await sandbox.serialize()
                ),
                workdir="/sandbox/workspace",
            )
            if plan.requires_fws:
                started = await sandbox.exec("fws server start", preserve_background_services=True, timeout_s=45)
                if started.error_type or started.return_code != 0:
                    raise handled_episode_failure(
                        stage="seed",
                        operation="episode",
                        terminal=True,
                        failure_reason="This task needs fws and its mock service in the agent image",
                    )
            turns, observations, trajectories = [], [], []
            deadline = asyncio.get_running_loop().time() + cfg.timeout_multiplier * plan.timeout
            groups = conversations(plan)
            for prompts in groups:
                session = AgentSession(self.server_client, cfg.agent_server, request)
                await session.seed(cleanup, sandbox_access=access, continuation=len(prompts) > 1)
                for prompt in prompts:
                    params = task.responses_create_params.model_copy(update={"input": prompt})
                    try:
                        turn = await session.activate(params, deadline=deadline)
                    except TurnDeadlineExpired as error:
                        raise handled_episode_failure(
                            stage="agent",
                            operation="episode",
                            terminal=False,
                            failure_reason="PinchBench task timed out",
                        ) from error
                    action = agent_outcome_policy(turn.response, config=cfg.agent_outcome_policy)
                    if action != OutcomeAction.VERIFY:
                        raise handled_episode_failure(
                            stage="agent",
                            operation="episode",
                            terminal=action == OutcomeAction.FAIL,
                            failure_reason=turn.response.error.message
                            if turn.response.error
                            else "PinchBench agent failed",
                        )
                    turns.append(turn)
                await session.close()
                if session.closed is None or session.closed.cleanup_confirmed is False:
                    raise handled_episode_failure(
                        stage="cleanup",
                        operation="episode",
                        terminal=False,
                        failure_reason="Agent termination was not confirmed",
                    )
                if session.closed.agent_observations is not None:
                    observations.append(session.closed.agent_observations)
                if session.closed.trajectory is not None:
                    trajectories.append(session.closed.trajectory)

            output = Path(directory) / "output"
            await download_dir(sandbox, "/sandbox/workspace", output)
            await upload_dir(verifier, output, "/sandbox/workspace")
        events = transcript(turns, agent=cfg.agent_server.name, session_id=request.episode_id.capture_key)
        await upload_text(verifier, path="/sandbox/pinchbench-episode/transcript.json", text=json.dumps(events))
        if plan.grading_type != "automated":
            await worker(
                "judge",
                env={
                    "JUDGE_MODEL": cfg.judge_model,
                    "JUDGE_BASE_URL": cfg.judge_base_url,
                    "JUDGE_API_KEY": cfg.judge_api_key.get_secret_value(),
                },
            )
        if plan.grading_type != "llm_judge":
            await worker("automated")
        grade = json.loads(await read_text(verifier, path="/sandbox/pinchbench-episode/grade.json"))
        return Response(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=Result(
                task_id=task_id,
                reward=grade["score"],
                grading_type=grade["grading_type"],
                grading_breakdown=grade["breakdown"],
                grading_notes=grade["notes"],
                response=turns[-1].response,
                turns=turns,
                session_observations=observations,
                session_trajectories=trajectories,
                ng_agent_observations=observations[0] if len(groups) == 1 and observations else None,
                ng_trajectory=trajectories[0] if len(groups) == 1 and trajectories else None,
            ),
        )


if __name__ == "__main__":
    PinchBenchEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PinchBenchEnvironmentServer.run_webserver()
