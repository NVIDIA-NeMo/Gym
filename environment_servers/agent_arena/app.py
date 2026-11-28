# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentArena as one Environment Server episode per row: this server sets the turn up, serves its simulated tools,
runs the agent and scores the turn. No Resources Server takes part in the episode.

``run`` for one row:

1. Builds the turn's session from the whole row (`TurnSimulation.new_session`): the row's harness, its history replayed
   into the model of the agent's files, the tool simulator's preamble and the armed obstacle.
2. Serves the row's tools for the episode (`serve_tools`, with a `TurnToolset`): the agent calls ``POST /<tool name>``
   with the episode token of its grant, and each call is answered as the resources server's ``/simulate_tool_call``
   answers it.
3. Seeds an agent session with that grant and the row's harness in its ``agent_context`` (`TurnHarnessContext`: the
   stop tools, the tool-call protocol and the tool names this server answers), and invokes the agent with the row's
   ``responses_create_params`` as they are. Then it closes the agent's session.
4. Scores the turn in process (`TurnSimulation.verify`: code gates, judges, masking). The result has the fields the
   ``agentic_turn_simulation`` resources server's ``/verify`` returns, so the metrics are the same, and a judge that
   gives no verdict masks the turn as that ``/verify`` does (``judge_failsafe``). With ``skip_verification`` (this
   server's, else the agent's), the turn gets the fixed reward instead, unscored.

The tool simulator, judge and scoring settings are those of the resources server the config names
(``resources_server``): overrides of that server's settings apply here too, and ``gym eval reverify`` through that
server scores a saved row with the same settings. The episode makes no call to it.

No operation runs in a sandbox: a call is answered from the session's model of the files or by a model call, so
``sandbox_provider`` is ``local`` and no episode starts one. Checkpointing is restart-only (the base default): after a
crash the row starts over from its input. With several workers, a tool call reaches the worker that runs its episode
(the base's session routing). Obstacle injection is refused at startup: a call reaches this server alone, without the
reply it belongs to, which an obstacle's decision needs.
"""

import asyncio
import hashlib
import json
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from omegaconf import OmegaConf
from pydantic import ConfigDict, Field, PositiveFloat, PrivateAttr, TypeAdapter, ValidationError

from environment_servers.agent_arena.toolset import TurnToolset
from nemo_gym.agents import (
    AGENT_ACTIVATION_HEADER,
    AGENT_OUTCOME_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentOutcomePolicyConfig,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    OutcomeAction,
    agent_outcome_policy,
    verification_failure_fields,
)
from nemo_gym.client import get_response_json, raise_boundary_failure
from nemo_gym.config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKIP_VERIFICATION_KEY_NAME,
    SKIP_VERIFICATION_REWARD_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
    TOKEN_CAPTURE_BLOCK,
    TOKEN_CAPTURE_PATH_SEGMENT,
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    ResourcesServerRef,
    get_first_server_config_dict,
)
from nemo_gym.core import (
    ENVIRONMENT_SERVER_FAILED,
    MODEL_SAMPLING_NULL,
    EpisodeId,
    FailureBoundary,
    FailureReportRow,
    NeMoGymResponse,
    NgFailureEvent,
    RequestFailureLog,
    maybe_rollout_id_from_run_body,
    model_call_sampling_config,
    model_sampling_context,
    model_sampling_from_run_body,
    model_sampling_path_segment,
    skips_token_capture,
    verifier_invocation_context,
)
from nemo_gym.data import materialize_task
from nemo_gym.data.task_materialization import TASK_ID_FIELDS
from nemo_gym.metrics import add_perf_metrics, compute_aggregate_metrics
from nemo_gym.server_runtime import (
    AgentContext,
    BaseEnvironmentServerWithTools,
    BaseEnvironmentServerWithToolsConfig,
    CleanupContext,
    SingleAgentTurnRequest,
    SingleAgentTurnResponse,
    SingleAgentTurnResult,
    ToolAccess,
    dependency_failure,
    handled_episode_failure,
    is_handled_episode_failure,
    is_retryable_dependency_error,
)
from nemo_gym.trajectory import (
    NG_AGENT_OBSERVATIONS_KEY,
    NG_TRAJECTORY_KEY,
    NG_VERIFIER_INVOCATIONS_KEY,
    AgentObservationBundle,
    TrajectoryRecord,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import judge_failsafe
from resources_servers.agentic_turn_simulation.schemas import (
    AgenticTurnSimulationRunRequest,
    AgenticTurnSimulationSettings,
    AgenticTurnSimulationVerifyRequest,
)
from resources_servers.agentic_turn_simulation.turns import OUTCOME_STAGE, TurnSession, TurnSimulation, harness_context


# Where a masked turn's outcome decision saves its originating stage (`applied_outcome_fields`).
FAILURE_STAGE_KEY = "_ng_failure_stage"


class AgentArenaEnvironmentServerConfig(BaseEnvironmentServerWithToolsConfig):
    """Bind the agent that plays the turn and the resources server whose settings score it."""

    model_config = ConfigDict(extra="forbid")

    agent_server: AgentServerRef
    # Its tool simulator, judge and scoring settings are the episode's. No call goes to it during an episode.
    resources_server: ResourcesServerRef
    agent_outcome_policy: AgentOutcomePolicyConfig = Field(default_factory=AgentOutcomePolicyConfig)
    # The limits of the legacy_agent relay this server replaces.
    default_episode_timeout_seconds: PositiveFloat | None = 21600
    cleanup_timeout_seconds: PositiveFloat = 180
    # Unset, each falls back to the bound agent's own setting, which also carries the global flag, so a run moved off
    # the legacy_agent relay keeps skipping as its agent's /run did (as in single_agent_turn).
    skip_verification: bool | None = None
    skip_verification_reward: float | None = None


@dataclass
class _AgentSession:
    """The episode's agent session: its caller-assigned id, and the cookies its replies set."""

    agent_session_id: str = field(default_factory=lambda: f"agent-session-{uuid4().hex}")
    cookies: dict[str, str] = field(default_factory=dict)


class AgentArenaEnvironmentServer(BaseEnvironmentServerWithTools[SingleAgentTurnRequest, SingleAgentTurnResponse]):
    """Run one simulated agent turn per episode and score it, serving the turn's tools itself."""

    config: AgentArenaEnvironmentServerConfig
    request_model = SingleAgentTurnRequest
    response_model = SingleAgentTurnResponse
    # The agent is the participant a training framework trains. The simulator and the judge keep their own sampling.
    applies_model_sampling = True

    _turns: TurnSimulation = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        settings = self._settings()
        if settings.obstacle_injection.enabled:
            # An obstacle's decision needs the turn's replies as the policy produced them (a stop tool beside a call,
            # sibling calls of one reply), and a call reaches this server alone. Refused at startup rather than
            # deciding obstacles from a history that would contradict the final scoring.
            raise ValueError(
                "agent_arena does not support obstacle_injection: run obstacle injection through the legacy_agent "
                "relay (benchmarks/agent_arena/config_legacy_agent.yaml or "
                "resources_servers/agentic_turn_simulation/configs/agentic_turn_simulation_legacy_agent.yaml)"
            )
        self._turns = TurnSimulation(settings, self.server_client)

    def _caller_sampling(self) -> AbstractContextManager[None]:
        """The sampling of this server's own model calls (the tool simulator and the judges) in a training run.

        In a run that sets ``model_call_sampling`` they send ``null``, as a resources server's calls do
        (``CallerSamplingMiddleware``): they keep their own sampling, and a trained agent's Model Server, which refuses
        a generation call without a segment, takes them when the simulator or a judge shares it. Only the agent's
        calls carry the training framework's values.
        """
        if model_call_sampling_config(self.server_client.global_config_dict) is None:
            return nullcontext()
        return model_sampling_context(MODEL_SAMPLING_NULL)

    def _settings(self) -> AgenticTurnSimulationSettings:
        """The settings of the resources server the config names, without that server's own fields."""
        raw = get_first_server_config_dict(self.server_client.global_config_dict, self.config.resources_server.name)
        raw = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else dict(raw)
        fields = AgenticTurnSimulationSettings.model_fields
        return AgenticTurnSimulationSettings.model_validate({k: v for k, v in raw.items() if k in fields})

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # A benchmark's flat row reaches /run too, so /run takes both shapes (run_legacy). The new route takes the
        # typed one's place, ahead of the tool route that takes every other one-segment POST path.
        routes = app.router.routes
        index = next(i for i, route in enumerate(routes) if isinstance(route, APIRoute) and route.path == "/run")
        app.post("/run")(self.run_legacy)
        routes[index] = routes.pop()
        return app

    # --- episode --------------------------------------------------------------------------------------------

    async def run(self, request: SingleAgentTurnRequest, cleanup: CleanupContext) -> SingleAgentTurnResponse:
        task_input = request.task.task_input
        try:
            row = AgenticTurnSimulationRunRequest.model_validate(
                task_input.task_data | {"responses_create_params": task_input.responses_create_params}
            )
            session = self._turns.new_session(row)
            agent_context = _agent_context(request, session)
            # A row tool that no grant can carry fails here (ToolSpec).
            toolset = TurnToolset(self._turns, session, row, call_context=self._caller_sampling)
        except ValueError as error:
            raise _row_setup_failure(error) from error
        try:
            # Registered first, so it runs last: the agent session closes before its tools go away.
            served = await self.serve_tools(request, toolset, cleanup)
        except ValueError as error:
            # A row tool named like one of this server's routes.
            raise _row_setup_failure(error) from error
        toolset.bind(served.calls)

        agent = _AgentSession()
        closed: list[AgentCloseSessionResponse] = []

        async def close_agent() -> None:
            closed.append(await self._close_agent(request, agent))

        # Registered before the seed, so a seed whose reply is lost is still closed.
        agent_cleanup = cleanup.register_cleanup("agent session", close_agent)
        await self._seed_agent(request, agent, served.tool_accesses, agent_context)
        try:
            response = await self._invoke_agent(request, agent)
        except NgFailureEvent as failure:
            if not is_handled_episode_failure(failure):
                raise
            # The agent's evidence of the failed turn arrives with its session close.
            try:
                await agent_cleanup.close()
            except Exception:
                pass  # Final unwind retries the close and logs its failure.
            if closed:
                failure.episode_evidence["agent_observations"] = closed[-1].agent_observations
                failure.episode_evidence["trajectory"] = closed[-1].trajectory
            raise
        try:
            await agent_cleanup.close()
        except Exception as error:
            raise dependency_failure(error, stage="cleanup", partial_response=response) from error
        observations, trajectory = closed[-1].agent_observations, closed[-1].trajectory

        action = agent_outcome_policy(response, config=self.config.agent_outcome_policy)
        if action != OutcomeAction.VERIFY:
            raise handled_episode_failure(
                boundary=FailureBoundary.ENVIRONMENT,
                stage="agent",
                operation="episode",
                failure_reason=(response.error.message if response.error else "Agent reported failure")[:2000],
                terminal=action == OutcomeAction.FAIL,
                failure_kind=_agent_failure_kind(response),
                evidence={"partial_response": response, "agent_observations": observations, "trajectory": trajectory},
            )

        skipped_reward = self._skipped_verification_reward()
        if skipped_reward is not None:
            # No scoring: the flag marks the fixed reward, so anything reading the result can tell it from a measured
            # one (as in single_agent_turn).
            result = SingleAgentTurnResult.model_validate(
                self._verify_body(request, response) | {"reward": skipped_reward, "verification_skipped": True}
            )
        else:
            result = await self._verify(request, response, session, observations, trajectory)
        return SingleAgentTurnResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=result.model_copy(update={NG_AGENT_OBSERVATIONS_KEY: observations, NG_TRAJECTORY_KEY: trajectory}),
        )

    async def _verify(
        self,
        request: SingleAgentTurnRequest,
        response: NeMoGymResponse,
        session: TurnSession,
        observations: AgentObservationBundle | None,
        trajectory: TrajectoryRecord | None,
    ) -> SingleAgentTurnResult:
        body = AgenticTurnSimulationVerifyRequest.model_validate(
            self._verify_body(request, response) | verification_failure_fields(response)
        )
        try:
            # The judges' calls belong to this verifier invocation, as a resources server's /verify calls do.
            with self._caller_sampling(), verifier_invocation_context(f"verify-{uuid4().hex}"):
                # A judge that gives no verdict masks the turn (reward 0, judge_failed), as the resources server's
                # /verify does: an inconclusive grade is not retried by running the whole turn again.
                verified = await judge_failsafe(self._turns.verify)(body=body, state=session)
        except Exception as error:
            # Any other failure of the scoring (judge failures are masked above) fails the row as the resources
            # server's failed /verify does: retried when another attempt may pass, else terminal.
            cause = error.__cause__ if isinstance(error, NgFailureEvent) and error.__cause__ is not None else error
            raise dependency_failure(
                error,
                stage="verification",
                terminal=not is_retryable_dependency_error(cause),
                partial_response=response,
                agent_observations=observations,
                trajectory=trajectory,
            ) from error
        payload = json.loads(verified.body) if isinstance(verified, JSONResponse) else verified.model_dump(mode="json")
        # Rollout collection keeps `_ng_failure_stage` for itself in an episode result; the flat result restores it
        # (`_legacy_result`), as the relay's row has it.
        payload.pop(FAILURE_STAGE_KEY, None)
        return SingleAgentTurnResult.model_validate(payload)

    def _skipped_verification_reward(self) -> float | None:
        """The fixed reward that replaces scoring, or None to score the turn.

        This server's own setting wins; otherwise the bound agent's applies (the global flag is placed on the agents).
        The agent's raw values are parsed as the agent's own config parses them, so a string "false" stays false.
        """
        agent_config = get_first_server_config_dict(
            self.server_client.global_config_dict, self.config.agent_server.name
        )
        skip = self.config.skip_verification
        if skip is None:
            skip = TypeAdapter(bool).validate_python(agent_config.get(SKIP_VERIFICATION_KEY_NAME, False))
        if not skip:
            return None
        reward = self.config.skip_verification_reward
        if reward is None:
            reward = TypeAdapter(float).validate_python(agent_config.get(SKIP_VERIFICATION_REWARD_KEY_NAME, 0.0))
        return reward

    async def _seed_agent(
        self,
        request: SingleAgentTurnRequest,
        agent: _AgentSession,
        tool_accesses: list[ToolAccess],
        agent_context: AgentContext,
    ) -> None:
        try:
            http = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions",
                json=AgentSeedSessionRequest(
                    agent_session_id=agent.agent_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    tool_accesses=tool_accesses,
                    # The row's harness, with the typed task's own context, which the agent validates or refuses.
                    agent_context=agent_context,
                ).model_dump(mode="json"),
            )
            await raise_boundary_failure(http)
            seeded = AgentSeedSessionResponse.model_validate(await get_response_json(http))
            if seeded.agent_session_id != agent.agent_session_id:
                raise ValueError("Agent seed returned a different agent_session_id")
            agent.cookies = _cookies(http)
        except Exception as error:
            raise dependency_failure(error, stage="agent") from error

    async def _invoke_agent(self, request: SingleAgentTurnRequest, agent: _AgentSession) -> NeMoGymResponse:
        try:
            http = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path=self._agent_responses_path(request),
                # The row's prompt only: the rest of the row stays here, and its harness went with the seed.
                json=request.task.task_input.responses_create_params,
                cookies=agent.cookies,
                # The episode's only activation.
                headers={AGENT_ACTIVATION_HEADER: "1"},
            )
            await raise_boundary_failure(http)
            agent.cookies = _cookies(http) or agent.cookies
            response = NeMoGymResponse.model_validate(await get_response_json(http))
            # Body-carried diagnostics, kept before the outcome policy can skip scoring.
            saved = (response.model_extra or {}).get("failure_report")
            log = RequestFailureLog.current()
            if saved is not None and log is not None:
                log.add_saved(saved)
            return response
        except Exception as error:
            raise dependency_failure(error, stage="agent") from error

    async def _close_agent(self, request: SingleAgentTurnRequest, agent: _AgentSession) -> AgentCloseSessionResponse:
        http = await self.server_client.post(
            server_name=self.config.agent_server.name,
            url_path="/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(
                agent_session_id=agent.agent_session_id,
                episode_id=request.episode_id,
            ).model_dump(mode="json"),
            cookies=agent.cookies,
        )
        await raise_boundary_failure(http)
        close = AgentCloseSessionResponse.model_validate(await get_response_json(http))
        if close.agent_session_id != agent.agent_session_id:
            raise ValueError("Agent close returned a different agent_session_id")
        return close

    def _agent_responses_path(self, request: SingleAgentTurnRequest) -> str:
        """The agent's /v1/responses under the episode's rollout prefix, with token capture and the training
        framework's sampling values as single_agent_turn sends them."""
        block = self.server_client.global_config_dict.get(TOKEN_CAPTURE_BLOCK) or {}
        agent_config = get_first_server_config_dict(
            self.server_client.global_config_dict, self.config.agent_server.name
        )
        token_capture = (
            not request.skip_token_capture
            and bool(block.get("enabled", False))
            and (bool(block.get("all_agents", False)) or bool(agent_config.get("token_id_capture", False)))
        )
        capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
        sampling_segment = model_sampling_path_segment(self.model_sampling_segment(request, trained=True))
        return f"/ng-rollout/{request.episode_id.capture_key}{capture_segment}{sampling_segment}/v1/responses"

    @staticmethod
    def _verify_body(request: SingleAgentTurnRequest, response: NeMoGymResponse) -> dict[str, Any]:
        """The resources server's flat verify body, as the agent's /run sends it."""
        task_input = request.task.task_input
        return task_input.task_data | {
            "responses_create_params": task_input.responses_create_params.model_dump(mode="json", exclude_unset=True),
            "response": response.model_dump(mode="json"),
        }

    # --- metrics --------------------------------------------------------------------------------------------

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """The resources server's metrics, computed here, plus the performance metrics of the whole batch. With
        verification skipped, only the performance metrics: a fixed reward is no measured turn score."""
        skipped = self._skipped_verification_reward() is not None

        def aggregate() -> AggregateMetrics:
            if skipped:
                return add_perf_metrics(AggregateMetrics(), body.verify_responses)
            # The resources server is sent the rows without their performance summary, which is reduced once below.
            rows = [{key: value for key, value in row.items() if key != "ng_perf"} for row in body.verify_responses]
            metrics = compute_aggregate_metrics(
                rows, compute_metrics_fn=self._turns.compute_metrics, get_key_metrics_fn=self._turns.get_key_metrics
            )
            return add_perf_metrics(metrics, body.verify_responses)

        # The reduction is CPU-bound over every row, so keep it off the server's event loop.
        return await asyncio.to_thread(aggregate)

    # --- flat rows ------------------------------------------------------------------------------------------

    async def run_legacy(self, row: dict[str, Any]) -> dict[str, Any]:
        """An episode request gets the episode reply; a flat row (a benchmark's) gets a flat result, as
        single_agent_turn_legacy answers it."""
        is_episode_request = "episode_id" in row and "task" in row
        try:
            request = (
                SingleAgentTurnRequest.model_validate(row)
                if is_episode_request
                else self._episode_request_from_row(row)
            )
        except ValidationError as error:
            # A request that fails the episode request model gets a terminal 422, as on the episode request route,
            # instead of a 500 that a caller retries.
            raise RequestValidationError(error.errors(include_url=False)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        response = await self.run_request(request)
        return response.model_dump(mode="json") if is_episode_request else self._legacy_result(response)

    def _episode_request_from_row(self, row: dict[str, Any]) -> SingleAgentTurnRequest:
        task_source = row.get(TASK_SOURCE_KEY_NAME, self.config.resources_server.name)
        bound = (self.config.resources_server.name, self.config.agent_server.name, self.config.name)
        if task_source not in bound:
            raise ValueError(
                f"Row task_source {task_source!r} names none of this Environment Server's instances: {list(bound)}"
            )
        agent_ref = row.get(AGENT_REF_KEY_NAME)
        if agent_ref is not None and (
            not isinstance(agent_ref, dict) or agent_ref.get("name") != self.config.agent_server.name
        ):
            raise ValueError(
                f"Row agent_ref {agent_ref!r} does not match configured agent server {self.config.agent_server.name!r}"
            )
        attempt = row.get(ATTEMPT_INDEX_KEY_NAME, 0)
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            raise ValueError(f"Invalid episode attempt: {attempt!r}")
        capture_key = maybe_rollout_id_from_run_body(row)
        if capture_key is not None:
            episode_id, task = EpisodeId.from_capture_key(capture_key), materialize_task(row, taskset=task_source)
        elif TASK_INDEX_KEY_NAME in row or ROLLOUT_INDEX_KEY_NAME in row:
            raise ValueError(f"A row with {TASK_INDEX_KEY_NAME} or {ROLLOUT_INDEX_KEY_NAME} needs both")
        else:
            episode_id, task = _direct_call_identity(row, task_source, attempt)
        return SingleAgentTurnRequest(
            episode_id=episode_id,
            task=task,
            model_sampling=model_sampling_from_run_body(row),
            skip_token_capture=skips_token_capture(row),
        )

    def _legacy_result(self, response: SingleAgentTurnResponse) -> dict[str, Any]:
        agent_ref = {"name": self.config.agent_server.name}
        if response.failure is not None:
            failure: dict[str, Any] = {
                "_ng_failure_class": response.failure.failure_kind or ENVIRONMENT_SERVER_FAILED,
                "_ng_failure_kind": response.failure.failure_kind,
                "_ng_failure_terminal": response.failure.terminal,
                "_ng_failure_message": response.failure.failure_reason,
                "_ng_failure_stage": response.failure.stage,
                "agent_ref": agent_ref,
            }
            if response.failure.partial_response is not None:
                failure["_ng_failure_partial_response"] = response.failure.partial_response.model_dump(mode="json")
            if response.failure.agent_observations is not None:
                failure[NG_AGENT_OBSERVATIONS_KEY] = response.failure.agent_observations.model_dump(mode="json")
            if response.failure.trajectory is not None:
                failure[NG_TRAJECTORY_KEY] = response.failure.trajectory.model_dump(mode="json")
            if response.failure.verifier_invocations is not None:
                failure[NG_VERIFIER_INVOCATIONS_KEY] = [
                    invocation.model_dump(mode="json") for invocation in response.failure.verifier_invocations
                ]
            if response.failure_report is not None:
                FailureReportRow.merged_into(failure, response.failure_report)
            return failure
        if response.result is None:
            raise ValueError("Successful episode response has no result")
        result = response.result.model_dump(mode="json")
        if result.get(NG_AGENT_OBSERVATIONS_KEY) is None:
            result.pop(NG_AGENT_OBSERVATIONS_KEY, None)
        if "failure_outcome_action" in result:
            # A masked turn's outcome decision, saved as the resources server's /verify saves it.
            result[FAILURE_STAGE_KEY] = OUTCOME_STAGE
        result["agent_ref"] = agent_ref
        if response.failure_report is not None:
            FailureReportRow.merged_into(result, response.failure_report)
        return result


def _agent_context(request: SingleAgentTurnRequest, session: TurnSession) -> AgentContext:
    """The agent session's context: the row's harness (`TurnHarnessContext`) beside the typed task's own context.

    A task context that sets one of the harness's keys is refused rather than overridden: the episode never runs
    without context it was given.
    """
    harness = harness_context(session.harness, session.tool_names).model_dump(mode="json")
    task = request.task.task_input.agent_seed_context or AgentContext()
    taken = sorted(task.fields().keys() & harness.keys())
    if taken:
        raise handled_episode_failure(
            stage="agent",
            operation="seed",
            failure_reason=f"The task's agent_context sets {taken}, which agent_arena sends from the row's harness",
            terminal=True,
        )
    return AgentContext(values={**task.values, **harness}, artifacts=task.artifacts)


def _direct_call_identity(row: dict[str, Any], taskset: str, attempt: int) -> tuple[EpisodeId, dict[str, Any]]:
    """The identities of a flat row posted straight to ``/run`` with no rollout id and neither collector index, as
    the relay accepted one: each call is an episode of its own, and a row without a task id field is named by its
    content, so the same row posted again is the same task."""
    # The index only satisfies materialize_task; the id it gives is replaced below when the row names no task.
    task = materialize_task(row, taskset=taskset, task_index=0)
    if not any(row.get(key) is not None for key in TASK_ID_FIELDS):
        content = json.dumps(task["task_input"], sort_keys=True, ensure_ascii=False).encode()
        task["task_id"]["task_id"] = f"direct-{hashlib.sha256(content).hexdigest()[:16]}"
    return EpisodeId(rollout_id=f"direct-{uuid4().hex}", attempt=attempt), task


def _row_setup_failure(error: ValueError) -> NgFailureEvent:
    """A row this server cannot set an episode up for (its tools, its harness context): terminal, since every attempt
    fails alike, and named, so the failure row says which row field is wrong. (A handled failure, such as
    `_agent_context`'s, is no ValueError and passes through.)"""
    return handled_episode_failure(
        stage="seed",
        operation="episode.setup",
        failure_reason=f"The row cannot be set up as an episode: {error}"[:2000],
        terminal=True,
    )


def _agent_failure_kind(response: NeMoGymResponse) -> str | None:
    """The failure kind a failed agent reply classified itself with, if it did."""
    try:
        return AgentOutcome.model_validate_json(
            (response.metadata or {}).get(AGENT_OUTCOME_METADATA_KEY, "")
        ).failure_kind
    except (ValidationError, TypeError):
        return None


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


if __name__ == "__main__":
    AgentArenaEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AgentArenaEnvironmentServer.run_webserver()  # noqa: F401
