# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentArena as one Environment Server episode per row: this server sets the turn up, serves its simulated tools,
runs the agent and scores the turn. No Resources Server takes part in the episode.

``run`` for one row:

1. Builds the turn's session from the whole row (`TurnSimulation.new_session`): the row's harness, its history replayed
   into the model of the agent's files, the tool simulator's preamble and the armed obstacle. The session stays in
   this process under a secret drawn for the episode.
2. Seeds an agent session whose direct HTTP grant points at this server, with the secret in its headers. The agent
   calls ``/harness`` and ``/simulate_tool_call`` through it; a call that carries no live episode's secret is refused.
3. Invokes the agent with the row's ``responses_create_params`` and closes its session.
4. Scores the turn in process (`TurnSimulation.verify`: code gates, judges, masking). The result has the fields the
   ``agentic_turn_simulation`` resources server's ``/verify`` returns, so the metrics are the same.
5. Drops the session in the episode's cleanup.

The tool simulator, judge and scoring settings are those of the resources server the config names
(``resources_server``): overrides of that server's settings apply here too, and ``gym eval reverify`` through that
server scores a saved row with the same settings. The episode makes no call to it.

Checkpointing is restart-only (the base default): after a crash the row starts over from its input.
"""

import asyncio
import secrets
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request
from omegaconf import OmegaConf
from pydantic import ConfigDict, Field, PositiveFloat, PrivateAttr, ValidationError

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
    EpisodeId,
    FailureBoundary,
    FailureReportRow,
    NeMoGymResponse,
    NgFailureEvent,
    RequestFailureLog,
    maybe_rollout_id_from_run_body,
    model_sampling_from_run_body,
    model_sampling_path_segment,
    skips_token_capture,
)
from nemo_gym.data import materialize_task
from nemo_gym.metrics import add_perf_metrics, compute_aggregate_metrics
from nemo_gym.server_runtime import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    DirectHTTPToolAccess,
    RolloutContextMiddleware,
    SingleAgentTurnRequest,
    SingleAgentTurnResponse,
    SingleAgentTurnResult,
    dependency_failure,
    handled_episode_failure,
    is_handled_episode_failure,
    is_retryable_dependency_error,
)
from nemo_gym.trajectory import NG_AGENT_OBSERVATIONS_KEY, NG_TRAJECTORY_KEY, NG_VERIFIER_INVOCATIONS_KEY
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.agentic_turn_simulation.schemas import (
    AgenticTurnSimulationRunRequest,
    AgenticTurnSimulationSettings,
    AgenticTurnSimulationVerifyRequest,
    HarnessResponse,
    SimulateToolCallRequest,
    SimulateToolCallResponse,
)
from resources_servers.agentic_turn_simulation.turns import TurnSession, TurnSimulation


# The header that carries an episode's secret on the agent's calls to /harness and /simulate_tool_call.
EPISODE_SECRET_HEADER = "x-agent-arena-episode"


class AgentArenaEnvironmentServerConfig(BaseEnvironmentServerConfig):
    """Bind the agent that plays the turn and the resources server whose settings score it."""

    model_config = ConfigDict(extra="forbid")

    agent_server: AgentServerRef
    # Its tool simulator, judge and scoring settings are the episode's. No call goes to it during an episode.
    resources_server: ResourcesServerRef
    agent_outcome_policy: AgentOutcomePolicyConfig = Field(default_factory=AgentOutcomePolicyConfig)
    # The limits of the legacy_agent relay this server replaces.
    default_episode_timeout_seconds: PositiveFloat | None = 21600
    cleanup_timeout_seconds: PositiveFloat = 180


@dataclass
class _AgentSession:
    """The episode's agent session: its caller-assigned id, and the cookies its replies set."""

    agent_session_id: str = field(default_factory=lambda: f"agent-session-{uuid4().hex}")
    cookies: dict[str, str] = field(default_factory=dict)


class AgentArenaEnvironmentServer(BaseEnvironmentServer[SingleAgentTurnRequest, SingleAgentTurnResponse]):
    """Run one simulated agent turn per episode and score it, serving the turn's tools itself."""

    config: AgentArenaEnvironmentServerConfig
    request_model = SingleAgentTurnRequest
    response_model = SingleAgentTurnResponse
    # The agent is the participant a training framework trains. The simulator and the judge keep their own sampling.
    applies_model_sampling = True

    _turns: TurnSimulation = PrivateAttr()
    # The live episodes' turn sessions, by the secret their agent sends.
    _sessions: dict[str, TurnSession] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._turns = TurnSimulation(self._settings(), self.server_client)

    def _settings(self) -> AgenticTurnSimulationSettings:
        """The settings of the resources server the config names, without that server's own fields."""
        raw = get_first_server_config_dict(self.server_client.global_config_dict, self.config.resources_server.name)
        raw = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else dict(raw)
        fields = AgenticTurnSimulationSettings.model_fields
        return AgenticTurnSimulationSettings.model_validate({k: v for k, v in raw.items() if k in fields})

    def multiple_workers_unsupported_reason(self) -> str:
        return "each episode's turn session lives in the worker that runs it, which the agent's tool calls must reach"

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        # Flat rows run the episode protocol through run_request too, so their episodes take part in checkpoints.
        self.setup_environment_checkpoint(app)
        app.post("/run")(self.run_legacy)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        app.post("/harness")(self.describe_harness)
        app.post("/simulate_tool_call")(self.simulate_tool_call)
        # With rollout checkpointing on, the agent names its attempt in the path of every tool call.
        app.add_middleware(RolloutContextMiddleware)
        return app

    # --- the agent's tool routes ----------------------------------------------------------------------------

    def _session_for(self, request: Request) -> TurnSession:
        session = self._sessions.get(request.headers.get(EPISODE_SECRET_HEADER, ""))
        if session is None:
            # Not 404: the agent reads a 404 from /harness as a bare request with no session.
            raise HTTPException(status_code=403, detail="This call carries no live AgentArena episode's secret.")
        return session

    async def describe_harness(self, request: Request) -> HarnessResponse:
        return self._turns.describe_harness(self._session_for(request))

    async def simulate_tool_call(self, request: Request, body: SimulateToolCallRequest) -> SimulateToolCallResponse:
        return await self._turns.simulate_tool_call(self._session_for(request), body)

    # --- episode --------------------------------------------------------------------------------------------

    async def run(self, request: SingleAgentTurnRequest, cleanup: CleanupContext) -> SingleAgentTurnResponse:
        task_input = request.task.task_input
        row = AgenticTurnSimulationRunRequest.model_validate(
            task_input.task_data | {"responses_create_params": task_input.responses_create_params}
        )
        session = self._turns.new_session(row)
        secret = secrets.token_urlsafe(32)
        self._sessions[secret] = session

        async def release_session() -> None:
            self._sessions.pop(secret, None)

        # Registered first, so it runs last: the agent session closes before its tools go away.
        cleanup.register_cleanup("turn session", release_session)

        agent = _AgentSession()
        closed: list[AgentCloseSessionResponse] = []

        async def close_agent() -> None:
            closed.append(await self._close_agent(request, agent))

        # Registered before the seed, so a seed whose reply is lost is still closed.
        agent_cleanup = cleanup.register_cleanup("agent session", close_agent)
        await self._seed_agent(request, agent, secret)
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

        body = AgenticTurnSimulationVerifyRequest.model_validate(
            self._verify_body(request, response) | verification_failure_fields(response)
        )
        try:
            verified = await self._turns.verify(body, session)
        except Exception as error:
            # A judge that could not be reached fails the row as the resources server's failed /verify does: another
            # attempt may reach it. A judge that refused the call (a 4xx, such as a conversation too long for it) would
            # refuse it again.
            cause = error.__cause__ if isinstance(error, NgFailureEvent) and error.__cause__ is not None else error
            raise dependency_failure(
                error,
                stage="verification",
                terminal=not is_retryable_dependency_error(cause),
                partial_response=response,
                agent_observations=observations,
                trajectory=trajectory,
            ) from error
        result = SingleAgentTurnResult.model_validate(verified.model_dump(mode="json"))
        return SingleAgentTurnResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=result.model_copy(update={NG_AGENT_OBSERVATIONS_KEY: observations, NG_TRAJECTORY_KEY: trajectory}),
        )

    async def _seed_agent(self, request: SingleAgentTurnRequest, agent: _AgentSession, secret: str) -> None:
        grant = DirectHTTPToolAccess(
            name=f"{self.config.name}.direct_http",
            required=True,
            base_url=self.server_client._resolve_base_url(self.config.name).rstrip("/"),
            headers={EPISODE_SECRET_HEADER: secret},
        )
        try:
            http = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions",
                json=AgentSeedSessionRequest(
                    agent_session_id=agent.agent_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    tool_accesses=[grant],
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
                # The row's prompt only: the turn's session is this server's, reached through the grant.
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
        """The resources server's metrics, computed here, plus the performance metrics of the whole batch."""

        def aggregate() -> AggregateMetrics:
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
        request = (
            SingleAgentTurnRequest.model_validate(row) if is_episode_request else self._episode_request_from_row(row)
        )
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
        capture_key = (
            maybe_rollout_id_from_run_body(row) or f"{row[TASK_INDEX_KEY_NAME]}-{row[ROLLOUT_INDEX_KEY_NAME]}"
        )
        return SingleAgentTurnRequest(
            episode_id=EpisodeId.from_capture_key(capture_key),
            task=materialize_task(row, taskset=task_source),
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
        result["agent_ref"] = agent_ref
        if response.failure_report is not None:
            FailureReportRow.merged_into(result, response.failure_report)
        return result


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
