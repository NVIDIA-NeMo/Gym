# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resources-backed single-agent environment server."""

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal
from uuid import uuid4

from aiohttp import ClientConnectorError, ConnectionTimeoutError
from fastapi import Body
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, TypeAdapter, ValidationError, model_validator

from nemo_gym.agents import (
    AGENT_ACTIVATION_HEADER,
    AGENT_EPISODE_STEP_LIMIT_HEADER,
    AGENT_OUTCOME_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentOutcomePolicyConfig,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    OutcomeAction,
    agent_outcome_policy,
    confirmed_policy_response_timeout,
    verification_failure_fields,
)
from nemo_gym.client import get_response_json, raise_boundary_failure, rollout_path_prefix
from nemo_gym.config import (
    AGENT_SESSION_AGENT_TYPES,
    SKIP_VERIFICATION_KEY_NAME,
    SKIP_VERIFICATION_REWARD_KEY_NAME,
    TOKEN_CAPTURE_BLOCK,
    TOKEN_CAPTURE_PATH_SEGMENT,
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    ResourcesServerRef,
    get_first_server_config_dict,
)
from nemo_gym.core import (
    MODEL_POLICY_RESPONSE_TIMEOUT,
    MODEL_REQUEST_ERROR,
    FailureBoundary,
    FailureReport,
    FailureReportRow,
    ModelFailureDetails,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NgFailureEvent,
    RequestFailureLog,
    accumulate_response_usage,
    model_sampling_path_segment,
    rebuild_failure,
)
from nemo_gym.metrics import add_perf_metrics
from nemo_gym.server_runtime import (
    CHECKPOINT_CLOSE_FOLLOWS_HEADER,
    RESOURCES_TASK_STATE_PATH,
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CheckpointMode,
    CleanupContext,
    CleanupHandle,
    DirectHTTPToolAccess,
    MCPStreamableHTTPConnection,
    MCPToolAccess,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesTaskState,
    SingleAgentTurnRequest,
    SingleAgentTurnResponse,
    SingleAgentTurnResult,
    StepType,
    ToolAccess,
    dependency_failure,
    handled_episode_failure,
    is_handled_episode_failure,
    seed_rebinds_tool_accesses,
    seed_restarts,
    seed_verify_step_type,
)
from nemo_gym.trajectory import (
    NG_AGENT_OBSERVATIONS_KEY,
    NG_TRAJECTORY_KEY,
    NG_VERIFIER_INVOCATIONS_KEY,
    AgentObservationBundle,
    TrajectoryRecord,
    VerifierInvocation,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


LOGGER = logging.getLogger(__name__)


class ToolSession(BaseModel):
    """An episode's session on a tool server, a Resources server whose MCP tools the agent is granted."""

    session_id: str
    cookies: dict[str, str] = Field(default_factory=dict)


class SessionHandles(BaseModel):
    """The sessions an episode owns: enough to reuse them after a restore and to close them."""

    resources_session_id: str
    resources_cookies: dict[str, str] = Field(default_factory=dict)
    agent_session_id: str
    agent_cookies: dict[str, str] = Field(default_factory=dict)
    # Whether a checkpoint may replay the resources server's /verify, as its seed reply reported.
    resources_verify_step_type: StepType = "wait"
    observation: str | None = None
    # Keyed by tool server name.
    tool_sessions: dict[str, ToolSession] = Field(default_factory=dict)
    # Whether the agent borrows a sandbox the resources server lent in its seed reply.
    # A restore rebuilds that sandbox under a new id, so the replacement attempt seeds both sessions again.
    borrowed_sandbox: bool = False

    @classmethod
    def new(cls, tool_servers: Sequence[str] = ()) -> "SessionHandles":
        # Every session ID is assigned before seeding, so a lost seed reply cannot hide one from cleanup.
        return cls(
            resources_session_id=f"resources-session-{uuid4().hex}",
            agent_session_id=f"agent-session-{uuid4().hex}",
            tool_sessions={name: ToolSession(session_id=f"tool-session-{uuid4().hex}") for name in tool_servers},
        )


class SingleAgentTurnEnvironmentServerConfig(BaseEnvironmentServerConfig):
    """Bind Resources and Agent Servers for one agent turn."""

    model_config = ConfigDict(extra="forbid")

    resources_server: ResourcesServerRef
    agent_server: AgentServerRef
    agent_outcome_policy: AgentOutcomePolicyConfig = Field(default_factory=AgentOutcomePolicyConfig)
    resources_tool_transports: list[Literal["direct_http", "mcp"]] = Field(default_factory=list)
    # Resources servers, such as a search server, whose MCP tools every episode grants the agent besides its own
    # Resources server's. Unset, the bound agent's own `tool_servers`, so an agent moved off the legacy_agent relay
    # keeps the tool servers its /run seeded.
    tool_servers: list[ResourcesServerRef] | None = None
    forward_agent_evidence: bool = False
    # Unset, each falls back to the bound agent's own setting, which also carries the global flag.
    # A pairing moved off the legacy_agent relay then keeps skipping as its agent's /run did.
    skip_verification: bool | None = None
    skip_verification_reward: float | None = None
    max_episode_steps: PositiveInt | None = Field(
        default=None,
        description=(
            "Drive the agent one step per activation (one model call and the tool calls it makes), up to this many "
            "steps in the episode. The agent must keep a session, run the tools it calls and stop after one step "
            "(simple_agent with max_steps: 1); the server checks at startup. The episode ends after the first step "
            "in which the model called no tool or that did not complete, and with end_on_task_state once the "
            "resources server reports that the task ended. Its episodes take part in checkpoints as restarts. "
            "Unset, the agent runs its own loop in one activation."
        ),
    )
    end_on_task_state: bool = Field(
        default=False,
        description=(
            "With max_episode_steps, read the resources session's task state after each step "
            f"(GET {RESOURCES_TASK_STATE_PATH}, ResourcesTaskState) and end the episode once the task has ended, "
            "before the next model call. For a resources server whose task ends through its state rather than "
            "through one tool's call"
        ),
    )

    @model_validator(mode="after")
    def _task_state_is_read_between_steps(self) -> "SingleAgentTurnEnvironmentServerConfig":
        if self.end_on_task_state and self.max_episode_steps is None:
            raise ValueError("end_on_task_state reads the task state between steps: set max_episode_steps too")
        return self


class SingleAgentTurnEnvironmentServer(BaseEnvironmentServer[SingleAgentTurnRequest, SingleAgentTurnResponse]):
    """Run one agent turn followed by Resources verification and cleanup."""

    config: SingleAgentTurnEnvironmentServerConfig
    request_model = SingleAgentTurnRequest
    checkpoint_mode = "exported"
    applies_model_sampling = True
    response_model = SingleAgentTurnResponse

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        # Fail at startup, not on the first episode, when a tool server cannot serve its tools.
        for name in self.tool_server_names:
            self._check_tool_server(name, self.config.resources_server.name)
        self._check_mcp_server_names()
        if self.config.max_episode_steps is not None:
            self._check_step_driven_agent()

    def checkpoint_mode_in_effect(self) -> CheckpointMode:
        # The step loop records no boundary between steps, so an episode it drives starts over after a crash.
        return "restart_only" if self.config.max_episode_steps is not None else self.checkpoint_mode

    def _check_step_driven_agent(self) -> None:
        """Refuse to drive an agent one step per activation that cannot take its steps so: the agent must keep a
        session (``AGENT_SESSION_AGENT_TYPES``), run the tools it calls, and stop after one step (``max_steps: 1``),
        or a step taken after the task ended, within the same activation, would count."""
        name = self.config.agent_server.name
        instance = self.server_client.global_config_dict.get(name)
        agents = instance.get("responses_api_agents") if isinstance(instance, (dict, DictConfig)) else None
        agent_type = next(iter(agents), None) if isinstance(agents, (dict, DictConfig)) and len(agents) == 1 else None
        config = get_first_server_config_dict(self.server_client.global_config_dict, name) if agent_type else {}
        problems = []
        if agent_type not in AGENT_SESSION_AGENT_TYPES:
            problems.append(f"its type {agent_type!r} keeps no agent session")
        if config.get("execute_tools") is False:
            problems.append("it does not run the tools it calls (execute_tools: false)")
        if config.get("max_steps") != 1:
            problems.append(f"it stops after max_steps: {config.get('max_steps')} steps per activation, not 1")
        if problems:
            raise ValueError(
                f"Environment Server {self.config.name!r} drives {name!r} one step per activation "
                f"(max_episode_steps: {self.config.max_episode_steps}), but " + "; ".join(problems)
            )

    def _check_mcp_server_names(self) -> None:
        """Refuse a pairing that grants two MCP endpoints listed under one name: MCP clients name each tool by the
        endpoint that serves it (``mcp__<name>__<tool>``), so the two endpoints' tools would collide. Each endpoint
        lists itself under its ``mcp_server_name``, else its server's name."""
        granted = [self.config.resources_server.name] if "mcp" in self.config.resources_tool_transports else []
        granted += self.tool_server_names
        listed: dict[str, str] = {}
        for name in granted:
            config = get_first_server_config_dict(self.server_client.global_config_dict, name)
            listed_as = config.get("mcp_server_name") or name
            if listed_as in listed:
                raise ValueError(
                    f"Environment Server {self.config.name!r} grants the MCP tools of {listed[listed_as]!r} and "
                    f"{name!r}, which both list themselves as {listed_as!r}, so their tools' names would collide; "
                    "give one of them another mcp_server_name"
                )
            listed[listed_as] = name

    def _check_tool_server(self, name: str, resources_server_name: str) -> None:
        server = f"Environment Server {self.config.name!r}"
        instance = self.server_client.global_config_dict.get(name)
        if instance is None:
            raise ValueError(f"{server} grants tools of an unknown server {name!r}")
        if not isinstance(instance, (dict, DictConfig)) or "resources_servers" not in instance:
            raise ValueError(f"{server} grants tools of {name!r}, which is not a Resources server")
        if name == resources_server_name:
            raise ValueError(
                f"{server} lists its Resources server {name!r} as a tool server; "
                "grant its tools with resources_tool_transports: [mcp] instead"
            )
        if not get_first_server_config_dict(self.server_client.global_config_dict, name).get("expose_tools_over_mcp"):
            raise ValueError(
                f"{server} grants the MCP tools of {name!r}, which does not serve them; "
                "set expose_tools_over_mcp: true on that server"
            )

    @property
    def tool_server_names(self) -> list[str]:
        """The tool servers each episode seeds and grants: this server's own setting, or else its agent's."""
        servers = self.config.tool_servers
        if servers is None:
            agent_config = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.agent_server.name,
            )
            raw = agent_config.get("tool_servers") or []
            servers = TypeAdapter(list[ResourcesServerRef]).validate_python(
                OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else raw
            )
        return [server.name for server in servers]

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Combine the resources server's benchmark metrics with performance metrics for the whole batch.

        Rollout collection derives each row's ``ng_perf`` from agent observations and model-call captures.
        Performance metrics count every supplied rollout, including masked and unverified ones.
        The resources server owns scoring, task-specific weighting, and headline metric selection.
        """
        metrics = AggregateMetrics()
        if self._skipped_verification_reward() is None:
            # Keep benchmark evidence (including response.usage) intact. The reserved
            # performance summary is reduced here once, independently of resources server filtering.
            benchmark_body = body.model_copy(
                update={
                    "verify_responses": [
                        {key: value for key, value in row.items() if key != "ng_perf"} for row in body.verify_responses
                    ]
                }
            )
            response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/aggregate_metrics",
                json=benchmark_body,
            )
            await raise_boundary_failure(response)
            metrics = AggregateMetrics.model_validate(await get_response_json(response))
        # The reduction is CPU-bound over every row, so keep it off the server's event loop.
        return await asyncio.to_thread(add_perf_metrics, metrics, body.verify_responses)

    async def run(
        self,
        request: SingleAgentTurnRequest,
        cleanup: CleanupContext,
    ) -> SingleAgentTurnResponse:
        # Steps: seed → invoke the agent → close the agent → verify → return.
        # A boundary before each step names the next step, the sessions to reuse, and what that step needs,
        # so a replacement attempt that continues a checkpoint resumes at that step.
        continuation = self.checkpoint_continuation(request) or {}
        stage = continuation.get("next", "seed")
        handles = (
            SessionHandles.model_validate(continuation["handles"])
            if "handles" in continuation
            else SessionHandles.new(self.tool_server_names)
        )
        agent_response = (
            NeMoGymResponse.model_validate(continuation["response"]) if continuation.get("response") else None
        )
        observations = (
            AgentObservationBundle.model_validate(continuation["observations"])
            if continuation.get("observations")
            else None
        )
        trajectory = (
            TrajectoryRecord.model_validate(continuation["trajectory"]) if continuation.get("trajectory") else None
        )
        verification = (
            SingleAgentTurnResult.model_validate(continuation["result"]) if continuation.get("result") else None
        )

        # Register cleanup before seeding so a lost seed reply cannot hide the caller-assigned session ID.
        # Final cleanup runs after run() returns, outside the episode deadline.
        resources_cleanup = cleanup.register_cleanup(
            "resources session", lambda: self._close_resources(request, handles)
        )
        for name in handles.tool_sessions:
            cleanup.register_cleanup(
                f"{name} tool session", lambda name=name: self._close_tool_session(request, handles, name)
            )
        closed_agent: list[AgentCloseSessionResponse] = []

        async def close_agent() -> None:
            closed_agent.append(await self._close_agent(request, handles))

        # Register the agent cleanup before seeding the agent,
        # so cancellation can close a remotely created session even if its reply is lost;
        # a continuation past seeding registers it at once.
        agent_cleanup = (
            cleanup.register_cleanup("agent session", close_agent)
            if stage in ("invoke_agent", "close_agent")
            else None
        )

        if stage == "seed":
            await self.checkpoint_boundary(request, {"next": "seed", "handles": handles.model_dump()})
            # Seeds are idempotent for a caller-assigned session ID, so a checkpoint need not wait for them.
            async with self.checkpoint_step(request, "replay"):
                agent_cleanup = await self.seed_episode_sessions(
                    request,
                    handles,
                    register_agent_cleanup=lambda: cleanup.register_cleanup("agent session", close_agent),
                )
            stage = "invoke_agent"
        elif continuation and stage in ("invoke_agent", "close_agent"):
            # Restored before the agent's work ended: the sandbox the agent borrowed was rebuilt under a new id.
            # Re-seeding is idempotent, so a checkpoint need not wait for it, and a crash during it re-seeds again.
            async with self.checkpoint_step(request, "replay"):
                await self.reseed_after_restore(request, handles)

        if stage == "invoke_agent":
            await self.checkpoint_boundary(request, {"next": "invoke_agent", "handles": handles.model_dump()})
            # The agent parks at its own boundaries and continues from them after a restore.
            try:
                async with self.checkpoint_step(request, "replay"):
                    agent_response = await self._invoke_agent(request, handles)
            except NgFailureEvent as failure:
                if not is_handled_episode_failure(failure):
                    raise
                # The agent's evidence of the failed turn arrives with its session close.
                # The close runs after the failed step, which parks until resume while a checkpoint is open.
                try:
                    await agent_cleanup.close()
                except Exception:
                    pass  # Final unwind retries the close and logs its failure.
                if closed_agent:
                    failure.episode_evidence["agent_observations"] = closed_agent[-1].agent_observations
                    failure.episode_evidence["trajectory"] = closed_agent[-1].trajectory
                raise
            stage = "close_agent"

        if stage == "close_agent":
            # Park before the close, never inside it:
            # a checkpoint that counted this episode before the activation never finds it inside a wait step,
            # and a replacement restored here closes nothing before resume.
            await self.checkpoint_boundary(
                request,
                lambda: {
                    "next": "close_agent",
                    "handles": handles.model_dump(),
                    "response": agent_response.model_dump(mode="json"),
                },
            )
            # Closing returns the agent's observations and final resources cookies exactly once,
            # so a checkpoint waits for it and records them.
            async with self.checkpoint_step(request, "wait"):
                try:
                    await agent_cleanup.close()
                    close = closed_agent[-1]
                except Exception as error:
                    raise dependency_failure(
                        error,
                        stage="cleanup",
                        partial_response=agent_response,
                    ) from error
            observations = close.agent_observations
            trajectory = close.trajectory
            stage = "verify"

        def dumped_observations() -> dict | None:
            return observations.model_dump(mode="json") if observations else None

        def dumped_trajectory() -> dict | None:
            return trajectory.model_dump(mode="json") if trajectory else None

        if stage == "verify":
            action = agent_outcome_policy(agent_response, config=self.config.agent_outcome_policy)
            cause = self._agent_failure(agent_response)
            if agent_response.status == "failed" and AGENT_OUTCOME_METADATA_KEY in (agent_response.metadata or {}):
                extra = agent_response.model_extra or {}
                report = extra.get("failure_report")
                # A report that selects nothing (e.g. a provider 400 without a Gym failure) keeps the policy action.
                unselected = (
                    isinstance(report, dict) and report.get("failure_id") is None and not report.get("omitted")
                )
                if cause is None and "failure_report" in extra and not unselected:
                    action = OutcomeAction.FAIL
                elif cause is not None and cause.failure_kind == MODEL_POLICY_RESPONSE_TIMEOUT:
                    action = self._timeout_action(action)
                elif cause is not None and cause.failure_kind == MODEL_REQUEST_ERROR and cause.http_status is None:
                    # An enclosing HTTP 500 is not a provider verdict or evidence of an unexecuted request.
                    action = action if self._model_request_not_sent(agent_response) else OutcomeAction.FAIL
            if action == OutcomeAction.COUNT_AS_ZERO:
                # Explicit policy completion is not a verifier execution.
                report = FailureReport.model_validate(agent_response.model_extra["failure_report"])
                verification = SingleAgentTurnResult.model_validate(
                    self._verify_body(request, agent_response)
                    | {
                        "reward": 0.0,
                        "_ng_failure_class": cause.failure_kind,
                        "failure_kind": cause.failure_kind,
                        "_ng_failure_terminal": True,
                        "_ng_failure_counts_as_zero": True,
                        **FailureReportRow.fields(report),
                    }
                )
                stage = "return"
            elif action != OutcomeAction.VERIFY:
                raise self._failure(
                    stage="agent",
                    failure_reason=agent_response.error.message if agent_response.error else "Agent reported failure",
                    terminal=action == OutcomeAction.FAIL,
                    failure_kind=cause.failure_kind if cause else None,
                    partial_response=agent_response,
                    agent_observations=observations,
                    trajectory=trajectory,
                ) from cause
        if stage == "verify":
            await self.checkpoint_boundary(
                request,
                lambda: {
                    "next": "verify",
                    "handles": handles.model_dump(),
                    "response": agent_response.model_dump(mode="json"),
                    "observations": dumped_observations(),
                    "trajectory": dumped_trajectory(),
                },
            )
            skipped_reward = self._skipped_verification_reward()
            if skipped_reward is None:
                async with self.checkpoint_step(request, handles.resources_verify_step_type):
                    verification = await self._verify(request, handles, agent_response, observations, trajectory)
            else:
                # The flag marks the fixed reward, so anything reading the result can tell it from a measured one.
                verification = SingleAgentTurnResult.model_validate(
                    self._verify_body(request, agent_response)
                    | {"reward": skipped_reward, "verification_skipped": True}
                )
            stage = "return"

        # Every path ends here, at the return stage.
        # Record the result so a restore never runs a state-changing verification twice,
        # and park before the final cleanup, so a replacement restored here does not end before resume.
        await self.checkpoint_boundary(
            request,
            lambda: {
                "next": "return",
                "handles": handles.model_dump(),
                "result": verification.model_dump(mode="json"),
                "observations": dumped_observations(),
                "trajectory": dumped_trajectory(),
            },
        )

        # Keep one bounded retry in final unwind without erasing a completed verdict.
        cleanup.register_cleanup("post-verification resources session", resources_cleanup.close)
        final_zero_report = (
            FailureReportRow.report(verification.model_dump(mode="json"))
            if (verification.model_extra or {}).get("_ng_failure_counts_as_zero") is True
            else None
        )
        return SingleAgentTurnResponse(
            episode_id=request.episode_id,
            failure_report=final_zero_report,
            task_id=request.task.task_id,
            result=verification.model_copy(
                update={NG_AGENT_OBSERVATIONS_KEY: observations, NG_TRAJECTORY_KEY: trajectory}
            ),
        )

    async def seed_episode_sessions(
        self,
        request: SingleAgentTurnRequest,
        handles: "SessionHandles",
        *,
        register_agent_cleanup: Callable[[], CleanupHandle] | None = None,
    ) -> CleanupHandle | None:
        """Seed the episode's resources and tool server sessions, then its agent session with what they lent.

        ``register_agent_cleanup`` runs before the agent seed, so a lost agent seed reply is still closed;
        its result is returned.
        An episode whose agent holds resources MCP grants restarts after a crash
        unless the agent borrows a sandbox the resources seed lent and re-binds them;
        one that grants a tool server's MCP tools always does.
        """
        seed = await self._seed_resources(request, handles)
        tool_accesses = await self._seed_tool_servers(request, handles)
        registered = register_agent_cleanup() if register_agent_cleanup is not None else None
        await self._seed_agent(request, handles, seed, tool_accesses)
        return registered

    async def reseed_after_restore(self, request: SingleAgentTurnRequest, handles: "SessionHandles") -> None:
        """Seed both sessions again for a replacement attempt that continues a checkpoint before the agent closed.

        The resources server returns the sandbox its restore forked, without setting the task up again;
        the agent re-binds its restored session to that sandbox and to this attempt's tool grants.
        Without a borrowed sandbox nothing changed hands, and nothing is seeded.
        """
        if not handles.borrowed_sandbox:
            return
        LOGGER.info(
            "Re-seeding the sessions of %s after a restore: the agent's sandbox was rebuilt",
            request.episode_id.capture_key,
        )
        await self.seed_episode_sessions(request, handles)

    async def _seed_resources(
        self, request: SingleAgentTurnRequest, handles: "SessionHandles"
    ) -> ResourcesSeedSessionResponse:
        task_input = request.task.task_input
        try:
            seed_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=ResourcesSeedSessionRequest(
                    resources_session_id=handles.resources_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    task_data=task_input.task_data,
                    # As _verify_body sends them: a server whose session is built from the task's history or tools
                    # reads them at seed. Tool servers' seeds do not get them.
                    responses_create_params=task_input.responses_create_params.model_dump(
                        mode="json", exclude_unset=True
                    ),
                ).model_dump(mode="json"),
            )
            await raise_boundary_failure(seed_http_response)
            handles.resources_cookies = _cookies(seed_http_response)
            handles.resources_verify_step_type = seed_verify_step_type(seed_http_response.headers)
            if seed_restarts(seed_http_response.headers):
                await self.checkpoint_restart(request)
            if not handles.resources_cookies:
                raise ValueError("Resources seed did not establish a session cookie")
            seed = ResourcesSeedSessionResponse.model_validate(await get_response_json(seed_http_response))
            if seed.resources_session_id != handles.resources_session_id:
                raise ValueError("Resources seed returned a different resources_session_id")
            handles.borrowed_sandbox = seed.sandbox_access is not None
        except Exception as error:
            raise dependency_failure(
                error,
                stage="seed",
            ) from error
        handles.observation = seed.observation
        return seed

    async def _seed_tool_servers(self, request: SingleAgentTurnRequest, handles: "SessionHandles") -> list[ToolAccess]:
        """Seed the episode's session on each tool server and grant the agent its MCP tools."""
        task_input = request.task.task_input
        tool_accesses: list[ToolAccess] = []
        for name, session in handles.tool_sessions.items():
            try:
                seed_http_response = await self.server_client.post(
                    server_name=name,
                    url_path="/seed_session",
                    json=ResourcesSeedSessionRequest(
                        resources_session_id=session.session_id,
                        episode_id=request.episode_id,
                        task_id=request.task.task_id,
                        task_data=task_input.task_data,
                    ).model_dump(mode="json"),
                )
                await raise_boundary_failure(seed_http_response)
                session.cookies = _cookies(seed_http_response)
                seed = ResourcesSeedSessionResponse.model_validate(await get_response_json(seed_http_response))
                if seed.resources_session_id != session.session_id:
                    raise ValueError(f"Tool server {name!r} seed returned a different resources_session_id")
            except Exception as error:
                raise dependency_failure(
                    error,
                    stage="seed",
                ) from error
            tool_accesses.append(self._mcp_tool_access(name, seed))
        if tool_accesses:
            # As for the Resources Server's own MCP grant: each token names this attempt.
            await self.checkpoint_restart(request)
        return tool_accesses

    def _mcp_tool_access(self, server_name: str, seed: ResourcesSeedSessionResponse) -> MCPToolAccess:
        """The required MCP grant for a seeded session on `server_name`."""
        if seed.resources_tools is None:
            raise self._failure(
                stage="seed",
                failure_reason=f"Seed of {server_name!r} did not return requested MCP metadata",
                terminal=True,
            )
        if seed.resources_tools.transport != "http":
            raise self._failure(
                stage="seed",
                failure_reason=f"Unsupported MCP transport of {server_name!r}: {seed.resources_tools.transport}",
                terminal=True,
            )
        base_url = self.server_client._resolve_base_url(server_name).rstrip("/")
        return MCPToolAccess(
            name=seed.resources_tools.server_name,
            required=True,
            connection=MCPStreamableHTTPConnection(
                url=f"{base_url}/{seed.resources_tools.url_path.lstrip('/')}",
                headers=seed.resources_tools.headers,
            ),
        )

    async def _seed_agent(
        self,
        request: SingleAgentTurnRequest,
        handles: "SessionHandles",
        seed: ResourcesSeedSessionResponse,
        tool_server_accesses: Sequence[ToolAccess] = (),
    ) -> None:
        resources_base_url = self.server_client._resolve_base_url(self.config.resources_server.name).rstrip("/")
        tool_accesses: list[ToolAccess] = []
        if "direct_http" in self.config.resources_tool_transports:
            # With checkpointing on, the Resources Server refuses a tool call that does not name its attempt,
            # so the grant carries this attempt's rollout prefix.
            # An agent that continues the session after a restore sends its calls under the replacement's prefix.
            tools_base_url = (
                f"{resources_base_url}{rollout_path_prefix(request.episode_id.capture_key)}"
                if self._checkpoint is not None
                else resources_base_url
            )
            tool_accesses.append(
                DirectHTTPToolAccess(
                    name=f"{self.config.resources_server.name}.direct_http",
                    required=True,
                    base_url=tools_base_url,
                    cookies=handles.resources_cookies,
                )
            )
        if "mcp" in self.config.resources_tool_transports:
            tool_accesses.append(self._mcp_tool_access(self.config.resources_server.name, seed))
            if seed.sandbox_access is None:
                # The MCP session token names this attempt, and only a restore's re-seed hands the agent the token
                # of the attempt that replaces it. A restore re-seeds only an episode whose agent borrows a sandbox,
                # so without one the agent would keep a token the Resources Server refuses, whatever it declares:
                # the episode restarts from its input after a crash.
                # Marked before the agent seed, so no checkpoint exports the agent session it creates.
                await self.checkpoint_restart(request)
        tool_accesses.extend(tool_server_accesses)

        try:
            agent_create_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions",
                json=AgentSeedSessionRequest(
                    agent_session_id=handles.agent_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    tool_accesses=tool_accesses,
                    sandbox_access=seed.sandbox_access,
                    agent_context=request.task.task_input.agent_seed_context,
                ).model_dump(mode="json"),
            )
            await raise_boundary_failure(agent_create_http_response)
            agent_session = AgentSeedSessionResponse.model_validate(
                await get_response_json(agent_create_http_response)
            )
            if agent_session.agent_session_id != handles.agent_session_id:
                raise ValueError("Agent seed returned a different agent_session_id")
            handles.agent_cookies = _cookies(agent_create_http_response)
            # The agent cannot capture its session, or holds an MCP session token it cannot exchange for the one a
            # restore's re-seed carries: the Resources Server refuses the token once a restore replaces this attempt,
            # so the episode restarts from its input after a crash.
            # Such an agent reports the restart on its own seed reply, so no checkpoint exports its session;
            # the MCP check covers an agent whose own check misses that case.
            if seed_restarts(agent_create_http_response.headers) or (
                "mcp" in self.config.resources_tool_transports
                and not seed_rebinds_tool_accesses(agent_create_http_response.headers)
            ):
                await self.checkpoint_restart(request)
        except Exception as error:
            raise dependency_failure(
                error,
                stage="agent",
            ) from error

    async def _invoke_agent(self, request: SingleAgentTurnRequest, handles: "SessionHandles") -> NeMoGymResponse:
        params = request.task.task_input.responses_create_params
        if handles.observation:
            input_items = params.input
            if isinstance(input_items, str):
                input_items = [NeMoGymEasyInputMessage(role="user", content=input_items)] if input_items else []
            params = params.model_copy(
                update={"input": [*input_items, NeMoGymEasyInputMessage(role="user", content=handles.observation)]}
            )
        if self.config.max_episode_steps is not None:
            return await self._run_steps(request, handles, params)
        # The episode's only activation.
        # A restore before its reply invokes it again, and the agent gives the reply it already gave.
        return await self._activate_agent(request, handles, params, activation=1)

    async def _run_steps(
        self, request: SingleAgentTurnRequest, handles: "SessionHandles", params: Any
    ) -> NeMoGymResponse:
        """Drive the agent one step per activation, up to ``max_episode_steps``.

        Each activation continues the conversation with every step before it. The episode ends after the first step
        in which the model called no tool or that did not complete, as the agent's own loop ends, and with
        ``end_on_task_state`` once the resources session reports that its task ended. The reply carries every step's
        output and their summed usage, as one activation's would; each step's failure report is saved as it arrives.
        """
        limit = self.config.max_episode_steps
        conversation = (
            ([NeMoGymEasyInputMessage(role="user", content=params.input)] if params.input else [])
            if isinstance(params.input, str)
            else list(params.input)
        )
        outputs: list[Any] = []
        usage = None
        for step in range(1, limit + 1):
            response = await self._activate_agent(
                request,
                handles,
                params.model_copy(update={"input": [*conversation, *outputs]}),
                activation=step,
                # The agent then counts its one step as completed, and keeps one record of the episode's invocation.
                headers={AGENT_EPISODE_STEP_LIMIT_HEADER: str(limit)},
            )
            outputs.extend(response.output)
            usage = accumulate_response_usage(usage, response.usage)
            # A step that did not complete ends the episode, as a step without a tool call does. A reply may omit its
            # status (a model server may send none, and the simple agent passes that on), so only a status other
            # than completed, or truncation, counts as not completing.
            if (
                response.status not in (None, "completed")
                or response.incomplete_details is not None
                or not any(item.type == "function_call" for item in response.output)
            ):
                break
            if self.config.end_on_task_state and await self._task_ended(handles):
                break
        return response.model_copy(update={"output": outputs, "usage": usage})

    async def _task_ended(self, handles: "SessionHandles") -> bool:
        """Whether the episode's task has ended, as its resources session reports it (``RESOURCES_TASK_STATE_PATH``)."""
        try:
            state_http_response = await self.server_client.get(
                server_name=self.config.resources_server.name,
                url_path=RESOURCES_TASK_STATE_PATH,
                cookies=handles.resources_cookies,
            )
            await raise_boundary_failure(state_http_response)
            return ResourcesTaskState.model_validate(await get_response_json(state_http_response)).task_ended
        except Exception as error:
            # Read during the agent's turn, between its steps.
            raise dependency_failure(error, stage="agent") from error

    async def _activate_agent(
        self,
        request: SingleAgentTurnRequest,
        handles: "SessionHandles",
        params: Any,
        *,
        activation: int,
        headers: dict[str, str] | None = None,
    ) -> NeMoGymResponse:
        """One activation of the agent session, number ``activation``; the failure report its reply carries is saved."""
        try:
            agent_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path=self._agent_responses_path(request),
                json=params,
                cookies=handles.agent_cookies,
                headers={AGENT_ACTIVATION_HEADER: str(activation), **(headers or {})},
            )
            await raise_boundary_failure(agent_http_response)
            response_cookies = _cookies(agent_http_response)
            if response_cookies:
                handles.agent_cookies = response_cookies
            response = NeMoGymResponse.model_validate(await get_response_json(agent_http_response))
            # Capture body-carried observations before close or outcome policy can bypass verification.
            # They are diagnostics, not a selection of the episode's terminal failure.
            saved = (response.model_extra or {}).get("failure_report")
            log = RequestFailureLog.current()
            if saved is not None and log is not None:
                log.add_saved(saved)
            return response
        except Exception as error:
            raise dependency_failure(
                error,
                stage="agent",
            ) from error

    async def _close_agent(
        self, request: SingleAgentTurnRequest, handles: "SessionHandles"
    ) -> AgentCloseSessionResponse:
        close_http_response = await self.server_client.post(
            server_name=self.config.agent_server.name,
            url_path="/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(
                agent_session_id=handles.agent_session_id,
                episode_id=request.episode_id,
            ).model_dump(mode="json"),
            cookies=handles.agent_cookies,
        )
        await raise_boundary_failure(close_http_response)
        close = AgentCloseSessionResponse.model_validate(await get_response_json(close_http_response))
        if close.agent_session_id != handles.agent_session_id:
            raise ValueError("Agent close returned a different agent_session_id")
        # Verify identifies the episode by the Resources session cookie.
        # An agent that held no Resources cookie jar returns None or an empty jar.
        # Neither may replace the cookie from the Resources seed.
        if close.resources_cookies:
            handles.resources_cookies = close.resources_cookies
        return close

    async def _close_resources(self, request: SingleAgentTurnRequest, handles: "SessionHandles") -> None:
        await self._close_session(
            request, self.config.resources_server.name, handles.resources_session_id, handles.resources_cookies
        )

    async def _close_tool_session(self, request: SingleAgentTurnRequest, handles: "SessionHandles", name: str) -> None:
        session = handles.tool_sessions[name]
        await self._close_session(request, name, session.session_id, session.cookies)

    async def _close_session(
        self, request: SingleAgentTurnRequest, server_name: str, session_id: str, cookies: dict[str, str]
    ) -> None:
        close_response = await self.server_client.post(
            server_name=server_name,
            url_path="/close_session",
            json=ResourcesCloseSessionRequest(
                resources_session_id=session_id,
                episode_id=request.episode_id,
            ).model_dump(mode="json"),
            cookies=cookies,
        )
        await raise_boundary_failure(close_response)

    async def _verify(
        self,
        request: SingleAgentTurnRequest,
        handles: "SessionHandles",
        agent_response: NeMoGymResponse,
        observations: AgentObservationBundle | None,
        trajectory: TrajectoryRecord | None = None,
    ) -> SingleAgentTurnResult:
        verifier_invocations = None
        try:
            verify_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                # The session cookie identifies the episode.
                json=self._verify_body(request, agent_response)
                | verification_failure_fields(agent_response)
                | (
                    {"ng_trajectory": trajectory.model_dump(mode="json")}
                    if self.config.forward_agent_evidence and trajectory is not None
                    else {}
                )
                | (
                    {"ng_agent_observations": observations.model_dump(mode="json")}
                    if self.config.forward_agent_evidence and observations is not None
                    else {}
                ),
                cookies=handles.resources_cookies,
                # Final cleanup closes the session, so the Resources Server keeps tracking it until then.
                headers={CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"},
            )
            # Read returned evidence before raising on an HTTP failure. A non-JSON
            # error body still follows the original HTTP-error handling path.
            try:
                payload = await get_response_json(verify_http_response)
                if isinstance(payload, dict) and payload.get(NG_VERIFIER_INVOCATIONS_KEY):
                    verifier_invocations = [
                        VerifierInvocation.model_validate(value) for value in payload[NG_VERIFIER_INVOCATIONS_KEY]
                    ]
            except (ValueError, TypeError):
                await raise_boundary_failure(verify_http_response, content=await verify_http_response.read())
                raise
            await raise_boundary_failure(verify_http_response, content=await verify_http_response.read())
            return SingleAgentTurnResult.model_validate(payload)
        except Exception as error:
            raise dependency_failure(
                error,
                stage="verification",
                partial_response=agent_response,
                agent_observations=observations,
                trajectory=trajectory,
                verifier_invocations=verifier_invocations,
            ) from error

    @staticmethod
    def _verify_body(request: SingleAgentTurnRequest, agent_response: NeMoGymResponse) -> dict[str, Any]:
        """The Resources Server's own flat verify body, as an Agent's /run sends it."""
        task_input = request.task.task_input
        return task_input.task_data | {
            "responses_create_params": task_input.responses_create_params.model_dump(mode="json", exclude_unset=True),
            "response": agent_response.model_dump(mode="json"),
        }

    @staticmethod
    def _agent_failure(response: NeMoGymResponse) -> NgFailureEvent | None:
        """Rebuild only the cause selected by this failed reply and matching its outcome."""
        if response.status != "failed":
            return None
        try:
            outcome = AgentOutcome.model_validate_json((response.metadata or {}).get(AGENT_OUTCOME_METADATA_KEY, ""))
            report = FailureReport.model_validate((response.model_extra or {}).get("failure_report"))
        except (ValidationError, TypeError):
            return None
        if report.omitted or report.failure_id is None:
            return None
        selected = report.events[report.failure_id]
        if selected.failure_kind != outcome.failure_kind:
            return None
        if selected.http_status is not None and selected.http_status != outcome.status_code:
            return None
        if selected.failure_kind == MODEL_POLICY_RESPONSE_TIMEOUT:
            if (
                selected.boundary != FailureBoundary.AGENT
                or selected.operation != "agent.policy_request"
                or not selected.operation_id
                or selected.cause_id is None
                or report.events[selected.cause_id].operation_id != selected.operation_id
                or not confirmed_policy_response_timeout(report.model_copy(update={"failure_id": selected.cause_id}))
            ):
                return None
        return rebuild_failure(report)

    @staticmethod
    def _model_request_not_sent(response: NeMoGymResponse) -> bool:
        """A selected connector failure is safe only if every attempt of that operation never connected.

        A connection that was refused and one that timed out connecting were both never made.
        """
        report = FailureReport.model_validate(response.model_extra["failure_report"])
        selected = report.events[report.failure_id]
        if (
            selected.boundary != FailureBoundary.MODEL
            or selected.operation != "model_request"
            or not selected.operation_id
            or not selected.attempt_id
        ):
            return False
        observed = [
            event
            for event in report.events.values()
            if event.boundary == FailureBoundary.MODEL and event.operation == "model_request"
        ]
        if any(not event.operation_id or not event.attempt_id for event in observed):
            return False
        attempts = [event for event in observed if event.operation_id == selected.operation_id]
        unconnected_types = {
            f"{error_type.__module__}.{error_type.__qualname__}"
            for error_type in (ClientConnectorError, ConnectionTimeoutError)
        }
        return len({event.attempt_id for event in attempts}) == len(attempts) and all(
            isinstance(event.details, ModelFailureDetails)
            and event.details.request_upload_complete is False
            and event.foreign_error is not None
            and event.foreign_error.exception_type in unconnected_types
            for event in attempts
        )

    def _timeout_action(self, fallback: OutcomeAction) -> OutcomeAction:
        """Keep the migrated agent's explicit timeout rule unless the environment overrides policy."""
        policy = self.config.agent_outcome_policy
        if MODEL_POLICY_RESPONSE_TIMEOUT in policy.by_failure_kind or "default_action" in policy.model_fields_set:
            return fallback
        agent = get_first_server_config_dict(self.server_client.global_config_dict, self.config.agent_server.name)
        rules = (agent.get("outcome_policy") or {}).get("by_failure_kind") or {}
        if MODEL_POLICY_RESPONSE_TIMEOUT in rules:
            return OutcomeAction(rules[MODEL_POLICY_RESPONSE_TIMEOUT])
        if "final_on_model_timeout" in agent:
            return (
                OutcomeAction.COUNT_AS_ZERO
                if TypeAdapter(bool).validate_python(agent["final_on_model_timeout"])
                else OutcomeAction.RETRY
            )
        # The migrated SimpleAgent /run boundary defaults confirmed timeouts to retry.
        return OutcomeAction((agent.get("outcome_policy") or {}).get("default_action", OutcomeAction.RETRY))

    def _skipped_verification_reward(self) -> float | None:
        """The fixed reward that replaces /verify, or None to verify.

        This server's own setting wins; otherwise the bound agent's applies.
        The agent's raw values are parsed as the agent's own config parses them, so a string "false" stays false.
        """
        agent_config = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.agent_server.name,
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

    def _agent_responses_path(self, request: SingleAgentTurnRequest) -> str:
        block = self.server_client.global_config_dict.get(TOKEN_CAPTURE_BLOCK) or {}
        agent_config = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.agent_server.name,
        )
        # A request whose rollout skips capture (a validation rollout) activates the agent without the tag.
        token_capture = (
            not request.skip_token_capture
            and bool(block.get("enabled", False))
            and (bool(block.get("all_agents", False)) or bool(agent_config.get("token_id_capture", False)))
        )
        capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
        # The agent is the one the training framework trains, so its model calls carry the episode's values.
        sampling_segment = model_sampling_path_segment(self.model_sampling_segment(request, trained=True))
        return f"/ng-rollout/{request.episode_id.capture_key}{capture_segment}{sampling_segment}/v1/responses"

    @staticmethod
    def _failure(
        *,
        stage: str,
        failure_reason: str,
        terminal: bool,
        failure_kind: str | None = None,
        partial_response: Any = None,
        agent_observations: Any = None,
        trajectory: Any = None,
        verifier_invocations: list[VerifierInvocation] | None = None,
    ) -> NgFailureEvent:
        # The harness's evidence of a failed turn (its observations and the decisions it took before the failure)
        # travels with the failure row, as it travels with a result.
        return handled_episode_failure(
            boundary=FailureBoundary.ENVIRONMENT,
            stage=stage,
            operation="episode",
            failure_reason=failure_reason[:2000],
            terminal=terminal,
            failure_kind=failure_kind,
            evidence={
                "partial_response": partial_response,
                "agent_observations": agent_observations,
                "trajectory": trajectory,
                "verifier_invocations": verifier_invocations,
            },
        )


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


if __name__ == "__main__":
    SingleAgentTurnEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SingleAgentTurnEnvironmentServer.run_webserver()  # noqa: F401
