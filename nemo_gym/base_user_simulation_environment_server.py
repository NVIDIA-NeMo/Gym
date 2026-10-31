# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared turn loop for environment servers that simulate the user.

The environment writes every user message. The agent under test answers each one in a turn: one activation of its
agent session, in which the agent runs its own tool loop with the episode's tools. A subclass supplies the user side
and the episode's tools through its hooks; the base class owns the agent session, the turn loop, the deadlines and
the failure stages, and serves the tools (``BaseEnvironmentServerWithTools``).
"""

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from nemo_gym.agent_activation_types import AgentContinuationRequirements
from nemo_gym.agent_outcome import AgentOutcomePolicyConfig, OutcomeAction, agent_outcome_policy
from nemo_gym.base_environment_server import (
    CleanupContext,
    EpisodeRequestT,
    EpisodeResponseT,
    HandledEpisodeError,
    dependency_failure,
)
from nemo_gym.base_environment_server_with_tools import (
    BaseEnvironmentServerWithTools,
    BaseEnvironmentServerWithToolsConfig,
)
from nemo_gym.base_responses_api_agent import (
    AGENT_ACTIVATION_HEADER,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.config_types import TOKEN_CAPTURE_PATH_SEGMENT, AgentServerRef
from nemo_gym.episode_tools import EpisodeToolset, ToolCallRecord
from nemo_gym.episode_types import EpisodeFailure
from nemo_gym.failure_kinds import FailureStage
from nemo_gym.global_config import TOKEN_ID_CAPTURE_BLOCK, get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseInput
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.tool_access import ToolAccess


EpisodeT = TypeVar("EpisodeT")


class UserSimulationEnvironmentServerConfig(BaseEnvironmentServerWithToolsConfig):
    """Bind the agent under test and bound its conversation."""

    agent_server: AgentServerRef
    max_turns: PositiveInt = Field(
        description="Most agent turns in one episode. Above 1, the agent must resume its session across turns."
    )
    turn_timeout_seconds: PositiveFloat
    interaction_timeout_seconds: PositiveFloat | None = Field(
        default=None,
        description="Bound on all turns together. When it expires, the agent is closed and the episode is graded.",
    )
    agent_outcome_policy: AgentOutcomePolicyConfig = Field(default_factory=AgentOutcomePolicyConfig)


class UserMessage(BaseModel):
    """The next user message, which is the whole input of the agent's next turn."""

    model_config = ConfigDict(extra="forbid")

    content: str


class Stop(BaseModel):
    """End the conversation. ``reason`` becomes the episode's ``stop_reason``."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)


@dataclass(frozen=True)
class AgentAccess:
    """Tools and a sandbox the agent gets at seed, beyond the episode's own tools (``episode_tools``)."""

    tool_accesses: list[ToolAccess] = field(default_factory=list)
    sandbox_access: SandboxAccess | None = None


class TurnRecord(BaseModel):
    """One agent turn: the input the agent received and the response it returned."""

    model_config = ConfigDict(extra="forbid")

    turn: PositiveInt
    input: Union[str, NeMoGymResponseInput]
    response: NeMoGymResponse
    duration_seconds: float


class UserSimulationResult(BaseModel):
    """The fields every user-simulation result has. Each server's result model extends it."""

    model_config = ConfigDict(extra="forbid")

    reward: float
    turns: list[TurnRecord]
    stop_reason: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    ng_agent_observations: AgentObservationBundle | None = None
    ng_trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)


class UserSimulationFailure(EpisodeFailure):
    """A failed episode, with the turns and tool calls completed before the failure and the agent's evidence."""

    turns: list[TurnRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    agent_observations: AgentObservationBundle | None = None
    trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)


@dataclass
class Transcript:
    """What ``finish_episode`` grades: the turns, why they stopped, the agent's close and the episode's tool calls."""

    turns: list[TurnRecord]
    stop_reason: str
    agent_close: AgentCloseSessionResponse | None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    def result_fields(self) -> dict[str, Any]:
        """The ``UserSimulationResult`` fields other than ``reward``."""
        return {
            "turns": self.turns,
            "stop_reason": self.stop_reason,
            "tool_calls": self.tool_calls,
            "ng_agent_observations": self.agent_close.agent_observations if self.agent_close else None,
            "ng_trajectory": self.agent_close.trajectory if self.agent_close else None,
        }


@dataclass
class _AgentSession:
    agent_session_id: str = field(default_factory=lambda: f"agent-session-{uuid4().hex}")
    cookies: dict[str, str] = field(default_factory=dict)
    close: AgentCloseSessionResponse | None = None


class _InteractionDeadlineExpired(Exception):
    pass


class BaseUserSimulationEnvironmentServer(
    BaseEnvironmentServerWithTools[EpisodeRequestT, EpisodeResponseT],
    Generic[EpisodeRequestT, EpisodeResponseT, EpisodeT],
):
    """Run a conversation in which the environment writes every user message.

    ``run_episode`` calls the hooks in this order: ``start_episode``, ``episode_tools``, ``agent_access``, the agent
    seed, ``responses_create_params``, then one ``next_user_turn`` after each agent turn until it returns ``Stop`` or
    ``max_turns`` is reached, the agent close, and ``finish_episode``. The server serves the episode's tools over
    each of ``tool_transports`` until the episode ends, and the agent's seed grants them. Each activation sends only its new input, so with ``max_turns`` above 1
    the seed requires an agent that resumes its session; any other agent fails the episode at seed.

    Episodes are restarts after a crash (``checkpoint_mode = "restart_only"``).
    Failures in calls to other servers are mapped with ``dependency_failure`` to the stages ``seed``, ``agent``,
    ``cleanup`` and ``verification``. A hook that calls another server raises ``self.failure(error, stage=...)``.
    """

    config: UserSimulationEnvironmentServerConfig

    @abstractmethod
    async def start_episode(self, request: EpisodeRequestT, cleanup: CleanupContext) -> EpisodeT:
        """Create the episode state, such as a Resources session or a sandbox. Register each cleanup in ``cleanup``."""

    async def episode_tools(self, episode: EpisodeT) -> EpisodeToolset | None:
        """Return the tools this server serves the agent for the episode, or ``None``. The default serves none.

        The tools act on the episode's state. Over direct HTTP, the agent calls only the tools that
        ``responses_create_params`` declares; over MCP, it lists them from the server.
        """
        return None

    async def agent_access(self, episode: EpisodeT) -> AgentAccess:
        """Return other tools and a sandbox the agent gets at seed. The default gives neither."""
        return AgentAccess()

    @abstractmethod
    async def responses_create_params(self, episode: EpisodeT) -> NeMoGymResponseCreateParamsNonStreaming:
        """Return the first turn's request: the opening messages, and the tools and settings the session keeps.

        It is the same object as a dataset row's ``responses_create_params``. Later turns send only a ``UserMessage``.
        """

    @abstractmethod
    async def next_user_turn(self, episode: EpisodeT, turn: TurnRecord) -> UserMessage | Stop:
        """Read the agent's turn and return the next user message, or ``Stop``.

        This hook is the user side. It can be a simulator, a fixed protocol or a grader. It is called after every
        turn, the last one included, so it can grade a final answer before ``max_turns`` ends the conversation.
        """

    @abstractmethod
    async def finish_episode(self, episode: EpisodeT, transcript: Transcript) -> EpisodeResponseT:
        """Grade the episode after the agent is closed, and return the response."""

    def failure(self, error: BaseException, *, stage: FailureStage, **fields: Any) -> HandledEpisodeError:
        """Map a failed call to another server to this protocol's failure (see ``dependency_failure``)."""
        return dependency_failure(error, failure_model=UserSimulationFailure, stage=stage, **fields)

    def agent_responses_path(self, request: EpisodeRequestT) -> str:
        """The agent's ``/v1/responses`` path under this episode's capture route."""
        global_config = self.server_client.global_config_dict
        block = global_config.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        agent_config = get_first_server_config_dict(global_config, self.config.agent_server.name)
        capture = bool(block.get("enabled")) and (block.get("all_agents") or agent_config.get("token_id_capture"))
        segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if capture else ""
        return f"/ng-rollout/{request.episode_id.capture_key}{segment}/v1/responses"

    async def run_episode(self, request: EpisodeRequestT, cleanup: CleanupContext) -> EpisodeResponseT:
        episode = await self.start_episode(request, cleanup)
        toolset = await self.episode_tools(episode)
        served = await self.serve_tools(request, toolset, cleanup) if toolset is not None else None
        tool_calls = served.calls if served is not None else []
        access = await self.agent_access(episode)
        tool_accesses = [*(served.tool_accesses if served is not None else []), *access.tool_accesses]
        session = _AgentSession()
        # Registered before the seed, so a cancellation closes a session whose seed reply was lost.
        agent_cleanup = cleanup.register_cleanup("agent session", lambda: self._close_agent(request, session))
        turns: list[TurnRecord] = []
        try:
            await self._seed_agent(request, session, tool_accesses, access.sandbox_access)
            stop_reason = await self._converse(request, episode, session, turns)
            # The agent stops before grading, so verification reads the state the agent left.
            try:
                await agent_cleanup.close()
            except Exception as error:
                raise self.failure(error, stage="cleanup") from error
            return await self.finish_episode(episode, Transcript(turns, stop_reason, session.close, list(tool_calls)))
        except HandledEpisodeError as error:
            # Every failure after the seed keeps the completed turns and the agent's evidence.
            if isinstance(error.failure, UserSimulationFailure):
                try:
                    await agent_cleanup.close()  # A no-op when the agent is already closed.
                except Exception:
                    pass  # Final cleanup retries the close and logs its failure.
                error.failure.turns = turns
                error.failure.tool_calls = list(tool_calls)
                if session.close is not None:
                    error.failure.agent_observations = session.close.agent_observations
                    error.failure.trajectory = session.close.trajectory
            raise

    async def _converse(
        self, request: EpisodeRequestT, episode: EpisodeT, session: _AgentSession, turns: list[TurnRecord]
    ) -> str:
        """Run turns until ``next_user_turn`` stops them or a limit does; return the stop reason."""
        loop = asyncio.get_running_loop()
        interaction_deadline = (
            loop.time() + self.config.interaction_timeout_seconds if self.config.interaction_timeout_seconds else None
        )
        params = await self.responses_create_params(episode)
        for turn in range(1, self.config.max_turns + 1):
            if interaction_deadline is not None and loop.time() >= interaction_deadline:
                return "interaction_timeout"
            try:
                record = await self._activate(request, session, params, turn, interaction_deadline)
            except _InteractionDeadlineExpired:
                return "interaction_timeout"
            turns.append(record)
            action = agent_outcome_policy(record.response, config=self.config.agent_outcome_policy)
            if action != OutcomeAction.VERIFY:
                error = record.response.error
                raise HandledEpisodeError(
                    UserSimulationFailure(
                        stage="agent",
                        failure_reason=(error.message if error else f"Agent reported failure in turn {turn}")[:2000],
                        terminal=action == OutcomeAction.FAIL,
                    )
                )
            step = await self.next_user_turn(episode, record)
            if isinstance(step, Stop):
                return step.reason
            params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": step.content}])
        return "max_turns"

    async def _seed_agent(
        self,
        request: EpisodeRequestT,
        session: _AgentSession,
        tool_accesses: list[ToolAccess],
        sandbox_access: SandboxAccess | None,
    ) -> None:
        continuation = AgentContinuationRequirements() if self.config.max_turns > 1 else None
        try:
            response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions",
                # Dumped in full: the client drops unset fields, such as each tool access's `kind` discriminator.
                json=AgentSeedSessionRequest(
                    agent_session_id=session.agent_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    tool_accesses=tool_accesses,
                    sandbox_access=sandbox_access,
                    continuation=continuation,
                ).model_dump(mode="json"),
            )
            await raise_for_status(response)
            session.cookies.update({name: cookie.value for name, cookie in response.cookies.items()})
            seeded = AgentSeedSessionResponse.model_validate(await get_response_json(response))
            if seeded.agent_session_id != session.agent_session_id:
                raise ValueError("Agent seed returned a different agent_session_id")
            if continuation is not None and seeded.capabilities is None:
                raise ValueError("Agent seed did not confirm session continuation")
        except Exception as error:
            raise self.failure(error, stage="seed") from error

    async def _activate(
        self,
        request: EpisodeRequestT,
        session: _AgentSession,
        params: NeMoGymResponseCreateParamsNonStreaming,
        turn: int,
        interaction_deadline: float | None,
    ) -> TurnRecord:
        loop = asyncio.get_running_loop()
        started = loop.time()
        turn_deadline = started + self.config.turn_timeout_seconds
        deadline = turn_deadline if interaction_deadline is None else min(turn_deadline, interaction_deadline)
        body = params.model_copy(
            update={"metadata": {**(params.metadata or {}), "agent_timeout_sec": str(deadline - started)}}
        )
        try:
            async with asyncio.timeout_at(deadline) as timer:
                response = await self.server_client.post(
                    server_name=self.config.agent_server.name,
                    url_path=self.agent_responses_path(request),
                    json=body,
                    headers={AGENT_ACTIVATION_HEADER: str(turn)},
                    cookies=session.cookies,
                )
                await raise_for_status(response)
                session.cookies.update({name: cookie.value for name, cookie in response.cookies.items()})
                agent_response = NeMoGymResponse.model_validate(await get_response_json(response))
        except TimeoutError as error:
            # The episode's own deadline also raises TimeoutError here; only this turn's is handled.
            if not timer.expired():
                raise
            if deadline != turn_deadline:
                raise _InteractionDeadlineExpired from error
            raise HandledEpisodeError(
                UserSimulationFailure(
                    stage="agent",
                    failure_reason=f"Agent turn {turn} timed out after {self.config.turn_timeout_seconds} seconds",
                    terminal=False,
                )
            ) from error
        except Exception as error:
            raise self.failure(error, stage="agent") from error
        return TurnRecord(
            turn=turn, input=params.input, response=agent_response, duration_seconds=loop.time() - started
        )

    async def _close_agent(self, request: EpisodeRequestT, session: _AgentSession) -> None:
        response = await self.server_client.post(
            server_name=self.config.agent_server.name,
            url_path="/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(agent_session_id=session.agent_session_id, episode_id=request.episode_id),
            cookies=session.cookies,
        )
        await raise_for_status(response)
        close = AgentCloseSessionResponse.model_validate(await get_response_json(response))
        if close.agent_session_id != session.agent_session_id:
            raise ValueError("Agent close returned a different agent_session_id")
        session.close = close
