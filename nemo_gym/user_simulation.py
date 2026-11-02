# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversations in which an Environment Server writes every user message to the agent under test.

The server seeds the agent's session (``AgentSessionClient``), then ``converse`` runs the turns: each turn activates
the agent with the user's latest message, and the server's user side reads the reply and writes the next message or
stops. The server keeps its episode's state, tools and grading; this module adds no base class.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from nemo_gym.agent_outcome import AgentOutcomePolicyConfig, OutcomeAction, agent_outcome_policy
from nemo_gym.agent_session_client import AgentSessionClient, TurnDeadlineExpired, TurnRecord
from nemo_gym.base_responses_api_agent import AgentCloseSessionResponse
from nemo_gym.config.config_types import AgentServerRef
from nemo_gym.core.responses_api_types import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime.base_environment_server import HandledEpisodeError
from nemo_gym.server_runtime.episode_tools import ToolCallRecord
from nemo_gym.server_runtime.episode_types import EpisodeFailure
from nemo_gym.trajectory.trajectory_record import AgentObservationBundle, TrajectoryRecord


class ConversationConfig(BaseModel):
    """Bind the agent under test and bound its conversation. An Environment Server's config inherits it."""

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


# The user side: reads the agent's turn and returns the next user message or Stop. It can be a simulator, a fixed
# protocol, a grader or another agent.
UserSide = Callable[[TurnRecord], Awaitable[UserMessage | Stop]]


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
    """What the server grades: the turns, why they stopped, the agent's close receipt and the episode's tool calls."""

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

    def failure_fields(self) -> dict[str, Any]:
        """The ``UserSimulationFailure`` fields that keep this evidence, for a failure in grading."""
        return _failure_fields(self.turns, self.agent_close, self.tool_calls)


def _failure_fields(
    turns: list[TurnRecord], agent_close: AgentCloseSessionResponse | None, tool_calls: list[ToolCallRecord]
) -> dict[str, Any]:
    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "agent_observations": agent_close.agent_observations if agent_close else None,
        "trajectory": agent_close.trajectory if agent_close else None,
    }


async def converse(
    agent: AgentSessionClient,
    user: UserSide,
    first: NeMoGymResponseCreateParamsNonStreaming,
    config: ConversationConfig,
    *,
    tool_calls: Sequence[ToolCallRecord] = (),
) -> Transcript:
    """Run the agent's turns until ``user`` stops them or a limit does, then close the agent.

    The first turn sends ``first``; each later turn sends only the user's new message, so above one turn the agent
    must have been seeded with ``continuation``. ``user`` reads every turn, the last one included. A turn that the
    agent outcome policy does not send to grading fails the episode at the stage ``agent``.

    The agent is closed before the server grades, so grading reads the state the agent left. A ``HandledEpisodeError``
    whose failure is a ``UserSimulationFailure`` leaves with the completed turns, ``tool_calls`` (a live list, such as
    ``ServedTools.calls``, is read when the conversation ends) and the agent's evidence.
    """
    try:
        stop_reason = await _run_turns(agent, user, first, config)
        await agent.close()
    except HandledEpisodeError as error:
        if isinstance(error.failure, UserSimulationFailure):
            try:
                await agent.close()  # A no-op when the agent is already closed.
            except Exception:
                pass  # The episode's cleanup retries the close and logs its failure.
            evidence = _failure_fields(list(agent.turns), agent.closed, list(tool_calls))
            error.failure = error.failure.model_copy(update=evidence)
        raise
    return Transcript(list(agent.turns), stop_reason, agent.closed, list(tool_calls))


async def _run_turns(
    agent: AgentSessionClient,
    user: UserSide,
    first: NeMoGymResponseCreateParamsNonStreaming,
    config: ConversationConfig,
) -> str:
    """Run turns until ``user`` stops them or a limit does; return the stop reason."""
    loop = asyncio.get_running_loop()
    interaction_deadline = (
        loop.time() + config.interaction_timeout_seconds if config.interaction_timeout_seconds else None
    )
    params = first
    for turn in range(1, config.max_turns + 1):
        now = loop.time()
        if interaction_deadline is not None and now >= interaction_deadline:
            return "interaction_timeout"
        turn_deadline = now + config.turn_timeout_seconds
        deadline = turn_deadline if interaction_deadline is None else min(turn_deadline, interaction_deadline)
        try:
            record = await agent.activate(params, deadline=deadline)
        except TurnDeadlineExpired as error:
            if deadline != turn_deadline:
                return "interaction_timeout"
            raise HandledEpisodeError(
                agent.failure_model(
                    stage="agent",
                    failure_reason=f"Agent turn {turn} timed out after {config.turn_timeout_seconds} seconds",
                    terminal=False,
                )
            ) from error
        action = agent_outcome_policy(record.response, config=config.agent_outcome_policy)
        if action != OutcomeAction.VERIFY:
            error = record.response.error
            raise HandledEpisodeError(
                agent.failure_model(
                    stage="agent",
                    failure_reason=(error.message if error else f"Agent reported failure in turn {turn}")[:2000],
                    terminal=action == OutcomeAction.FAIL,
                )
            )
        step = await user(record)
        if isinstance(step, Stop):
            return step.reason
        params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": step.content}])
    return "max_turns"
