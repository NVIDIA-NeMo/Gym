# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One agent session that an Environment Server drives during an episode: seed it, run its turns, close it."""

import asyncio
from collections.abc import Sequence
from typing import Any, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, PositiveInt

from nemo_gym.agent_activation_types import AgentContinuationRequirements
from nemo_gym.base_responses_api_agent import (
    AGENT_ACTIVATION_HEADER,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.client.http_client import get_response_json, raise_for_status
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import TOKEN_CAPTURE_PATH_SEGMENT, AgentServerRef
from nemo_gym.config.global_config import TOKEN_CAPTURE_BLOCK, get_first_server_config_dict
from nemo_gym.core.failure_kinds import FailureStage
from nemo_gym.core.responses_api_types import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInput,
)
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.server_runtime.base_environment_server import (
    CleanupContext,
    CleanupHandle,
    HandledEpisodeError,
    dependency_failure,
)
from nemo_gym.server_runtime.episode_types import BaseEpisodeRequest, EpisodeFailure
from nemo_gym.server_runtime.tool_access import ToolAccess


class TurnRecord(BaseModel):
    """One agent turn: the input the agent received and the response it returned."""

    model_config = ConfigDict(extra="forbid")

    turn: PositiveInt
    input: Union[str, NeMoGymResponseInput]
    response: NeMoGymResponse
    duration_seconds: float


class TurnDeadlineExpired(Exception):
    """The deadline of a turn passed before the agent answered."""


class AgentSessionClient:
    """One agent session that an Environment Server drives during an episode.

    ``seed`` opens the session with the episode's grants, ``activate`` runs one turn and records it, and ``close`` ends
    the session and keeps its close receipt (``closed``), which carries the agent's observations and trajectory. Each
    turn goes through the episode's capture route. A failed call raises ``HandledEpisodeError`` with a
    ``failure_model``, at the stage ``seed``, ``agent`` or ``cleanup``.
    """

    def __init__(
        self,
        server_client: ServerClient,
        agent: AgentServerRef,
        request: BaseEpisodeRequest[Any],
        *,
        failure_model: type[EpisodeFailure] = EpisodeFailure,
    ) -> None:
        self.server_client = server_client
        self.agent = agent
        self.request = request
        self.failure_model = failure_model
        self.agent_session_id = f"agent-session-{uuid4().hex}"
        self.turns: list[TurnRecord] = []
        self.closed: AgentCloseSessionResponse | None = None
        self._cookies: dict[str, str] = {}
        self._cleanup: CleanupHandle | None = None

    def failure(self, error: BaseException, *, stage: FailureStage) -> HandledEpisodeError:
        """Map a failed call to the agent to the episode's failure (see ``dependency_failure``)."""
        return dependency_failure(error, failure_model=self.failure_model, stage=stage)

    def responses_path(self) -> str:
        """The agent's ``/v1/responses`` path under the episode's capture route."""
        global_config = self.server_client.global_config_dict
        block = global_config.get(TOKEN_CAPTURE_BLOCK) or {}
        agent_config = get_first_server_config_dict(global_config, self.agent.name)
        capture = bool(block.get("enabled")) and (block.get("all_agents") or agent_config.get("token_id_capture"))
        segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if capture else ""
        return f"/ng-rollout/{self.request.episode_id.capture_key}{segment}/v1/responses"

    async def seed(
        self,
        cleanup: CleanupContext,
        *,
        tool_accesses: Sequence[ToolAccess] = (),
        sandbox_access: SandboxAccess | None = None,
        continuation: bool = False,
    ) -> None:
        """Open the session with ``tool_accesses`` and ``sandbox_access``; ``cleanup`` closes it when the episode ends.

        With ``continuation``, the session must resume across turns, so each turn can send only its new input. An agent
        that cannot fails the seed.
        """
        # Registered before the seed, so a cancellation closes a session whose seed reply was lost.
        self._cleanup = cleanup.register_cleanup("agent session", self._close)
        requirements = AgentContinuationRequirements() if continuation else None
        try:
            response = await self.server_client.post(
                server_name=self.agent.name,
                url_path="/v1/agent_sessions",
                # Dumped in full: the client drops unset fields, such as each tool access's `kind` discriminator.
                json=AgentSeedSessionRequest(
                    agent_session_id=self.agent_session_id,
                    episode_id=self.request.episode_id,
                    task_id=self.request.task.task_id,
                    tool_accesses=list(tool_accesses),
                    sandbox_access=sandbox_access,
                    continuation=requirements,
                ).model_dump(mode="json"),
            )
            await raise_for_status(response)
            self._cookies.update({name: cookie.value for name, cookie in response.cookies.items()})
            seeded = AgentSeedSessionResponse.model_validate(await get_response_json(response))
            if seeded.agent_session_id != self.agent_session_id:
                raise ValueError("Agent seed returned a different agent_session_id")
            if requirements is not None and seeded.capabilities is None:
                raise ValueError("Agent seed did not confirm session continuation")
        except Exception as error:
            raise self.failure(error, stage="seed") from error

    async def activate(self, params: NeMoGymResponseCreateParamsNonStreaming, *, deadline: float) -> TurnRecord:
        """Run the session's next turn on ``params``, and record it.

        ``deadline`` is in event loop time (``loop.time()``); the agent learns how long it has from
        ``metadata.agent_timeout_sec``. Raises ``TurnDeadlineExpired`` when the deadline passes first.
        """
        turn = len(self.turns) + 1
        loop = asyncio.get_running_loop()
        started = loop.time()
        body = params.model_copy(
            update={"metadata": {**(params.metadata or {}), "agent_timeout_sec": str(deadline - started)}}
        )
        try:
            async with asyncio.timeout_at(deadline) as timer:
                response = await self.server_client.post(
                    server_name=self.agent.name,
                    url_path=self.responses_path(),
                    json=body,
                    headers={AGENT_ACTIVATION_HEADER: str(turn)},
                    cookies=self._cookies,
                )
                await raise_for_status(response)
                self._cookies.update({name: cookie.value for name, cookie in response.cookies.items()})
                agent_response = NeMoGymResponse.model_validate(await get_response_json(response))
        except TimeoutError as error:
            # An outer deadline, such as the episode's own, also raises TimeoutError here; only this turn's is handled.
            if not timer.expired():
                raise
            raise TurnDeadlineExpired(f"Agent turn {turn} passed its deadline") from error
        except Exception as error:
            raise self.failure(error, stage="agent") from error
        record = TurnRecord(
            turn=turn, input=params.input, response=agent_response, duration_seconds=loop.time() - started
        )
        self.turns.append(record)
        return record

    async def close(self) -> None:
        """Close the session now; the episode's cleanup then finds it closed. A no-op once it is closed."""
        if self._cleanup is None:
            return
        try:
            await self._cleanup.close()
        except Exception as error:
            raise self.failure(error, stage="cleanup") from error

    async def _close(self) -> None:
        response = await self.server_client.post(
            server_name=self.agent.name,
            url_path="/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(agent_session_id=self.agent_session_id, episode_id=self.request.episode_id),
            cookies=self._cookies,
        )
        await raise_for_status(response)
        closed = AgentCloseSessionResponse.model_validate(await get_response_json(response))
        if closed.agent_session_id != self.agent_session_id:
            raise ValueError("Agent close returned a different agent_session_id")
        self.closed = closed
