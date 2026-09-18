# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-activation sessions for existing local, verifier-only CLI harnesses."""

import asyncio
import os
import signal
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from fastapi import Body, HTTPException, Request
from pydantic import ConfigDict

from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.episode import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap
from nemo_gym.server_utils import SESSION_ID_KEY


_AGENT_SESSION_ACTIVE_KEY = "nemo_gym_agent_session"


def kill_cli_process_group(process: asyncio.subprocess.Process) -> None:
    """Stop a CLI launched with start_new_session, including its npm children."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@dataclass
class CLIActivation:
    """Keep the activation alive until close has observed its cleanup."""

    task: asyncio.Task[NeMoGymResponse] | None = None
    observations: AgentObservationBundle | None = None
    cancel_requested: bool = False


@dataclass
class _CLISession:
    request: AgentSeedSessionRequest
    state: CLIActivation
    activation_started: bool = False


class CLIResponsesAPIAgent(SimpleResponsesAPIAgent):
    """Opt local CLI harnesses into the agent session protocol.

    Capabilities: one Responses activation, local scratch workspace, no runtime
    resources tools, no borrowed sandbox. Required tool accesses are rejected;
    optional accesses are not configured. Legacy /run callers keep the original
    Responses path without session interception.
    """

    observation_source: ClassVar[str]
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sem: asyncio.Semaphore | None = None
    _agent_sessions: dict[str, _CLISession]

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._agent_sessions = {}

    async def seed_agent_session(self, request: Request, body: AgentSeedSessionRequest) -> AgentSeedSessionResponse:
        """Create worker-local state without starting a harness."""
        session_id = request.session[SESSION_ID_KEY]
        if session_id in self._agent_sessions:
            raise HTTPException(409, f"Agent session already exists: {session_id}")
        state = await self.initialize_agent_session_state(session_id, body)
        self._agent_sessions[session_id] = _CLISession(request=body, state=state)
        request.session[_AGENT_SESSION_ACTIVE_KEY] = True
        return AgentSeedSessionResponse(agent_session_id=session_id)

    async def close_agent_session(self, request: Request, body: AgentCloseSessionRequest) -> AgentCloseSessionResponse:
        """Retain session state until activation cleanup has succeeded."""
        if body.agent_session_id != self.agent_session_id_from_request(request):
            raise HTTPException(409, "agent_session_id does not match the session cookie")
        session = self.require_agent_session(body.agent_session_id)
        if body.episode_id != session.request.episode_id:
            raise HTTPException(409, "episode_id does not match the seeded agent session")
        observations = await self.close_agent_session_state(body.agent_session_id, session)
        del self._agent_sessions[body.agent_session_id]
        request.session.pop(_AGENT_SESSION_ACTIVE_KEY, None)
        return AgentCloseSessionResponse(agent_session_id=body.agent_session_id, agent_observations=observations)

    def require_agent_session(self, agent_session_id: str) -> _CLISession:
        """Resolve a session owned by this worker."""
        try:
            return self._agent_sessions[agent_session_id]
        except KeyError as error:
            raise HTTPException(409, f"Unknown agent_session_id: {agent_session_id}") from error

    def begin_agent_activation(self, agent_session_id: str, rollout_id: str) -> _CLISession:
        """Validate correlation and reserve the session's single activation."""
        session = self.require_agent_session(agent_session_id)
        if rollout_id != session.request.episode_id.capture_key:
            raise ValueError("Agent-session episode_id does not match the rollout route")
        if session.activation_started:
            raise ValueError("Agent session has already been activated")
        session.activation_started = True
        return session

    @staticmethod
    def agent_session_id_from_request(request: Request | None) -> str | None:
        """Distinguish native sessions from direct Responses calls."""
        if request is None:
            return None
        try:
            session = request.session
        except (AssertionError, AttributeError):
            return None
        if not isinstance(session, Mapping) or session.get(_AGENT_SESSION_ACTIVE_KEY) is not True:
            return None
        session_id = session.get(SESSION_ID_KEY)
        return session_id if isinstance(session_id, str) else None

    async def initialize_agent_session_state(
        self, agent_session_id: str, body: AgentSeedSessionRequest
    ) -> CLIActivation:
        """Validate supported access and workspace isolation before admission."""
        # Multi-worker direct /run deployments remain valid; only native seed
        # needs worker affinity for its in-memory session state.
        if self.config.num_workers not in (None, 1):
            raise HTTPException(422, "Native CLI sessions require num_workers=1")
        if body.sandbox_access is not None:
            raise HTTPException(422, f"{self.observation_source} does not support borrowed SandboxAccess yet")
        required_tools = [access.name for access in self.effective_tool_accesses(body) if access.required]
        if required_tools:
            raise HTTPException(
                422,
                f"{self.observation_source} native sessions do not support required runtime tools: {required_tools}",
            )
        # Persistent workspace overrides cannot provide episode isolation.
        for option in ("cwd", "repo_dir"):
            if getattr(self.config, option, None):
                raise HTTPException(422, f"Native CLI sessions require an isolated workspace; unset {option}")
        return CLIActivation()

    @abstractmethod
    async def _execute_responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        """Execute the harness for native and compatibility calls, including observation hooks."""
        raise NotImplementedError

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self.agent_session_id_from_request(request)
        if session_id is None:
            return await self._execute_responses(request, body)
        rollout_id = request.path_params.get("rollout_id")
        if not isinstance(rollout_id, str):
            raise HTTPException(409, "Native CLI activation requires a rollout-prefixed Responses route")
        try:
            session = self.begin_agent_activation(session_id, rollout_id)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        state = session.state
        if not isinstance(state, CLIActivation):
            raise TypeError("CLI session has invalid activation state")

        async def activate() -> NeMoGymResponse:
            if self.sem is None:
                raise RuntimeError("CLI concurrency semaphore is not initialized")
            async with self.sem:
                response = await self._execute_responses(request, body)
            response = response.model_copy(deep=True)
            raw = (response.model_extra or {}).get("_ng_agent_observations")
            if raw is not None:
                state.observations = AgentObservationBundle.model_validate(raw)
                response.__pydantic_extra__.pop("_ng_agent_observations")
            else:
                # These are output-only observations, not an invented complete
                # model transcript or inferred model-call ownership.
                state.observations = AgentObservationBundle(
                    source=self.observation_source,
                    records=[AgentInvocation(invocation_id=rollout_id, conversation=response.output)],
                    gaps=[
                        ObservationGap(code="agent_transcript_input_unavailable"),
                        ObservationGap(code="model_call_ownership_unavailable"),
                        ObservationGap(code="tool_timing_unavailable"),
                        ObservationGap(code="no_sandbox_runtime"),
                    ],
                )
            return response

        state.task = asyncio.create_task(activate())
        try:
            return await asyncio.shield(state.task)
        except asyncio.CancelledError:
            # Do not interrupt cleanup if close has already cancelled the task.
            if not state.task.done() and not state.task.cancelling():
                state.cancel_requested = True
                state.task.cancel()
            raise

    async def close_agent_session_state(
        self, agent_session_id: str, session: _CLISession
    ) -> AgentObservationBundle | None:
        """Join activation cleanup and collect its observations for verification."""
        state = session.state
        if not isinstance(state, CLIActivation):
            raise TypeError("CLI session has invalid activation state")
        task = state.task
        if task is not None:
            if not task.done() and not task.cancelling():
                state.cancel_requested = True
                task.cancel()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    # Close itself was cancelled. Leave state available for retry.
                    raise
            except Exception:
                if state.cancel_requested:
                    # A failed cancellation cleanup cannot authorize verification.
                    raise
                # Activation failures already propagate to /responses; the task
                # must finish its subprocess/workspace finally before close.
                pass
            if state.observations is None:
                state.observations = AgentObservationBundle(
                    source=self.observation_source,
                    records=[
                        AgentInvocation(
                            invocation_id=session.request.episode_id.capture_key,
                            status="incomplete" if task.cancelled() else "failed",
                        )
                    ],
                    gaps=[ObservationGap(code="agent_activation_interrupted")],
                )
        return state.observations
