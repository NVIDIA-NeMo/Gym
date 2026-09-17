# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-activation sessions for existing local, verifier-only CLI harnesses."""

import asyncio
import os
import signal
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from fastapi import Body, HTTPException, Request
from pydantic import ConfigDict

from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.episode import AgentSeedSessionRequest, MCPResourcesToolAccess
from nemo_gym.episode_sessions import AgentCloseSessionResult, AgentSession
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap


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


class CLIResponsesAPIAgent(SimpleResponsesAPIAgent):
    """Opt local CLI harnesses into the M1 session protocol.

    Capabilities: one Responses activation, local scratch workspace, no runtime
    resources tools, no borrowed sandbox. Direct resources metadata is retained
    but unused for verifier-only tasks; MCP access is rejected. Legacy /run
    callers keep the original Responses path without session interception.
    """

    supports_agent_sessions: ClassVar[bool] = True
    observation_source: ClassVar[str]
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sem: asyncio.Semaphore | None = None

    def model_post_init(self, context: object) -> None:
        # Preserve multi-worker legacy /run deployments. These CLI classes had
        # no superclass initialization; check worker affinity at native seed,
        # rather than rejecting an otherwise unchanged legacy configuration.
        self._agent_sessions = {}

    async def initialize_agent_session_state(
        self, agent_session_id: str, body: AgentSeedSessionRequest
    ) -> CLIActivation:
        if self.config.num_workers not in (None, 1):
            raise HTTPException(422, "Native CLI sessions require num_workers=1")
        if body.sandbox_access is not None:
            raise HTTPException(422, f"{self.observation_source} does not support borrowed SandboxAccess yet")
        if isinstance(body.resources_access, MCPResourcesToolAccess):
            raise HTTPException(422, f"{self.observation_source} native sessions do not support runtime MCP tools yet")
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

    async def close_agent_session_state(self, agent_session_id: str, session: AgentSession) -> AgentCloseSessionResult:
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
        return AgentCloseSessionResult(agent_observations=state.observations)
