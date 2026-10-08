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

from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    SimpleResponsesAPIAgent,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap


def kill_cli_process_group(process: asyncio.subprocess.Process) -> None:
    """Stop a CLI launched with start_new_session, including its npm children."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@dataclass
class CLIActivation(AgentSessionState):
    """Keep the activation alive until close has observed its cleanup."""

    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    observations: AgentObservationBundle | None = None
    cancel_requested: bool = False


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

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> CLIActivation:
        """Validate supported access and workspace isolation before admission."""
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
        return CLIActivation(request=body)

    @abstractmethod
    async def _execute_responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        """Execute the harness for native and compatibility calls, including observation hooks."""
        raise NotImplementedError

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            return await self._execute_responses(request, body)
        rollout_id = request.path_params.get("rollout_id")
        if not isinstance(rollout_id, str):
            raise HTTPException(409, "Native CLI activation requires a rollout-prefixed Responses route")
        state = self._require_agent_session(session_id)
        if not isinstance(state, CLIActivation):
            raise TypeError("CLI session has invalid activation state")
        if rollout_id != state.request.episode_id.capture_key:
            raise HTTPException(409, "Agent-session episode_id does not match the rollout route")
        if state.task is not None:
            if body != state.activation_request:
                raise HTTPException(409, "Agent session has already been activated with a different request")
            return (await asyncio.shield(state.task)).model_copy(deep=True)
        state.activation_request = body.model_copy(deep=True)

        async def activate() -> NeMoGymResponse:
            if self.sem is None:
                raise RuntimeError("CLI concurrency semaphore is not initialized")
            async with self.sem:
                response = await self._execute_responses(request, state.activation_request.model_copy(deep=True))
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
        # A disconnected HTTP waiter must not cancel the invocation: ServerClient
        # can retry it. Only explicit session close owns cancellation and cleanup.
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Join activation cleanup and collect its observations for verification."""
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
                            invocation_id=state.request.episode_id.capture_key,
                            status="incomplete" if task.cancelled() else "failed",
                        )
                    ],
                    gaps=[ObservationGap(code="agent_activation_interrupted")],
                )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, agent_observations=state.observations
        )
