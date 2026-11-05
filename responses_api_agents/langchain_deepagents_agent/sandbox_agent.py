# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepAgents agent that runs each episode's deepagents graph inside a task sandbox.

The sandbox is the one the Resources Server lends the session, or one the agent starts from `sandbox_provider`. The
agent server installs deepagents in it, stages `sandbox_runner.py` and runs it under Gym's process supervisor. The
graph's file tools and `execute` tool work on the sandbox's filesystem, and its model calls go from the sandbox to the
Gym Model Server. `DeepAgentsAgent` (app.py) runs a graph with custom tools in process instead.
"""

import asyncio
import time
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from langchain_core.messages import SystemMessage, messages_from_dict, messages_to_dict
from pydantic import Field, model_validator

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SandboxSession,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
    resolve_agent_timeout,
)
from nemo_gym.config import ModelServerRef, get_global_config_dict
from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider, resolve_provider_config
from nemo_gym.server_runtime import BaseRunRequest
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap, SandboxObservation
from responses_api_agents.langchain_deepagents_agent.responses_langchain_bridge import to_langchain, to_responses
from responses_api_agents.langchain_deepagents_agent.sandbox import DeepAgentsSandboxSession


OBSERVATION_SOURCE = "langchain_deepagents_agent"
# How the graph's run ended, as the runner reports it, for the sandbox record of the close bundle.
# A run that hit the graph's step limit ended on its own, so its sandbox work completed.
_SANDBOX_OUTCOMES = {
    "completed": "completed",
    "max_iterations": "completed",
    "failed": "failed",
    "wall_time": "timeout",
    "cancelled": "cancelled",
}


class SandboxedDeepAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    # Required, no default: without it, deepagents' SummarizationMiddleware falls back to a hardcoded,
    # model-agnostic threshold (170k tokens, keep last 6 messages) unrelated to the real model.
    max_input_tokens: int = Field(
        gt=0, description="The model's context window, which sets when deepagents summarizes"
    )
    system_prompt: Optional[str] = Field(default=None, description="Added to deepagents' own system prompt")
    shell_timeout_seconds: int = Field(
        default=120, gt=0, description="Default time limit of one `execute` command; the model may pass its own"
    )
    concurrency: int = Field(default=32, gt=0, description="Maximum concurrent runs")
    sandbox_provider: Optional[str] = Field(
        default=None, description="Sandbox provider config to start a sandbox from when the session lends none."
    )
    sandbox_config: dict[str, Any] = Field(
        default_factory=dict, description="SandboxSpec fields for a sandbox started from sandbox_provider."
    )
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    sandbox_runner_timeout_seconds: float = Field(
        default=12600,
        gt=0,
        allow_inf_nan=False,
        description="Time limit of one run; a task's shorter metadata.agent_timeout_sec applies instead.",
    )

    @model_validator(mode="after")
    def require_started_sandbox_workdir(self) -> "SandboxedDeepAgentConfig":
        if self.sandbox_provider is not None:
            workdir = self.sandbox_config.get("workdir")
            if not isinstance(workdir, str) or not PurePosixPath(workdir).is_absolute():
                raise ValueError("sandbox_config.workdir must be an absolute path when sandbox_provider is set")
        return self


def _usage(output: Mapping[str, Any]) -> Optional[NeMoGymResponseUsage]:
    """The run's total usage, or None unless every model call returned with usage."""
    usages = output.get("usages") or []
    if len(usages) != output.get("model_calls") or any(usage is None for usage in usages):
        return None
    total = None
    for usage in usages:
        total = accumulate_response_usage(total, NeMoGymResponseUsage.model_validate(usage))
    return total


def _model_error_type(output: Mapping[str, Any]) -> Optional[str]:
    if not output.get("model_error"):
        return None
    return "ModelTransportError" if output.get("model_transport_error") else "ModelRequestError"


class SandboxedDeepAgent(SimpleResponsesAPIAgent):
    """Run deepagents' own graph, with its file and shell tools, in a borrowed or started task sandbox."""

    config: SandboxedDeepAgentConfig
    sem: Any = Field(default=None, exclude=True)

    def model_post_init(self, context: Any) -> None:
        self.sem = asyncio.Semaphore(self.config.concurrency)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> DeepAgentsSandboxSession:
        required = [access.name for access in self.effective_tool_accesses(body) if access.required]
        if required:
            # The graph runs only deepagents' own tools, so a required grant would go unused.
            raise AgentSeedRejectedError(
                f"Sandboxed DeepAgents does not support required tool access: {', '.join(required)}"
            )
        access = body.sandbox_access
        if access is None:
            if self.config.sandbox_provider is None:
                raise AgentSeedRejectedError("Sandboxed DeepAgents requires sandbox_access or a sandbox_provider")
            provider_ref, workdir = self.config.sandbox_provider, self.config.sandbox_config["workdir"]
        else:
            if access.user is not None:
                raise AgentSeedRejectedError("Sandboxed DeepAgents does not support a sandbox user override")
            if not PurePosixPath(access.workdir).is_absolute():
                raise AgentSeedRejectedError("Sandboxed DeepAgents requires an absolute sandbox workdir")
            provider_ref, workdir = access.connection.provider_config_ref, access.workdir
        provider_config = resolve_provider_config(provider_ref, get_global_config_dict())
        try:
            # The runner calls the Model Server from inside the sandbox, which cannot reach a loopback address.
            assert_model_url_reachable_from_sandbox(
                self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key),
                provider_name=next(iter(provider_config), None),
            )
        except ValueError as error:
            raise AgentSeedRejectedError(str(error)) from error
        provider = create_provider(provider_config)
        try:
            if access is None:
                sandbox = AsyncSandbox(provider)
                sandbox_config = dict(self.config.sandbox_config)
                # Provider TTLs outlive this server; reserve ten minutes for setup and cleanup.
                sandbox_config.setdefault(
                    "ttl_s",
                    self.config.sandbox_install_timeout_seconds + self.config.sandbox_runner_timeout_seconds + 600,
                )
                await sandbox.start(SandboxSpec(**sandbox_config))
            else:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        state = DeepAgentsSandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                workdir=workdir,
                session_dir=f"/tmp/nemo-gym-deepagents-sessions/{uuid4().hex}",
                owns_sandbox=access is None,
                harness="DeepAgents",
            ),
            sandbox_provider=sandbox.provider_name,
            sandbox_id=sandbox.sandbox_id,
        )
        try:
            await state.install_runtime(timeout=self.config.sandbox_install_timeout_seconds)
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, DeepAgentsSandboxSession):
            raise TypeError("Expected a DeepAgents sandbox session state")
        await state.close(self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, agent_observations=self._observations(state)
        )

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, DeepAgentsSandboxSession) else None

    @staticmethod
    def _observations(state: DeepAgentsSandboxSession) -> AgentObservationBundle:
        """The sandbox the graph ran in and how its run ended; the graph itself reports no invocation records."""
        output = state.session.artifacts or {}
        ending = output.get("stop_reason") or output.get("status")
        record = SandboxObservation(
            role="agent",
            provider=state.sandbox_provider,
            sandbox_id=state.sandbox_id,
            # A session closed before its run reported anything was cancelled.
            outcome=_SANDBOX_OUTCOMES.get(ending, "unknown") if output else "cancelled",
            error_type=_model_error_type(output),
        )
        return AgentObservationBundle(
            source=OBSERVATION_SOURCE,
            records=[record],
            gaps=[
                ObservationGap(
                    code="agent_transcript_unavailable",
                    detail="the deepagents graph reports no invocation or tool-call records",
                ),
                ObservationGap(code="model_call_ownership_unavailable"),
            ],
        )

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """The Model Server's HTTP status, or a transport failure, that ended the run, as the runner reported it."""
        status = native_result.get("model_error_status")
        if native_result.get("model_error") and type(status) is int and 400 <= status <= 599:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=status)
        if native_result.get("model_transport_error"):
            return AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
        return super().classify_outcome(native_result)

    async def _run_in_sandbox(
        self, body: NeMoGymResponseCreateParamsNonStreaming, state: DeepAgentsSandboxSession
    ) -> NeMoGymResponse:
        if body.tools:
            raise HTTPException(
                422, "Sandboxed DeepAgents runs only deepagents' own tools; the request declares tools"
            )
        input_items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        messages = to_langchain(input_items)
        if body.instructions:
            messages.insert(0, SystemMessage(content=body.instructions))
        capture_key = state.request.episode_id.capture_key
        sampling = {
            "temperature": body.temperature,
            "top_p": body.top_p,
            "max_output_tokens": body.max_output_tokens,
            "reasoning": body.reasoning,
        }
        access = state.request.sandbox_access
        payload = {
            "messages": messages_to_dict(messages),
            # The sandbox reaches the Model Server directly; the rollout prefix keeps its calls correlated.
            "model_url": f"{self.resolve_model_base_url(self.config.model_server.name, capture_key)}/responses",
            "request_defaults": {name: value for name, value in sampling.items() if value is not None},
            "max_input_tokens": self.config.max_input_tokens,
            "system_prompt": self.config.system_prompt,
            "shell_timeout_seconds": self.config.shell_timeout_seconds,
            "workdir": state.session.workdir,
            "task_env": dict(access.env) if access is not None else {},
            "rollout_id": capture_key,
            "stop_request_path": state.session.stop_request_path,
        }
        try:
            output = await state.execute(
                payload,
                timeout=resolve_agent_timeout(body.metadata, ceiling=self.config.sandbox_runner_timeout_seconds),
                close_timeout=self.config.session_close_timeout_seconds,
            )
        except TimeoutError:
            # The runner normally stops at the deadline and writes its partial report. If the supervisor overran
            # too, the episode is still a timeout when that report is there, and an infrastructure failure when not.
            output = state.session.artifacts
            if not (isinstance(output, dict) and output.get("stop_reason") == "wall_time"):
                raise
        response = to_responses(messages_from_dict(output["messages"]), self.config.model_server.name)
        response.update(
            id=f"deepagents-{capture_key}",
            created_at=int(time.time()),
            status=output["status"],
            usage=_usage(output),
            metadata={"stop_reason": output["stop_reason"]} if output["stop_reason"] else None,
            error={"code": "server_error", "message": output["model_error"]} if output["model_error"] else None,
        )
        response = self.report_rollout_stop(NeMoGymResponse.model_validate(response), state.session)
        return self.report_outcome(response, output)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        agent_session_id = self._agent_session_id_from_request(request)
        if agent_session_id is None:
            raise HTTPException(409, "Sandboxed DeepAgents runs only in agent sessions; use an environment server")
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        state = self._require_agent_session(agent_session_id)
        if not isinstance(state, DeepAgentsSandboxSession):
            raise TypeError("Expected a DeepAgents sandbox session state")
        if state.request.episode_id.capture_key != rollout_id:
            raise HTTPException(409, "Agent-session episode_id does not match the rollout route")
        if state.task is None:
            # No await until the task and its request binding are installed.
            state.activation_request = body.model_copy(deep=True)

            async def activate() -> NeMoGymResponse:
                async with self.sem:
                    return await self._run_in_sandbox(body, state)

            state.task = asyncio.create_task(activate())
        elif body != state.activation_request:
            raise HTTPException(409, "Sandboxed DeepAgents sessions run one activation; retry the same request")
        # Session close owns cancellation; a disconnected waiter must not cancel the activation.
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def run(self, request: Request, body: BaseRunRequest = Body()):
        raise HTTPException(409, "Sandboxed DeepAgents runs only in agent sessions; use an environment server")


if __name__ == "__main__":
    SandboxedDeepAgent.run_webserver()
