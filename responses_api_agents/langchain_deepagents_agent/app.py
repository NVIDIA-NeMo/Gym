# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepAgents adapter with one entrypoint for its sandbox and Tavily search profiles.

The builtin profile runs deepagents in a borrowed or agent-owned sandbox. The tavily_search profile preserves
our in-process search graph and legacy /run integration. Both use the same agent-session endpoints and bridge.
Custom in-process graphs may still override build_agent(model).
"""

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal, Optional
from uuid import uuid4

from deepagents import create_deep_agent
from fastapi import Body, HTTPException, Request, Response
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, messages_from_dict, messages_to_dict
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from pydantic import Field, PrivateAttr, model_validator

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    SandboxSession,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
    resolve_agent_timeout,
)
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import (
    MODEL_CONTEXT_LIMIT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
    NgFailureEvent,
    accumulate_response_usage,
    observe_failure,
    sdk_error_details,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider, resolve_provider_config, resolve_provider_metadata
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap, SandboxObservation
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.langchain_deepagents_agent.responses_langchain_bridge import (
    GymResponsesChatModel,
    to_langchain,
    to_responses,
)
from responses_api_agents.langchain_deepagents_agent.sandbox import DeepAgentsSandboxSession
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
    SimpleAgentSessionState,
    SimpleAgentVerifyResponse,
)


LOG = logging.getLogger(__name__)


OBSERVATION_SOURCE = "langchain_deepagents_agent"
# How the graph's run ended, as the runner reports it, for the sandbox record of the close bundle.
# A run that hit the graph's step limit ended on its own, so its sandbox work completed.
_SANDBOX_OUTCOMES = {
    "completed": "completed",
    "max_iterations": "completed",
    "context_limit": "completed",
    "failed": "failed",
    "wall_time": "timeout",
    "cancelled": "cancelled",
}


@dataclass
class DeepAgentsSessionState(SimpleAgentSessionState):
    """One agent session and its single activation.

    The first activation request starts `activation`. An identical retry joins it, a different request is refused,
    and close cancels it.
    """

    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    activation: asyncio.Task[NeMoGymResponse] | None = None


class DeepAgentsAgentConfig(SimpleAgentConfig):
    """Select the graph's tools while sharing model routing and agent-session configuration."""

    tool_profile: Literal["builtin", "tavily_search"] = Field(
        default="builtin",
        description="Built-in file/shell tools in a task sandbox, or the existing Tavily search graph in process.",
    )
    tavily_api_key: Optional[str] = None
    max_search_results: int = 5
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
    def validate_tool_profile(self) -> "DeepAgentsAgentConfig":
        if self.tool_profile == "tavily_search":
            if not self.tavily_api_key:
                raise ValueError("tavily_api_key is required for tool_profile=tavily_search")
            if self.sandbox_provider is not None:
                raise ValueError(
                    "tool_profile=tavily_search runs in process; sandbox_provider requires tool_profile=builtin"
                )
        elif self.tavily_api_key is not None:
            raise ValueError("tavily_api_key requires tool_profile=tavily_search")
        if self.sandbox_provider is not None:
            workdir = self.sandbox_config.get("workdir")
            if not isinstance(workdir, str) or not PurePosixPath(workdir).is_absolute():
                raise ValueError("sandbox_config.workdir must be an absolute path when sandbox_provider is set")
        return self

    @model_validator(mode="after")
    def reject_max_steps(self) -> "DeepAgentsAgentConfig":
        if self.max_steps is not None:
            raise ValueError(
                "max_steps is not supported by DeepAgentsAgent: deepagents runs its own internal tool loop, "
                "with no per-model-call step counter for this to bound. Remove max_steps from this agent's "
                "config."
            )
        return self

    @model_validator(mode="after")
    def reject_execute_tools_false(self) -> "DeepAgentsAgentConfig":
        # SimpleAgent can hand unresolved tool calls back to the Environment Server; this graph cannot,
        # because deepagents always runs its own tools inside ainvoke().
        if not self.execute_tools:
            raise ValueError(
                "execute_tools=false is not supported by DeepAgentsAgent: the deepagents graph always runs its "
                "own tools. Remove execute_tools from this agent's config."
            )
        return self


def _usage(output: Mapping[str, Any]) -> Optional[NeMoGymResponseUsage]:
    """The run's total usage, or None unless every model call returned with usage."""
    if output.get("usage_incomplete"):
        return None
    usages = output.get("usages") or []
    if len(usages) != output.get("model_calls") or any(usage is None for usage in usages):
        return None
    total = None
    for usage in usages:
        total = accumulate_response_usage(total, NeMoGymResponseUsage.model_validate(usage))
    return total


def _reject_response_budget(body: NeMoGymResponseCreateParamsNonStreaming) -> None:
    """Reject an aggregate budget before starting resources or binding graph work."""
    if body.max_output_tokens is not None:
        raise HTTPException(
            422,
            "DeepAgents cannot enforce a response-wide max_output_tokens budget; "
            "configure per-call output limits on the Model Server instead",
        )


def _model_error_type(output: Mapping[str, Any]) -> Optional[str]:
    if not output.get("model_error"):
        return None
    return "ModelTransportError" if output.get("model_transport_error") else "ModelRequestError"


class DeepAgentsAgent(SimpleAgent):
    """Dispatch DeepAgents sessions to the configured sandbox or in-process tool profile.

    The graph cannot export a partial activation, so checkpoint recovery restarts it from the input.
    Close reports the observation gaps of both profiles and the sandbox outcome when applicable.
    """

    checkpoint_mode = "restart_only"
    config: DeepAgentsAgentConfig
    agent: Any = Field(default=None, exclude=True)
    _sem: asyncio.Semaphore = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sem = asyncio.Semaphore(self.config.concurrency)
        if self.config.tool_profile == "tavily_search":
            self.agent = self.build_agent(GymResponsesChatModel(agent=self))

    async def retire_agent_session(self, session_key: str) -> None:
        """Release the graph and sandbox before retiring a checkpoint attempt."""
        await SimpleResponsesAPIAgent.retire_agent_session(self, session_key)

    def build_agent(self, model: BaseChatModel) -> Any:
        """Build the in-process graph once; subclasses may still provide their own graph."""
        model.profile = {"max_input_tokens": self.config.max_input_tokens}
        return create_deep_agent(
            model=model,
            tools=[
                TavilySearch(max_results=self.config.max_search_results, tavily_api_key=self.config.tavily_api_key)
            ],
            system_prompt=self.config.system_prompt,
        )

    async def _seed_agent_session_state(
        self, body: AgentSeedSessionRequest
    ) -> DeepAgentsSessionState | DeepAgentsSandboxSession:
        if self.config.tool_profile == "tavily_search":
            return self._new_session_state(body)
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
        try:
            global_config = get_global_config_dict()
            provider_config = resolve_provider_config(provider_ref, global_config)
            # The runner calls the Model Server from inside the sandbox, which cannot reach a loopback address.
            assert_model_url_reachable_from_sandbox(
                self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key),
                provider_name=next(iter(provider_config), None),
            )
            spec = None
            if access is None:
                sandbox_config = dict(self.config.sandbox_config)
                # Provider TTLs outlive this server; reserve ten minutes for setup and cleanup.
                sandbox_config.setdefault(
                    "ttl_s",
                    self.config.sandbox_install_timeout_seconds + self.config.sandbox_runner_timeout_seconds + 600,
                )
                sandbox_config["metadata"] = {
                    **resolve_provider_metadata(provider_ref, global_config),
                    **(sandbox_config.get("metadata") or {}),
                    "gym-agent": "deepagents",
                    "gym-session": body.agent_session_id[:63],
                    "gym-rollout": body.episode_id.capture_key[:63],
                }
                spec = SandboxSpec(**sandbox_config)
            provider = create_provider(provider_config)
        except (TypeError, ValueError) as error:
            raise AgentSeedRejectedError(str(error)) from error
        try:
            if access is None:
                sandbox = AsyncSandbox(provider)
            else:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
        except BaseException:
            # Cleanup must not replace the original connection/start failure.
            try:
                await provider.aclose()
            except BaseException as cleanup_error:
                LOG.exception("DeepAgents provider cleanup failed after sandbox setup failed")
                if isinstance(cleanup_error, NgFailureEvent):
                    observe_failure(cleanup_error)
                else:
                    cleanup_failure = NgFailureEvent(
                        boundary="sandbox",
                        failure_kind="session_release_failed",
                        failure_reason="DeepAgents provider cleanup failed",
                        operation="aclose",
                        stage="cleanup",
                    )
                    cleanup_failure.__cause__ = cleanup_error
                    observe_failure(cleanup_failure)
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
            if access is None:
                await sandbox.start(spec)
                state.sandbox_id = sandbox.sandbox_id
            await state.install_runtime(timeout=self.config.sandbox_install_timeout_seconds)
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    def _new_session_state(self, body: AgentSeedSessionRequest) -> DeepAgentsSessionState:
        """Accept a session only for what the graph uses: its own tools, in process.

        The graph never calls a granted Resources tool, so a required grant is refused rather than silently
        dropped. An optional grant is skipped. The session holds no Resources cookie jar, so close leaves the
        Environment Server's jar for /verify unchanged. A lent task sandbox is refused: this graph runs in the
        Agent Server, and the builtin profile runs deepagents in the sandbox.
        """
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError(
                "DeepAgents agent runs in process and cannot use a task sandbox; use tool_profile=builtin"
            )
        required = [access.name for access in self.effective_tool_accesses(body) if access.required]
        if required:
            raise AgentSeedRejectedError(
                f"DeepAgents agent does not support required tool access: {', '.join(required)}"
            )
        return DeepAgentsSessionState(
            request=body,
            tool_access=None,
            resources_cookies=None,
            observations=AgentObservationBundle(
                source=OBSERVATION_SOURCE,
                gaps=[
                    ObservationGap(
                        code="agent_transcript_unavailable",
                        detail="the deepagents graph reports no invocation or tool-call records",
                    ),
                    ObservationGap(code="model_call_ownership_unavailable"),
                ],
            ),
        )

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Stop the activation. A close whose activation does not stop in time raises, which keeps the session for the
        Environment Server's close retry."""
        if isinstance(state, DeepAgentsSandboxSession):
            if state.provider_wait_failure is not None:
                observe_failure(state.provider_wait_failure)
            await state.close(self.config.session_close_timeout_seconds)
            return AgentCloseSessionResponse(
                agent_session_id=state.request.agent_session_id, agent_observations=self._observations(state)
            )
        if not isinstance(state, DeepAgentsSessionState):
            raise TypeError("Expected a DeepAgents session state")
        activation = state.activation
        if activation is not None and not activation.done():
            activation.cancel()
            await asyncio.wait({activation}, timeout=self.config.session_close_timeout_seconds)
            if not activation.done():
                raise TimeoutError("DeepAgents activation did not stop within session_close_timeout_seconds")
        return await super()._close_agent_session_state(state)

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
        if native_result.get("model_context_limit"):
            return AgentOutcome(failure_kind=MODEL_CONTEXT_LIMIT)
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
        except TimeoutError as failure:
            # The runner normally stops at the deadline and writes its partial report. If the supervisor overran
            # too, the episode is still a timeout when that report is there, and an infrastructure failure when not.
            output = state.session.artifacts
            if not (isinstance(output, dict) and output.get("stop_reason") == "wall_time"):
                raise
            state.provider_wait_failure = NgFailureEvent(
                boundary="sandbox",
                failure_kind="transport_timeout",
                failure_reason="Sandbox provider wait expired after DeepAgents reported its deadline",
                operation="exec_wait",
                stage="agent",
                foreign_error=sdk_error_details(failure, message=str(failure)),
            )
            state.provider_wait_failure.__cause__ = failure
            observe_failure(state.provider_wait_failure)
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
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        _reject_response_budget(body)
        # Same extraction SimpleAgent.responses() does, including the "unscoped" fallback.
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        model_url_path = self.url_path_for_request("/v1/responses", request)
        agent_session_id = self._agent_session_id_from_request(request)
        if agent_session_id is None:
            if self.config.tool_profile == "builtin":
                raise HTTPException(409, "Sandboxed DeepAgents runs only in agent sessions; use an environment server")
            # Legacy self-dispatch from run(). Different servers have different session-middleware keys
            # (get_session_middleware_key() is f"{class_name}___{server_name}"), so a cookie from the resources
            # server's /seed_session does not automatically ride through to /verify unless re-set here.
            for key, value in request.cookies.items():
                response.set_cookie(key, value)
            return await self._run_graph(body, model_url_path=model_url_path, rollout_id=rollout_id)

        # An activation of a seeded session: the Environment Server holds the Resources cookies and sends them to
        # /verify itself, so nothing is mirrored back. An unknown or closing session answers 409.
        state = self._require_agent_session(agent_session_id)
        if not isinstance(state, (DeepAgentsSessionState, DeepAgentsSandboxSession)):
            raise TypeError("Expected a DeepAgents session state")
        if rollout_id != state.request.episode_id.capture_key:
            raise HTTPException(409, "Activation rollout does not match the seeded agent session")
        if state.activation is None:
            if isinstance(state, DeepAgentsSandboxSession) and body.tools:
                raise HTTPException(
                    422, "Sandboxed DeepAgents runs only deepagents' own tools; the request declares tools"
                )
            # Bound before the first await, so a retry that arrives meanwhile joins this run.
            state.activation_request = body.model_copy(deep=True)

            async def activate() -> NeMoGymResponse:
                if isinstance(state, DeepAgentsSandboxSession):
                    async with self._sem:
                        return await self._run_in_sandbox(body, state)
                return await self._run_graph(body, model_url_path=model_url_path, rollout_id=rollout_id)

            state.activation = asyncio.create_task(activate())
        elif body != state.activation_request:
            raise HTTPException(409, "DeepAgents sessions run one activation; retry the same request")
        # A disconnected HTTP client must not cancel the shared run; only close cancels it.
        return (await asyncio.shield(state.activation)).model_copy(deep=True)

    async def _run_graph(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_url_path: str,
        rollout_id: str | None,
    ) -> NeMoGymResponse:
        if isinstance(body.input, str):
            input_items = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        else:
            input_items = body.input
        input_messages = to_langchain(input_items)

        # `model_cookies` starts empty, not from `request.cookies` (those are resources-server session
        # cookies chained from /seed_session — see the cookie-mirroring loop in responses() — and are unrelated
        # to the model server's own session). `GymResponsesChatModel._agenerate()` fills this holder in from
        # each model response and reads it back on the next internal call within this rollout; see
        # responses_langchain_bridge.py. `model_usage` similarly accumulates usage from every internal
        # model call so the final agent response reports the whole rollout.
        model_usage = {"usage": None}
        run_config: RunnableConfig = {
            "configurable": {
                "model_url_path": model_url_path,
                "model_cookies": {"cookies": None},
                "model_usage": model_usage,
                # deepagents' SummarizationMiddleware offloads evicted history to
                # `/conversation_history/{thread_id}.md`, appending to that same path on each later
                # summarization so a rollout accumulates one running log. Without this key it falls back to
                # a *freshly generated* `session_<uuid>` per summarization event: every eviction lands in a
                # new file, and the pointer it hands the model — under a prompt claiming the full history
                # was saved there — only ever covers the most recent one, orphaning everything older.
                # Safe to share the "unscoped" fallback across rollouts even though `thread_id` is
                # LangGraph's checkpointing key: the graph is compiled without a checkpointer, and the
                # default StateBackend keeps its virtual filesystem in per-`ainvoke` graph state.
                "thread_id": rollout_id or "unscoped",
                # Forwarded verbatim to every internal model call (see GymResponsesChatModel._agenerate()),
                # so a caller requesting e.g. {"summary": "auto"} gets reasoning summaries back on every
                # turn — unlike SimpleAgent, which forwards this for free via body.model_copy(), this agent
                # rebuilds each internal request from scratch and has to thread it through explicitly.
                # `body.reasoning` is already a plain dict at runtime (its field type, Reasoning, is a
                # TypedDict, not a BaseModel — verified via NeMoGymResponseCreateParamsNonStreaming; no
                # .model_dump() needed, and calling one would raise AttributeError on a plain dict).
                "model_reasoning": body.reasoning,
            }
        }
        final_state = await self.agent.ainvoke({"messages": input_messages}, config=run_config)

        new_messages = final_state["messages"][len(input_messages) :]
        response_body = to_responses(new_messages, self.config.model_server.name)
        response_body["usage"] = model_usage["usage"]
        return NeMoGymResponse.model_validate(response_body)

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        """Keep legacy /run for the search profile; sandbox episodes require an Environment Server."""
        if self.config.tool_profile == "builtin":
            raise HTTPException(409, "Sandboxed DeepAgents runs only in agent sessions; use an environment server")
        _reject_response_budget(body.responses_create_params)
        return await super().run(request, body)


if __name__ == "__main__":
    DeepAgentsAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = DeepAgentsAgent.run_webserver()
