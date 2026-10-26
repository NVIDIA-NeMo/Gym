# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stirrup agent that runs each episode inside the task sandbox it borrows from the environment, or one it starts.

The agent server installs Stirrup in the sandbox, stages ``sandbox_runner.py`` and runs it under Gym's process
supervisor. Shell commands run in the sandbox; task tools are the resources server's routes, reached over HTTP.
"""

import asyncio
import time
from asyncio import Semaphore
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, model_validator

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.agent_tools import sandbox_server_url
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers import create_provider
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPToolAccess
from responses_api_agents.stirrup_agent.sandbox import StirrupSessionState
from responses_api_agents.stirrup_agent.stirrup_utils import messages_from_input


def _usage(usages: list[dict[str, Any]]) -> Optional[NeMoGymResponseUsage]:
    """Sum the chat-completions usage of the episode's model calls; calls that reported none are left out."""
    total = None
    for usage in usages:
        total = accumulate_response_usage(
            total,
            NeMoGymResponseUsage(
                input_tokens=usage["prompt_tokens"],
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                ),
                output_tokens=usage["completion_tokens"],
                output_tokens_details=NeMoGymResponseOutputTokensDetails(
                    reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                ),
                total_tokens=usage["total_tokens"],
            ),
        )
    return total


def _zero_usage() -> NeMoGymResponseUsage:
    """The usage of an episode whose only model request was rejected, so nothing was processed."""
    return NeMoGymResponseUsage(
        input_tokens=0,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=None),
        output_tokens=0,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=None),
        total_tokens=0,
    )


def _observations(output: Optional[dict[str, Any]], model_server: ModelServerRef) -> AgentObservationBundle:
    """Validate the evidence the sandbox runner reported into an observation bundle.

    Each part is validated on its own, so a part that fails becomes an ``observation_capture_failed`` gap naming
    it and the rest is kept.
    """
    raw = (output or {}).get("observations")
    if not isinstance(raw, dict):
        gap = ObservationGap(code="observation_capture_failed", detail="the runner reported no observations")
        return AgentObservationBundle(source="stirrup", gaps=[gap])
    records, gaps = [], []

    def failed(part: str, error: Exception) -> None:
        detail = f"{part}: {type(error).__name__}"
        gaps.append(
            ObservationGap(code="observation_capture_failed", invocation_id=raw.get("invocation_id"), detail=detail)
        )

    try:
        invocation = AgentInvocation(
            invocation_id=raw["invocation_id"],
            status=raw["status"],
            # Every call goes to the configured model server, so a response ID identifies the call.
            model_calls=[ModelCallRef(model_ref=model_server, response_id=r) for r in raw["model_response_ids"]],
        )
    except Exception as error:
        failed("invocation", error)
    else:
        try:
            conversation = output.get("input_items", []) + output.get("output_items", [])
            invocation = AgentInvocation.model_validate({**invocation.model_dump(), "conversation": conversation})
        except Exception as error:
            failed("conversation", error)
        records.append(invocation)
    for index, tool in enumerate(raw.get("tool_calls") or []):
        try:
            records.append(ToolCallObservation.model_validate(tool))
        except Exception as error:
            failed(f"tool_calls[{index}]", error)
    return AgentObservationBundle(source="stirrup", records=records, gaps=gaps)


_REMOVED_OPTIONS = [
    "judge_only",
    "execute_only",
    "rerun_incomplete",
    "persist_deliverables_dir",
    "tavily_api_key",
    "tavily_max_sweeps",
    "gdpval_container_path",
    "system_prompt_template",
    "user_prompt_template",
    "model_id",
    "task",
]


class StirrupAgentWrapperConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_options(cls, data: Any) -> Any:
        removed = [f"`{name}`" for name in _REMOVED_OPTIONS if name in (data or {})]
        if removed:
            raise ValueError(
                f"Stirrup config includes removed options that are no longer supported: {', '.join(removed)}\n"
                "See migration guide for details: benchmarks/gdpval/README.md"
            )
        return data

    agent_max_turns: int = Field(default=250, description="Maximum turns for the Stirrup agent")
    concurrency: int = Field(default=32, description="Maximum concurrent runs")
    temperature: float = Field(default=0.6, description="Sampling temperature for the agent model")

    completion_token_buffer: int = Field(
        default=1000,
        description="Token budget reserved on top of input_tokens when computing per-call "
        "max_completion_tokens. Absorbs the residual gap between our estimate "
        "(messages + tool-schema JSON) and the exact prompt the server sees after chat-template "
        "rendering. See ``nemo_client.DynamicMaxTokensChatCompletionsClient``.",
    )
    context_window_tokens: int = Field(
        default=262144,
        ge=1,
        description="Model context-window size used for dynamic per-call completion budgeting. "
        "This is independent of the Responses API request's max_output_tokens, which limits "
        "output rather than describing model capacity.",
    )
    top_p: float = Field(
        default=0.95,
        description="Top-p sampling cutoff for the policy model. Forwarded to the LLM client.",
    )
    enable_thinking: bool = Field(
        default=True,
        description="Whether to enable reasoning tokens (sets "
        "``extra_body.chat_template_kwargs.enable_thinking``). Reasoning-trained models default to True.",
    )
    max_completion_tokens_cap: int = Field(
        default=64000,
        ge=1,
        description="Hard ceiling on per-call ``max_completion_tokens``. Dynamic sizing computes "
        "context_window - input_tokens - completion_token_buffer, then caps to this value. "
        "Set to match the training-side response-length budget for RL.",
    )
    min_completion_tokens: int = Field(
        default=1024,
        ge=1,
        description="Target minimum per-call ``max_completion_tokens``. The hard cap always applies; "
        "the approximate estimate preserves this floor even when it exceeds context. "
        "Long-horizon benchmarks can opt into a larger usable floor.",
    )
    prompt_estimator_truncate_history_thinking: Optional[bool] = Field(
        default=None,
        description="Optional prompt-estimator setting for checkpoints whose template omits historical "
        "reasoning before the last user turn. This is never forwarded to the model request.",
    )
    truncation_recovery: bool = Field(
        default=False,
        description="When a model call spends its entire completion budget without emitting a "
        "tool call, run the next call with thinking disabled plus a transient instruction not to "
        "restart the analysis. Targets a measured failure mode where the model loops on unbounded "
        "reasoning: on a 200-task GDPVal run these turns burned 15.8% of all model time. The token "
        "budget is deliberately NOT reduced, because recovery turns are usually large single-shot "
        "deliverable writes (median 34.6k tokens). The instruction is sent to the server but not "
        "recorded in the trajectory, so it is off by default to keep rollouts unsteered; the GDPVal "
        "benchmark config opts in.",
    )
    min_compaction_summary_words: int = Field(
        default=1,
        ge=1,
        description="Minimum whitespace-delimited word count accepted from context compaction. "
        "The generic default rejects only empty output; long-horizon benchmarks can require a "
        "more complete summary.",
    )
    finish_tool_names: Optional[List[str]] = Field(
        default=None,
        description="Request tools that end the episode when they succeed. None uses Stirrup's own finish tool.",
    )
    code_exec: bool = Field(
        default=True,
        description="Give the model the code_exec shell tool. Off for tasks whose only tools are the request's.",
    )
    sandbox_provider: Optional[str] = Field(
        default=None, description="Sandbox provider config to start a sandbox from when the session lends none."
    )
    sandbox_config: Dict[str, Any] = Field(
        default_factory=dict, description="SandboxSpec fields for a sandbox started from sandbox_provider."
    )
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    sandbox_runner_timeout_seconds: float = Field(default=12600, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)


class StirrupAgentWrapper(SimpleResponsesAPIAgent):
    config: StirrupAgentWrapperConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> StirrupSessionState:
        owns_sandbox = body.sandbox_access is None
        if owns_sandbox:
            if self.config.sandbox_provider is None:
                raise HTTPException(422, "Stirrup requires sandbox_access or a configured sandbox_provider")
            provider_ref = self.config.sandbox_provider
            workdir = self.config.sandbox_config.get("workdir")
        else:
            connection = body.sandbox_access.connection
            if not isinstance(connection, DirectSandboxConnection):
                raise HTTPException(422, "Stirrup supports only direct sandbox connections")
            provider_ref = connection.provider_config_ref
            workdir = body.sandbox_access.workdir
        accesses = self.effective_tool_accesses(body)
        unsupported = [access.name for access in accesses if isinstance(access, MCPToolAccess) and access.required]
        if unsupported:
            raise HTTPException(422, f"Stirrup does not support required MCP tool access: {', '.join(unsupported)}")
        direct_accesses = [access for access in accesses if isinstance(access, DirectHTTPToolAccess)]
        if len(direct_accesses) > 1:
            raise HTTPException(422, "Stirrup supports at most one direct HTTP tool access")
        tool_access = direct_accesses[0] if direct_accesses else None

        provider = create_provider(resolve_provider_config(provider_ref, get_global_config_dict()))
        try:
            if provider.name != "local":
                # The runner calls the model server from the sandbox, which cannot reach a loopback address.
                sandbox_server_url(self.config.model_server.name, require_reachable=True)
            if owns_sandbox:
                sandbox = AsyncSandbox(provider)
                sandbox_config = self.config.sandbox_config.copy()
                # Provider TTLs outlive this server; reserve ten minutes for setup/cleanup overhead.
                sandbox_config.setdefault(
                    "ttl_s",
                    self.config.sandbox_install_timeout_seconds + self.config.sandbox_runner_timeout_seconds + 600,
                )
                await sandbox.start(SandboxSpec(**sandbox_config))
            else:
                sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        state = StirrupSessionState(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                workdir=workdir,
                session_dir=f"/tmp/nemo-gym-stirrup-sessions/{uuid4().hex}",
                owns_sandbox=owns_sandbox,
                harness="Stirrup",
            ),
            tool_access=tool_access,
            resources_cookies=dict(tool_access.cookies) if tool_access else {},
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
        if not isinstance(state, StirrupSessionState):
            raise TypeError("Expected Stirrup session state")
        await state.close(self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=_observations(state.session.artifacts, self.config.model_server),
            resources_cookies=state.resources_cookies,
        )

    async def _run_sandbox_episode(
        self, body: NeMoGymResponseCreateParamsNonStreaming, state: StirrupSessionState
    ) -> NeMoGymResponse:
        if body.tools and state.tool_access is None:
            raise HTTPException(422, "The request has tools, but the session grants no tool access to call them")
        params = body.model_dump(mode="json")
        # Checked here, before the runner starts, so a request that can never run is a terminal 422 and not a
        # retryable 500 that re-seeds a sandbox and installs the runtime again.
        try:
            messages_from_input(params["input"], params["instructions"])
        except (ValueError, NotImplementedError) as error:
            raise HTTPException(422, str(error)) from error
        missing = set(self.config.finish_tool_names or []) - {tool["name"] for tool in params["tools"] or []}
        if missing:
            raise HTTPException(422, f"Finish tools {sorted(missing)} are not among the request's tools")
        max_completion_tokens_cap = self.config.max_completion_tokens_cap
        if body.max_output_tokens is not None:
            max_completion_tokens_cap = min(max_completion_tokens_cap, body.max_output_tokens)
        model_name = body.model or "default"
        payload = {
            "input": params["input"],
            "instructions": params["instructions"],
            "tools": params["tools"],
            "finish_tool_names": self.config.finish_tool_names,
            "code_exec": self.config.code_exec,
            "tool_access": state.tool_access.model_dump(mode="json") if state.tool_access else None,
            "workdir": state.session.workdir,
            "stop_request_path": state.session.stop_request_path,
            "max_turns": self.config.agent_max_turns,
            "min_compaction_summary_words": self.config.min_compaction_summary_words,
            "client": {
                "model": model_name,
                # The sandbox reaches the Model Server directly; the rollout prefix keeps its calls correlated.
                "base_url": self.resolve_model_base_url(
                    self.config.model_server.name, state.request.episode_id.capture_key
                ),
                "max_tokens": self.config.context_window_tokens,
                "completion_token_buffer": self.config.completion_token_buffer,
                "temperature": self.config.temperature if body.temperature is None else body.temperature,
                "top_p": self.config.top_p if body.top_p is None else body.top_p,
                "enable_thinking": self.config.enable_thinking,
                "max_completion_tokens_cap": max_completion_tokens_cap,
                "min_completion_tokens": self.config.min_completion_tokens,
                "prompt_estimator_truncate_history_thinking": self.config.prompt_estimator_truncate_history_thinking,
                "truncation_recovery": self.config.truncation_recovery,
            },
        }
        try:
            output = await state.execute(
                payload,
                timeout=self.config.sandbox_runner_timeout_seconds,
                close_timeout=self.config.session_close_timeout_seconds,
            )
        except TimeoutError:
            # The runner normally stops at the deadline and writes its partial report. If the supervisor overran
            # too, the episode is still a timeout when that report is there, and an infrastructure failure when not.
            artifacts = state.session.artifacts
            if not (isinstance(artifacts, dict) and artifacts.get("stop_reason") == "wall_time"):
                raise
            state.resources_cookies = artifacts["resources_cookies"]
            output = artifacts
        usage = _usage(output["usages"])
        if usage is None and output.get("model_error") and not output["observations"]["model_response_ids"]:
            # The only request was rejected before any processing: a known zero. A request killed in flight may
            # have used tokens, so that case stays unknown.
            usage = _zero_usage()
        return NeMoGymResponse(
            id=f"stirrup-{state.request.episode_id.capture_key}",
            created_at=int(time.time()),
            model=model_name,
            object="response",
            output=output["input_items"] + output["output_items"],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            metadata={
                "elapsed_seconds": str(output["elapsed_seconds"]),
                **({"stop_reason": output["stop_reason"]} if output["stop_reason"] else {}),
            },
            status=output["observations"]["status"],
            error={"code": "server_error", "message": output["model_error"]} if output.get("model_error") else None,
            usage=usage,
        )

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        agent_session_id = self._agent_session_id_from_request(request)
        if agent_session_id is None:
            raise HTTPException(409, "Stirrup runs only in agent sessions; use an environment server")
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        state = self._require_agent_session(agent_session_id)
        if state.request.episode_id.capture_key != rollout_id:
            raise HTTPException(409, "Agent-session episode_id does not match the rollout route")
        if state.task is None:
            # No await until the task and its request binding are installed.
            state.activation_request = body.model_copy(deep=True)

            async def activate() -> NeMoGymResponse:
                async with self.sem:
                    return await self._run_sandbox_episode(body, state)

            state.task = asyncio.create_task(activate())
        elif body != state.activation_request:
            raise HTTPException(409, "Stirrup sandbox sessions support one activation; retry the same request")
        # Session close owns cancellation; a disconnected waiter must not cancel the activation.
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def run(self, request: Request, body: BaseRunRequest = Body()):
        raise HTTPException(409, "Stirrup runs only in agent sessions; use an environment server")


if __name__ == "__main__":
    StirrupAgentWrapper.run_webserver()
