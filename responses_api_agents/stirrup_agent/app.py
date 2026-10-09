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
"""Stirrup agent that runs each episode inside the task sandbox it borrows from the environment.

The agent server installs Stirrup in the sandbox, stages ``sandbox_runner.py`` and runs it under Gym's process
supervisor. Shell commands run in the sandbox; task tools are the resources server's routes, reached over HTTP.
"""

import asyncio
import time
from asyncio import Semaphore
from typing import Any, List, Mapping, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field

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
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.agent_tools import sandbox_server_url
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers import create_provider
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPToolAccess
from responses_api_agents.stirrup_agent.sandbox import StirrupSessionState


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


class StirrupAgentWrapperConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef

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
        if body.sandbox_access is None:
            raise ValueError("Stirrup requires sandbox access")
        connection = body.sandbox_access.connection
        if not isinstance(connection, DirectSandboxConnection):
            raise ValueError("Stirrup supports only direct sandbox connections")
        accesses = self.effective_tool_accesses(body)
        unsupported = [access.name for access in accesses if isinstance(access, MCPToolAccess) and access.required]
        if unsupported:
            raise ValueError(f"Stirrup does not support required MCP tool access: {', '.join(unsupported)}")
        direct_accesses = [access for access in accesses if isinstance(access, DirectHTTPToolAccess)]
        if len(direct_accesses) != 1:
            raise ValueError("Stirrup requires exactly one direct HTTP tool access")

        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            if provider.name != "local":
                # The runner calls the model server from the sandbox, which cannot reach a loopback address.
                sandbox_server_url(self.config.model_server.name, require_reachable=True)
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        state = StirrupSessionState(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                workdir=body.sandbox_access.workdir,
                session_dir=f"/tmp/nemo-gym-stirrup-sessions/{uuid4().hex}",
                harness="Stirrup",
            ),
            tool_access=direct_accesses[0],
            resources_cookies=dict(direct_accesses[0].cookies),
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
        max_completion_tokens_cap = self.config.max_completion_tokens_cap
        if body.max_output_tokens is not None:
            max_completion_tokens_cap = min(max_completion_tokens_cap, body.max_output_tokens)
        model_name = body.model or "default"
        params = body.model_dump(mode="json")
        payload = {
            "input": params["input"],
            "instructions": params["instructions"],
            "tools": params["tools"],
            "finish_tool_names": self.config.finish_tool_names,
            "tool_access": state.tool_access.model_dump(mode="json"),
            "workdir": state.session.workdir,
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
                "temperature": body.temperature or self.config.temperature,
                "top_p": body.top_p or self.config.top_p,
                "enable_thinking": self.config.enable_thinking,
                "max_completion_tokens_cap": max_completion_tokens_cap,
                "min_completion_tokens": self.config.min_completion_tokens,
                "prompt_estimator_truncate_history_thinking": self.config.prompt_estimator_truncate_history_thinking,
                "truncation_recovery": self.config.truncation_recovery,
            },
        }
        output = await state.execute(
            payload,
            timeout=self.config.sandbox_runner_timeout_seconds,
            close_timeout=self.config.session_close_timeout_seconds,
        )
        return NeMoGymResponse(
            id=f"stirrup-{state.request.episode_id.capture_key}",
            created_at=int(time.time()),
            model=model_name,
            object="response",
            output=output["input_items"] + output["output_items"],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            metadata={"elapsed_seconds": str(output["elapsed_seconds"])},
            usage=_usage(output["usages"]),
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
