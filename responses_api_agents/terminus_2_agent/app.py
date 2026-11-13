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

"""Run Harbor's Terminus-2 loop as a NeMo Gym agent server.

Agent sessions run Terminus-2 inside the task sandbox; requests without a session run it on this host.
"""

import asyncio
import json
import logging
import shutil
import tempfile
from asyncio import Semaphore
from collections.abc import Mapping
from pathlib import Path
from time import monotonic, time
from typing import Any, ClassVar, Literal, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from harbor.models.agent.context import AgentContext
from openai.types.responses import ResponseError
from pydantic import ConfigDict, Field, field_validator

from nemo_gym.agents import (
    AGENT_TIMEOUT_METADATA_KEY,
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
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef, get_global_config_dict
from nemo_gym.core import (
    AGENT_INVALID_OUTPUT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
    NgFailureEvent,
    observe_failure,
    sdk_error_details,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    SandboxObservation,
    ToolCallObservation,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.terminus_2_agent.llm import NemoGymLLM, chat_completion_params
from responses_api_agents.terminus_2_agent.output import trajectory_to_responses
from responses_api_agents.terminus_2_agent.sandbox import Terminus2SandboxSession
from responses_api_agents.terminus_2_agent.terminus import LocalEnvironment, StandaloneTerminus2


LOG = logging.getLogger(__name__)

# How the sandbox runner's stop reasons end the response and the invocation; any other ending is a failure.
_STATUS_BY_STOP_REASON: dict[str, Literal["completed", "incomplete"]] = {
    "completed": "completed",
    "max_turns": "incomplete",
    "wall_time": "incomplete",
    "cancelled": "incomplete",
    "terminal_exited": "incomplete",
    # A SIGTERM from inside the task, such as `pkill python3`, ended the runner; like an exited terminal, the task's
    # own commands ended the run.
    "terminated": "incomplete",
}
_SANDBOX_OUTCOME_BY_STOP_REASON = {"wall_time": "timeout", "cancelled": "cancelled"}


def _message_text(item: Any) -> str:
    content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") for part in content)


def _extract_instruction(body_input: Any) -> tuple[str, Optional[str]]:
    """Flatten Responses input into Terminus-2's single instruction."""
    if isinstance(body_input, str):
        return body_input, None

    messages: list[tuple[str, str]] = []
    for item in body_input or []:
        role = getattr(item, "role", None) if not isinstance(item, dict) else item.get("role")
        text = _message_text(item)
        if role and text:
            messages.append((role, text))

    conversation = [(role, text) for role, text in messages if role in {"user", "assistant"}]
    if len(conversation) <= 1 and (not conversation or conversation[0][0] == "user"):
        user_message = conversation[0][1] if conversation else ""
        system_text = "\n\n".join(text for role, text in messages if role in {"system", "developer"})
        return user_message, system_text or None
    return "\n\n".join(f"{role.title()}: {text}" for role, text in messages), None


class Terminus2AgentConfig(BaseResponsesAPIAgentConfig):
    # Only the /run path seeds and verifies through a resources server; agent sessions get their task, and
    # usually their sandbox, from the environment server that drives them.
    resources_server: ResourcesServerRef | None = None
    model_server: ModelServerRef
    concurrency: int = 1
    model: Optional[str] = None
    workspace_root: Optional[str] = None
    system_prompt: Optional[str] = None
    max_turns: Optional[int] = None
    parser_name: Literal["json", "xml"] = "json"
    temperature: float = 0.7
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "default"]] = None
    collect_rollout_details: bool = False
    enable_summarize: bool = True
    proactive_summarization_threshold: int = 8000
    max_thinking_tokens: Optional[int] = None
    model_info: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_input_tokens": 262144,
            "max_output_tokens": 81920,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        }
    )
    trajectory_config: dict[str, Any] = Field(default_factory=lambda: {"raw_content": False})
    tmux_pane_width: int = 160
    tmux_pane_height: int = 40
    store_all_messages: bool = False
    record_terminal_session: bool = False
    interleaved_thinking: bool = False
    model_timeout_sec: float = Field(default=2400, gt=0, allow_inf_nan=False)
    command_timeout_sec: float = Field(default=1800, gt=0, allow_inf_nan=False)
    # The run's wall-clock budget; a task's own agent timeout lowers it for an agent session.
    timeout: float = Field(default=10800, gt=0, allow_inf_nan=False)
    keep_logs: bool = False
    logs_root: str = "outputs/terminus_2_agent/runs"
    # Agent sessions without sandbox_access start a sandbox from this provider block and SandboxSpec fields.
    sandbox_provider: str | None = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)

    @field_validator("reasoning_effort", "max_thinking_tokens")
    @classmethod
    def _refuse_unapplied_settings(cls, value: Any, info: Any) -> Any:
        # Harbor applies these only to the LiteLLM client it builds itself; Gym's Model Server client never sees them.
        if value is not None:
            raise ValueError(
                f"Terminus 2 cannot apply {info.field_name}; leave it null and set the request's reasoning or the "
                "Model Server's settings instead"
            )
        return value


class Terminus2AgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class Terminus2AgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    agent_timeout_error: int = 0
    context_length_exceeded_error: int = 0


class Terminus2Agent(SimpleResponsesAPIAgent):
    """NeMo Gym Responses API wrapper for the Terminus-2 terminal loop."""

    config: Terminus2AgentConfig
    # Agent sessions run Terminus inside the task sandbox, which calls the model server from there.
    calls_model_from_sandbox: ClassVar[bool] = True
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.sem = Semaphore(self.config.concurrency)

    def _logs_dir(self) -> Path:
        if not self.config.keep_logs:
            return Path(tempfile.mkdtemp(prefix="nemo-gym-terminus-2-"))
        root = Path(self.config.logs_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        path = root / f"run-{int(time())}-{uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _model_name(self, body: NeMoGymResponseCreateParamsNonStreaming) -> str:
        global_model = getattr(self.server_client, "global_config_dict", {}).get("policy_model_name")
        return self.config.model or body.model or global_model or "model"

    def _rollout_id(self, request: Optional[Request]) -> Optional[str]:
        path_params = getattr(request, "path_params", None)
        return path_params.get("rollout_id") if isinstance(path_params, Mapping) else None

    @staticmethod
    def _validate_output_budget(body: NeMoGymResponseCreateParamsNonStreaming) -> None:
        """Refuse an aggregate budget neither supported execution mode can enforce."""
        if body.max_output_tokens is not None:
            raise HTTPException(
                422,
                "Terminus 2 cannot enforce request max_output_tokens across model calls; "
                "omit it and configure a per-call limit on the Model Server instead",
            )

    def _prepare_request(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> tuple[NeMoGymResponseCreateParamsNonStreaming, str]:
        """Apply the configured temperature and flatten the instructions and input into Terminus-2's instruction."""
        self._validate_output_budget(body)
        body = body.model_copy(deep=True)
        if body.temperature is None:
            body.temperature = self.config.temperature
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_instruction, input_system = _extract_instruction(body.input)
        instruction_parts = [
            part for part in (self.config.system_prompt, body.instructions, input_system, user_instruction) if part
        ]
        return body, "\n\n".join(instruction_parts)

    def _terminus_settings(self, body: NeMoGymResponseCreateParamsNonStreaming) -> dict[str, Any]:
        """Terminus-2's constructor arguments that come from configuration, shared by host and sandbox runs."""
        return {
            "max_turns": self.config.max_turns,
            "parser_name": self.config.parser_name,
            "temperature": body.temperature if body.temperature is not None else self.config.temperature,
            "reasoning_effort": self.config.reasoning_effort,
            "collect_rollout_details": self.config.collect_rollout_details,
            "enable_summarize": self.config.enable_summarize,
            "proactive_summarization_threshold": self.config.proactive_summarization_threshold,
            "max_thinking_tokens": self.config.max_thinking_tokens,
            "model_info": self.config.model_info,
            "trajectory_config": self.config.trajectory_config,
            "tmux_pane_width": self.config.tmux_pane_width,
            "tmux_pane_height": self.config.tmux_pane_height,
            "store_all_messages": self.config.store_all_messages,
            "interleaved_thinking": self.config.interleaved_thinking,
        }

    def _build_agent(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        logs_dir: Path,
        api_base: str,
    ) -> StandaloneTerminus2:
        model_name = self._model_name(body)
        llm = NemoGymLLM(
            model_name=model_name,
            api_base=api_base,
            collect_rollout_details=self.config.collect_rollout_details,
            model_info=self.config.model_info,
            responses_create_params=body.model_dump(exclude_none=True),
            timeout_sec=self.config.model_timeout_sec,
        )
        return StandaloneTerminus2(
            logs_dir=logs_dir,
            model_name=model_name,
            api_base=api_base,
            record_terminal_session=self.config.record_terminal_session,
            llm=llm,
            **self._terminus_settings(body),
        )

    async def _run_terminus(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        instruction: str,
        api_base: str,
    ) -> tuple[dict[str, Any], AgentContext, dict[str, bool], bool, bool, str | None, dict[str, Any]]:
        """Run Terminus on the Gym host; the last item is the model-usage accounting of ``NemoGymLLM``."""
        logs_dir = self._logs_dir()
        temporary_workspace: Optional[Path] = None
        try:
            if self.config.workspace_root:
                workspace = Path(self.config.workspace_root).expanduser().resolve()
                if not workspace.is_dir():
                    raise FileNotFoundError(f"Terminus-2 workspace does not exist: {workspace}")
            else:
                workspace = temporary_workspace = Path(tempfile.mkdtemp(prefix="nemo-gym-terminus-2-workspace-"))

            environment = LocalEnvironment(workspace, self.config.command_timeout_sec)
            context = AgentContext()
            agent = self._build_agent(body, logs_dir, api_base)
            timed_out = False
            try:
                await agent.setup(environment)
                try:
                    await asyncio.wait_for(agent.run(instruction, environment, context), timeout=self.config.timeout)
                except asyncio.TimeoutError:
                    timed_out = True
                    LOG.warning("Terminus-2 timed out after %gs", self.config.timeout)
            finally:
                await agent.close(environment)

            trajectory_path = logs_dir / "trajectory.json"
            trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else {"steps": []}
            flags = {"context_length_exceeded": agent.context_length_exceeded}
            llm = getattr(agent, "_llm", None)
            terminal_response_id = llm.last_response_id if isinstance(llm, NemoGymLLM) else None
            accounting = {
                # A call cancelled by the timeout stays counted, so its unknown usage keeps the total unknown.
                "calls_without_usage": llm.calls_without_usage if isinstance(llm, NemoGymLLM) else None,
                "cached_usage_known": llm.cached_usage_known if isinstance(llm, NemoGymLLM) else False,
            }
            return trajectory, context, flags, timed_out, agent.finished_naturally, terminal_response_id, accounting
        finally:
            if not self.config.keep_logs:
                shutil.rmtree(logs_dir, ignore_errors=True)
            if temporary_workspace is not None:
                shutil.rmtree(temporary_workspace, ignore_errors=True)

    def _response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        trajectory: dict[str, Any],
        context: AgentContext,
        flags: dict[str, bool],
        timed_out: bool,
        finished_naturally: bool,
        terminal_response_id: str | None,
        usage_known: bool,
        cached_usage_known: bool,
    ) -> NeMoGymResponse:
        """Convert a Terminus-2 trajectory and its context into a Responses API result.

        Usage is reported only when every model call had a usage record. Harbor keeps no reasoning-token counts, and
        cache counts only when every response reported them; neither becomes a measured zero.
        """
        output_items = trajectory_to_responses(trajectory)
        if not output_items:
            output_items = [
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                ).model_dump()
            ]

        final_metrics = trajectory.get("final_metrics") or {}
        input_tokens = final_metrics.get("total_prompt_tokens", 0)
        output_tokens = final_metrics.get("total_completion_tokens", 0)
        cached_tokens = final_metrics.get("total_cached_tokens", 0)
        if input_tokens == 0 and output_tokens == 0:
            input_tokens = context.n_input_tokens or 0
            output_tokens = context.n_output_tokens or 0
            cached_tokens = context.n_cache_tokens or 0

        return NeMoGymResponse(
            id=terminal_response_id or f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self._model_name(body),
            object="response",
            output=output_items,
            parallel_tool_calls=False,
            temperature=body.temperature,
            top_p=body.top_p,
            tool_choice=body.tool_choice,
            tools=body.tools,
            background=False,
            reasoning={"effort": None, "generate_summary": None, "summary": None},
            service_tier="default",
            status="completed",
            text={"format": {"type": "text"}, "verbosity": "medium"},
            top_logprobs=0,
            truncation="disabled",
            store=True,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cached_tokens=cached_tokens if cached_usage_known else None
                ),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=None),
                total_tokens=input_tokens + output_tokens,
            )
            if usage_known
            else None,
            metadata={
                "terminus_2": json.dumps(
                    {
                        "context": context.model_dump(mode="json"),
                        "agent_error_flags": flags,
                        "timed_out": timed_out,
                        "finished_naturally": finished_naturally,
                    }
                )
            },
        )

    # ----- Agent sessions: Terminus-2 in a borrowed or owned task sandbox -----

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> Terminus2SandboxSession:
        """Connect to or start the episode's sandbox and prepare Terminus-2 in it before activation."""
        required = [access.name for access in self.effective_tool_accesses(body) if access.required]
        if required:
            raise AgentSeedRejectedError(
                "Terminus 2 runs only its terminal; required tool grants it cannot use: " + ", ".join(sorted(required))
            )
        owns_sandbox = body.sandbox_access is None
        if owns_sandbox:
            if self.config.sandbox_provider is None:
                raise AgentSeedRejectedError(
                    "Terminus 2 needs sandbox_access or a configured sandbox_provider for an agent session"
                )
            provider_ref = self.config.sandbox_provider
            workdir = self.config.sandbox_config.get("workdir")
        else:
            connection = body.sandbox_access.connection
            if not isinstance(connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("Terminus 2 supports only direct sandbox connections")
            provider_ref = connection.provider_config_ref
            workdir = body.sandbox_access.workdir
        if not isinstance(workdir, str) or not workdir.startswith("/"):
            raise AgentSeedRejectedError(f"Terminus 2 needs an absolute sandbox workdir, got {workdir!r}")

        global_config = get_global_config_dict()
        try:
            provider_config = resolve_provider_config(provider_ref, global_config)
            # The runner inside the sandbox calls the Model Server itself, so a loopback URL (the default
            # without use_absolute_ip) would make every model call fail.
            assert_model_url_reachable_from_sandbox(
                self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key),
                provider_name=next(iter(provider_config), None),
            )
            spec = None
            if owns_sandbox:
                sandbox_config = self.config.sandbox_config.copy()
                sandbox_config["metadata"] = {
                    **resolve_provider_metadata(provider_ref, global_config),
                    **(sandbox_config.get("metadata") or {}),
                    "gym-agent": "terminus_2",
                    "gym-session": body.agent_session_id[:63],
                    "gym-rollout": body.episode_id.capture_key[:63],
                }
                # Provider TTLs outlive this server; reserve both setup steps (runtime and tmux install), the run,
                # and ten minutes for cleanup.
                sandbox_config.setdefault(
                    "ttl_s", 2 * self.config.sandbox_install_timeout_seconds + self.config.timeout + 600
                )
                spec = SandboxSpec(**sandbox_config)
            provider = create_provider(provider_config)
        except (TypeError, ValueError) as error:
            raise AgentSeedRejectedError(str(error)) from error
        try:
            if owns_sandbox:
                sandbox = AsyncSandbox(provider)
                await sandbox.start(spec)
            else:
                sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            try:
                await provider.aclose()
            except BaseException as cleanup_error:
                LOG.exception("Terminus 2 provider cleanup failed after sandbox setup failed")
                if isinstance(cleanup_error, NgFailureEvent):
                    observe_failure(cleanup_error)
                else:
                    cleanup_failure = NgFailureEvent(
                        boundary="sandbox",
                        failure_kind="session_release_failed",
                        failure_reason="Terminus 2 provider cleanup failed",
                        operation="aclose",
                        stage="cleanup",
                    )
                    cleanup_failure.__cause__ = cleanup_error
                    observe_failure(cleanup_failure)
            raise

        state = Terminus2SandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                workdir=workdir,
                session_dir=f"/tmp/nemo-gym-terminus-2-sessions/{uuid4().hex}",
                owns_sandbox=owns_sandbox,
                harness="Terminus 2",
            ),
            sandbox_id=sandbox.sandbox_id,
            provider=provider_ref,
        )
        try:
            await state.install_runtime(install_timeout=self.config.sandbox_install_timeout_seconds)
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                LOG.exception(
                    "Could not clean failed Terminus 2 setup %s; retaining session for close", body.agent_session_id
                )
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    def _require_agent_session(self, agent_session_id: str) -> Terminus2SandboxSession:
        state = super()._require_agent_session(agent_session_id)
        if not isinstance(state, Terminus2SandboxSession):
            raise TypeError("Expected a Terminus 2 agent session state")
        return state

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, Terminus2SandboxSession) else None

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, Terminus2SandboxSession):
            raise TypeError("Expected a Terminus 2 agent session state")
        if state.provider_wait_failure is not None:
            observe_failure(state.provider_wait_failure)
        await state.close(self.config.session_close_timeout_seconds)
        observations = state.observations
        if observations is None and isinstance(state.session.artifacts, dict):
            # Cancellation can bypass response conversion; the runner's result was still captured.
            observations = self._observations(state, state.session.artifacts)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=observations
            or AgentObservationBundle(source="terminus2", gaps=[ObservationGap(code="observation_capture_failed")]),
        )

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Read the error kind the sandbox runner recorded for a run that ended early."""
        kind = native_result.get("error_kind")
        status = native_result.get("error_status")
        if kind == "model_request_error" and isinstance(status, int) and 400 <= status <= 599:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=status)
        if kind == "model_transport_error":
            return AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
        if kind == "output_limit":
            return AgentOutcome(failure_kind=AGENT_INVALID_OUTPUT)
        return AgentOutcome(failure_kind="unknown")

    @staticmethod
    def _stop_reason(output: Mapping[str, Any]) -> str:
        return (
            "context_limit"
            if output.get("error_kind") == "context_limit"
            else str(output.get("stop_reason") or "error")
        )

    @staticmethod
    def _status(output: Mapping[str, Any]) -> Literal["completed", "incomplete", "failed"]:
        # A context overflow that summarization could not resolve is a limit of the run, graded like a timeout.
        if output.get("error_kind") == "context_limit":
            return "incomplete"
        return _STATUS_BY_STOP_REASON.get(str(output.get("stop_reason")), "failed")

    def _observations(
        self,
        state: Terminus2SandboxSession,
        output: Mapping[str, Any],
        *,
        conversation: list[dict[str, Any]] | None = None,
        duration_ms: float | None = None,
    ) -> AgentObservationBundle:
        """Evidence from the runner's result: the invocation, its commands, model calls, summarizations and sandbox.

        ``conversation`` is the response output; without one it is converted again from the runner's trajectory.
        """
        invocation_id = state.invocation_id or "terminus2"
        try:
            stop_reason = self._stop_reason(output)
            status = self._status(output)
            if conversation is None:
                conversation = trajectory_to_responses(output.get("trajectory") or {"steps": []})
            # Terminus runs a turn's whole command batch, then records one terminal result for it. Every command of a
            # turn with a result ran; a turn without one was still running when the run stopped.
            tool_status: dict[str, str] = {}
            unfinished = {"wall_time": "timeout", "cancelled": "cancelled"}.get(stop_reason, "failed")
            turn: list[str] = []
            for item in [*conversation, {"type": "message"}]:
                if item.get("type") == "function_call":
                    turn.append(item["call_id"])
                elif item.get("type") == "function_call_output":
                    tool_status.update(dict.fromkeys(turn, "completed"))
                elif item.get("type") == "message":
                    tool_status.update({call_id: unfinished for call_id in turn if call_id not in tool_status})
                    turn = []
            model_calls = [
                ModelCallRef(model_ref=self.config.model_server, response_id=response_id)
                for response_id in output.get("response_ids") or []
                if isinstance(response_id, str) and response_id
            ]
            records: list[Any] = [
                AgentInvocation(
                    invocation_id=invocation_id,
                    status=status,
                    stop_reason=stop_reason,
                    duration_ms=duration_ms,
                    error_type=output.get("error_type")
                    or {"wall_time": "timeout", "cancelled": "cancelled"}.get(stop_reason),
                    model_calls=model_calls,
                    conversation=conversation,
                )
            ]
            # Terminus runs the keystroke batches it parses from the model's text; the response presents each batch
            # as a function call with the terminal output Terminus recorded for it.
            records.extend(
                ToolCallObservation(
                    invocation_id=invocation_id,
                    tool_call_id=item["call_id"],
                    sandbox_id=state.sandbox_id,
                    tool_name=item["name"],
                    timing_source="harness",
                    status=tool_status[item["call_id"]],
                )
                for item in conversation
                if item.get("type") == "function_call"
            )
            for compaction in output.get("compactions") or []:
                records.append(
                    ContextCompactionObservation(
                        invocation_id=invocation_id,
                        observed_at=compaction.get("observed_at"),
                        trigger="harbor_summarize",
                        outcome=compaction.get("outcome", "unknown"),
                        model_calls=[
                            ModelCallRef(model_ref=self.config.model_server, response_id=response_id)
                            for response_id in compaction.get("response_ids") or []
                        ],
                    )
                )
            cleanup = state.session.cleanup
            records.append(
                SandboxObservation(
                    role="agent",
                    provider=state.provider,
                    sandbox_id=state.sandbox_id,
                    outcome=_SANDBOX_OUTCOME_BY_STOP_REASON.get(
                        stop_reason, "failed" if status == "failed" else "completed"
                    ),
                    exit_code=cleanup["return_code"] if cleanup is not None else None,
                    error_type=output.get("error_kind") if status == "failed" else None,
                )
            )
            gaps = [
                # Terminus records no start or end time for a command batch.
                ObservationGap(code="tool_timing_unavailable", invocation_id=invocation_id),
            ]
            gaps.append(ObservationGap(code="reasoning_token_usage_unavailable", invocation_id=invocation_id))
            if not output.get("cached_usage_known", False):
                gaps.append(ObservationGap(code="cached_token_usage_unavailable", invocation_id=invocation_id))
            calls_without_usage = output.get("calls_without_usage")
            if calls_without_usage != 0:
                gaps.append(
                    ObservationGap(
                        code="model_usage_unavailable",
                        invocation_id=invocation_id,
                        detail=f"{calls_without_usage} model responses had no usage"
                        if calls_without_usage
                        else "the runner reported no usage accounting",
                    )
                )
            return AgentObservationBundle(source="terminus2", records=records, gaps=gaps)
        except Exception as error:
            # A malformed result, which a task command could also have written, is a gap in the evidence; it must
            # not fail the activation or every close retry.
            LOG.exception("Could not build Terminus 2 observations from the runner's result")
            return AgentObservationBundle(
                source="terminus2",
                gaps=[
                    ObservationGap(
                        code="observation_capture_failed", invocation_id=invocation_id, detail=type(error).__name__
                    )
                ],
            )

    async def _run_sandbox_activation(
        self,
        state: Terminus2SandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        instruction: str,
        chat_params: dict[str, Any],
    ) -> NeMoGymResponse:
        """Run a prepared request (see ``_prepare_request``) in the session's sandbox."""
        invocation_id = state.invocation_id = f"terminus2_{uuid4().hex}"
        payload = {
            "instruction": instruction,
            "model": self._model_name(body),
            # The sandbox reaches the Model Server directly; the rollout prefix keeps its calls correlated.
            "model_base_url": self.resolve_model_base_url(
                self.config.model_server.name, state.request.episode_id.capture_key
            ),
            "chat_params": chat_params,
            "invocation_id": invocation_id,
            "model_timeout_sec": self.config.model_timeout_sec,
            "command_timeout_sec": self.config.command_timeout_sec,
            "terminus": self._terminus_settings(body),
        }
        timeout = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
        async with self.sem:
            # The budget runs from launch: time spent waiting for a run slot is not the harness's.
            started = monotonic()
            try:
                output = await state.execute(
                    payload, timeout=timeout, close_timeout=self.config.session_close_timeout_seconds
                )
            except TimeoutError as error:
                # Only the supervisor receipt establishes a harness deadline. A provider timeout alone does not.
                output = state.session.artifacts
                cleanup = state.session.cleanup
                if not isinstance(output, dict) or cleanup is None or not cleanup["timed_out"]:
                    raise
                state.provider_wait_failure = NgFailureEvent(
                    boundary="sandbox",
                    failure_kind="transport_timeout",
                    failure_reason="Sandbox provider wait expired after Terminus 2's confirmed deadline",
                    operation="exec_wait",
                    stage="agent",
                    foreign_error=sdk_error_details(error, message=str(error)),
                )
                state.provider_wait_failure.__cause__ = error
                observe_failure(state.provider_wait_failure)
                output = {**output, "stop_reason": "wall_time"}
        duration_ms = (monotonic() - started) * 1000
        cleanup = state.session.cleanup
        if output.get("stop_reason") == "wall_time" and not (cleanup and cleanup["timed_out"]):
            # The runner reads any SIGTERM without a close request as its deadline, but the supervisor did not stop
            # it: a task command, such as `pkill python3`, did. That is no timeout.
            output = {**output, "stop_reason": "terminated"}

        stop_reason = self._stop_reason(output)
        status = self._status(output)
        context = AgentContext.model_validate(output.get("context") or {})
        response = self._response(
            body,
            trajectory=output.get("trajectory") or {"steps": []},
            context=context,
            flags={"context_length_exceeded": bool(output.get("context_length_exceeded"))},
            timed_out=stop_reason == "wall_time",
            finished_naturally=bool(output.get("finished_naturally")),
            terminal_response_id=output.get("terminal_response_id"),
            # Pending, failed, or partially accounted attempts make aggregate usage unknown, and so does a result
            # without the count, such as the runner's report of its own crash.
            usage_known=output.get("calls_without_usage") == 0,
            cached_usage_known=bool(output.get("cached_usage_known", False)),
        )
        state.observations = self._observations(
            state,
            output,
            conversation=[item.model_dump(mode="json", exclude_none=True) for item in response.output],
            duration_ms=duration_ms,
        )
        error = output.get("error")
        response = response.model_copy(
            update={
                "status": status,
                "error": ResponseError(code="server_error", message=str(error)[:2000])
                if status == "failed" and error
                else None,
                "metadata": {
                    **(response.metadata or {}),
                    "harness_execution": "sandbox",
                    "stop_reason": stop_reason,
                    AGENT_TIMEOUT_METADATA_KEY: str(timeout),
                },
            }
        )
        response = self.report_outcome(response, output)
        return self.report_rollout_stop(response, state.session)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        agent_session_id = self._agent_session_id_from_request(request)
        if agent_session_id is not None:
            rollout_id = self._rollout_id(request)
            if not isinstance(rollout_id, str):
                raise HTTPException(409, "Agent sessions require an attempt-qualified rollout path")
            state = self._require_agent_session(agent_session_id)
            if state.request.episode_id.capture_key != rollout_id:
                raise HTTPException(409, "Agent-session episode_id does not match the rollout route")
            if state.task is None:
                prepared, instruction = self._prepare_request(body)
                try:
                    # Terminus types keystrokes and calls no tools, so the request's tool fields stay out of its
                    # model calls. A field the converter cannot represent is refused before any sandbox work.
                    chat_params = chat_completion_params(
                        prepared.model_dump(
                            mode="json", exclude_none=True, exclude={"tools", "tool_choice", "parallel_tool_calls"}
                        ),
                        collect_rollout_details=self.config.collect_rollout_details,
                    )
                except NotImplementedError as error:
                    raise HTTPException(422, f"Terminus 2 cannot run this request: {error}") from error
                # No await until the task and its request binding are installed: a retry arriving meanwhile must
                # find them.
                state.activation_request = body.model_copy(deep=True)
                state.task = asyncio.create_task(
                    self._run_sandbox_activation(state, prepared, instruction=instruction, chat_params=chat_params)
                )
            elif body != state.activation_request:
                raise HTTPException(409, "Terminus 2 sandbox sessions support one activation; retry the same request")
            # A disconnected HTTP waiter must not cancel the shared activation. Session close owns cancellation;
            # identical retries join this task or replay its result or error.
            return (await asyncio.shield(state.task)).model_copy(deep=True)

        body, instruction = self._prepare_request(body)
        api_base = self.resolve_model_base_url(
            self.config.model_server.name,
            self._rollout_id(request),
        )
        async with self.sem:
            (
                trajectory,
                context,
                flags,
                timed_out,
                finished_naturally,
                terminal_response_id,
                accounting,
            ) = await self._run_terminus(body, instruction, api_base)
        return self._response(
            body,
            trajectory=trajectory,
            context=context,
            flags=flags,
            timed_out=timed_out,
            finished_naturally=finished_naturally,
            terminal_response_id=terminal_response_id,
            usage_known=accounting["calls_without_usage"] == 0,
            cached_usage_known=accounting["cached_usage_known"],
        )

    async def run(self, request: Request, body: Terminus2AgentRunRequest) -> Terminus2AgentVerifyResponse:
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "Use the agent session responses and close routes")
        if self.config.resources_server is None:
            raise HTTPException(422, "Terminus 2 /run requires resources_server; use agent sessions instead")
        self._validate_output_budget(body.responses_create_params)
        cookies = request.cookies
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        agent_response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(agent_response)
        cookies = agent_response.cookies
        response_json = await get_response_json(agent_response)

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": response_json},
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        verify_json = await get_response_json(verify_response)

        gym_response = NeMoGymResponse.model_validate(response_json)
        turns = sum(
            1
            for item in gym_response.output
            if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
        )
        terminus_metadata_json = (gym_response.metadata or {}).get("terminus_2", "{}")
        try:
            terminus_metadata = json.loads(terminus_metadata_json)
        except (json.JSONDecodeError, TypeError):
            terminus_metadata = {}
        flags = terminus_metadata.get("agent_error_flags", {})
        timed_out = bool(terminus_metadata.get("timed_out"))
        naturally = bool(terminus_metadata.get("finished_naturally"))

        return Terminus2AgentVerifyResponse.model_validate(
            verify_json
            | {
                "turns_used": turns,
                "finished_naturally": naturally and not timed_out,
                "agent_timeout_error": int(timed_out),
                "context_length_exceeded_error": int(flags.get("context_length_exceeded", False)),
            }
        )


if __name__ == "__main__":
    Terminus2Agent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = Terminus2Agent.run_webserver()  # noqa: F401
