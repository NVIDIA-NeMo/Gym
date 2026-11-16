# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CrewAI harness adapter: one crew turn per agent session, run inside the episode's task sandbox.

The sandbox is the one the Resources Server lends the session, or, when it lends none and ``sandbox_provider`` is set,
one the agent starts and stops itself. A lent sandbox is never stopped. At seed the Agent Server installs a private
Python runtime with CrewAI in the sandbox; the activation stages ``sandbox_runner.py`` and runs it under Gym's process
supervisor. The crew's shell tool runs bash in the task's working directory, its grant tools call the episode's MCP
grants, and its model calls go from the sandbox to the Gym Model Server at the rollout's Chat Completions route. The
Agent Server turns the runner's report into the response and the close bundle.
"""

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from time import time
from typing import Any, ClassVar
from uuid import uuid4

from fastapi import HTTPException, Request
from openai.types.responses import ResponseError
from pydantic import Field, PrivateAttr

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SandboxNetworkAccess,
    SandboxSession,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
    resolve_agent_timeout,
    restrict_sandbox_network,
)
from nemo_gym.config import ModelServerRef, ResourcesServerRef, get_global_config_dict
from nemo_gym.core import (
    AGENT_INVALID_OUTPUT,
    AGENT_RUN_ERROR,
    MODEL_CONTEXT_LIMIT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    FailureHeader,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
    NgFailureEvent,
    SdkErrorDetails,
    failure_report_for,
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
from nemo_gym.server_runtime import MCPToolAccess
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    SandboxObservation,
    TrajectoryToolCall,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.crewai_agent.sandbox import SESSION_ROOT, CrewAISessionState


LOG = logging.getLogger(__name__)


OBSERVATION_SOURCE = "crewai_agent"
# Request fields a crew run applies; any other field set to a non-default value is refused.
_APPLIED_REQUEST_FIELDS = frozenset(
    {
        "input",
        "instructions",
        "model",
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "metadata",
        "tools",
    }
)


class CrewAIAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef | None = Field(
        default=None,
        description="Unused: the agent runs only in agent sessions, where the Environment Server binds the benchmark.",
    )
    model_server: ModelServerRef
    model: str | None = Field(
        default=None,
        description="Model name sent on each request when the request names none. The vLLM Model Server sends its own.",
    )
    concurrency: int = Field(default=16, ge=1, description="Crew runs at once.")
    timeout: float = Field(
        default=1800,
        gt=0,
        description="Seconds a crew run may take. A task's agent_timeout_sec, from the seed or the request, lowers it.",
    )
    max_iter: int = Field(
        default=50, ge=1, description="CrewAI Agent.max_iter: model calls before CrewAI asks for a final answer."
    )
    max_model_retries: int = Field(
        default=5,
        ge=0,
        description="Additional attempts for model HTTP 408, 425, 429 and 5xx responses; each wait is capped at 60s.",
    )
    context_window_size: int | None = Field(
        default=None,
        gt=0,
        description="The model's context window in tokens. CrewAI summarizes the conversation as it nears 85% of it.",
    )
    role: str = "Software engineer"
    goal: str = "Complete the task you are given, exactly as it asks."
    backstory: str = "You do the work with the tools you are given, and you check the result before you answer."
    expected_output: str = (
        "The task done. Reply with the final answer in the form the task asks for, or a short summary of what you did."
    )
    shell_timeout_seconds: float = Field(default=600, gt=0, description="Deadline for one shell-tool command.")
    shell_max_output_bytes: int = Field(
        default=30000, gt=0, description="Output of one command shown to the model; its head and tail are kept."
    )
    mcp_timeout_seconds: float = Field(default=300, gt=0, description="Deadline for one MCP listing or tool call.")
    sandbox_provider: str | None = Field(
        default=None,
        description=(
            "Top-level sandbox provider block for a sandbox the agent starts itself, used only when the Environment "
            "Server lends none. Without it, such a session is refused."
        ),
    )
    sandbox_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "SandboxSpec fields for the agent's own sandbox. workdir defaults to /app, and ttl_s to the install and "
            "run timeouts plus ten minutes, so the provider reclaims a sandbox the agent never stopped."
        ),
    )
    sandbox_install_timeout_seconds: float = Field(
        default=900, gt=0, allow_inf_nan=False, description="Time limit for installing CrewAI's runtime at seed."
    )
    network_access: SandboxNetworkAccess = Field(
        default="inherit",
        description=(
            "Egress of a sandbox the agent starts: `model_only` allows only the model server's host, "
            "`model_and_tools` also each MCP grant's host. A lent sandbox keeps its owner's network, so a session "
            "that borrows one refuses any other value than `inherit`."
        ),
    )


@dataclass(frozen=True)
class _Instruction:
    """What one activation asks of the crew, read from its request."""

    description: str
    system: str
    # The grant tools the request offers the model, or None for every grant tool.
    tools: list[str] | None
    model: str | None
    sampling: dict[str, Any]
    parallel_tool_calls: bool
    timeout: float


def _text(content: Any) -> str | None:
    """A message's text, or None when it holds anything other than text."""
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        part_type = part.get("type") if isinstance(part, Mapping) else getattr(part, "type", None)
        text = part.get("text") if isinstance(part, Mapping) else getattr(part, "text", None)
        if part_type != "input_text" or not isinstance(text, str):
            return None
        texts.append(text)
    return "\n".join(texts)


def _usage(usage: Mapping[str, int | None] | None) -> NeMoGymResponseUsage | None:
    if usage is None:
        return None
    return NeMoGymResponseUsage(
        input_tokens=usage["input"],
        output_tokens=usage["output"],
        total_tokens=usage["total"],
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=usage["cached"]),
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=usage["reasoning"]),
    )


def _response_status(report: Mapping[str, Any]) -> tuple[str, dict[str, str] | None]:
    """The activation's status and incomplete details: a final answer cut off at the output cap is incomplete."""
    if report["status"] == "completed" and report["finish_reason"] == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    return report["status"], None


def _run_end(report: Mapping[str, Any] | None) -> tuple[str, str | None]:
    """How the crew's run in its sandbox ended (outcome, error type), from the runner's report."""
    if report is None:
        # Close came before the runner reported anything.
        return "cancelled", None
    if report.get("error") is not None:
        return "failed", "runner_error"
    if report.get("infrastructure_error") is not None:
        return "failed", "infrastructure_error"
    if report.get("configuration_error") is not None:
        return "failed", "configuration_error"
    if report["status"] == "failed":
        return "failed", report["error_type"]
    if report["stop_reason"] == "wall_time":
        return "timeout", None
    if report["stop_reason"] == "cancelled":
        return "cancelled", None
    return "completed", None


def _infrastructure_failure(report: Mapping[str, Any]) -> NgFailureEvent:
    """Project the runner's selected origin/cause facts once; the session reuses these event identities."""
    failure = NgFailureEvent(
        boundary="harness",
        failure_kind=AGENT_RUN_ERROR,
        failure_reason=report["message"],
        operation=report["operation"],
        operation_id=(report.get("grant_name") or "")[:128] or None,
        stage="agent",
    )
    parent = failure
    for cause in report["causes"]:
        observed = NgFailureEvent(
            boundary="harness",
            failure_kind=cause["failure_kind"],
            failure_reason=cause["message"],
            operation=report["operation"],
            stage="agent",
            foreign_error=SdkErrorDetails(
                exception_type=cause["exception_type"], message=cause["message"], http_status=cause["http_status"]
            ),
        )
        parent.__cause__ = observed
        parent = observed
    return failure


class CrewAIAgent(SimpleResponsesAPIAgent):
    config: CrewAIAgentConfig
    # The crew runs inside the task sandbox and calls the model server from there.
    calls_model_from_sandbox: ClassVar[bool] = True
    _semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        # A session lives in the process that seeded it.
        if self.config.num_workers not in (None, 1):
            raise ValueError("CrewAI agent sessions are process-local and require num_workers=1")
        self._semaphore = asyncio.Semaphore(self.config.concurrency)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> CrewAISessionState:
        accesses = self.effective_tool_accesses(body)
        unusable = sorted(
            access.name for access in accesses if access.required and not isinstance(access, MCPToolAccess)
        )
        if unusable:
            raise AgentSeedRejectedError(
                f"CrewAI consumes only MCP tool grants; required grants it cannot use: {', '.join(unusable)}"
            )
        access = body.sandbox_access
        spec = None
        if access is not None:
            if not isinstance(access.connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("CrewAI supports only direct sandbox connections")
            if not PurePosixPath(access.workdir).is_absolute():
                raise AgentSeedRejectedError("CrewAI needs an absolute sandbox workdir")
            if access.user is not None:
                # The runtime and the session's files are written as the sandbox's default user.
                raise AgentSeedRejectedError("CrewAI does not support a SandboxAccess user override")
            if self.config.network_access != "inherit":
                raise AgentSeedRejectedError(
                    "CrewAI network_access applies only to a sandbox it starts, not a lent one"
                )
            provider_ref, workdir = access.connection.provider_config_ref, access.workdir
        elif self.config.sandbox_provider is not None:
            try:
                spec = SandboxSpec(**{"workdir": "/app", **self.config.sandbox_config})
            except (TypeError, ValueError) as error:
                raise AgentSeedRejectedError(str(error)) from error
            if not isinstance(spec.workdir, str) or not PurePosixPath(spec.workdir).is_absolute():
                raise AgentSeedRejectedError("CrewAI needs an absolute sandbox_config.workdir")
            provider_ref, workdir = self.config.sandbox_provider, spec.workdir
        else:
            raise AgentSeedRejectedError(
                "CrewAI runs in a task sandbox: the session lends none and the agent has no sandbox_provider"
            )
        global_config = get_global_config_dict()
        try:
            provider_config = resolve_provider_config(provider_ref, global_config)
            # The runner calls the Model Server from inside the sandbox, which cannot reach a loopback address.
            assert_model_url_reachable_from_sandbox(
                self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key),
                provider_name=next(iter(provider_config), None),
            )
            provider = create_provider(provider_config)
        except (TypeError, ValueError) as error:
            raise AgentSeedRejectedError(str(error)) from error
        grants = [
            {
                "name": grant.name,
                "url": str(grant.connection.url),
                "headers": dict(grant.connection.headers),
                "required": grant.required,
            }
            for grant in accesses
            if isinstance(grant, MCPToolAccess)
        ]

        try:
            if spec is None:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
            else:
                spec = restrict_sandbox_network(
                    replace(
                        spec,
                        # The provider reclaims a sandbox this process never stopped.
                        ttl_s=self.config.sandbox_config.get(
                            "ttl_s", self.config.sandbox_install_timeout_seconds + self.config.timeout + 600
                        ),
                        metadata=resolve_provider_metadata(provider_ref, global_config)
                        | dict(spec.metadata)
                        | {"nemo_gym_agent": self.config.name, "agent_session": body.agent_session_id[:63]},
                    ),
                    network_access=self.config.network_access,
                    provider=provider,
                    model_base_url=self.resolve_model_base_url(self.config.model_server.name, trace_context=False),
                    tool_urls=[grant["url"] for grant in grants],
                    required_tool_urls=[grant["url"] for grant in grants if grant["required"]],
                )
                sandbox = AsyncSandbox(provider)
                await sandbox.start(spec)
        except BaseException:
            # Cleanup must not replace the original connection/start failure.
            try:
                await provider.aclose()
            except BaseException as cleanup_error:
                LOG.exception("CrewAI provider cleanup failed after sandbox setup failed")
                if isinstance(cleanup_error, NgFailureEvent):
                    observe_failure(cleanup_error)
                else:
                    cleanup_failure = NgFailureEvent(
                        boundary="sandbox",
                        failure_kind="session_release_failed",
                        failure_reason="CrewAI provider cleanup failed",
                        operation="aclose",
                        stage="cleanup",
                    )
                    cleanup_failure.__cause__ = cleanup_error
                    observe_failure(cleanup_failure)
            raise
        state = CrewAISessionState(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                workdir=workdir,
                session_dir=f"{SESSION_ROOT}/{uuid4().hex}",
                owns_sandbox=spec is not None,
                harness="CrewAI",
            ),
            grants=grants,
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
        if not isinstance(state, CrewAISessionState):
            raise TypeError("Expected a CrewAI session state")
        if state.provider_wait_failure is not None:
            observe_failure(state.provider_wait_failure)
        if state.infrastructure_failure is not None:
            observe_failure(state.infrastructure_failure)
        await state.close(self.config.session_close_timeout_seconds)
        # No trajectory: rollout collection builds the turns from the captured calls, and joins every call, failed
        # attempts included, to the invocation by its session header.
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=self._observations(state),
            cleanup_confirmed=state.session.cleanup_confirmed,
        )

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, CrewAISessionState) else None

    def _instruction(self, state: CrewAISessionState, body: NeMoGymResponseCreateParamsNonStreaming) -> _Instruction:
        """Read the activation's request, refusing what a crew run cannot apply."""
        for name, request_field in type(body).model_fields.items():
            if name in _APPLIED_REQUEST_FIELDS or getattr(body, name) == request_field.get_default(
                call_default_factory=True
            ):
                continue
            raise HTTPException(422, f"CrewAI does not support request field {name}")
        items = [{"role": "user", "content": body.input}] if isinstance(body.input, str) else body.input
        user, system = [], [body.instructions] if body.instructions else []
        for item in items:
            role = item.get("role") if isinstance(item, Mapping) else getattr(item, "role", None)
            item_type = (item.get("type") if isinstance(item, Mapping) else getattr(item, "type", None)) or "message"
            content = item.get("content") if isinstance(item, Mapping) else getattr(item, "content", None)
            text = _text(content) if item_type == "message" and content is not None else None
            if text is None or role not in ("user", "system", "developer"):
                raise HTTPException(422, "CrewAI accepts only text system, developer and user messages")
            (user if role == "user" else system).append(text)
        if not user:
            raise HTTPException(422, "CrewAI needs a user message to run")
        tools = None
        if body.tools:
            # A task's function tools are Resources tools; the crew calls them through the MCP grant that serves them.
            if not state.grants:
                raise HTTPException(422, "CrewAI runs request tools only through an MCP grant; the session has none")
            tools = []
            for tool in body.tools:
                tool_type = tool.get("type") if isinstance(tool, Mapping) else getattr(tool, "type", None)
                name = tool.get("name") if isinstance(tool, Mapping) else getattr(tool, "name", None)
                if tool_type != "function" or not name:
                    raise HTTPException(422, f"CrewAI runs only function tools, through an MCP grant: {tool_type}")
                tools.append(name)
        timeout = self.config.timeout
        if state.request.agent_timeout_sec is not None:
            timeout = min(timeout, state.request.agent_timeout_sec)
        sampling = {
            key: value
            for key, value in (
                ("temperature", body.temperature),
                ("top_p", body.top_p),
            )
            if value is not None
        }
        return _Instruction(
            description="\n\n".join(user),
            system="\n\n".join(system),
            tools=tools,
            model=body.model or self.config.model,
            sampling=sampling,
            parallel_tool_calls=body.parallel_tool_calls,
            timeout=resolve_agent_timeout(body.metadata, ceiling=timeout),
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        agent_session_id = self._agent_session_id_from_request(request)
        if agent_session_id is None:
            raise HTTPException(409, "CrewAI runs only in agent sessions; seed one through an Environment Server")
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        if not isinstance(rollout_id, str):
            raise HTTPException(409, "Agent sessions require an attempt-qualified rollout path")
        state = self._require_agent_session(agent_session_id)
        if not isinstance(state, CrewAISessionState):
            raise TypeError("Expected a CrewAI session state")
        if state.request.episode_id.capture_key != rollout_id:
            raise HTTPException(409, "Agent-session episode_id does not match the rollout route")
        if state.task is None:
            # No await until the task and its request are bound: a retry arriving meanwhile must find them.
            instruction = self._instruction(state, body)
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._activate(state, instruction))
        elif body != state.activation_request:
            raise HTTPException(409, "CrewAI sessions support one activation; retry the same request")
        # A disconnected HTTP waiter must not cancel the shared run. Close owns cancellation; identical retries
        # join this task or replay its result or error.
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def _activate(self, state: CrewAISessionState, instruction: _Instruction) -> NeMoGymResponse:
        async with self._semaphore:
            capture_key = state.request.episode_id.capture_key
            access = state.request.sandbox_access
            payload = {
                "invocation_id": capture_key,
                # The sandbox reaches the Model Server directly; the rollout prefix keeps its calls correlated.
                "model_url": (
                    f"{self.resolve_model_base_url(self.config.model_server.name, capture_key)}/chat/completions"
                ),
                "model": instruction.model,
                "llm_model": instruction.model or self.config.model_server.name,
                "sampling": instruction.sampling,
                "parallel_tool_calls": instruction.parallel_tool_calls,
                "max_model_retries": self.config.max_model_retries,
                "context_window_size": self.config.context_window_size,
                "max_iter": self.config.max_iter,
                "role": self.config.role,
                "goal": self.config.goal,
                "backstory": "\n\n".join(part for part in (self.config.backstory, instruction.system) if part),
                "description": instruction.description,
                "expected_output": self.config.expected_output,
                "workdir": state.session.workdir,
                "task_env": dict(access.env) if access is not None else {},
                "shell_timeout_seconds": self.config.shell_timeout_seconds,
                "shell_max_output_bytes": self.config.shell_max_output_bytes,
                "grants": state.grants,
                "tools": instruction.tools,
                "mcp_timeout_seconds": self.config.mcp_timeout_seconds,
                "stop_request_path": state.session.stop_request_path,
            }
            try:
                report = await state.execute(
                    payload, timeout=instruction.timeout, close_timeout=self.config.session_close_timeout_seconds
                )
            except TimeoutError as failure:
                # The runner normally stops at the deadline and writes its partial report. If the supervisor overran
                # too, the episode is still a timeout when that report is there, and an infrastructure failure when not.
                report = state.session.artifacts
                if not (isinstance(report, dict) and report.get("stop_reason") == "wall_time"):
                    raise
                state.provider_wait_failure = NgFailureEvent(
                    boundary="sandbox",
                    failure_kind="transport_timeout",
                    failure_reason="Sandbox provider wait expired after CrewAI reported its deadline",
                    operation="exec_wait",
                    stage="agent",
                    foreign_error=sdk_error_details(failure, message=str(failure)),
                )
                state.provider_wait_failure.__cause__ = failure
                observe_failure(state.provider_wait_failure)
            if report.get("error") is not None:
                raise RuntimeError(f"CrewAI sandbox runner failed: {report['error']}\n{report.get('traceback', '')}")
            if report.get("configuration_error") is not None:
                raise HTTPException(422, report["configuration_error"])
            if report["infrastructure_error"] is not None:
                state.infrastructure_failure = _infrastructure_failure(report["infrastructure_error"])
                observe_failure(state.infrastructure_failure)
                raise HTTPException(
                    503,
                    f"CrewAI episode infrastructure failed: {state.infrastructure_failure.failure_reason}",
                    headers=FailureHeader.encode(failure_report_for(state.infrastructure_failure)),
                ) from state.infrastructure_failure
            return self.report_rollout_stop(self._response(state, instruction, report), state.session)

    def _response(
        self, state: CrewAISessionState, instruction: _Instruction, report: Mapping[str, Any]
    ) -> NeMoGymResponse:
        status, incomplete_details = _response_status(report)
        metadata = {"crewai_model_calls": str(report["model_calls"]), "crewai_tool_calls": str(report["tool_runs"])}
        if status == "incomplete" and incomplete_details is None:
            metadata["crewai_stop_reason"] = "context_limit" if report["stop_reason"] == "context_limit" else "timeout"
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=instruction.model or self.config.model_server.name,
            object="response",
            output=report["output"],
            status=status,
            incomplete_details=incomplete_details,
            error=ResponseError(code="server_error", message=report["error_message"]) if status == "failed" else None,
            metadata=metadata,
            tool_choice=state.activation_request.tool_choice,
            tools=state.activation_request.tools,
            parallel_tool_calls=state.activation_request.parallel_tool_calls,
            usage=_usage(report["usage"]),
        )
        return self.report_outcome(response, report)

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """A crew run that raised: the model request that failed, if one did; an empty answer is the model's invalid
        output; CrewAI's other errors are unknown."""
        failure = native_result.get("model_failure") or {}
        status = failure.get("status")
        if failure.get("context_limit"):
            return AgentOutcome(failure_kind=MODEL_CONTEXT_LIMIT)
        if type(status) is int and 400 <= status <= 599:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=status)
        if failure.get("transport"):
            return AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
        if native_result.get("error_type") == "EmptyLLMResponseError":
            # The model answered with neither text nor a tool call, also when CrewAI asked again.
            return AgentOutcome(failure_kind=AGENT_INVALID_OUTPUT)
        return super().classify_outcome(native_result)

    def _observations(self, state: CrewAISessionState) -> AgentObservationBundle:
        """The crew's invocation and tool calls, from the runner's report, and the sandbox it ran in."""
        report = state.session.artifacts if isinstance(state.session.artifacts, dict) else None
        outcome, error_type = _run_end(report)
        if state.session.execution_oom is not None:
            outcome, error_type = "oom", "oom_killed"
        sandbox = SandboxObservation(
            role="agent",
            provider=state.sandbox_provider,
            sandbox_id=state.sandbox_id,
            outcome=outcome,
            error_type=error_type,
            teardown_confirmed=True if state.session.sandbox_stopped else None,
        )
        if report is None or "output" not in report:
            # Close before any activation, or a runner that left no transcript: there is nothing the crew did.
            detail = "no activation" if state.activation_request is None else "no runner report"
            return AgentObservationBundle(
                source=OBSERVATION_SOURCE,
                records=[sandbox],
                gaps=[ObservationGap(code="observation_capture_failed", detail=detail)],
            )
        input_items = state.activation_request.input
        if isinstance(input_items, str):
            input_items = [{"role": "user", "content": input_items, "type": "message"}]
        invocation_id = report["invocation_id"]
        invocation = AgentInvocation(
            invocation_id=invocation_id,
            # The response's status: a final answer cut off at the output cap makes it incomplete.
            status=_response_status(report)[0],
            stop_reason="timeout"
            if report["stop_reason"] == "wall_time"
            else "context_limit"
            if report["stop_reason"] == "context_limit"
            else None,
            error_type=error_type if outcome in ("failed", "oom") else None,
            duration_ms=report["duration_ms"],
            model_calls=[
                ModelCallRef(model_ref=self.config.model_server, response_id=response_id)
                for response_id in report["response_ids"]
            ],
            conversation=[*input_items, *report["output"]],
        )
        tool_calls = [
            TrajectoryToolCall(invocation_id=invocation_id, timing_source="executor", **record)
            for record in report["tool_calls"]
        ]
        return AgentObservationBundle(
            source=OBSERVATION_SOURCE,
            records=[invocation, *tool_calls, sandbox],
            gaps=[ObservationGap(**gap) for gap in report["gaps"]],
        )


if __name__ == "__main__":
    CrewAIAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CrewAIAgent.run_webserver()  # noqa: F401
