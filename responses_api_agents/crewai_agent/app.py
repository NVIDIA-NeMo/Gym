# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CrewAI harness adapter: one crew turn per agent session, run in the Agent Server.

The crew's model is the Gym Model Server, reached through the activation's rollout-prefixed Chat Completions route.
Its tools are a shell that runs in the episode's task sandbox, and the tools of the episode's MCP grants. The sandbox
is the one the Resources Server lends the session, or, when it lends none and ``sandbox_provider`` is set, one the
agent starts and stops itself. A lent sandbox is never stopped.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from time import monotonic, perf_counter, time
from typing import Any
from uuid import uuid4

import aiohttp
from fastapi import HTTPException, Request
from mcp import McpError
from mcp.types import CallToolResult, Tool
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
    SimpleResponsesAPIAgent,
    resolve_agent_timeout,
)
from nemo_gym.agents.utils import lifecycle_observation
from nemo_gym.client import get_response_json
from nemo_gym.config import ModelServerRef, ResourcesServerRef, get_global_config_dict
from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import BaseRunRequest, MCPToolAccess, is_retryable_dependency_status
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    SandboxObservation,
    TrajectoryToolCall,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.crewai_agent import mcp_client
from responses_api_agents.crewai_agent.crew import (
    CrewRun,
    GrantTool,
    GymChatLLM,
    ShellTool,
    build_crew,
    grant_tool_args_schema,
    grant_tool_result,
    tool_name,
)
from responses_api_agents.crewai_agent.sandbox_shell import SandboxShell, ShellResult, ShellRunnerError


LOG = logging.getLogger(__name__)

SHELL_TOOL_NAME = "shell"
# Request fields a crew run applies; any other field set to a non-default value is refused.
_APPLIED_REQUEST_FIELDS = frozenset(
    {
        "input",
        "instructions",
        "model",
        "temperature",
        "top_p",
        "max_output_tokens",
        "parallel_tool_calls",
        "metadata",
        "tools",
    }
)
# CrewAI adds keys of its own to messages, such as cache_breakpoint; the Model Server validates messages strictly.
_CHAT_MESSAGE_KEYS = ("role", "content", "tool_calls", "tool_call_id")


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
    concurrency: int = Field(
        default=16,
        ge=1,
        description=(
            "Crews that run at once. CrewAI holds a thread of the event loop's default executor for each model or "
            "tool call in flight."
        ),
    )
    timeout: float = Field(
        default=1800,
        gt=0,
        description="Seconds a crew run may take. A task's agent_timeout_sec, from the seed or the request, lowers it.",
    )
    max_iter: int = Field(
        default=50, ge=1, description="CrewAI Agent.max_iter: model calls before CrewAI asks for a final answer."
    )
    max_model_retries: int = Field(
        default=2,
        ge=0,
        description="Additional attempts for model HTTP 408, 425, 429 and 5xx responses, with bounded backoff.",
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
            "Server lends none. Without it, such a session runs with grant tools only."
        ),
    )
    sandbox_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "SandboxSpec fields for the agent's own sandbox. workdir defaults to /app, and ttl_s to timeout plus ten "
            "minutes so the provider reclaims a sandbox the agent never stopped."
        ),
    )


@dataclass
class _GrantServer:
    access: MCPToolAccess
    tools: list[Tool]


@dataclass(frozen=True)
class _Instruction:
    """What one activation asks of the crew, read from its request."""

    description: str
    system: str
    # The grant tools the request offers the model, or None for every grant tool.
    tools: frozenset[str] | None
    model: str | None
    sampling: dict[str, Any]
    parallel_tool_calls: bool
    timeout: float


@dataclass
class _ToolRun:
    name: str
    arguments: dict[str, Any]
    started_at: float
    completed_at: float = 0.0
    duration_ms: float = 0.0
    status: str = "unknown"
    error_type: str | None = None
    output: Any = None


class _InfrastructureError(Exception):
    """The task sandbox or a tool grant failed underneath the crew; the episode cannot be graded."""


class _ModelRequestError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        # The upstream body stays in the message: CrewAI reads a context-limit error from it and summarizes.
        super().__init__(f"Model HTTP {status}: {detail}")
        self.status = status


class _Episode:
    """One activation's crew run and what it observed, recorded on the event loop."""

    def __init__(self, *, invocation_id: str, rollout_id: str, model_ref: ModelServerRef) -> None:
        self.crew_run = CrewRun(asyncio.get_running_loop())
        self.invocation_id = invocation_id
        self.rollout_id = rollout_id
        self.model_ref = model_ref
        # CrewAI's name for each tool the crew has, to the tool's own name, which the transcript records.
        self.tool_names: dict[str, str] = {}
        self.started = monotonic()
        self.output: list[Any] = []
        self.model_calls: list[ModelCallRef] = []
        self.turn_count = 0
        self.tool_runs: list[_ToolRun] = []
        self.gaps: list[ObservationGap] = []
        self.model_cookies: dict[str, str] = {}
        self.usage: dict[str, int] | None = {"input": 0, "output": 0, "total": 0, "cached": 0, "reasoning": 0}
        self.cached_known = True
        self.reasoning_known = True
        self.finish_reason: str | None = None
        # The failure behind a crew run that raised: the last model request that failed, or an infrastructure error.
        self.model_failure: AgentOutcome | None = None
        self.infrastructure_error: str | None = None
        self._answered_calls: set[str] = set()

    def record_tool_results(self, messages: list[dict[str, Any]]) -> None:
        """Tool results the model is about to see, as CrewAI formatted them, in the order it sends them."""
        for message in messages:
            call_id = message.get("tool_call_id")
            if message.get("role") != "tool" or not call_id or call_id in self._answered_calls:
                continue
            self._answered_calls.add(call_id)
            content = message.get("content")
            self.output.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=content if isinstance(content, str) else json.dumps(content),
                )
            )

    def record_model_response(self, data: dict[str, Any]) -> dict:
        """Add a Chat Completions response to the transcript; return its assistant message for CrewAI."""
        choice = (data.get("choices") or [{}])[0]
        message = dict(choice.get("message") or {})
        self.finish_reason = choice.get("finish_reason")
        self.turn_count += 1
        response_id = data.get("id")
        if response_id:
            self.model_calls.append(ModelCallRef(model_ref=self.model_ref, response_id=response_id))
        else:
            self.gaps.append(
                ObservationGap(
                    code="model_call_reference_unavailable",
                    invocation_id=self.invocation_id,
                    detail=f"turn:{self.turn_count}",
                )
            )
        self._add_usage(data.get("usage"))

        items: list[Any] = []
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            items.append(
                NeMoGymResponseReasoningItem(
                    id=f"rs_{uuid4().hex}", summary=[NeMoGymSummary(text=reasoning, type="summary_text")]
                )
            )
        content = message.get("content")
        if isinstance(content, str) and content:
            items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}", content=[NeMoGymResponseOutputText(text=content, annotations=[])]
                )
            )
        tool_calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            # Some servers omit a call's ID; CrewAI and the transcript need the same one.
            call = {
                "id": call.get("id") or f"call_{uuid4().hex}",
                "type": "function",
                "function": {
                    "name": function.get("name") or "",
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
                },
            }
            tool_calls.append(call)
            name = call["function"]["name"]
            items.append(
                NeMoGymResponseFunctionToolCall(
                    call_id=call["id"],
                    # Verifiers compare the tool's own name; CrewAI matched the call by its sanitized form.
                    name=self.tool_names.get(tool_name(name), name),
                    arguments=call["function"]["arguments"],
                    status="completed",
                )
            )
        message["tool_calls"] = tool_calls
        self.output.extend(items)
        return message

    def _add_usage(self, usage: Any) -> None:
        if self.usage is None:
            return
        if not isinstance(usage, Mapping) or not all(
            isinstance(usage.get(key), int) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            # Without every call's usage the total is unknown, not a subtotal.
            self.usage = None
            self.gaps.append(ObservationGap(code="model_call_usage_unavailable", invocation_id=self.invocation_id))
            return
        self.usage["input"] += usage["prompt_tokens"]
        self.usage["output"] += usage["completion_tokens"]
        self.usage["total"] += usage["total_tokens"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if isinstance(cached, int):
            self.usage["cached"] += cached
        else:
            self.cached_known = False
        if isinstance(reasoning, int):
            self.usage["reasoning"] += reasoning
        else:
            self.reasoning_known = False

    def response_usage(self) -> NeMoGymResponseUsage | None:
        if self.usage is None:
            return None
        return NeMoGymResponseUsage(
            input_tokens=self.usage["input"],
            output_tokens=self.usage["output"],
            total_tokens=self.usage["total"],
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=self.usage["cached"]),
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=self.usage["reasoning"]),
        )

    async def run_tool(
        self, name: str, arguments: dict[str, Any], execute: Callable[[], Awaitable[tuple[Any, str, str | None]]]
    ) -> Any:
        """Run one tool call and record its timing and outcome. ``execute`` returns (result, status, error type)."""
        run = _ToolRun(name=name, arguments=arguments, started_at=time())
        self.tool_runs.append(run)
        started = perf_counter()
        try:
            result, run.status, run.error_type = await execute()
            # A ToolFailure carries the text the model sees in its message.
            run.output = result if isinstance(result, str) else str(getattr(result, "message", result))
            return result
        except asyncio.CancelledError:
            run.status, run.error_type = "cancelled", "cancelled"
            raise
        except _InfrastructureError as error:
            run.status, run.error_type = "failed", "infrastructure_error"
            # The crew would show the model this error and carry on; stop it at its next call instead, so the
            # episode is not graded. This call still returns its error to CrewAI.
            self.infrastructure_error = str(error)
            self.crew_run.refuse_new_calls()
            raise
        finally:
            run.completed_at = max(run.started_at, time())
            run.duration_ms = (perf_counter() - started) * 1000

    def tool_observations(self) -> list[TrajectoryToolCall]:
        """Join each executed tool run to the model's call: CrewAI runs a call without passing its ID to the tool."""
        unmatched = list(self.tool_runs)
        records = []
        for call in (item for item in self.output if isinstance(item, NeMoGymResponseFunctionToolCall)):
            try:
                arguments = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = None
            run = next(
                (run for run in unmatched if run.name == call.name and run.arguments == arguments), None
            ) or next((run for run in unmatched if run.name == call.name), None)
            if run is None:
                # CrewAI answered the call itself (an unknown tool or invalid arguments) without running a tool.
                continue
            unmatched.remove(run)
            records.append(
                TrajectoryToolCall(
                    invocation_id=self.invocation_id,
                    tool_call_id=call.call_id,
                    tool_name=run.name,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                    duration_ms=run.duration_ms,
                    timing_source="executor",
                    status=run.status,
                    error_type=run.error_type,
                    output=run.output,
                )
            )
        for run in unmatched:
            self.gaps.append(
                ObservationGap(
                    code="tool_call_identity_unavailable", invocation_id=self.invocation_id, detail=f"tool:{run.name}"
                )
            )
        return records


@dataclass
class CrewAISessionState(AgentSessionState):
    shell: SandboxShell | None
    grants: list[_GrantServer]
    # Whether the agent started the shell's sandbox, and so stops it at close.
    owns_sandbox: bool = False
    gaps: list[ObservationGap] = field(default_factory=list)
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    episode: _Episode | None = None
    observations: AgentObservationBundle | None = None
    # How the crew's use of its sandbox ended (outcome, error type): cancelled unless an activation finished.
    run_end: tuple[str, str | None] = ("cancelled", None)


def _chat_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [{key: value for key, value in dict(message).items() if key in _CHAT_MESSAGE_KEYS} for message in messages]


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


class CrewAIAgent(SimpleResponsesAPIAgent):
    config: CrewAIAgentConfig
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
        connection = spec = None
        if body.sandbox_access is not None:
            connection = body.sandbox_access.connection
            if not isinstance(connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("CrewAI supports only direct sandbox connections")
            if not PurePosixPath(body.sandbox_access.workdir).is_absolute():
                raise AgentSeedRejectedError("CrewAI needs an absolute sandbox workdir")
            if body.sandbox_access.user is not None:
                # The shell's session files are written as the sandbox's default user.
                raise AgentSeedRejectedError("CrewAI does not support a SandboxAccess user override")
        elif self.config.sandbox_provider is not None:
            spec = SandboxSpec(**{"workdir": "/app", **self.config.sandbox_config})
            if not PurePosixPath(spec.workdir or "").is_absolute():
                raise AgentSeedRejectedError("CrewAI needs an absolute sandbox_config.workdir")

        gaps: list[ObservationGap] = []
        grants = []
        for access in accesses:
            if not isinstance(access, MCPToolAccess):
                continue
            try:
                tools = await asyncio.wait_for(
                    mcp_client.list_tools(access, timeout_seconds=self.config.mcp_timeout_seconds),
                    self.config.mcp_timeout_seconds,
                )
            except Exception as error:
                if access.required:
                    # A connection failure may pass: the seed answers 500, which the Environment Server retries.
                    raise RuntimeError(f"Required MCP grant {access.name} did not list its tools: {error}") from error
                LOG.warning("Skipping optional MCP grant %s: %s", access.name, error)
                gaps.append(ObservationGap(code="mcp_server_skipped", detail=f"{access.name}: {type(error).__name__}"))
                continue
            grants.append(_GrantServer(access=access, tools=tools))
        # Grant tools keep their own names, which verifiers compare, rather than CrewAI's server-prefixed MCP names.
        names = [SHELL_TOOL_NAME] if connection is not None or spec is not None else []
        names += [tool_name(tool.name) for grant in grants for tool in grant.tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise AgentSeedRejectedError(f"CrewAI tool names collide after sanitizing: {', '.join(duplicates)}")

        state = CrewAISessionState(request=body, shell=None, grants=grants, gaps=gaps)
        if spec is not None:
            await self._start_sandbox(state, spec)
            return state
        if connection is None:
            return state
        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            # No sandbox owns the provider yet.
            await provider.aclose()
            raise
        state.shell = SandboxShell(
            sandbox,
            workdir=body.sandbox_access.workdir,
            env=dict(body.sandbox_access.env),
            timeout_seconds=self.config.shell_timeout_seconds,
            max_output_bytes=self.config.shell_max_output_bytes,
        )
        try:
            await state.shell.setup()
        except BaseException as error:
            try:
                await self._release_sandbox(state)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _start_sandbox(self, state: CrewAISessionState, spec: SandboxSpec) -> None:
        """Start the agent's own sandbox for a session the Environment Server lent none."""
        global_config = get_global_config_dict()
        provider_ref = self.config.sandbox_provider
        provider = create_provider(resolve_provider_config(provider_ref, global_config))
        spec = replace(
            spec,
            # The provider reclaims a sandbox this process never stopped.
            ttl_s=self.config.sandbox_config.get("ttl_s", self.config.timeout + 600),
            metadata=resolve_provider_metadata(provider_ref, global_config)
            | dict(spec.metadata)
            | {"nemo_gym_agent": self.config.name, "agent_session": state.request.agent_session_id[:63]},
        )
        sandbox = AsyncSandbox(provider)
        state.shell = SandboxShell(
            sandbox,
            workdir=spec.workdir,
            env={},
            timeout_seconds=self.config.shell_timeout_seconds,
            max_output_bytes=self.config.shell_max_output_bytes,
        )
        state.owns_sandbox = True
        try:
            # A setup that fails stops the sandbox.
            await sandbox.start_with_setup(spec, lambda _: state.shell.setup())
        except BaseException as error:
            try:
                # Releases the provider's client when the create itself failed.
                await asyncio.wait_for(sandbox.stop(), self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            state.shell = None
            raise

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, CrewAISessionState):
            raise TypeError("Expected a CrewAI session state")
        if state.episode is not None:
            # No model or tool call starts after this, and the ones in flight are cancelled.
            state.episode.crew_run.cancel()
        if state.task is not None and not state.task.done():
            state.task.cancel()
            await asyncio.wait({state.task}, timeout=self.config.session_close_timeout_seconds)
            if not state.task.done():
                raise RuntimeError("The CrewAI run did not stop; close again to retry")
        sandbox_record = None
        if state.shell is not None:
            sandbox_record = await self._release_sandbox(state)
        if state.observations is None:
            if state.episode is not None:
                # Close cut the activation short: keep what the crew did until then.
                state.episode.gaps.append(
                    ObservationGap(code="agent_activation_interrupted", invocation_id=state.episode.invocation_id)
                )
            self._record_observations(state, status="incomplete")
        if sandbox_record is not None:
            if sandbox_record["outcome"] == "unknown":
                # Teardown went through, so the sandbox's outcome is that of the crew's work in it.
                sandbox_record["outcome"], error_type = state.run_end
                sandbox_record["error_type"] = sandbox_record["error_type"] or error_type
            state.observations.records.append(SandboxObservation.model_validate(sandbox_record))
        # No trajectory: rollout collection builds the turns from the captured calls, and joins every call, failed
        # attempts included, to the invocation by its session header.
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=state.observations,
            cleanup_confirmed=True,
        )

    async def _release_sandbox(self, state: CrewAISessionState) -> dict[str, Any] | None:
        """Kill shell commands still running and remove the session's files, then stop the agent's own sandbox or
        disconnect from a lent one. Returns the agent's observation of the sandbox."""
        shell = state.shell
        timeout = self.config.session_close_timeout_seconds
        try:
            await asyncio.wait_for(shell.release(timeout_seconds=timeout), timeout)
        except Exception:
            if not state.owns_sandbox:
                raise
            # Stopping the agent's own sandbox ends its commands too.
            LOG.warning("Could not release the shell session; stopping the sandbox instead", exc_info=True)
        if state.owns_sandbox:
            await asyncio.wait_for(shell.sandbox.stop(), timeout)
            record = lifecycle_observation(shell.sandbox, role="agent")
        else:
            # Before disconnecting, while the connection still names the provider.
            record = lifecycle_observation(shell.sandbox, role="agent")
            await asyncio.wait_for(shell.sandbox.disconnect(), timeout)
        state.shell = None
        return record

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
            granted = {tool.name for grant in state.grants for tool in grant.tools}
            tools = set()
            for tool in body.tools:
                tool_type = tool.get("type") if isinstance(tool, Mapping) else getattr(tool, "type", None)
                name = tool.get("name") if isinstance(tool, Mapping) else getattr(tool, "name", None)
                if tool_type != "function" or name not in granted:
                    raise HTTPException(
                        422, f"CrewAI runs request tools only through an MCP grant; none serves {name or tool_type}"
                    )
                tools.add(name)
            tools = frozenset(tools)
        timeout = self.config.timeout
        if state.request.agent_timeout_sec is not None:
            timeout = min(timeout, state.request.agent_timeout_sec)
        sampling = {
            key: value
            for key, value in (
                ("temperature", body.temperature),
                ("top_p", body.top_p),
                # A crew makes many model calls: the request's output cap applies to each of them.
                ("max_tokens", body.max_output_tokens),
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
            url_path = self.url_path_for_request("/v1/chat/completions", request)
            state.task = asyncio.create_task(self._activate(state, instruction, url_path))
        elif body != state.activation_request:
            raise HTTPException(409, "CrewAI sessions support one activation; retry the same request")
        # A disconnected HTTP waiter must not cancel the shared run. Close owns cancellation; identical retries
        # join this task or replay its result or error.
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def _activate(self, state: CrewAISessionState, instruction: _Instruction, url_path: str) -> NeMoGymResponse:
        async with self._semaphore:
            episode = _Episode(
                invocation_id=state.request.episode_id.capture_key,
                rollout_id=state.request.episode_id.capture_key,
                model_ref=self.config.model_server,
            )
            state.episode = episode
            llm = GymChatLLM(
                crew_run=episode.crew_run,
                complete=lambda messages, tools: self._complete(episode, instruction, url_path, messages, tools),
                model=instruction.model or self.config.model_server.name,
                context_window_size=self.config.context_window_size,
            )
            crew = build_crew(
                llm=llm,
                tools=self._crew_tools(state, episode, instruction),
                role=self.config.role,
                goal=self.config.goal,
                backstory="\n\n".join(part for part in (self.config.backstory, instruction.system) if part),
                description=instruction.description,
                expected_output=self.config.expected_output,
                max_iter=self.config.max_iter,
            )
            status, error = "completed", None
            deadline = asyncio.timeout(instruction.timeout)
            try:
                async with deadline:
                    await crew.akickoff()
            except Exception as raised:
                if isinstance(raised, TimeoutError) and deadline.expired():
                    # The time budget ran out: the partial work is graded, as for a harness at its own limit.
                    status = "incomplete"
                    episode.gaps.append(ObservationGap(code="agent_run_timeout", invocation_id=episode.invocation_id))
                else:
                    # Including a model request that timed out, which CrewAI raises as it is.
                    status, error = "failed", raised
            finally:
                episode.crew_run.cancel()
            if episode.infrastructure_error is not None:
                state.run_end = ("failed", "infrastructure_error")
                self._record_observations(state, status="failed", error_type="infrastructure_error")
                raise HTTPException(503, f"CrewAI episode infrastructure failed: {episode.infrastructure_error}")
            error_type = type(error).__name__ if error else None
            state.run_end = ("timeout" if status == "incomplete" else status, error_type)
            response = self._response(state, instruction, status=status, error=error)
            self._record_observations(state, status=status, error_type=error_type)
            return response

    def _crew_tools(self, state: CrewAISessionState, episode: _Episode, instruction: _Instruction) -> list[Any]:
        tools: list[Any] = []
        shell = state.shell
        if shell is not None:
            episode.tool_names[SHELL_TOOL_NAME] = SHELL_TOOL_NAME

            async def run_command(command: str) -> str:
                async def execute() -> tuple[str, str, str | None]:
                    try:
                        result: ShellResult = await shell.run(command)
                    except ShellRunnerError as error:
                        raise _InfrastructureError(f"shell command failed in the sandbox: {error}") from error
                    except Exception as error:
                        raise _InfrastructureError(f"sandbox exec failed: {type(error).__name__}: {error}") from error
                    if result.timed_out:
                        outcome = ("timeout", "timeout")
                    elif result.exit_code:
                        outcome = ("failed", f"exit_code_{result.exit_code}")
                    else:
                        outcome = ("completed", None)
                    return result.for_model(shell.timeout_seconds), *outcome

                return await episode.run_tool(SHELL_TOOL_NAME, {"command": command}, execute)

            tools.append(
                ShellTool(
                    crew_run=episode.crew_run,
                    execute=run_command,
                    description=(
                        f"Run a bash command in the task's environment. Each call starts a fresh bash in {shell.workdir}"
                        ", so cd and variables do not carry over. Returns the combined stdout and stderr and the exit "
                        f"code. A command is killed after {shell.timeout_seconds:g} seconds."
                    ),
                )
            )
        for grant in state.grants:
            for tool in grant.tools:
                if instruction.tools is None or tool.name in instruction.tools:
                    episode.tool_names[tool_name(tool.name)] = tool.name
                    tools.append(self._grant_tool(episode, grant.access, tool))
        return tools

    def _grant_tool(self, episode: _Episode, access: MCPToolAccess, tool: Tool) -> GrantTool:
        timeout = self.config.mcp_timeout_seconds

        async def call(arguments: dict[str, Any]) -> Any:
            async def execute() -> tuple[Any, str, str | None]:
                try:
                    result: CallToolResult = await asyncio.wait_for(
                        mcp_client.call_tool(access, tool.name, arguments, timeout_seconds=timeout), timeout
                    )
                except TimeoutError:
                    message = f"The tool call timed out after {timeout:g} seconds"
                    return grant_tool_result(message, failed=True), "timeout", "timeout"
                except McpError as error:
                    # A JSON-RPC error from the server, such as invalid arguments: the model sees it.
                    return grant_tool_result(str(error), failed=True), "failed", "mcp_error"
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    raise _InfrastructureError(
                        f"MCP grant {access.name} failed: {type(error).__name__}: {error}"
                    ) from error
                if result.isError:
                    return grant_tool_result(result.content, failed=True), "failed", "tool_error"
                return grant_tool_result(result.content, failed=False), "completed", None

            return await episode.run_tool(tool.name, arguments, execute)

        name = tool_name(tool.name)
        return GrantTool(
            crew_run=episode.crew_run,
            execute=call,
            name=name,
            description=tool.description or f"Tool {tool.name} from {access.name}",
            args_schema=grant_tool_args_schema(name, tool.inputSchema),
        )

    async def _complete(
        self,
        episode: _Episode,
        instruction: _Instruction,
        url_path: str,
        messages: list[Any],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """One model call of the crew, with bounded retries of transient HTTP statuses."""
        question = _chat_messages(messages)
        episode.record_tool_results(question)
        payload: dict[str, Any] = {"messages": question, **instruction.sampling}
        if instruction.model:
            payload["model"] = instruction.model
        if tools:
            payload["tools"] = tools
            if not instruction.parallel_tool_calls:
                payload["parallel_tool_calls"] = False
        retry = 0
        while True:
            try:
                response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path=url_path,
                    json=payload,
                    cookies=episode.model_cookies,
                    # Rollout collection assigns each captured call, failed ones included, to this invocation.
                    headers={"x-session-id": episode.invocation_id},
                )
            except (aiohttp.ClientError, TimeoutError):
                episode.model_failure = AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
                raise
            if response.ok:
                break
            detail = (await response.content.read()).decode(errors="replace")
            if retry < self.config.max_model_retries and is_retryable_dependency_status(response.status):
                LOG.warning(
                    "Retrying model HTTP %s (%s/%s) rollout_id=%s",
                    response.status,
                    retry + 1,
                    self.config.max_model_retries,
                    episode.rollout_id,
                )
                await asyncio.sleep(0.5 * 2 ** min(retry, 4))
                retry += 1
                continue
            episode.model_failure = AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=response.status)
            raise _ModelRequestError(response.status, detail)
        data = await get_response_json(response)
        episode.model_cookies.update({name: morsel.value for name, morsel in response.cookies.items()})
        # A model call that went through clears an earlier failure that CrewAI recovered from.
        episode.model_failure = None
        return episode.record_model_response(data)

    def _response(
        self, state: CrewAISessionState, instruction: _Instruction, *, status: str, error: Exception | None
    ) -> NeMoGymResponse:
        episode = state.episode
        metadata = {
            "crewai_model_calls": str(len(episode.model_calls)),
            "crewai_tool_calls": str(len(episode.tool_runs)),
        }
        incomplete_details = None
        if status == "completed" and episode.finish_reason == "length":
            # The final answer was cut off at the output cap.
            status, incomplete_details = "incomplete", {"reason": "max_output_tokens"}
        if status == "incomplete" and incomplete_details is None:
            metadata["crewai_stop_reason"] = "timeout"
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=instruction.model or self.config.model_server.name,
            object="response",
            output=episode.output,
            status=status,
            incomplete_details=incomplete_details,
            error=ResponseError(code="server_error", message=f"{type(error).__name__}: {error}"[:2000])
            if error is not None
            else None,
            metadata=metadata,
            tool_choice=state.activation_request.tool_choice,
            tools=state.activation_request.tools,
            parallel_tool_calls=state.activation_request.parallel_tool_calls,
            usage=episode.response_usage(),
        )
        if error is not None:
            return self.report_outcome(response, {"outcome": episode.model_failure})
        return response

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """A crew run that raised: the model request that failed, if one did; CrewAI's own errors are unknown."""
        outcome = native_result.get("outcome")
        return outcome if isinstance(outcome, AgentOutcome) else AgentOutcome(failure_kind="unknown")

    def _record_observations(self, state: CrewAISessionState, *, status: str, error_type: str | None = None) -> None:
        episode = state.episode
        if episode is None:
            # Close before any activation: there is nothing the crew did.
            state.observations = AgentObservationBundle(
                source="crewai_agent",
                gaps=[*state.gaps, ObservationGap(code="observation_capture_failed", detail="no activation")],
            )
            return
        tool_calls = episode.tool_observations()
        if episode.usage is not None and not episode.cached_known:
            episode.gaps.append(
                ObservationGap(code="cached_token_usage_unavailable", invocation_id=episode.invocation_id)
            )
        if episode.usage is not None and not episode.reasoning_known:
            episode.gaps.append(
                ObservationGap(code="reasoning_token_usage_unavailable", invocation_id=episode.invocation_id)
            )
        input_items = state.activation_request.input
        if isinstance(input_items, str):
            input_items = [{"role": "user", "content": input_items, "type": "message"}]
        invocation = AgentInvocation(
            invocation_id=episode.invocation_id,
            status=status,
            error_type=error_type,
            duration_ms=(monotonic() - episode.started) * 1000,
            model_calls=episode.model_calls,
            conversation=[*input_items, *episode.output],
        )
        state.observations = AgentObservationBundle(
            source="crewai_agent", records=[invocation, *tool_calls], gaps=[*state.gaps, *episode.gaps]
        )

    async def run(self, request: Request, body: BaseRunRequest = Body()) -> None:
        raise HTTPException(409, "CrewAI runs only in agent sessions; use an Environment Server")


if __name__ == "__main__":
    CrewAIAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CrewAIAgent.run_webserver()  # noqa: F401
