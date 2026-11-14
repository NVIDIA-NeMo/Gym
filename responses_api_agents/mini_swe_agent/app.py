# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""mini-SWE agent: SWE-Gym ``/run`` on the agent server, and agent sessions in the task sandbox.

``/run`` runs the SWE-Gym pipeline of the NVIDIA mini-swe-agent fork installed in this server's environment, through
a Ray worker in Docker or Singularity, and grades the result itself. Agent sessions run mini-SWE 2.4.6 in the
sandbox the Resources Server lends, or in one this agent starts from ``sandbox_provider``; the Environment Server
seeds, activates, closes and verifies them. The two paths share no runtime: the session runtime is built from
``runtime/requirements.txt`` and installed in the sandbox at seed.
"""

import asyncio
import json
import logging
import sys
from asyncio import Semaphore
from collections.abc import Mapping
from os import environ, getenv, makedirs
from pathlib import Path, PurePosixPath
from shlex import quote
from time import time
from typing import Any, Callable, ClassVar, Literal, Optional
from uuid import uuid4

import ray
import yaml
from fastapi import HTTPException, Request
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.run.extra.swegym_runner import _main as run_swegym
from pydantic import ConfigDict, Field, JsonValue

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
    SandboxNetworkAccess,
    SandboxSession,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
    resolve_agent_timeout,
    restrict_sandbox_network,
)
from nemo_gym.config import (
    ModelServerRef,
    ResourcesServerRef,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.core import (
    AGENT_INVALID_OUTPUT,
    MODEL_CONTEXT_LIMIT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NgFailureEvent,
    observe_failure,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    MCPToolAccess,
)
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap, SandboxObservation
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.mini_swe_agent.sandbox import (
    MINI_SWE_VERSION,
    SESSION_ROOT,
    MiniSWEArtifacts,
    MiniSWESession,
)
from responses_api_agents.mini_swe_agent.trajectory import observation_records, project
from responses_api_agents.mini_swe_agent.utils import MiniSWEAgentUtils


LOG = logging.getLogger(__name__)

# Request fields an agent session forwards to every model call, as mini-SWE's model_kwargs.
FORWARDED_REQUEST_FIELDS = frozenset(
    {"instructions", "temperature", "top_p", "max_output_tokens", "reasoning", "parallel_tool_calls", "tool_choice"}
)
# Read by the adapter: the input is the task, metadata may carry the task's time budget, and model must match.
# ``tools`` is not forwarded: mini-SWE offers the model only its bash tool, and granted MCP tools are called through it.
CONSUMED_REQUEST_FIELDS = frozenset({"input", "metadata", "model", "tools", "stream"})
# mini-SWE 2.4.6 exit statuses that stop the loop at a limit; the run is the agent's own result and is verified.
LIMIT_EXIT_STATUSES = frozenset({"LimitsExceeded", "TimeExceeded", "ContextWindowExceededError"})
# The LiteLLM exceptions mini-SWE records as its exit status when a model call fails, with the HTTP status each means.
MODEL_ERROR_STATUS_CODES = {
    "BadRequestError": 400,
    "AuthenticationError": 401,
    "PaymentRequiredError": 402,
    "PermissionDeniedError": 403,
    "NotFoundError": 404,
    "UnprocessableEntityError": 422,
    "RateLimitError": 429,
    "InternalServerError": 500,
    "BadGatewayError": 502,
    "ServiceUnavailableError": 503,
}
MODEL_TRANSPORT_EXIT_STATUSES = frozenset({"APIConnectionError", "Timeout", "APIError"})
# LiteLLM exceptions that end a model call with no HTTP answer, whatever status attribute they carry.
MODEL_NO_ANSWER_EXIT_STATUSES = frozenset({"APIConnectionError", "Timeout"})
# Set by Gym's vLLM Model Server on the empty reply it returns for a prompt over the context window.
CONTEXT_WINDOW_EXCEEDED_METADATA_KEY = "nemo_gym_context_window_exceeded"
# The sandbox runner's own exit statuses (see sandbox_runner.py).
RUNNER_MCP_UNAVAILABLE_EXIT = 3


class MiniSWEAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    concurrency: int
    # Seconds a bash command may run, and the most model calls in one run (0: no limit); both paths.
    step_timeout: int = 600
    step_limit: int = 250
    # /run only: the SWE-Gym pipeline of the NVIDIA mini-swe-agent fork.
    env: Optional[Literal["docker", "singularity"]] = None
    cache_dir_template: Optional[str] = None
    run_golden: bool = False
    eval_timeout: int = 1800
    skip_if_exists: bool = False
    collapse_limit: int = 3
    # Agent sessions only. /run grades itself and never calls a Resources Server; agent sessions leave this null.
    resources_server: Optional[ResourcesServerRef] = None
    # The model name mini-SWE sends to the Model Server; defaults to the model server's name.
    model: Optional[str] = None
    # An agent-owned sandbox, for an episode whose Resources Server lends none.
    sandbox_provider: Optional[str] = None
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)
    network_access: SandboxNetworkAccess = Field(
        default="inherit",
        description=(
            "Egress of a sandbox mini-SWE starts: `model_only` allows only the Model Server's host, `model_and_tools` "
            "also each MCP grant's host. A borrowed task sandbox keeps its owner's network, so a session that borrows "
            "one refuses any value other than `inherit`."
        ),
    )
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    # The longest a run may take; a task's own agent timeout can only shorten it.
    sandbox_runner_timeout_seconds: float = Field(default=28800, gt=0, allow_inf_nan=False)


class MiniSWEAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    # Optional per-request policy endpoint override. External trainers that
    # route each episode through its own OpenAI-compatible URL (e.g. RL
    # trainers recording rollouts via a per-episode proxy) set these per /run;
    # when unset, the configured model_server is used, as before.
    policy_base_url: Optional[str] = None
    policy_api_key: Optional[str] = None


class MiniSWEAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class MiniSWEAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)


def _instruction(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """The task: the text of every input message, in order."""
    parts: list[str] = []
    for item in [body.input] if isinstance(body.input, str) else body.input:
        if isinstance(item, str):
            parts.append(item)
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                value = part if isinstance(part, dict) else part.model_dump()
                if value.get("type") not in ("input_text", "output_text"):
                    raise HTTPException(422, "mini-SWE accepts text input only")
                parts.append(value.get("text") or "")
        else:
            raise HTTPException(422, f"mini-SWE accepts message input only, not {getattr(item, 'type', item)!r}")
    instruction = "\n\n".join(part for part in parts if part)
    if not instruction.strip():
        raise HTTPException(422, "mini-SWE requires a nonempty task instruction")
    return instruction


def _context_overflow(exit_status: str | None, error: str) -> bool:
    """Whether a failed model call reported the model's context window as exceeded (mini-SWE 2.4.6, LiteLLM)."""
    if exit_status == "ContextWindowExceededError":
        return True
    lowered = error.lower()
    return exit_status == "BadRequestError" and (
        "context_length_exceeded" in error
        or "context length" in lowered
        or "maximum model length" in lowered
        or ("max_tokens" in error and "too large" in lowered)
    )


class MiniSWEAgent(SimpleResponsesAPIAgent):
    ray_enabled = True
    # Agent sessions run mini-SWE inside the task sandbox, which calls the Model Server from there.
    calls_model_from_sandbox: ClassVar[bool] = True
    config: MiniSWEAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.sem = Semaphore(self.config.concurrency)

    def supports_task_tool_context(self) -> bool:
        """mini-SWE reads the task's MCP servers and skills directory from ``task.json`` at seed."""
        return True

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Read the exit status mini-SWE 2.4.6 records when a model call fails; tested on that release."""
        exit_status = native_result.get("exit_status")
        if _context_overflow(exit_status, str(native_result.get("error") or "")):
            return AgentOutcome(failure_kind=MODEL_CONTEXT_LIMIT)
        status = native_result.get("status_code")
        if exit_status not in MODEL_NO_ANSWER_EXIT_STATUSES and type(status) is int and 400 <= status <= 599:
            # The status of the HTTP answer that ended the run, from the LiteLLM exception that recorded it.
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=status)
        if exit_status in MODEL_ERROR_STATUS_CODES:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=MODEL_ERROR_STATUS_CODES[exit_status])
        if exit_status in MODEL_TRANSPORT_EXIT_STATUSES:
            return AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
        if exit_status in ("RepeatedFormatError", "FormatError"):
            return AgentOutcome(failure_kind=AGENT_INVALID_OUTPUT)
        return AgentOutcome(failure_kind="unknown")

    # -- agent sessions ------------------------------------------------------------------------------------------

    def _model_name(self) -> str:
        return self.config.model or str(self.config.model_server.name)

    def _require_agent_session(self, agent_session_id: str) -> MiniSWESession:
        state = super()._require_agent_session(agent_session_id)
        if not isinstance(state, MiniSWESession):
            raise TypeError("Expected a mini-SWE agent session state")
        return state

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, MiniSWESession) else None

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> MiniSWESession:
        """Connect to or start the task sandbox, install the runtime, and read the task's own tool declarations."""
        grants = self.effective_tool_accesses(body)
        unsupported = [access.name for access in grants if access.required and not isinstance(access, MCPToolAccess)]
        if unsupported:
            raise AgentSeedRejectedError(
                "mini-SWE supports only MCP tool grants; required grants it cannot use: "
                + ", ".join(sorted(unsupported))
            )
        mcp_grants = [access for access in grants if isinstance(access, MCPToolAccess)]
        owns_sandbox = body.sandbox_access is None
        if owns_sandbox:
            if self.config.sandbox_provider is None:
                raise AgentSeedRejectedError("mini-SWE requires sandbox_access or a configured sandbox_provider")
            provider_ref = self.config.sandbox_provider
            workdir = self.config.sandbox_config.get("workdir") or "/app"
        else:
            access = body.sandbox_access
            if not isinstance(access.connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("mini-SWE supports only direct sandbox connections")
            if access.user is not None:
                raise AgentSeedRejectedError("mini-SWE runs as the sandbox's default user; a task user is unsupported")
            if self.config.network_access != "inherit":
                raise AgentSeedRejectedError(
                    "mini-SWE network_access applies only to a sandbox it starts, not a borrowed one"
                )
            provider_ref = access.connection.provider_config_ref
            workdir = access.workdir
        if (
            not isinstance(workdir, str)
            or not workdir.startswith("/")
            or ".." in PurePosixPath(workdir).parts
            or PurePosixPath(SESSION_ROOT).is_relative_to(workdir)
            or PurePosixPath(workdir).is_relative_to(SESSION_ROOT)
        ):
            raise AgentSeedRejectedError("mini-SWE needs an absolute task workdir disjoint from its session directory")

        global_config = get_global_config_dict()
        # mini-SWE calls the Model Server from inside the sandbox, where a loopback URL is the sandbox itself.
        try:
            provider_config = resolve_provider_config(provider_ref, global_config)
            assert_model_url_reachable_from_sandbox(
                self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key),
                provider_name=next(iter(provider_config), None),
            )
            spec = None
            if owns_sandbox:
                settings = dict(self.config.sandbox_config)
                metadata = settings.pop("metadata", None) or {}
                settings["metadata"] = {
                    **resolve_provider_metadata(provider_ref, global_config),
                    **metadata,
                    "agent_session_id": body.agent_session_id[:63],
                }
                settings.setdefault("workdir", workdir)
                # Provider TTLs outlive this server; reserve ten minutes for setup and cleanup.
                settings.setdefault(
                    "ttl_s",
                    self.config.sandbox_install_timeout_seconds + self.config.sandbox_runner_timeout_seconds + 600,
                )
                spec = SandboxSpec(**settings)
            provider = create_provider(provider_config)
        except (TypeError, ValueError) as error:
            raise AgentSeedRejectedError(str(error)) from error
        if spec is not None:
            try:
                spec = restrict_sandbox_network(
                    spec,
                    network_access=self.config.network_access,
                    provider=provider,
                    model_base_url=self.resolve_model_base_url(self.config.model_server.name, trace_context=False),
                    tool_urls=[str(access.connection.url) for access in mcp_grants],
                    # model_only blocks these hosts, so a required grant is refused before the sandbox exists.
                    required_tool_urls=[str(access.connection.url) for access in mcp_grants if access.required],
                )
            except BaseException:
                await provider.aclose()
                raise
        try:
            if owns_sandbox:
                sandbox = AsyncSandbox(provider)
            else:
                sandbox = await AsyncSandbox.connect(body.sandbox_access.connection.descriptor, provider=provider)
        except BaseException:
            try:
                await provider.aclose()
            except BaseException as cleanup_error:
                LOG.exception("mini-SWE provider cleanup failed after sandbox setup failed")
                if isinstance(cleanup_error, NgFailureEvent):
                    observe_failure(cleanup_error)
                else:
                    cleanup_failure = NgFailureEvent(
                        boundary="sandbox",
                        failure_kind="session_release_failed",
                        failure_reason="mini-SWE provider cleanup failed",
                        operation="aclose",
                        stage="cleanup",
                    )
                    cleanup_failure.__cause__ = cleanup_error
                    observe_failure(cleanup_failure)
            raise
        state = MiniSWESession(
            request=body,
            provider_name=provider_ref,
            sandbox_id=sandbox.sandbox_id,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=f"{SESSION_ROOT}/{uuid4().hex}",
                workdir=workdir,
                harness="mini-SWE",
                owns_sandbox=owns_sandbox,
            ),
        )
        try:
            if owns_sandbox:
                await sandbox.start(spec)
                state.sandbox_id = sandbox.sandbox_id
                prepared = await sandbox.exec(f"mkdir -p -- {quote(workdir)}", cwd="/", timeout_s=30)
                if prepared.return_code != 0 or prepared.error_type is not None:
                    raise RuntimeError(f"Could not prepare the mini-SWE task workdir {workdir}")
            await state.install(timeout=self.config.sandbox_install_timeout_seconds)
            state.task_context = {} if owns_sandbox else await state.read_task_context()
            state.mcp_servers, state.required_mcp_servers = self._mcp_servers(mcp_grants, state.task_context)
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                LOG.exception(
                    "Could not clean failed mini-SWE setup %s; retaining it for close", body.agent_session_id
                )
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    @staticmethod
    def _mcp_servers(
        grants: list[MCPToolAccess], task_context: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """The MCP servers the task-local CLI connects to: the episode's grants, then the task's own servers.

        Every granted server that is required must connect before the first model call. The task's own servers are
        required too, as the task declares them for its tools.
        """
        servers = [
            {
                "name": grant.name,
                "transport": grant.connection.transport,
                "url": str(grant.connection.url),
                "headers": grant.connection.headers,
            }
            for grant in grants
        ]
        required = [grant.name for grant in grants if grant.required]
        granted = {grant.name for grant in grants}
        for server in task_context.get("mcp_servers") or []:
            if not isinstance(server, dict) or not isinstance(server.get("name"), str):
                raise AgentSeedRejectedError("The task declares an MCP server without a name")
            if server["name"] in granted:
                # A Harbor environment server can also grant the task's own server, under its own name.
                required.extend([server["name"]] if server["name"] not in required else [])
                continue
            servers.append(server)
            required.append(server["name"])
        names = [server["name"] for server in servers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise AgentSeedRejectedError("MCP server names collide: " + ", ".join(duplicates))
        return servers, required

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, MiniSWESession):
            raise TypeError("Expected a mini-SWE agent session state")
        await state.close(self.config.session_close_timeout_seconds)
        # A run that was stopped or failed before its response was built still left its artifacts.
        if state.observations is None and state.session.artifacts is not None:
            task, cleanup = state.task, state.session.cleanup
            failed = task is not None and task.done() and not task.cancelled() and task.exception() is not None
            stopped = None if failed else "timeout" if cleanup and cleanup["timed_out"] else "cancelled"
            try:
                state.observations = self._observations(
                    state, state.session.artifacts, stopped=stopped, status="failed" if failed else "incomplete"
                )
            except Exception:
                LOG.exception("Could not build mini-SWE observations at close")
        observations = state.observations or AgentObservationBundle(
            source="mini_swe", gaps=[ObservationGap(code="observation_capture_failed")]
        )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=observations,
            cleanup_confirmed=state.session.cleanup_confirmed,
        )

    @staticmethod
    def _validate_output_budget(body: NeMoGymResponseCreateParamsNonStreaming) -> None:
        """Refuse a token limit the legacy SWE-Gym fork cannot forward."""
        if body.max_output_tokens is not None:
            raise HTTPException(
                422,
                "mini-SWE legacy /run does not support request max_output_tokens; "
                "use agent sessions for a per-call limit, or configure it on the legacy model endpoint",
            )

    def _validate_request(self, body: NeMoGymResponseCreateParamsNonStreaming) -> None:
        """Apply, consume or reject every request field, so no control is dropped silently."""
        if body.model is not None and body.model != self._model_name():
            raise HTTPException(422, "mini-SWE request model must match the configured model")
        if body.stream:
            raise HTTPException(422, "mini-SWE agent sessions do not stream")
        if body.tool_choice not in (None, "auto", "required"):
            # The model is offered only mini-SWE's bash tool, so it can neither name another tool nor call none.
            raise HTTPException(422, "mini-SWE accepts only tool_choice auto or required: its only tool is bash")
        if body.metadata and set(body.metadata) - {AGENT_TIMEOUT_METADATA_KEY}:
            raise HTTPException(422, f"mini-SWE accepts only {AGENT_TIMEOUT_METADATA_KEY} in the request metadata")
        for name, field in type(body).model_fields.items():
            if name in FORWARDED_REQUEST_FIELDS or name in CONSUMED_REQUEST_FIELDS:
                continue
            if getattr(body, name) != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"mini-SWE agent sessions do not support request field {name}")
        _instruction(body)

    def _agent_timeout_sec(self, state: MiniSWESession, body: NeMoGymResponseCreateParamsNonStreaming) -> float:
        """The run's budget: the configured ceiling, shortened by the task's own agent timeout."""
        ceiling = min(
            self.config.sandbox_runner_timeout_seconds,
            state.request.agent_timeout_sec or self.config.sandbox_runner_timeout_seconds,
        )
        return resolve_agent_timeout(body.metadata, ceiling=ceiling)

    def _activation_payload(
        self, state: MiniSWESession, body: NeMoGymResponseCreateParamsNonStreaming, *, timeout: float
    ) -> dict[str, Any]:
        """The runner's input: the task, mini-SWE's configuration overlay, and the session's files."""
        model_kwargs = {
            name: value
            for name, value in body.model_dump(mode="json", exclude_none=True).items()
            if name in FORWARDED_REQUEST_FIELDS
        }
        # LiteLLM raises small OpenAI token caps to 16; extra_body preserves the caller's exact per-call limit.
        if "max_output_tokens" in model_kwargs:
            model_kwargs["extra_body"] = {"max_output_tokens": model_kwargs.pop("max_output_tokens")}
        directory = state.session.session_dir
        capture_key = state.request.episode_id.capture_key
        model_kwargs.update(
            # The sandbox reaches the Model Server directly; the rollout prefix keeps its calls correlated.
            api_base=self.resolve_model_base_url(self.config.model_server.name, capture_key),
            api_key="gym",
            timeout=max(1, timeout),
            max_retries=0,
            # Ownership of every model call, failed ones included, for the invocation reported at close.
            extra_headers={"x-session-id": capture_key},
        )
        context = state.task_context
        environment_variables = dict(state.request.sandbox_access.env) if state.request.sandbox_access else {}
        return {
            "task": _instruction(body),
            "skills_dir": context.get("skills_dir"),
            "config": {
                "agent": {"step_limit": self.config.step_limit, "cost_limit": 0},
                "environment": {
                    "cwd": state.session.workdir,
                    "timeout": max(1, int(min(timeout, self.config.step_timeout))),
                    **({"env": environment_variables} if environment_variables else {}),
                },
                "model": {
                    "model_class": "litellm_response",
                    "model_name": f"openai/{self._model_name()}",
                    "cost_tracking": "ignore_errors",
                    "model_kwargs": model_kwargs,
                },
            },
            "trajectory_path": f"{directory}/trajectory.json",
            "mcp": (
                {"servers": [server["name"] for server in state.mcp_servers], "required": state.required_mcp_servers}
                if state.mcp_servers
                else None
            ),
            "stop_request_path": state.session.stop_request_path,
        }

    def _observations(
        self, state: MiniSWESession, artifacts: MiniSWEArtifacts, *, stopped: str | None, status: str = "incomplete"
    ) -> AgentObservationBundle:
        projection = project(artifacts.trajectory, artifacts.mcp_calls)
        invocation = AgentInvocation(
            invocation_id=state.request.episode_id.capture_key,
            status=status,
            stop_reason={"timeout": "wall_time"}.get(stopped, stopped),
        )
        records, gaps = observation_records(
            projection,
            invocation=invocation,
            model_server=self.config.model_server,
            sandbox_id=state.sandbox_id,
            stopped=stopped,
        )
        if artifacts.trajectory is None:
            gaps.append(ObservationGap(code="native_trajectory_unavailable", invocation_id=invocation.invocation_id))
        elif stopped is not None:
            gaps.append(
                ObservationGap(
                    code="native_trajectory_partial",
                    invocation_id=invocation.invocation_id,
                    detail="mini-SWE saves after each model call and each step; work the stop interrupted is absent.",
                )
            )
        if state.runtime_info is None:
            gaps.append(ObservationGap(code="runtime_info_unavailable"))
        if artifacts.mcp_calls_unread:
            gaps.append(
                ObservationGap(
                    code="mcp_call_records_unavailable",
                    invocation_id=invocation.invocation_id,
                    detail="mcp_calls.jsonl was larger than the size Gym reads, so its MCP calls are not reported",
                )
            )
        unavailable = ((artifacts.output or {}).get("mcp") or {}).get("unavailable") or []
        if unavailable:
            # Optional MCP servers that did not connect; the prompt left them out.
            gaps.append(
                ObservationGap(
                    code="mcp_server_unavailable",
                    invocation_id=invocation.invocation_id,
                    detail=json.dumps(unavailable)[:2000],
                )
            )
        if artifacts.invalid_mcp_calls:
            # Lines of mcp_calls.jsonl that are not a call record; left out.
            gaps.append(
                ObservationGap(
                    code="mcp_call_record_invalid",
                    invocation_id=invocation.invocation_id,
                    detail=str(artifacts.invalid_mcp_calls),
                )
            )
        cleanup = state.session.cleanup
        sandbox_outcome = stopped or ("completed" if status in ("completed", "incomplete") else "failed")
        return AgentObservationBundle(
            source="mini_swe",
            records=[
                invocation,
                *records,
                SandboxObservation(
                    role="agent",
                    provider=state.provider_name,
                    sandbox_id=state.sandbox_id,
                    outcome=sandbox_outcome,
                    exit_code=cleanup["return_code"] if cleanup else None,
                    # The category the invocation carries for the stop or failure.
                    error_type={"completed": None, "failed": "harness_failed"}.get(sandbox_outcome, sandbox_outcome),
                ),
            ],
            gaps=gaps,
        )

    async def _activate(self, state: MiniSWESession, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        timeout = self._agent_timeout_sec(state, body)
        async with self.sem:
            artifacts = await state.session.execute(
                stage_activation=lambda: state.stage_activation(
                    self._activation_payload(state, body, timeout=timeout)
                ),
                collect=state.collect,
                timeout=timeout,
                close_timeout=self.config.session_close_timeout_seconds,
            )
        cleanup = state.session.cleanup
        if cleanup is None or not cleanup["cleanup_confirmed"]:
            raise RuntimeError("mini-SWE ended without confirmed supervisor cleanup")
        output = artifacts.output or {}
        trajectory = artifacts.trajectory or {}
        info = trajectory.get("info") or {}
        exit_status = info.get("exit_status") or None
        messages = trajectory.get("messages") or [{}]
        error = str((messages[-1].get("extra") or {}).get("exception_str") or "")
        timed_out = cleanup["timed_out"]
        # Close stopped the run; the activation can still return before close cancels it.
        cancelled = not timed_out and (state.session.closing or output.get("stop_reason") == "cancelled")
        if cleanup["return_code"] == RUNNER_MCP_UNAVAILABLE_EXIT and not timed_out:
            # No model call was made: the episode's tools were unavailable, not the agent's doing.
            message = str((output.get("error") or {}).get("message"))[:2000]
            raise RuntimeError(f"mini-SWE task MCP servers are unavailable: {message}")
        if artifacts.trajectory is None and not (timed_out or cancelled):
            # The runner's error message, without its traceback, or else the end of the run's logs.
            reason = (output.get("error") or {}).get("message") if isinstance(output.get("error"), dict) else None
            raise RuntimeError(
                f"mini-SWE saved no trajectory: {str(reason)[:2000] if reason else artifacts.log[-2000:]}"
            )

        projection = project(artifacts.trajectory, artifacts.mcp_calls)
        if projection.invalid_response:
            raise RuntimeError("mini-SWE saved a model response that is not a valid Responses API object")
        runner_error = output.get("error") if isinstance(output.get("error"), dict) else {}
        native_result = {
            "exit_status": exit_status,
            "error": error,
            # Only the exception that ended mini-SWE's loop, which it records as the exit status, says the status.
            "status_code": runner_error.get("status_code") if runner_error.get("type") == exit_status else None,
        }
        # Gym's vLLM Model Server answers a prompt over the context window with an empty reply that it marks; mini-SWE
        # rejects that reply for its format until it gives up.
        context_limit = _context_overflow(exit_status, error) or (
            exit_status in ("RepeatedFormatError", "FormatError")
            and bool(projection.responses)
            and (projection.responses[-1].metadata or {}).get(CONTEXT_WINDOW_EXCEEDED_METADATA_KEY) == "true"
        )
        failed = False
        if timed_out and not projection.responses and not state.session.expired:
            # The supervisor proves the deadline, not whether a model request started or failed in transport.
            # Keep the retryable failed result without inventing mini-SWE's native model Timeout exception.
            exit_status, error = "AgentDeadline", "no model call returned before the run's deadline"
            native_result = {"exit_status": exit_status, "error": error}
            termination, status, failed = exit_status, "failed", True
        elif timed_out:
            termination, status = "timeout", "incomplete"
        elif exit_status == "Submitted":
            termination, status = "completed", "completed"
        elif context_limit or exit_status in LIMIT_EXIT_STATUSES:
            termination, status = "ContextWindowExceededError" if context_limit else exit_status, "incomplete"
        elif exit_status is None:
            # mini-SWE records an exit status whenever its loop ends, even on an exception, so a saved trajectory
            # without one means the run was stopped by close or killed mid-step: by the agent's own command, or by an
            # OOM kill.
            termination, status = "cancelled" if cancelled else "killed", "incomplete"
        elif self.classify_outcome(native_result).failure_kind != "unknown":
            termination, status, failed = exit_status, "failed", True
        else:
            raise RuntimeError(f"mini-SWE stopped with {exit_status}: {error[:2000]}")

        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self._model_name(),
            object="response",
            output=projection.output(),
            usage=projection.usage,
            status=status,
            error={"code": "server_error", "message": f"{exit_status}: {error[:2000]}"} if failed else None,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            metadata={
                "harness_execution": "sandbox",
                "mini_swe_version": MINI_SWE_VERSION,
                "mini_swe_termination": termination,
                AGENT_TIMEOUT_METADATA_KEY: str(timeout),
                **({"stop_reason": "wall_time"} if timed_out else {}),
                **(
                    {"harness_hostname": state.runtime_info.hostname, "harness_pid": str(state.runtime_info.pid)}
                    if state.runtime_info
                    else {}
                ),
            },
        )
        # A run that ended on its own before close stopped it keeps its own outcome.
        stopped = "timeout" if timed_out else "cancelled" if cancelled and exit_status is None else None
        state.observations = self._observations(
            state, artifacts, stopped=stopped, status="failed" if failed else status
        )
        if projection.usage is None and (projection.responses or projection.model_calls):
            state.observations.gaps.append(ObservationGap(code="usage_unavailable"))
        response = self.report_rollout_stop(response, state.session)
        return self.report_outcome(response, native_result)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        agent_session_id = self._agent_session_id_from_request(request)
        if agent_session_id is None:
            # mini-SWE has no host path: /run runs the SWE-Gym pipeline, agent sessions run in the task sandbox.
            raise HTTPException(422, "mini-SWE responses require an agent session; use an Environment Server")
        self._validate_request(body)
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        state = self._require_agent_session(agent_session_id)
        if body.tools and not state.mcp_servers:
            raise HTTPException(
                422, "mini-SWE reaches request tools only through MCP grants; set resources_tool_transports: [mcp]"
            )
        if state.request.episode_id.capture_key != rollout_id:
            raise HTTPException(409, "Agent-session episode_id does not match the rollout route")
        if state.task is None:
            # No await until the task and its request binding are installed.
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._activate(state, state.activation_request))
        elif body != state.activation_request:
            raise HTTPException(409, "mini-SWE sessions support one activation; retry the same request")
        # A disconnected waiter must not cancel the shared activation; close owns cancellation.
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    # -- /run: the SWE-Gym pipeline ------------------------------------------------------------------------------

    async def run(self, body: MiniSWEAgentRunRequest) -> MiniSWEAgentVerifyResponse:
        if self.config.env is None:
            raise HTTPException(422, "mini-SWE /run requires env (docker or singularity); agent sessions do not")
        self._validate_output_budget(body.responses_create_params)
        async with self.sem:
            model_server_name = self.config.model_server.name
            global_config_dict = self.server_client.global_config_dict

            model_server_config = get_first_server_config_dict(
                global_config_dict,
                model_server_name,
            )

            policy_model_name = global_config_dict["policy_model_name"]

            ##### MINI-SWE-AGENT CONFIG #####
            subset = body.subset
            split = body.split
            workers = 1
            cache_dir_template = self.config.cache_dir_template
            run_golden = self.config.run_golden
            if body.policy_base_url:
                # The caller owns routing (and any per-rollout correlation) for
                # its own endpoint; Gym's URL segments only apply to the built-in
                # model server.
                base_url = body.policy_base_url
            else:
                base_url = f"http://{model_server_config['host']}:{model_server_config['port']}"
                base_url = f"{self.base_url_for_run(base_url, body)}/v1"
            api_key = body.policy_api_key or "dummy_key"
            model_name = f"hosted_vllm/{policy_model_name}"
            step_timeout = self.config.step_timeout
            eval_timeout = self.config.eval_timeout
            env = self.config.env
            step_limit = self.config.step_limit
            collapse_limit = self.config.collapse_limit

            instance_id = body.instance_id

            mini_swe_config_path = builtin_config_dir / "extra" / "swebench.yaml"
            config = yaml.safe_load(get_config_path(mini_swe_config_path).read_text())

            default_model_kwargs = config["model"]["model_kwargs"]
            temperature = body.responses_create_params.temperature or default_model_kwargs["temperature"]
            top_p = body.responses_create_params.top_p or default_model_kwargs["top_p"]

            output_file_dir = f"{Path.cwd()}/results/{subset}/{policy_model_name}"

            if self.config.skip_if_exists:
                if Path(f"{output_file_dir}/{instance_id}/{instance_id}.json").exists():
                    with open(f"{output_file_dir}/{instance_id}/{instance_id}.json", "r") as f:
                        print(f"Skipping {instance_id} because it already exists")
                        verify_response = MiniSWEAgentVerifyResponse.model_validate_json(f.read())
                    return verify_response

            env_vars = environ.copy()
            if env == "singularity":
                slurm_job_id = getenv("SLURM_JOB_ID", str(uuid4()))
                env_vars.update(
                    {
                        "SINGULARITY_CACHEDIR": f"/tmp/singularity_cache_${slurm_job_id}_$$",
                        "APPTAINER_CACHEDIR": f"/tmp/apptainer_cache_${slurm_job_id}_$$",
                        "SINGULARITY_TMPDIR": f"/tmp/singularity_tmp_${slurm_job_id}_$$",
                        "APPTAINER_TMPDIR": f"/tmp/apptainer_tmp_${slurm_job_id}_$$",
                    }
                )
                for var in [
                    "SINGULARITY_CACHEDIR",
                    "APPTAINER_CACHEDIR",
                    "SINGULARITY_TMPDIR",
                    "APPTAINER_TMPDIR",
                ]:
                    makedirs(env_vars[var], exist_ok=True)

            #### RUN MINI-SWE-AGENT #####
            reseponses_create_params_dict = body.responses_create_params.model_dump()
            try:
                params = dict(
                    subset=subset,
                    split=split,
                    workers=workers,
                    output=output_file_dir,
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    cache_dir_template=cache_dir_template,
                    env=env,
                    run_golden=run_golden,
                    instance_id=instance_id,
                    # TODO: add this later
                    instance_dict=body.model_dump(),
                    responses_create_params=json.dumps(reseponses_create_params_dict),
                    step_timeout=step_timeout,
                    eval_timeout=eval_timeout,
                    step_limit=step_limit,
                    collapse_limit=collapse_limit,
                )
                future = runner_ray_remote.remote(run_swegym, params)
                result = await asyncio.to_thread(ray.get, future)
                result = result[instance_id]
                messages = result["messages"]
                responses = result["responses"]
                reward = 1.0 if MiniSWEAgentUtils.is_resolved(instance_id, result["eval_report"]) else 0.0

            except Exception as e:
                print(f"Error running swegym: {e}")
                result = None
                messages = []
                responses = []
                reward = 0.0

            # The first two messages are the system and user message generated by the harness
            # TODO(sugam): what if the user only provides the system/user message
            body.responses_create_params.input = messages[:2]

            response = MiniSWEAgentUtils.get_default_response_object()
            response["model"] = policy_model_name
            response["temperature"] = temperature
            response["top_p"] = top_p

            # Wrap output messages in responses format
            response["output"] = MiniSWEAgentUtils.chat_cmp_to_responses(messages[2:], responses)

            verify_response = MiniSWEAgentVerifyResponse(
                responses_create_params=body.responses_create_params,
                reward=reward,
                response=response,
                instance_id=instance_id,
                metadata=result.get("eval_report", {}) if result else {},
            )

            output_path = Path(f"{output_file_dir}/{instance_id}")
            output_path.mkdir(parents=True, exist_ok=True)

            with open(f"{output_file_dir}/{instance_id}/{instance_id}.json", "w") as f:
                json.dump(verify_response.model_dump(), f)

            return verify_response


if __name__ == "__main__":
    MiniSWEAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = MiniSWEAgent.run_webserver()  # noqa: F401
