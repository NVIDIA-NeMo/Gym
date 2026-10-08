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
import asyncio
import atexit
import logging
import os
import shutil
import sys
import tempfile
from asyncio import Semaphore
from collections.abc import Mapping
from time import time
from typing import Any, Callable, Optional
from uuid import uuid4

import model_tools  # noqa: F401  # fail-fast if hermes-agent isn't installed  # pyright: ignore[reportMissingImports]
from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field
from toolsets import TOOLSETS  # pyright: ignore[reportMissingImports]

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers import create_provider
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.tool_access import MCPToolAccess
from responses_api_agents.hermes_agent.model_kwargs import _model_api_kwargs
from responses_api_agents.hermes_agent.observability import HermesAgentObserver, normalize_hermes_messages
from responses_api_agents.hermes_agent.sandbox import HarnessProcessInfo, HermesSandboxSession


def _usage_from_result(result: dict[str, Any]) -> Optional[NeMoGymResponseUsage]:
    # Hermes' prompt total already includes cache reads/writes, and its completion
    # total includes reasoning. Early returns can omit these native aggregates.
    prompt_tokens = result.get("prompt_tokens")
    completion_tokens = result.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    return NeMoGymResponseUsage(
        input_tokens=prompt_tokens,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=result.get("cache_read_tokens")),
        output_tokens=completion_tokens,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=result.get("reasoning_tokens")),
        total_tokens=prompt_tokens + completion_tokens,
    )


def _trajectory_to_output_items(messages, n_input):
    output_items = []
    for item in messages[n_input:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "") or ""
        if isinstance(content, list):
            content = "".join(c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "") for c in content)
        if role == "assistant":
            reasoning_text = item.get("reasoning") or ""
            if reasoning_text:
                content = ResponsesConverter._parse_think_tags(content)[1]
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rsn-{len(output_items)}",
                        summary=[NeMoGymSummary(type="summary_text", text=reasoning_text)],
                        type="reasoning",
                    )
                )
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=content, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=item.get("prompt_token_ids") or [],
                    generation_token_ids=item.get("generation_token_ids") or [],
                    generation_log_probs=item.get("generation_log_probs") or [],
                    routed_experts=item.get("routed_experts"),
                )
            )
            for tc in item.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not fn:
                    continue
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=fn.get("arguments", ""),
                        call_id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        type="function_call",
                        id=tc.get("id"),
                        status="completed",
                    )
                )
        elif role == "tool":
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=item.get("tool_call_id", ""),
                    output=content,
                    status="completed",
                )
            )
    return output_items


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"


def _gym_mcp_tool_name(name: str, server_names: list[str]) -> str:
    """Rename Hermes' ``mcp_<server>_<tool>`` to Gym's ``mcp__<server>__<tool>``, which Gym strips before verify.

    Hermes replaces "-" and "." with "_" in both parts. The server part is matched against the granted names,
    longest first so one name that extends another cannot capture its tools; a tool name that contained "-" or
    "." keeps Hermes' replacement.
    """
    for server in sorted(server_names, key=len, reverse=True):
        prefix = "mcp_" + server.replace("-", "_").replace(".", "_") + "_"
        if name.startswith(prefix):
            return f"mcp__{server}__{name[len(prefix) :]}"
    return name


# if ray close sys.stderr mid-request, write to the original fd
class _SafeStderrHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = sys.__stderr__
            if stream is None:
                return
            stream.write(msg + "\n")
            stream.flush()
        except Exception:
            pass


if not LOG.handlers:
    LOG.addHandler(_SafeStderrHandler(level=logging.WARNING))


def _split_input_to_user_and_history(input_items) -> tuple[str, list[dict], Optional[str]]:
    if isinstance(input_items, str):
        return input_items, [], None
    items = list(input_items)
    system_message: Optional[str] = None
    if items:
        first = items[0]
        first_role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        first_content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
        if first_role == "system":
            if isinstance(first_content, list):
                first_content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in first_content
                )
            system_message = first_content or ""
            items = items[1:]

    user_message = ""
    history: list[dict] = []
    for idx, item in enumerate(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if isinstance(content, list):
            content = "".join((p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content)
        content = content or ""
        if idx == len(items) - 1 and role == "user":
            user_message = content
        else:
            history.append({"role": role, "content": content})
    return user_message, history, system_message


class HermesAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef | None = None
    model_server: ModelServerRef
    model: Optional[str] = None
    concurrency: int = 32
    max_turns: int = 90
    max_tokens: Optional[int] = None
    enabled_toolsets: Optional[list[str]] = None
    disabled_toolsets: Optional[list[str]] = None
    temperature: float | None = None
    terminal_backend: str = "local"
    terminal_timeout: int = 180
    sandbox_provider: str | None = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    sandbox_runner_timeout_seconds: float = Field(default=21600, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    system_prompt: Optional[str] = None
    compression_enabled: bool = True
    compression_threshold: float = 0.85
    chat_template_kwargs_enabled: bool = True
    api_key: Optional[str] = None
    delegation_max_iterations: int = 50
    checkpoints_enabled: bool = False


class HermesAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HermesAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: AgentObservationBundle | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class HermesAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: HermesAgentConfig
    sem: Semaphore = None
    # Set of agents currently running run_conversation, plus a flag tracking whether the single
    # shared SIGTERM dispatcher has been installed on the event loop. See _ensure_sigterm_handler.
    active_agents: set = None
    interrupted_agents: set = None
    sigterm_installed: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> HermesSandboxSession:
        tool_accesses = self.effective_tool_accesses(body)
        unsupported = [
            access.name for access in tool_accesses if access.required and not isinstance(access, MCPToolAccess)
        ]
        if unsupported:
            raise HTTPException(
                422,
                "Hermes Agent supports only MCP tool grants; required grants it cannot use: "
                + ", ".join(sorted(unsupported)),
            )
        # Hermes exposes each MCP server as a toolset of the same name, so a name must not shadow a built-in one.
        colliding = [
            access.name for access in tool_accesses if isinstance(access, MCPToolAccess) and access.name in TOOLSETS
        ]
        if colliding:
            raise ValueError("MCP tool grants collide with Hermes toolsets: " + ", ".join(sorted(colliding)))
        return await self._initialize_agent_session_state(body.agent_session_id, body)

    def _require_agent_session(self, agent_session_id: str) -> HermesSandboxSession:
        state = super()._require_agent_session(agent_session_id)
        if not isinstance(state, HermesSandboxSession):
            raise TypeError("Expected Hermes agent session state")
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, HermesSandboxSession):
            raise TypeError("Expected Hermes agent session state")
        await state.close(self.config.session_close_timeout_seconds)
        # Errors/cancellation can bypass response parsing. The common session
        # still captures available output before releasing the sandbox.
        output = state.session.artifacts
        if state.observations is None and output is not None:
            result = output.get("result")
            state.observations = self._sandbox_observations(
                result if isinstance(result, dict) else {"failed": True},
                output.get("observations"),
                runtime_info=state.runtime_info,
            )
        observations = state.observations or AgentObservationBundle(
            source="hermes", gaps=[ObservationGap(code="observation_capture_failed")]
        )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, agent_observations=observations
        )

    def _ensure_sigterm_handler(self) -> None:
        """Install exactly one SIGTERM handler on the event loop that interrupts *every* in-flight
        agent. Registering a fresh per-call handler is unsafe under concurrency: add_signal_handler
        replaces the previous handler, so concurrent responses() calls clobber each other and the
        first to finish removes the only remaining handler — leaving later SIGTERMs unhandled and
        their trajectories lost. A single dispatcher over `active_agents` avoids that race."""
        if self.sigterm_installed:
            return
        import signal

        def _dispatch():
            for ag in list(self.active_agents):
                self.interrupted_agents.add(id(ag))
                if hasattr(ag, "interrupt"):
                    ag.interrupt("timeout")

        try:
            asyncio.get_event_loop().add_signal_handler(signal.SIGTERM, _dispatch)
            self.sigterm_installed = True
        except (NotImplementedError, OSError):
            pass  # not supported on this platform (e.g. Windows, non-main thread)

    def _build_config(self, mcp_accesses: list[MCPToolAccess] | None = None) -> str:
        import yaml

        config: dict[str, Any] = {
            "model": self._model_name(),
            "provider": "auto",
            "toolsets": ["hermes-cli"],
            "agent": {"max_turns": self.config.max_turns},
            "memory": {
                "memory_enabled": False,
                "user_profile_enabled": False,
            },
            "compression": {
                "enabled": self.config.compression_enabled,
                "threshold": self.config.compression_threshold,
            },
            "terminal": {
                "backend": self.config.terminal_backend,
                "timeout": self.config.terminal_timeout,
            },
            "delegation": {
                "max_iterations": self.config.delegation_max_iterations,
            },
            "checkpoints": {
                "enabled": self.config.checkpoints_enabled,
            },
        }
        if mcp_accesses:
            # A grant is for tools, so Hermes' resource and prompt helper tools stay off.
            config["mcp_servers"] = {
                access.name: {
                    "url": str(access.connection.url),
                    "headers": access.connection.headers,
                    "tools": {"resources": False, "prompts": False},
                }
                for access in mcp_accesses
            }
        return yaml.dump(config, default_flow_style=False)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.sem = Semaphore(self.config.concurrency)
        self.active_agents = set()
        self.interrupted_agents = set()
        # hermes-agent reads these from env (cli.py / batch_runner.py); env vars are
        # process-global, so multiple HermesAgent instances in one process share them
        os.environ["TERMINAL_ENV"] = self.config.terminal_backend
        os.environ["TERMINAL_TIMEOUT"] = str(self.config.terminal_timeout)

        # Build config.yaml with config parameters
        hermes_home = tempfile.mkdtemp(prefix="hermes_agent_")
        atexit.register(shutil.rmtree, hermes_home, True)
        with open(os.path.join(hermes_home, "config.yaml"), "w") as _f:
            _f.write(self._build_config())
        os.environ["HERMES_HOME"] = hermes_home

    async def _initialize_agent_session_state(
        self,
        agent_session_id: str,
        body: AgentSeedSessionRequest,
    ) -> HermesSandboxSession:
        owns_sandbox = body.sandbox_access is None
        if owns_sandbox:
            if self.config.sandbox_provider is None:
                raise ValueError(
                    "Hermes requires sandbox_access or a configured sandbox_provider for an episode session"
                )
            provider_ref = self.config.sandbox_provider
            workdir = self.config.sandbox_config.get("workdir")
        else:
            connection = body.sandbox_access.connection
            if not isinstance(connection, DirectSandboxConnection):
                raise ValueError("Hermes currently supports only direct sandbox connections")
            provider_ref = connection.provider_config_ref
            workdir = body.sandbox_access.workdir

        provider_config = resolve_provider_config(provider_ref, get_global_config_dict())
        provider = create_provider(provider_config)
        try:
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

        session_dir = f"/tmp/nemo-gym-hermes-sessions/{uuid4().hex}"
        state = HermesSandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                workdir=workdir,
                session_dir=session_dir,
                owns_sandbox=owns_sandbox,
                harness="Hermes",
            ),
        )
        try:
            await state.install_runtime(install_timeout=self.config.sandbox_install_timeout_seconds)
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                LOG.exception("Could not clean failed Hermes setup %s; retaining session for close", agent_session_id)
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    def _model_name(self) -> str:
        return self.config.model or str(self.config.model_server.name)

    def _sandbox_observations(
        self,
        result: dict[str, Any],
        raw_observations: Any,
        *,
        runtime_info: HarnessProcessInfo | None,
    ) -> AgentObservationBundle:
        gaps = [] if runtime_info is not None else [ObservationGap(code="runtime_info_unavailable")]
        if isinstance(raw_observations, dict):
            try:
                records: list[AgentInvocation | ToolCallObservation | ContextCompactionObservation] = []
                for raw_invocation in raw_observations.get("invocations") or []:
                    response_ids = raw_invocation.get("model_response_ids") or []
                    invocation_id = str(raw_invocation["invocation_id"])
                    records.append(
                        AgentInvocation(
                            invocation_id=invocation_id,
                            parent_invocation_id=raw_invocation.get("parent_invocation_id"),
                            status=raw_invocation.get("status", "unknown"),
                            # Every call goes to the configured Model Server, so a response ID identifies the call.
                            model_calls=[
                                ModelCallRef(model_ref=self.config.model_server, response_id=response_id)
                                for response_id in response_ids
                                if isinstance(response_id, str) and response_id
                            ],
                            conversation=normalize_hermes_messages(
                                raw_invocation.get("messages") or [],
                                id_prefix=invocation_id,
                            ),
                        )
                    )
                records.extend(
                    ToolCallObservation.model_validate(tool) for tool in raw_observations.get("tools") or []
                )
                records.extend(
                    ContextCompactionObservation.model_validate(compaction)
                    for compaction in raw_observations.get("compactions") or []
                )
                return AgentObservationBundle(
                    source="hermes",
                    records=records,
                    gaps=[*gaps, *(ObservationGap.model_validate(gap) for gap in raw_observations.get("gaps") or [])],
                )
            except Exception as error:
                LOG.exception("failed to validate sandbox Hermes observations")
                return AgentObservationBundle(
                    source="hermes",
                    gaps=[
                        *gaps,
                        ObservationGap(
                            code="observation_capture_failed",
                            detail=type(error).__name__,
                        ),
                    ],
                )

        messages = result.get("messages") or []
        invocation_status = (
            "failed"
            if result.get("error") or result.get("failed")
            else ("completed" if result.get("completed", True) else "incomplete")
        )
        records: list[AgentInvocation | ToolCallObservation] = [
            AgentInvocation(
                invocation_id="root",
                status=invocation_status,
                conversation=normalize_hermes_messages(messages),
            )
        ]
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") if isinstance(tool_call, dict) else None
                if not isinstance(function, dict):
                    continue
                records.append(
                    ToolCallObservation(
                        invocation_id="root",
                        tool_call_id=str(tool_call.get("id") or ""),
                        tool_name=str(function.get("name") or ""),
                        timing_source="harness",
                        status="completed",
                    )
                )
        return AgentObservationBundle(source="hermes", records=records, gaps=gaps)

    async def _run_sandbox_episode(
        self,
        *,
        body: NeMoGymResponseCreateParamsNonStreaming,
        agent_session_id: str,
        state: HermesSandboxSession,
    ) -> AgentEpisode:
        params = self._conversation_params(body)
        mcp_accesses = [
            access for access in self.effective_tool_accesses(state.request) if isinstance(access, MCPToolAccess)
        ]
        enabled_toolsets = self.config.enabled_toolsets
        if enabled_toolsets is not None:
            # A restricted tool list would otherwise hide the tools this episode was granted.
            enabled_toolsets = [*enabled_toolsets, *(access.name for access in mcp_accesses)]
        payload = {
            "agent_session_id": agent_session_id,
            "chat_template_kwargs_enabled": self.config.chat_template_kwargs_enabled,
            "config_yaml": self._build_config(mcp_accesses),
            "disabled_toolsets": self.config.disabled_toolsets,
            "enabled_toolsets": enabled_toolsets,
            "mcp_servers": [access.name for access in mcp_accesses],
            "required_mcp_servers": [access.name for access in mcp_accesses if access.required],
            **params,
            "max_turns": self.config.max_turns,
            "model": self._model_name(),
            # The sandbox reaches the Model Server directly; the rollout prefix keeps its calls correlated.
            "model_base_url": self.resolve_model_base_url(
                self.config.model_server.name, state.request.episode_id.capture_key
            ),
            "terminal_timeout": self.config.terminal_timeout,
        }
        output = await state.execute(
            payload,
            timeout=self.config.sandbox_runner_timeout_seconds,
            close_timeout=self.config.session_close_timeout_seconds,
        )
        result = output.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Hermes sandbox runner returned an invalid output")
        runtime = state.runtime_info
        response = self._response_from_result(
            body=body,
            result=result,
            model_name=self._model_name(),
            n_input=len(params["history"]) + 1,
        )
        # Verifiers see Gym's MCP naming; the model's own names stay in the captured model calls.
        server_names = [access.name for access in mcp_accesses]
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                item.name = _gym_mcp_tool_name(item.name, server_names)
        response.metadata = {
            **(response.metadata or {}),
            "harness_execution": "sandbox",
        }
        if runtime is not None:
            response.metadata.update(
                harness_hostname=runtime.hostname,
                harness_pid=str(runtime.pid),
                harness_python=runtime.python or "",
            )
        return AgentEpisode(
            response=response,
            observations=self._sandbox_observations(result, output.get("observations"), runtime_info=runtime),
        )

    def _validate_request(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        """Validate once at the HTTP boundary, identically for host and sandbox execution."""
        if body.model is not None and body.model != self._model_name():
            raise HTTPException(422, "Hermes request model must match the configured model")
        if body.max_output_tokens is not None:
            raise HTTPException(
                422,
                "Hermes does not support the total max_output_tokens budget; "
                "configure max_tokens for a per-model-call limit instead",
            )
        supported = {"input", "instructions", "temperature", "model"}
        # Fail closed for new schema fields instead of silently accepting unimplemented controls.
        for name, field in type(body).model_fields.items():
            if name in supported:
                continue
            value = getattr(body, name)
            if name in ("stream", "background") and value is False:
                continue  # Explicit synchronous, non-streaming execution is supported.
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"Hermes does not support request field {name}")
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        roles = [getattr(item, "role", None) for item in body.input]
        conversation_roles = roles[1:] if roles and roles[0] == "system" else roles
        if (
            not conversation_roles
            or conversation_roles[-1] != "user"
            or any(role not in ("user", "assistant") for role in conversation_roles)
        ):
            raise HTTPException(422, "Hermes accepts text history ending with a user message")
        for item in body.input:
            if not isinstance(item.content, str) and any(
                (part.get("type") if isinstance(part, dict) else getattr(part, "type", None))
                not in ("input_text", "output_text")
                for part in item.content
            ):
                raise HTTPException(422, "Hermes only supports text input")
        return body

    def _conversation_params(self, body: NeMoGymResponseCreateParamsNonStreaming) -> dict[str, Any]:
        user_message, history, input_system = _split_input_to_user_and_history(body.input)
        return {
            "user_message": user_message,
            "history": history,
            "system_message": "\n\n".join(
                part for part in (self.config.system_prompt, body.instructions, input_system) if part
            )
            or None,
            "temperature": body.temperature if body.temperature is not None else self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    def _response_from_result(
        self,
        *,
        body: NeMoGymResponseCreateParamsNonStreaming,
        result: dict[str, Any],
        model_name: str,
        interrupted_by_dispatch: bool = False,
        n_input: int = 0,
    ) -> NeMoGymResponse:
        # The pinned Hermes marks provider/API failures with `failed`, but model-limit and
        # invalid-tool outcomes with `partial`. Keep those partial patches gradable. Its one
        # model-caused `failed` outcome is first-response truncation (run_agent.py).
        if result.get("failed") and result.get("error") != "First response truncated due to output length limit":
            raise RuntimeError(f"Hermes agent failed: {result.get('error') or 'unknown provider/API failure'}")

        messages = result.get("messages") or []
        output_items = _trajectory_to_output_items(messages, n_input)
        has_assistant_message = any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        )
        if not has_assistant_message:
            LOG.warning(
                "Hermes agent ended without an assistant message. Padding empty assistant message: error=%r",
                result.get("error"),
            )
            last_valid = next(
                (
                    message
                    for message in reversed(messages)
                    if isinstance(message, dict)
                    and message.get("role") == "assistant"
                    and message.get("generation_token_ids")
                ),
                None,
            )
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text=result.get("error") or "", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=last_valid["prompt_token_ids"] if last_valid else [],
                    generation_token_ids=last_valid["generation_token_ids"] if last_valid else [],
                    generation_log_probs=(last_valid.get("generation_log_probs") if last_valid else None) or [],
                    routed_experts=last_valid.get("routed_experts") if last_valid else None,
                )
            )

        agent_completed = bool(result.get("completed", True))
        was_interrupted = bool(result.get("interrupted")) or interrupted_by_dispatch
        harness_error = result.get("error")
        agent_failed = bool(harness_error) or bool(result.get("failed"))
        metadata: dict[str, str] = {
            "interrupted": "true" if was_interrupted else "false",
            "failed": "true" if result.get("failed") else "false",
            "partial": "true" if result.get("partial") else "false",
        }
        if isinstance(result.get("api_calls"), int):
            metadata["turns"] = str(result["api_calls"])
        if result.get("stop_reason"):
            metadata["stop_reason"] = str(result["stop_reason"])
        if harness_error:
            metadata["hermes_error"] = str(harness_error)[:2000]

        response_error = None
        if harness_error:
            from openai.types.responses import ResponseError  # pyright: ignore[reportMissingImports]

            response_error = ResponseError(code="server_error", message=str(harness_error)[:2000])

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            status="failed" if agent_failed else ("completed" if agent_completed else "incomplete"),
            error=response_error,
            metadata=metadata,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=_usage_from_result(result),
        )

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: Optional[str] = None,
        observation_collector: Optional[Callable[[AgentObservationBundle], None]] = None,
    ) -> NeMoGymResponse:
        from run_agent import AIAgent  # from hermes-agent on path  # pyright: ignore[reportMissingImports]

        params = self._conversation_params(body)

        base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        model_name = self._model_name()

        agent = AIAgent(
            base_url=base_url,
            api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY", "gym"),  # pragma: allowlist secret
            model=model_name,
            use_streaming=False,
            temperature=params["temperature"],
            insert_reasoning=True,
            max_iterations=self.config.max_turns,
            max_tokens=params["max_tokens"],
            enabled_toolsets=self.config.enabled_toolsets,
            disabled_toolsets=self.config.disabled_toolsets,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            persist_session=False,
            save_trajectories=False,
        )
        _original_build_api_kwargs = agent._build_api_kwargs

        def _patched_build_api_kwargs(api_messages: list[dict[str, Any]]) -> dict[str, Any]:
            return _model_api_kwargs(
                _original_build_api_kwargs(api_messages),
                preserve_reasoning_history=self.config.chat_template_kwargs_enabled,
            )

        agent._build_api_kwargs = _patched_build_api_kwargs
        observer = None
        if observation_collector is not None:
            try:
                observer = HermesAgentObserver(model_ref=self.config.model_server).instrument(agent)
            except Exception:
                LOG.exception("failed to initialize Hermes observability")

        # Interrupt the agent cleanly on SIGTERM so run_conversation returns with partial messages
        # instead of being killed mid-turn (which would leave response.json unwritten). A single
        # shared dispatcher interrupts every in-flight agent; we just register this one in the set.
        self._ensure_sigterm_handler()
        agent_id = id(agent)
        self.active_agents.add(agent)

        result = None
        agent_error: Optional[BaseException] = None
        interrupted_by_dispatch = False
        try:
            result = await asyncio.to_thread(
                agent.run_conversation,
                params["user_message"],
                params["system_message"],
                params["history"],
                task_id=None,
            )
        except BaseException as exc:
            agent_error = exc
            raise
        finally:
            self.active_agents.discard(agent)
            interrupted_by_dispatch = agent_id in self.interrupted_agents
            self.interrupted_agents.discard(agent_id)
            if observation_collector is not None:
                try:
                    observations = (
                        observer.finish(result, error=agent_error)
                        if observer is not None
                        else AgentObservationBundle(
                            source="hermes",
                            gaps=[ObservationGap(code="observation_capture_failed")],
                        )
                    )
                except Exception:
                    LOG.exception("failed to finish Hermes observability")
                    observations = AgentObservationBundle(
                        source="hermes",
                        gaps=[ObservationGap(code="observation_capture_failed")],
                    )
                try:
                    observation_collector(observations)
                except Exception:
                    LOG.exception("failed to return Hermes observations")

        return self._response_from_result(
            body=body,
            result=result,
            model_name=model_name,
            interrupted_by_dispatch=interrupted_by_dispatch,
            n_input=len(params["history"]) + 1,
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = self._validate_request(body)
        agent_session_id = self._agent_session_id_from_request(request)
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        if isinstance(agent_session_id, str):
            if not isinstance(rollout_id, str):
                raise HTTPException(409, "Agent sessions require an attempt-qualified rollout path")
            state = self._require_agent_session(agent_session_id)
            if state.request.episode_id.capture_key != rollout_id:
                raise HTTPException(409, "Agent-session episode_id does not match the rollout route")
            if state.task is None:
                # No await until the task and its immutable request binding are installed.
                state.activation_request = body.model_copy(deep=True)

                async def activate() -> NeMoGymResponse:
                    async with self.sem:
                        episode = await self._run_sandbox_episode(
                            body=body,
                            agent_session_id=agent_session_id,
                            state=state,
                        )
                        state.observations = episode.observations
                        return episode.response

                state.task = asyncio.create_task(activate())
            elif body != state.activation_request:
                raise HTTPException(409, "Hermes sandbox sessions support one activation; retry the same request")
            assert state.task is not None
            # A disconnected HTTP waiter must not cancel the shared activation. Session close
            # owns cancellation; identical retries join this task or replay its result/error.
            return (await asyncio.shield(state.task)).model_copy(deep=True)
        if not isinstance(rollout_id, str):
            return await self._create_response(body)
        episode = await self._create_episode(body, rollout_id=rollout_id)
        return episode.response.model_copy(
            update={_INTERNAL_OBSERVATIONS_KEY: episode.observations.model_dump(mode="json")}
        )

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: str,
    ) -> AgentEpisode:
        observations: Optional[AgentObservationBundle] = None

        def collect(bundle: AgentObservationBundle) -> None:
            nonlocal observations
            observations = bundle

        response = await self._create_response(
            body,
            rollout_id=rollout_id,
            observation_collector=collect,
        )
        if observations is None:
            observations = AgentObservationBundle(
                source="hermes",
                gaps=[ObservationGap(code="observation_capture_failed")],
            )
        observations.gaps.append(
            ObservationGap(
                code=(
                    "no_sandbox_runtime"
                    if self.config.terminal_backend == "local"
                    else "sandbox_observation_unavailable"
                ),
                detail=(
                    None
                    if self.config.terminal_backend == "local"
                    else f"terminal_backend={self.config.terminal_backend}"
                ),
            )
        )
        return AgentEpisode(response=response, observations=observations)

    async def run(self, request: Request, body: HermesAgentRunRequest) -> HermesAgentVerifyResponse:
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "Use the agent session responses and close routes")
        if self.config.resources_server is None:
            raise HTTPException(422, "Hermes /run requires resources_server; use Environment Server /run for sessions")
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            rollout_id = self.rollout_id_from_run(body)
            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)
            raw_observations = (
                agent_resp_json.pop(_INTERNAL_OBSERVATIONS_KEY, None) if rollout_id is not None else None
            )
            observations = (
                AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
            )

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            result = verify_json | {"turns_used": turns, "finished_naturally": naturally}
            if observations is not None:
                result["ng_agent_observations"] = observations.model_dump(mode="json")
            return HermesAgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    HermesAgent.run_webserver()
