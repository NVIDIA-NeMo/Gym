# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen Code CLI adapter for Environment Server episodes, run headless inside the task sandbox."""

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from shlex import quote
from time import time
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import Field, JsonValue, model_validator

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
    SandboxCommand,
    SandboxNetworkAccess,
    SandboxSession,
    SimpleResponsesAPIAgent,
    resolve_agent_timeout,
    restrict_sandbox_network,
)
from nemo_gym.config import ModelServerRef, get_global_config_dict
from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
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
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap, SandboxObservation
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.qwen_code_agent.conversion import _deep_merge, _text, parse_stream, summarize_subagents
from responses_api_agents.qwen_code_agent.sandbox import (
    CLI,
    QWEN_VERSION,
    RUNTIME_ROOT,
    SESSION_ROOT,
    QwenArtifacts,
    QwenSession,
)


LOG = logging.getLogger(__name__)

# Tools that start other agents; removing them is the solo control.
SUBAGENT_TOOLS = ["agent", "list_agents"]
API_KEY_ENV = "NEMO_GYM_MODEL_API_KEY"  # pragma: allowlist secret
SUBAGENT_PROMPT = Path(__file__).with_name("general-purpose-agent.md")
# Qwen Code takes the system prompt only as one argv value, which Linux caps at 128 KiB (MAX_ARG_STRLEN).
MAX_SYSTEM_PROMPT_BYTES = 128 * 1024
# harness_runner.sh exit statuses when Qwen Code itself never ran.
RUNNER_FAILURES = (125, 126, 127)
# Qwen Code's own limit stops: 53 is --max-session-turns, 55 is --max-tool-calls.
LIMIT_EXITS = (53, 55)


class QwenCodeAgentConfig(BaseResponsesAPIAgentConfig):
    """Qwen Code release, model, subagent and sandbox settings; the Environment Server binds the benchmark."""

    model_server: ModelServerRef
    # Route id Qwen Code sends as the model name. The Gym model server serves its configured model.
    model: str
    qwen_version: str = QWEN_VERSION
    concurrency: int = Field(default=32, gt=0)
    # Seconds per activation, children included, enforced by the sandbox's process supervisor.
    timeout: float = Field(default=3600, gt=0, allow_inf_nan=False)
    # false removes the agent tools (solo control).
    subagents: bool = True
    # A second route for the built-in general-purpose subagent. Forks and teammates keep the root model.
    subagent_model: str | None = None
    subagent_model_server: ModelServerRef | None = None
    max_subagent_depth: int | None = Field(default=None, gt=0, le=100)
    max_parallel_agents: int | None = Field(default=None, gt=0)
    agent_team: bool = False
    max_session_turns: int | None = Field(default=None, gt=0)
    # Qwen Code counts tool calls in the root session only, so subagent calls are not limited.
    max_tool_calls: int | None = Field(default=None, gt=0)
    # Headless runs cannot answer approval prompts.
    approval_mode: Literal["yolo", "auto-edit", "auto"] = "yolo"
    system_prompt: str | None = None
    # Web tools are off so a task is solved from the sandbox, not looked up online. A configured list replaces this one.
    exclude_tools: list[str] = Field(default_factory=lambda: ["web_fetch", "web_search"])
    # Merged under the generated settings.json, which keeps the model routes and settings it sets.
    # model.generationConfig is copied into every model route, where Qwen Code reads it.
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    # Agent-server tar.gz from build_runtime.sh, uploaded into sandboxes without internet access.
    runtime_archive: str | None = None
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    # Only for an agent-owned sandbox, when Resources supplies no sandbox_access.
    sandbox_provider: str | None = None
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)
    network_access: SandboxNetworkAccess = Field(
        default="inherit",
        description=(
            "Egress of a sandbox Qwen Code creates: `model_only` allows only the model servers' host. Qwen Code uses no "
            "tool grants, so `model_and_tools` is rejected. A borrowed task sandbox keeps its owner's network, so a "
            "session that borrows one rejects any other value than `inherit`."
        ),
    )

    @model_validator(mode="after")
    def check_subagent_settings(self) -> "QwenCodeAgentConfig":
        if not self.subagents and (self.subagent_model or self.agent_team):
            raise ValueError("subagent_model and agent_team require subagents: true")
        if self.subagent_model_server and not self.subagent_model:
            raise ValueError("subagent_model_server requires subagent_model")
        if self.subagent_model and not self.subagent_model_server:
            raise ValueError(
                "subagent_model requires subagent_model_server, since the policy server serves only its own model"
            )
        if self.subagent_model == self.model:
            raise ValueError("subagent_model must differ from model; leave it null to inherit the root model")
        if self.network_access == "model_only" and not self.runtime_archive:
            raise ValueError("network_access model_only requires runtime_archive, since npm cannot reach its registry")
        return self


class QwenCodeAgent(SimpleResponsesAPIAgent):
    """Run one headless Qwen Code session per agent session, inside the borrowed (or agent-owned) task sandbox."""

    config: QwenCodeAgentConfig
    calls_model_from_sandbox: ClassVar[bool] = True

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sem = asyncio.Semaphore(self.config.concurrency)

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        # Qwen Code 0.25.1 reports a model HTTP error as "[API Error: <status> ...]" in the result event.
        match = re.match(r"\[API Error: ([45]\d{2})\b", native_result.get("error_message", ""))
        if match:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=int(match[1]))
        return AgentOutcome(failure_kind=native_result.get("failure_kind", "unknown"))

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, QwenSession) else None

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> QwenSession:
        """Connect to (or start) the task sandbox and install Qwen Code before the activation."""
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError(
                "Qwen Code sessions use Qwen Code's own tools; required tool grants are unsupported"
            )
        if self.config.qwen_version != QWEN_VERSION:
            raise AgentSeedRejectedError(f"Qwen Code sessions pin qwen_version {QWEN_VERSION}")
        access = body.sandbox_access
        owns_sandbox = access is None
        if self.config.subagent_model_server and self._token_id_capture_enabled():
            raise AgentSeedRejectedError("Qwen Code token capture is unsupported with subagent_model_server")
        if self.config.network_access == "model_and_tools":
            raise AgentSeedRejectedError("Qwen Code uses no tool grants; set network_access to model_only or inherit")
        if not owns_sandbox and self.config.network_access != "inherit":
            raise AgentSeedRejectedError(
                "Qwen Code network_access applies only to a sandbox it creates, not a borrowed one"
            )
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise AgentSeedRejectedError("Qwen Code requires sandbox_access or a configured sandbox_provider")
            provider_ref = self.config.sandbox_provider
            workdir = self.config.sandbox_config.get("workdir") or "/app"
        else:
            if not isinstance(access.connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("Qwen Code requires direct SandboxAccess")
            if access.user is not None:
                raise AgentSeedRejectedError("Qwen Code does not support a SandboxAccess user override")
            provider_ref = access.connection.provider_config_ref
            workdir = access.workdir
        path = PurePosixPath(workdir) if isinstance(workdir, str) else None
        if (
            path is None
            or not path.is_absolute()
            or ".." in path.parts
            or path in (PurePosixPath("/"), PurePosixPath("/tmp"))
            or str(path).startswith((SESSION_ROOT, RUNTIME_ROOT))
        ):
            raise AgentSeedRejectedError(
                "Qwen Code task workdir must be absolute and separate from its session storage"
            )

        global_config = get_global_config_dict()
        provider = create_provider(resolve_provider_config(provider_ref, global_config))
        if owns_sandbox:
            settings = dict(self.config.sandbox_config)
            metadata = settings.pop("metadata", None) or {}
            settings["metadata"] = {
                **resolve_provider_metadata(provider_ref, global_config),
                **metadata,
                "agent_session_id": body.agent_session_id[:63],
            }
            settings["workdir"] = str(path)
            settings.setdefault("ttl_s", int(self.config.sandbox_install_timeout_seconds + self.config.timeout + 600))
            try:
                model_url = self.resolve_model_base_url(self.config.model_server.name, trace_context=False)
                subagent = self.config.subagent_model_server
                # The policy allows one model host, so a subagent server elsewhere would fail only mid-run.
                if (
                    self.config.network_access != "inherit"
                    and subagent
                    and urlsplit(self.resolve_model_base_url(subagent.name, trace_context=False)).hostname
                    != urlsplit(model_url).hostname
                ):
                    raise AgentSeedRejectedError(
                        "network_access allows one model host; subagent_model_server is on another"
                    )
                spec = restrict_sandbox_network(
                    SandboxSpec(**settings),
                    network_access=self.config.network_access,
                    provider=provider,
                    model_base_url=model_url,
                )
            except BaseException:
                await provider.aclose()
                raise
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        state = QwenSession(
            request=body,
            provider_name=provider.name,
            sandbox_id=sandbox.sandbox_id,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=f"{SESSION_ROOT}/{uuid4().hex}",
                workdir=str(path),
                harness="Qwen Code",
                owns_sandbox=owns_sandbox,
            ),
        )
        try:
            async with asyncio.timeout(self.config.sandbox_install_timeout_seconds):
                if owns_sandbox:
                    await sandbox.start(spec)
                    state.sandbox_id = sandbox.sandbox_id
                    made = await sandbox.exec(f"mkdir -p -- {quote(str(path))}", cwd="/", timeout_s=30)
                    if made.return_code != 0 or made.error_type:
                        raise RuntimeError(f"Cannot create the Qwen Code task workdir {path}")
                await state.install(
                    runtime_archive=self.config.runtime_archive, timeout=self.config.sandbox_install_timeout_seconds
                )
        except BaseException as error:
            try:
                await state.close(timeout=self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, QwenSession):
            raise TypeError("Expected QwenSession")
        await state.close(timeout=self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.session.cleanup_confirmed,
            agent_observations=state.observations
            or AgentObservationBundle(source="qwen_code", gaps=[ObservationGap(code="observation_capture_failed")]),
        )

    def _conversation(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
        """Split the request into Qwen Code's task (stdin) and appended system prompt, rejecting other controls."""
        if body.model is not None and body.model != self.config.model:
            raise HTTPException(422, "Qwen Code model is selected by the agent configuration")
        if body.max_output_tokens is not None:
            raise HTTPException(
                422, "Qwen Code cannot enforce a total max_output_tokens budget; configure per-call limits in settings"
            )
        supported = {"input", "instructions", "model", "temperature", "top_p", "max_tool_calls"}
        for name, field in type(body).model_fields.items():
            value = getattr(body, name)
            if name in supported or (name in ("stream", "background") and value is False):
                continue
            if name == "metadata" and value and set(value) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"Qwen Code does not support request field {name}")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"], ["developer", "user"]):
            raise HTTPException(422, "Qwen Code accepts one user message with an optional system or developer message")
        prompt = _text(items[-1].content)
        if not prompt.strip():
            raise HTTPException(422, "Qwen Code requires a nonempty task")
        system = [self.config.system_prompt, body.instructions] + [_text(item.content) for item in items[:-1]]
        system_text = "\n\n".join(part for part in system if part)
        if len(system_text.encode()) + 1 > MAX_SYSTEM_PROMPT_BYTES:
            raise HTTPException(
                422, f"Qwen Code system prompt exceeds the {MAX_SYSTEM_PROMPT_BYTES} byte argument limit"
            )
        return prompt, system_text

    def qwen_settings(self, state: QwenSession, body: NeMoGymResponseCreateParamsNonStreaming) -> dict[str, Any]:
        """Point every Qwen Code model route at Gym's model servers, with this rollout's capture prefix."""
        rollout_id = state.request.episode_id.capture_key

        def base_url(server: ModelServerRef) -> str:
            return self.model_endpoint(model_server=server, rollout_id=rollout_id).base_url

        # A provider route's generationConfig wins per field, so each route carries the configured block and sampling.
        generation = deepcopy((self.config.settings.get("model") or {}).get("generationConfig") or {})
        sampling = {
            key: value
            for key, value in (("temperature", body.temperature), ("top_p", body.top_p))
            if value is not None
        }
        if sampling:
            _deep_merge(generation, {"samplingParams": sampling})
        routes = [{"id": self.config.model, "baseUrl": base_url(self.config.model_server)}]
        if self.config.subagent_model:
            routes.append({"id": self.config.subagent_model, "baseUrl": base_url(self.config.subagent_model_server)})
        settings: dict[str, Any] = {
            "modelProviders": {
                "openai": [route | {"envKey": API_KEY_ENV, "generationConfig": generation} for route in routes]
            },
            "security": {"auth": {"selectedType": "openai"}},
            "model": {"name": self.config.model},
            "general": {"enableAutoUpdate": False},
            "privacy": {"usageStatisticsEnabled": False},
            # Background memory extraction makes extra model calls outside the task.
            "memory": {"enableManagedAutoMemory": False},
            # A recurring cron job would keep a headless run alive until the timeout.
            "experimental": {"cron": False, "agentTeam": self.config.agent_team},
        }
        if self.config.max_parallel_agents:
            settings["agents"] = {"maxParallelAgents": self.config.max_parallel_agents}
        return _deep_merge(deepcopy(self.config.settings), settings)

    def _command(self, state: QwenSession, system: str, max_tool_calls: int | None) -> list[str]:
        command = [f"{state.runtime}/bin/node", f"{state.runtime}/{CLI}"]
        command += ["--approval-mode", self.config.approval_mode, "-o", "stream-json"]
        for flag, value in (
            ("--max-session-turns", self.config.max_session_turns),
            # The request's own tool-call limit wins over the configured one.
            ("--max-tool-calls", max_tool_calls if max_tool_calls is not None else self.config.max_tool_calls),
            ("--max-subagent-depth", self.config.max_subagent_depth),
        ):
            if value is not None:
                command += [flag, str(value)]
        for tool in self.config.exclude_tools + ([] if self.config.subagents else SUBAGENT_TOOLS):
            command += ["--exclude-tools", tool]
        return command + (["--append-system-prompt", system] if system else [])

    def _env(self, state: QwenSession) -> dict[str, str]:
        access = state.request.sandbox_access
        env = {
            key: value
            for key, value in (access.env if access else {}).items()
            if not key.startswith(("QWEN_", "OPENAI_")) and key != API_KEY_ENV
        }
        # Qwen Code's settings, agents and transcripts live in the session directory, so tool shells keep the task's HOME.
        return env | {
            "QWEN_HOME": state.qwen_home,
            # System scope outranks a task repository's .qwen/settings.json.
            "QWEN_CODE_SYSTEM_SETTINGS_PATH": state.settings_path,
            "QWEN_CODE_MODELS_DEV_REFRESH": "off",
            API_KEY_ENV: "EMPTY",
            "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1",
        }

    async def _stage(self, state: QwenSession, body: NeMoGymResponseCreateParamsNonStreaming) -> SandboxCommand:
        prompt, system = self._conversation(body)
        subagent = None
        if self.config.subagent_model:
            subagent = SUBAGENT_PROMPT.read_text().replace("{model}", self.config.subagent_model)
        return await state.stage(
            command=self._command(state, system, body.max_tool_calls),
            env=self._env(state),
            prompt=prompt,
            settings=json.dumps(self.qwen_settings(state, body)),
            subagent=subagent,
        )

    def _observations(
        self, state: QwenSession, *, status: str, error_type: str | None, output: list[Any], usage_known: bool
    ) -> AgentObservationBundle:
        cleanup = state.session.cleanup
        conversation = []
        if state.activation_request is not None:
            prompt, _ = self._conversation(state.activation_request)
            conversation.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        # Qwen Code 0.25.1 sends no per-call session header to the model server.
        gaps = ["model_call_ownership_unavailable", "harness_tool_timing_unavailable", "child_invocations_unavailable"]
        if not usage_known:
            gaps.append("usage_unavailable")
        gaps += ["optional_tool_grant_skipped" for _ in self.effective_tool_accesses(state.request)]
        return AgentObservationBundle(
            source="qwen_code",
            records=[
                AgentInvocation(
                    invocation_id=state.request.episode_id.capture_key,
                    status=status,
                    error_type=error_type,
                    conversation=conversation + output,
                ),
                SandboxObservation(
                    role="agent",
                    provider=state.provider_name,
                    sandbox_id=state.sandbox_id,
                    outcome="timeout"
                    if cleanup and cleanup["timed_out"]
                    else "failed"
                    if status == "failed"
                    else "completed",
                    exit_code=cleanup["return_code"] if cleanup else None,
                    error_type=error_type,
                ),
            ],
            gaps=[ObservationGap(code=code) for code in sorted(set(gaps))],
        )

    async def _activate(self, state: QwenSession, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        timeout = resolve_agent_timeout(
            body.metadata, ceiling=min(self.config.timeout, state.request.agent_timeout_sec or self.config.timeout)
        )
        try:
            async with self._sem:
                artifacts: QwenArtifacts = await state.session.execute(
                    stage_activation=lambda: self._stage(state, body),
                    collect=state.collect,
                    timeout=timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            state.observations = self._observations(
                state, status="failed", error_type=type(error).__name__, output=[], usage_known=False
            )
            raise
        cleanup = state.session.cleanup
        if cleanup is None or not cleanup["cleanup_confirmed"]:
            raise RuntimeError("Qwen Code ended without confirmed supervisor cleanup")
        if cleanup["error"]:
            raise RuntimeError(f"Qwen Code supervision failed: {cleanup['error']}")
        return_code, timed_out = cleanup["return_code"], cleanup["timed_out"]
        if return_code in RUNNER_FAILURES and not timed_out:
            raise RuntimeError(f"Qwen Code did not start (runner exit {return_code}): {artifacts.stderr[-4096:]}")
        output, result = parse_stream(artifacts.stdout)
        subagents = await asyncio.to_thread(summarize_subagents, artifacts.sessions) if artifacts.sessions else {}

        status, failure_kind, error_type = "completed", None, None
        error = (result or {}).get("error")
        error_message = str(error.get("message") or "") if isinstance(error, dict) else ""
        if timed_out:
            # Includes a deadline or rollout stop before any model output.
            status, error_type = "incomplete", "timeout"
        elif return_code in LIMIT_EXITS or error_message.startswith("Loop detection halted the run"):
            status, error_type = "incomplete", "agent_limit"
        elif return_code != 0 or result is None or result.get("is_error"):
            # Unexplained endings, a bare exit 137 included, stay unknown. Model HTTP errors carry their status.
            status, failure_kind = "failed", "unknown"
            error_type = (result or {}).get("subtype") or f"exit_{return_code}"
        elif not any(getattr(item, "type", None) == "message" for item in output):
            status, failure_kind, error_type = "failed", "agent_invalid_output", "no_answer"

        # The result event's usage covers every model call, subagents included.
        usage = None
        if result and isinstance(result.get("usage"), dict):
            values = result["usage"]
            input_tokens, output_tokens = int(values.get("input_tokens", 0)), int(values.get("output_tokens", 0))
            usage = NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cached_tokens=int(values.get("cache_read_input_tokens", 0))
                ),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            )
        state.observations = self._observations(
            state, status=status, error_type=error_type, output=output, usage_known=usage is not None
        )
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            object="response",
            model=self.config.model,
            output=output,
            usage=usage,
            status=status,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            metadata={
                "harness_execution": "sandbox",
                "qwen_version": QWEN_VERSION,
                "qwen_exit_code": str(return_code),
                **{key: str(value) for key, value in subagents.items()},
            },
        )
        if status == "failed":
            LOG.warning("Qwen Code run failed (%s); stderr tail: %s", error_type, artifacts.stderr[-2048:])
            return self.report_outcome(response, {"failure_kind": failure_kind, "error_message": error_message})
        return self.report_rollout_stop(response, state.session)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(422, "Qwen Code runs only in an agent session managed by an Environment Server")
        state = self._require_agent_session(session_id)
        if not isinstance(state, QwenSession):
            raise TypeError("Expected QwenSession")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Qwen Code activation must use its seeded rollout route")
        if state.session.closing:
            raise HTTPException(409, "Qwen Code session is closing")
        self._conversation(body)
        if state.task is None:
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._activate(state, state.activation_request))
        elif body != state.activation_request:
            raise HTTPException(409, "Qwen Code sessions support one activation; retries must repeat the same request")
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        """Qwen Code runs through an Environment Server, which owns verification."""
        raise HTTPException(422, "Run Qwen Code through an Environment Server such as single_agent_turn")


if __name__ == "__main__":
    QwenCodeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = QwenCodeAgent.run_webserver()
