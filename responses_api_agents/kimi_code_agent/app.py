# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Code CLI adapter for Environment Server episodes, run inside the task sandbox with its Agent and AgentSwarm
subagents."""

import asyncio
import io
import logging
import re
import tarfile
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from shlex import quote
from time import time
from typing import Any, ClassVar
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
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
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
from responses_api_agents.kimi_code_agent.conversion import (
    deep_merge,
    max_concurrency,
    parse_session,
    parse_stream_json,
    stream_session_id,
    to_toml,
)
from responses_api_agents.kimi_code_agent.sandbox import (
    KIMI_VERSION,
    RUNTIME_ROOT,
    SESSION_ROOT,
    KimiArtifacts,
    KimiSession,
)


LOG = logging.getLogger(__name__)

# Kimi takes the task only as the `-p` argv value, which Linux caps at 128 KiB (MAX_ARG_STRLEN).
MAX_PROMPT_BYTES = 128 * 1024
# harness_runner.sh exit statuses when Kimi itself never ran.
RUNNER_FAILURES = (125, 126, 127)
# A failed print turn ends with "error: failed to run prompt: <code>: <message>" on stderr.
TURN_FAILURE = re.compile(r"failed to run prompt: ([\w.]+): (.*)")
# Native limit stops, graded as incomplete.
LIMIT_CODES = ("context.overflow", "loop.max_steps_exceeded")


class KimiCodeAgentConfig(BaseResponsesAPIAgentConfig):
    """Kimi Code release, model, subagent and sandbox settings; the Environment Server binds the benchmark."""

    model_server: ModelServerRef
    # The model name Kimi sends. The Gym model server serves its configured model.
    model: str
    # Children use this server by default; the root may still pick "policy" per spawn unless forced.
    subagent_model_server: ModelServerRef | None = None
    subagent_model: str | None = None
    subagent_model_force: bool = False
    # false disables the Agent and AgentSwarm tools (solo control).
    subagents: bool = True
    swarm_max_concurrency: int | None = Field(default=None, gt=0)
    max_steps_per_turn: int | None = Field(default=None, ge=0)
    # Bounds the extra root turns that background subagent completions trigger in print mode.
    print_max_turns: int | None = Field(default=None, gt=0)
    thinking_effort: str | None = None
    max_context_size: int = Field(default=262144, gt=0)
    # Web tools are off so a task is solved from the sandbox, not looked up online.
    disabled_tools: list[str] = Field(
        default_factory=lambda: ["CronCreate", "CronList", "CronDelete", "WebSearch", "FetchURL"]
    )
    permission_rules: list[dict[str, str]] = Field(default_factory=list)
    # Deep-merged under the generated config.toml, which keeps the model providers and tool settings it sets.
    kimi_config: dict[str, JsonValue] = Field(default_factory=dict)
    kimi_version: str = KIMI_VERSION
    concurrency: int = Field(default=32, gt=0)
    # Seconds per activation, enforced by the sandbox's process supervisor.
    timeout: float = Field(default=7200, gt=0, allow_inf_nan=False)
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    # Only for an agent-owned sandbox, when Resources supplies no sandbox_access.
    sandbox_provider: str | None = None
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)
    network_access: SandboxNetworkAccess = Field(
        default="inherit",
        description=(
            "Egress of a sandbox Kimi Code creates: `model_only` allows only the model servers' host. Kimi Code uses no "
            "tool grants, so `model_and_tools` is rejected. A borrowed task sandbox keeps its owner's network, so a "
            "session that borrows one rejects any other value than `inherit`."
        ),
    )

    @model_validator(mode="after")
    def require_subagent_server(self) -> "KimiCodeAgentConfig":
        # Gym model servers overwrite the request's model, so these act only on a separate subagent server.
        if (self.subagent_model or self.subagent_model_force) and not self.subagent_model_server:
            raise ValueError("subagent_model and subagent_model_force require subagent_model_server")
        return self


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        value = part if isinstance(part, dict) else part.model_dump()
        if value.get("type") != "input_text":
            raise HTTPException(422, "Kimi Code accepts text input only")
        parts.append(value["text"])
    return "\n".join(parts)


class KimiCodeAgent(SimpleResponsesAPIAgent):
    """Run one Kimi Code print-mode session per agent session, inside the borrowed (or agent-owned) task sandbox."""

    config: KimiCodeAgentConfig
    calls_model_from_sandbox: ClassVar[bool] = True

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sem = asyncio.Semaphore(self.config.concurrency)

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        return AgentOutcome(
            failure_kind=native_result.get("failure_kind", "unknown"), status_code=native_result.get("status_code")
        )

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, KimiSession) else None

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> KimiSession:
        """Connect to (or start) the task sandbox and install Kimi Code before the activation."""
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError(
                "Kimi Code sessions use Kimi's own tools; required tool grants are unsupported"
            )
        if self.config.kimi_version != KIMI_VERSION:
            raise AgentSeedRejectedError(f"Kimi Code sessions pin release checksums for kimi_version {KIMI_VERSION}")
        if self.config.subagent_model_server and self._token_id_capture_enabled():
            raise AgentSeedRejectedError("Kimi Code token capture would record subagent_model_server calls as policy")
        access = body.sandbox_access
        owns_sandbox = access is None
        if self.config.network_access == "model_and_tools":
            raise AgentSeedRejectedError("Kimi Code uses no tool grants; set network_access to model_only or inherit")
        if not owns_sandbox and self.config.network_access != "inherit":
            raise AgentSeedRejectedError(
                "Kimi Code network_access applies only to a sandbox it creates, not a borrowed one"
            )
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise AgentSeedRejectedError("Kimi Code requires sandbox_access or a configured sandbox_provider")
            provider_ref = self.config.sandbox_provider
            workdir = self.config.sandbox_config.get("workdir") or "/app"
        else:
            if not isinstance(access.connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("Kimi Code requires direct SandboxAccess")
            if access.user is not None:
                raise AgentSeedRejectedError("Kimi Code does not support a SandboxAccess user override")
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
                "Kimi Code task workdir must be absolute and separate from its session storage"
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
        state = KimiSession(
            request=body,
            provider_name=provider.name,
            sandbox_id=sandbox.sandbox_id,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=f"{SESSION_ROOT}/{uuid4().hex}",
                workdir=str(path),
                harness="Kimi Code",
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
                        raise RuntimeError(f"Cannot create the Kimi Code task workdir {path}")
                await state.install(timeout=self.config.sandbox_install_timeout_seconds)
        except BaseException as error:
            try:
                await state.close(timeout=self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, KimiSession):
            raise TypeError("Expected KimiSession")
        await state.close(timeout=self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.session.cleanup_confirmed,
            agent_observations=state.observations
            or AgentObservationBundle(source="kimi_code", gaps=[ObservationGap(code="observation_capture_failed")]),
        )

    def _conversation(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
        """Split the request into Kimi's task (`-p`) and global instructions (AGENTS.md), rejecting other controls."""
        if body.model is not None and body.model != self.config.model:
            raise HTTPException(422, "Kimi Code model is selected by the agent configuration")
        supported = {"input", "instructions", "model"}
        for name, field in type(body).model_fields.items():
            value = getattr(body, name)
            if name in supported or (name in ("stream", "background") and value is False):
                continue
            if name == "metadata" and value and set(value) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"Kimi Code does not support request field {name}")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"], ["developer", "user"]):
            raise HTTPException(422, "Kimi Code accepts one user message with an optional system or developer message")
        prompt = _text(items[-1].content)
        if not prompt.strip():
            raise HTTPException(422, "Kimi Code requires a nonempty task")
        if len(prompt.encode()) + 1 > MAX_PROMPT_BYTES:
            raise HTTPException(422, f"Kimi Code task exceeds the {MAX_PROMPT_BYTES} byte argument limit")
        instructions = [body.instructions] + [_text(item.content) for item in items[:-1]]
        return prompt, "\n\n".join(part for part in instructions if part)

    def kimi_config(self, state: KimiSession) -> dict[str, Any]:
        """Point Kimi at Gym's model servers; every root and child call keeps this rollout's capture prefix."""
        rollout_id = state.request.episode_id.capture_key

        def provider(server: ModelServerRef) -> dict[str, str]:
            base_url = self.model_endpoint(model_server=server, rollout_id=rollout_id).base_url
            return {
                "type": "openai",
                "base_url": base_url,
                "api_key": "gym",  # pragma: allowlist secret
                "custom_headers": {"x-session-id": rollout_id},
            }

        model = {"max_context_size": self.config.max_context_size, "capabilities": ["thinking", "tool_use"]}
        config: dict[str, Any] = {
            "default_model": "policy",
            "telemetry": False,
            "auto_session_title": False,
            "providers": {"policy": provider(self.config.model_server)},
            "models": {"policy": model | {"provider": "policy", "model": self.config.model}},
            "tools": {"disabled": list(self.config.disabled_tools)},
        }
        if self.config.max_steps_per_turn is not None:
            config["loop_control"] = {"max_steps_per_turn": self.config.max_steps_per_turn}
        if self.config.print_max_turns is not None:
            config["background"] = {"print_max_turns": self.config.print_max_turns}
        if self.config.thinking_effort:
            config["thinking"] = {"effort": self.config.thinking_effort}
        if not self.config.subagents:
            config["tools"]["disabled"] += ["Agent", "AgentSwarm"]
        if server := self.config.subagent_model_server:
            config["providers"]["subagent"] = provider(server)
            config["models"]["subagent"] = model | {
                "provider": "subagent",
                "model": self.config.subagent_model or self.config.model,
            }
            config["secondary_model"] = {"default_model": "subagent", "force": self.config.subagent_model_force}
        if self.config.permission_rules:
            config["permission"] = {"rules": self.config.permission_rules}
        return deep_merge(deepcopy(self.config.kimi_config), config)

    def _env(self, state: KimiSession) -> dict[str, str]:
        access = state.request.sandbox_access
        env = {
            key: value
            for key, value in (access.env if access else {}).items()
            if not key.startswith(("KIMI_", "OPENAI_", "MOONSHOT_"))
        }
        env |= {
            # Kimi's config, sessions and logs live in the session directory, so tool shells keep the task's HOME.
            "KIMI_CODE_HOME": state.home,
            "KIMI_DISABLE_TELEMETRY": "1",
            "KIMI_CODE_NO_AUTO_UPDATE": "1",
            "KIMI_DISABLE_CRON": "1",
            # Native assets go to the runtime's cache, which the installer's version check fills, not the task's HOME.
            "KIMI_CODE_CACHE_DIR": f"{state.runtime}/cache",
        }
        if self.config.swarm_max_concurrency:
            env["KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY"] = str(self.config.swarm_max_concurrency)
        return env

    async def _stage(self, state: KimiSession, body: NeMoGymResponseCreateParamsNonStreaming) -> SandboxCommand:
        prompt, instructions = self._conversation(body)
        return await state.stage(
            command=[f"{state.runtime}/kimi", "-p", prompt, "--output-format", "stream-json"],
            env=self._env(state),
            config=to_toml(self.kimi_config(state)),
            agents_md=instructions or None,
        )

    @staticmethod
    def _read_sessions(archive: bytes, session_id: str | None) -> tuple[NeMoGymResponseUsage | None, list[dict]]:
        with tempfile.TemporaryDirectory(prefix="gym-kimi-sessions-") as temporary:
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(temporary, filter="data")
            # Without the resume hint (a timed-out or failed run), only a single session is unambiguous.
            matches = [path for path in Path(temporary, "sessions").glob(f"*/{session_id or '*'}") if path.is_dir()]
            return parse_session(matches[0]) if len(matches) == 1 else (None, [])

    def _observations(
        self,
        state: KimiSession,
        *,
        status: str,
        error_type: str | None,
        output: list[Any],
        subagents: list[dict[str, Any]],
        usage_known: bool,
    ) -> AgentObservationBundle:
        cleanup = state.session.cleanup
        root = state.request.episode_id.capture_key
        conversation = []
        if state.activation_request is not None:
            prompt, _ = self._conversation(state.activation_request)
            conversation.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        records: list[Any] = [
            AgentInvocation(
                invocation_id=root, status=status, error_type=error_type, conversation=conversation + output
            )
        ]
        for child in subagents:
            started, ended = child.get("started"), child.get("ended")
            records.append(
                AgentInvocation(
                    invocation_id=f"{root}/{child['id']}",
                    parent_invocation_id=root if child["parent"] in (None, "main") else f"{root}/{child['parent']}",
                    spawned_by_tool_call_id=child["tool_call_id"],
                    status=child["status"],
                    duration_ms=ended - started if started is not None and ended is not None else None,
                )
            )
        records.append(
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
            )
        )
        gaps = ["harness_tool_timing_unavailable"]
        if self.config.subagents:
            # Kimi sends one static x-session-id per provider, so child calls are attributed to the root.
            gaps.append("model_call_ownership_unavailable")
        if not usage_known:
            gaps.append("usage_unavailable")
        gaps += ["optional_tool_grant_skipped" for _ in self.effective_tool_accesses(state.request)]
        return AgentObservationBundle(
            source="kimi_code", records=records, gaps=[ObservationGap(code=code) for code in sorted(set(gaps))]
        )

    async def _activate(self, state: KimiSession, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        timeout = resolve_agent_timeout(
            body.metadata, ceiling=min(self.config.timeout, state.request.agent_timeout_sec or self.config.timeout)
        )
        try:
            async with self._sem:
                artifacts: KimiArtifacts = await state.session.execute(
                    stage_activation=lambda: self._stage(state, body),
                    collect=state.collect,
                    timeout=timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            state.observations = self._observations(
                state, status="failed", error_type=type(error).__name__, output=[], subagents=[], usage_known=False
            )
            raise
        cleanup = state.session.cleanup
        if cleanup is None or not cleanup["cleanup_confirmed"]:
            raise RuntimeError("Kimi Code ended without confirmed supervisor cleanup")
        if cleanup["error"]:
            raise RuntimeError(f"Kimi Code supervision failed: {cleanup['error']}")
        return_code, timed_out = cleanup["return_code"], cleanup["timed_out"]
        if return_code in RUNNER_FAILURES and not timed_out:
            raise RuntimeError(f"Kimi Code did not start (runner exit {return_code}): {artifacts.stderr[-4096:]}")
        output = parse_stream_json(artifacts.stdout)
        usage, subagents = (None, [])
        if artifacts.sessions is not None:
            usage, subagents = await asyncio.to_thread(
                self._read_sessions, artifacts.sessions, stream_session_id(artifacts.stdout)
            )
        answered = any(getattr(item, "type", None) == "message" for item in output)

        status, outcome = "completed", {}
        cause = TURN_FAILURE.findall(artifacts.stderr) if return_code else []
        code, message = cause[-1] if cause else (None, "")
        http = re.match(r"(?:Error: )?([45]\d\d)\b", message)
        if timed_out or code in LIMIT_CODES:
            # A deadline or rollout stop keeps whatever the model did, even before Kimi printed anything.
            status = "incomplete"
        elif code and code.startswith("provider.") and http:
            status, outcome = "failed", {"failure_kind": "model_request_error", "status_code": int(http[1])}
        elif return_code != 0:
            status, outcome = "failed", {"failure_kind": "unknown"}
        elif not answered:
            status, outcome = "failed", {"failure_kind": "agent_invalid_output"}

        state.observations = self._observations(
            state,
            status=status,
            error_type=None if status == "completed" else f"exit_{return_code}" if return_code else None,
            output=output,
            subagents=subagents,
            usage_known=usage is not None,
        )
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            object="response",
            model=self.config.model,
            output=output,
            usage=usage,
            status=status,
            error={
                "code": "server_error",
                "message": f"{code}: {message}" if code else f"Kimi Code exit {return_code}",
            }
            if status == "failed"
            else None,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            metadata={
                "harness_execution": "sandbox",
                "kimi_version": KIMI_VERSION,
                "kimi_exit_code": str(return_code),
                "kimi_num_subagents": str(len(subagents)),
                "kimi_max_concurrent_subagents": str(max_concurrency(subagents)),
            },
        )
        if status == "failed":
            LOG.warning("Kimi Code run failed (exit %s); stderr tail: %s", return_code, artifacts.stderr[-2048:])
            return self.report_outcome(response, outcome)
        return self.report_rollout_stop(response, state.session)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(422, "Kimi Code runs only in an agent session managed by an Environment Server")
        state = self._require_agent_session(session_id)
        if not isinstance(state, KimiSession):
            raise TypeError("Expected KimiSession")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Kimi Code activation must use its seeded rollout route")
        if state.session.closing:
            raise HTTPException(409, "Kimi Code session is closing")
        self._conversation(body)
        if state.task is None:
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._activate(state, state.activation_request))
        elif body != state.activation_request:
            raise HTTPException(409, "Kimi Code sessions support one activation; retries must repeat the same request")
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        """Kimi Code runs through an Environment Server, which owns verification."""
        raise HTTPException(422, "Run Kimi Code through an Environment Server such as single_agent_turn")


if __name__ == "__main__":
    KimiCodeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = KimiCodeAgent.run_webserver()
