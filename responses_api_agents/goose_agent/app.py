# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goose CLI adapter for Environment Server episodes, run inside the task sandbox."""

import asyncio
import io
import json
import logging
import re
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from shlex import quote
from time import time
from typing import Any, ClassVar
from uuid import uuid4

from fastapi import HTTPException, Request
from openai.types.responses.response import IncompleteDetails
from pydantic import Field, JsonValue

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
from nemo_gym.core import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
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
from responses_api_agents.goose_agent.conversion import (
    goose_messages_to_output,
    goose_stop_reason,
    max_concurrent,
    model_server_failure,
    read_goose_sessions,
)
from responses_api_agents.goose_agent.sandbox import (
    GOOSE_VERSION,
    RUNTIME_ROOT,
    SESSION_ROOT,
    GooseArtifacts,
    GooseSession,
)


LOG = logging.getLogger(__name__)

# Goose takes the system prompt only as one argv value, which Linux caps at 128 KiB (MAX_ARG_STRLEN).
MAX_SYSTEM_PROMPT_BYTES = 128 * 1024
# harness_runner.sh exit statuses when Goose itself never ran.
RUNNER_FAILURES = (125, 126, 127)


class GooseAgentConfig(BaseResponsesAPIAgentConfig):
    """Goose release, model and sandbox settings; the Environment Server binds the benchmark."""

    model_server: ModelServerRef
    # Sent as GOOSE_MODEL. The Gym model server serves its configured model.
    model: str
    goose_version: str = GOOSE_VERSION
    concurrency: int = Field(default=32, gt=0)
    # Seconds per activation, enforced by the sandbox's process supervisor.
    timeout: float = Field(default=3600, gt=0, allow_inf_nan=False)
    max_turns: int = Field(default=1000, gt=0)
    # false disables Goose's summon extension (delegate/load), for a solo control.
    delegate: bool = True
    # GOOSE_CONTEXT_LIMIT: the served model's context window, for Goose's compaction.
    context_limit: int | None = Field(default=None, gt=0)
    system_prompt: str | None = None
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    # Only for an agent-owned sandbox, when Resources supplies no sandbox_access.
    sandbox_provider: str | None = None
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)
    network_access: SandboxNetworkAccess = Field(
        default="inherit",
        description=(
            "Egress of a sandbox Goose creates: `model_only` allows only the model server's host. Goose uses no tool "
            "grants, so `model_and_tools` is rejected. A borrowed task sandbox keeps its owner's network, so a "
            "session that borrows one rejects any other value than `inherit`."
        ),
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        value = part if isinstance(part, dict) else part.model_dump()
        if value.get("type") != "input_text":
            raise HTTPException(422, "Goose accepts text input only")
        parts.append(value["text"])
    return "\n".join(parts)


class GooseAgent(SimpleResponsesAPIAgent):
    """Run one Goose activation per agent session, inside the borrowed (or agent-owned) task sandbox."""

    config: GooseAgentConfig
    calls_model_from_sandbox: ClassVar[bool] = True

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sem = asyncio.Semaphore(self.config.concurrency)

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """The failure kind read from Goose 1.54.0's own stop notice."""
        return AgentOutcome(
            failure_kind=native_result.get("failure_kind") or "unknown", status_code=native_result.get("status_code")
        )

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, GooseSession) else None

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> GooseSession:
        """Connect to (or start) the task sandbox and install Goose before the activation."""
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError("Goose sessions use Goose's own tools; required tool grants are unsupported")
        if self.config.goose_version != GOOSE_VERSION:
            raise AgentSeedRejectedError(f"Goose sessions pin release checksums for goose_version {GOOSE_VERSION}")
        access = body.sandbox_access
        owns_sandbox = access is None
        if self.config.network_access == "model_and_tools":
            raise AgentSeedRejectedError("Goose uses no tool grants; set network_access to model_only or inherit")
        if not owns_sandbox and self.config.network_access != "inherit":
            raise AgentSeedRejectedError(
                "Goose network_access applies only to a sandbox it creates, not a borrowed one"
            )
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise AgentSeedRejectedError("Goose requires sandbox_access or a configured sandbox_provider")
            provider_ref = self.config.sandbox_provider
            workdir = self.config.sandbox_config.get("workdir") or "/app"
        else:
            if not isinstance(access.connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("Goose requires direct SandboxAccess")
            if access.user is not None:
                raise AgentSeedRejectedError("Goose does not support a SandboxAccess user override")
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
            raise AgentSeedRejectedError("Goose task workdir must be absolute and separate from its session storage")

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
                spec = restrict_sandbox_network(
                    SandboxSpec(**settings),
                    network_access=self.config.network_access,
                    provider=provider,
                    model_base_url=self.resolve_model_base_url(self.config.model_server.name, trace_context=False),
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
        state = GooseSession(
            request=body,
            provider_name=provider.name,
            sandbox_id=sandbox.sandbox_id,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=f"{SESSION_ROOT}/{uuid4().hex}",
                workdir=str(path),
                harness="Goose",
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
                        raise RuntimeError(f"Cannot create the Goose task workdir {path}")
                await state.install(timeout=self.config.sandbox_install_timeout_seconds)
        except BaseException as error:
            try:
                await state.close(timeout=self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, GooseSession):
            raise TypeError("Expected GooseSession")
        await state.close(timeout=self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.session.cleanup_confirmed,
            agent_observations=state.observations
            or AgentObservationBundle(source="goose", gaps=[ObservationGap(code="observation_capture_failed")]),
        )

    def _conversation(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
        """Split the request into Goose's task (stdin) and system prompt (--system), rejecting other controls."""
        if body.model is not None and body.model != self.config.model:
            raise HTTPException(422, "Goose model is selected by the agent configuration")
        supported = {"input", "instructions", "model", "temperature"}
        for name, field in type(body).model_fields.items():
            value = getattr(body, name)
            if name in supported or (name in ("stream", "background") and value is False):
                continue
            if name == "metadata" and value and set(value) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"Goose does not support request field {name}")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"], ["developer", "user"]):
            raise HTTPException(422, "Goose accepts one user message with an optional system or developer message")
        prompt = _text(items[-1].content)
        if not prompt.strip():
            raise HTTPException(422, "Goose requires a nonempty task")
        system = [self.config.system_prompt, body.instructions] + [_text(item.content) for item in items[:-1]]
        system_text = "\n\n".join(part for part in system if part)
        if len(("--system=" + system_text).encode()) + 1 > MAX_SYSTEM_PROMPT_BYTES:
            raise HTTPException(422, f"Goose system prompt exceeds the {MAX_SYSTEM_PROMPT_BYTES} byte argument limit")
        return prompt, system_text

    def _command(self, state: GooseSession, system: str) -> list[str]:
        command = [
            f"{state.runtime}/goose",
            "run",
            "--name",
            "gym",
            # The task goes in on stdin: Goose drops --system for `-i <file>`.
            "-i",
            "-",
            "--max-turns",
            str(self.config.max_turns),
            "--output-format",
            "stream-json",
        ]
        # `--system=` keeps a prompt that starts with "-" from being parsed as an option.
        return command + ([f"--system={system}"] if system else [])

    def _env(self, state: GooseSession, body: NeMoGymResponseCreateParamsNonStreaming) -> dict[str, str]:
        access = state.request.sandbox_access
        env = {
            key: value
            for key, value in (access.env if access else {}).items()
            if not key.startswith(("GOOSE_", "OPENAI_", "OTEL_", "LANGFUSE_"))
        }
        env |= {
            # Goose's config, sessions and state live in the session directory, so tool shells keep the task's HOME.
            "GOOSE_PATH_ROOT": state.root,
            "GOOSE_PROVIDER": "openai",
            "GOOSE_MODEL": self.config.model,
            # The environment outranks a provider or model the policy passes to `delegate`.
            "GOOSE_SUBAGENT_PROVIDER": "openai",
            "GOOSE_SUBAGENT_MODEL": self.config.model,
            # Headless runs cannot answer approval prompts.
            "GOOSE_MODE": "auto",
            "GOOSE_DISABLE_KEYRING": "1",
            "GOOSE_TELEMETRY_OFF": "1",
            "OPENAI_BASE_URL": self.model_endpoint(
                model_server=self.config.model_server, rollout_id=state.request.episode_id.capture_key
            ).base_url,
            "OPENAI_API_KEY": "gym",  # pragma: allowlist secret
            # Correlates every model call, including failed ones and delegates', with this invocation.
            "OPENAI_CUSTOM_HEADERS": f"x-session-id={state.request.episode_id.capture_key}",
        }
        if self.config.context_limit:
            env["GOOSE_CONTEXT_LIMIT"] = str(self.config.context_limit)
        if body.temperature is not None:
            env["GOOSE_TEMPERATURE"] = str(body.temperature)
        return env

    async def _stage(self, state: GooseSession, body: NeMoGymResponseCreateParamsNonStreaming) -> SandboxCommand:
        prompt, system = self._conversation(body)
        # JSON is valid YAML, which Goose reads its config as. Goose fills in its other extensions with defaults.
        # The solo control also disables extensionmanager, which could re-enable summon.
        enabled = {"enabled": self.config.delegate}
        config = json.dumps({"extensions": {"summon": enabled, "extensionmanager": enabled}})
        return await state.stage(
            command=self._command(state, system), env=self._env(state, body), prompt=prompt, config=config
        )

    @staticmethod
    def _read_sessions(archive: bytes) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="gym-goose-sessions-") as temporary:
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(temporary, filter="data")
            return read_goose_sessions(Path(temporary) / "sessions" / "sessions.db")

    def _observations(
        self, state: GooseSession, *, status: str, error_type: str | None, output: list[Any]
    ) -> AgentObservationBundle:
        cleanup = state.session.cleanup
        conversation = []
        if state.activation_request is not None:
            prompt, _ = self._conversation(state.activation_request)
            conversation.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        gaps = [
            "usage_unavailable",
            "harness_tool_timing_unavailable",
            "child_invocations_unavailable",
        ]
        gaps += ["optional_tool_grant_skipped" for _ in self.effective_tool_accesses(state.request)]
        return AgentObservationBundle(
            source="goose",
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

    async def _activate(self, state: GooseSession, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        timeout = resolve_agent_timeout(
            body.metadata, ceiling=min(self.config.timeout, state.request.agent_timeout_sec or self.config.timeout)
        )
        try:
            async with self._sem:
                artifacts: GooseArtifacts = await state.session.execute(
                    stage_activation=lambda: self._stage(state, body),
                    collect=state.collect,
                    timeout=timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            state.observations = self._observations(state, status="failed", error_type=type(error).__name__, output=[])
            raise
        cleanup = state.session.cleanup
        if cleanup is None or not cleanup["cleanup_confirmed"]:
            raise RuntimeError("Goose ended without confirmed supervisor cleanup")
        if cleanup["error"]:
            raise RuntimeError(f"Goose supervision failed: {cleanup['error']}")
        return_code, timed_out = cleanup["return_code"], cleanup["timed_out"]
        if return_code in RUNNER_FAILURES and not timed_out:
            raise RuntimeError(f"Goose did not start (runner exit {return_code}): {artifacts.stderr[-4096:]}")
        if artifacts.sessions is None and not timed_out:
            raise RuntimeError(f"Goose left no session store: {artifacts.stderr[-4096:]}")
        failed_extensions = re.findall(r"Failed to start extension '([^']+)'", artifacts.stderr)
        if failed_extensions:
            raise RuntimeError(f"Goose extensions could not start: {', '.join(sorted(set(failed_extensions)))}")
        sessions = await asyncio.to_thread(self._read_sessions, artifacts.sessions) if artifacts.sessions else []
        roots = [session for session in sessions if not session.get("parent_session_id")]
        if len(roots) > 1 or (not roots and not timed_out):
            raise RuntimeError(f"Expected one root Goose session, found {len(roots)}")
        output = goose_messages_to_output(roots[0]["messages"]) if roots else []
        children = [session for session in sessions if session.get("parent_session_id")]

        status, incomplete, failure_kind, status_code = "completed", None, None, None
        stop, notice = (None, "") if timed_out else goose_stop_reason(artifacts.stdout, sessions)
        if timed_out or stop in ("max_turns", "context_limit"):
            # A deadline or rollout stop, even before any output, and Goose's own limits are graded as they stand.
            status = "incomplete"
        elif stop == "model_server":
            status = "failed"
            failure_kind, status_code = model_server_failure(notice)
        elif stop == "length":
            status, incomplete = "incomplete", IncompleteDetails(reason="max_output_tokens")
        elif stop == "empty_response":
            status, failure_kind = "failed", "agent_invalid_output"
        elif stop == "model_error" or return_code != 0:
            # Unexplained, including a bare exit 137: a provider-confirmed OOM arrives from SandboxSession instead.
            status, failure_kind = "failed", "unknown"

        state.observations = self._observations(state, status=status, error_type=stop, output=output)
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            object="response",
            model=self.config.model,
            output=output,
            # Goose's usage ledger is not tied to individual model calls, so totals stay unknown.
            usage=None,
            status=status,
            error={"code": "server_error", "message": notice or f"Goose exited with code {return_code}"}
            if status == "failed"
            else None,
            incomplete_details=incomplete,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            metadata={
                "harness_execution": "sandbox",
                "goose_version": GOOSE_VERSION,
                "goose_exit_code": str(return_code),
                "goose_stop_reason": stop or "",
                "goose_subagents": str(len(children)),
                "goose_max_concurrent_subagents": str(
                    max_concurrent(
                        [
                            (min(m["created"] for m in s["messages"]), max(m["created"] for m in s["messages"]))
                            for s in children
                            if s["messages"]
                        ]
                    )
                ),
            },
        )
        if status == "failed":
            LOG.warning("Goose run failed (%s, exit %s); stderr tail: %s", stop, return_code, artifacts.stderr[-2048:])
            return self.report_outcome(response, {"failure_kind": failure_kind, "status_code": status_code})
        return self.report_rollout_stop(response, state.session)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(422, "Goose runs only in an agent session managed by an Environment Server")
        state = self._require_agent_session(session_id)
        if not isinstance(state, GooseSession):
            raise TypeError("Expected GooseSession")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Goose activation must use its seeded rollout route")
        if state.session.closing:
            raise HTTPException(409, "Goose session is closing")
        self._conversation(body)
        if state.task is None:
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._activate(state, state.activation_request))
        elif body != state.activation_request:
            raise HTTPException(409, "Goose sessions support one activation; retries must repeat the same request")
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        """Goose runs through an Environment Server, which owns verification."""
        raise HTTPException(422, "Run Goose through an Environment Server such as single_agent_turn")


if __name__ == "__main__":
    GooseAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = GooseAgent.run_webserver()
