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
"""Run Factory Droid (``droid exec``) in the task sandbox of an Environment Server agent session."""

import asyncio
import logging
import re
from asyncio import Semaphore
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from shlex import quote
from time import time
from typing import Any, ClassVar, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import ConfigDict, Field

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AGENT_TIMEOUT_METADATA_KEY,
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
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
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    MODEL_CONTEXT_LIMIT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
    NgFailureEvent,
    observe_failure,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxExecResult,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap, SandboxObservation
from nemo_gym.utils import bounded_text
from responses_api_agents.droid_agent.parsing import DroidRun, parse_events
from responses_api_agents.droid_agent.runtime import install_sandbox_runtime
from responses_api_agents.droid_agent.sandbox import RUNTIME_PREFIX, SESSION_ROOT, DroidSandboxSession


LOG = logging.getLogger(__name__)
DEFAULT_DROID_VERSION = "1.1.0"
# The bootstrap's exit status for an image Droid cannot run in, which seed refuses as final.
_UNSERVABLE_IMAGE = 78
# Bootstrap with POSIX sh; the runtime installer, the harness runner and the supervisor require Bash. There is no
# apk branch: Droid has no musl build, so an Alpine image is refused either way, before bash is installed into it.
_BASH_BOOTSTRAP = f"""if ! command -v bash >/dev/null 2>&1; then
    [ "$(id -u)" = 0 ] || {{ echo "Droid sandbox sessions require bash: preinstall it or use a root image." >&2; exit {_UNSERVABLE_IMAGE}; }}
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bash || exit 1
    else
        echo "Droid sandbox sessions require bash: preinstall it (automatic installation requires apt-get)." >&2
        exit {_UNSERVABLE_IMAGE}
    fi
fi
"""
# Environment servers keep this many characters of a failed response's error message as the failure reason.
_FAILURE_REASON_CHARS = 2000
# The release whose error event messages were read from runs against a scripted model server. Another release
# may word them differently, so its failures are reported as unknown.
_CLASSIFIED_DROID_VERSION = "1.1.0"
# A rejected model request: the HTTP status, a space, then the message of the response's error body.
_MODEL_REQUEST_STATUS = re.compile(r"^([45]\d{2}) ")
# A refused or reset connection to the model server.
_TRANSPORT_ERROR = "Connection error."
# Droid compacts the conversation when the model rejects its context, and stops with this when the compaction
# request does not fit either.
_CONTEXT_LIMIT = (
    "The existing summary and compaction instructions do not fit the selected model input window. "
    "Choose a compaction model with a larger context window."
)


class DroidAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    model: str
    droid_version: str = Field(default=DEFAULT_DROID_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    concurrency: int = Field(default=32, gt=0)
    # Ceiling on one activation's model and tool work; a task's metadata.agent_timeout_sec can lower it.
    timeout: float = Field(default=1800, gt=0, allow_inf_nan=False)
    # Droid sends this as max_tokens on every model call (BYOK maxOutputTokens).
    max_output_tokens: int = Field(default=8192, gt=0)
    # The model's context window (BYOK maxContextLimit), from which Droid decides when to compact.
    context_window: Optional[int] = Field(default=None, gt=0)
    system_prompt: Optional[str] = None

    # Used only when an episode does not supply a borrowed SandboxAccess.
    sandbox_provider: Optional[str] = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_install_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)


class DroidAgent(SimpleResponsesAPIAgent):
    """One ``droid exec`` run per agent session, inside the task sandbox; there is no host or ``/run`` path."""

    config: DroidAgentConfig
    # The CLI runs inside the task sandbox and calls the model server from there.
    calls_model_from_sandbox: ClassVar[bool] = True
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.router.routes = [route for route in app.router.routes if getattr(route, "path", None) != "/run"]
        return app

    async def run(self, body: object) -> None:
        raise NotImplementedError("Use the single_agent_turn Environment Server")

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Match the error event against the messages Droid 1.1.0 was seen to write; tested on that release only."""
        if self.config.droid_version != _CLASSIFIED_DROID_VERSION:
            return AgentOutcome(failure_kind="unknown")
        message = str(native_result.get("error") or "")
        if message == _CONTEXT_LIMIT:
            return AgentOutcome(failure_kind=MODEL_CONTEXT_LIMIT)
        if message == _TRANSPORT_ERROR:
            return AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
        match = _MODEL_REQUEST_STATUS.match(message)
        if match:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=int(match[1]))
        return AgentOutcome(failure_kind="unknown")

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, DroidSandboxSession) else None

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> DroidSandboxSession:
        """Install or reuse the pinned runtime in the task sandbox before the first activation."""
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError("Droid runs its own tools; required Resources tool grants are unsupported")
        access = body.sandbox_access
        owns_sandbox = access is None
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise AgentSeedRejectedError("Droid requires sandbox_access or a configured sandbox_provider")
            settings = {"workdir": "/app", **self.config.sandbox_config}
            workdir, provider_ref = settings["workdir"], self.config.sandbox_provider
        else:
            if not isinstance(access.connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("Droid supports only direct sandbox connections")
            workdir, provider_ref = access.workdir, access.connection.provider_config_ref
        if not isinstance(workdir, str):
            raise AgentSeedRejectedError("Droid sandbox workdir must be an absolute path string")
        path = PurePosixPath(workdir)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or "\x00" in str(path)
            or path in (PurePosixPath("/"), PurePosixPath("/tmp"))
            or str(path).startswith((SESSION_ROOT, RUNTIME_PREFIX))
        ):
            raise AgentSeedRejectedError(
                "Droid sandbox workdir must be absolute and separate from runtime/session storage"
            )
        global_config = self.server_client.global_config_dict
        try:
            provider_config = resolve_provider_config(provider_ref, global_config)
            # Droid calls the Model Server from inside the sandbox, where a loopback URL is the sandbox itself.
            assert_model_url_reachable_from_sandbox(
                self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key),
                provider_name=next(iter(provider_config), None),
            )
            if owns_sandbox:
                settings["metadata"] = {
                    **resolve_provider_metadata(provider_ref, global_config),
                    **(settings.get("metadata") or {}),
                    "gym-agent": "droid",
                    "gym-session": body.agent_session_id[:63],
                    "gym-rollout": body.episode_id.capture_key[:63],
                }
                # Provider TTLs outlive this server: cover setup, one activation and cleanup.
                settings.setdefault("ttl_s", self.config.sandbox_install_timeout_seconds + self.config.timeout + 600)
                spec = SandboxSpec(**settings)
            # An unknown provider or invalid provider settings fail every retry the same way.
            provider = create_provider(provider_config)
        except (TypeError, ValueError) as error:
            raise AgentSeedRejectedError(str(error)) from error
        if owns_sandbox:
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
            except BaseException:
                try:
                    await provider.aclose()
                except BaseException as cleanup_error:
                    LOG.exception("Droid provider cleanup failed after connect failed")
                    if isinstance(cleanup_error, NgFailureEvent):
                        observe_failure(cleanup_error)
                    else:
                        cleanup_failure = NgFailureEvent(
                            boundary="sandbox",
                            failure_kind="session_release_failed",
                            failure_reason=f"Droid provider cleanup failed: {type(cleanup_error).__name__}",
                            operation="aclose",
                            stage="cleanup",
                        )
                        cleanup_failure.__cause__ = cleanup_error
                        observe_failure(cleanup_failure)
                raise
        directory = f"{SESSION_ROOT}/{uuid4().hex}"
        state = DroidSandboxSession(
            request=body,
            runtime=RUNTIME_PREFIX + self.config.droid_version,
            session=SandboxSession(
                sandbox=sandbox, session_dir=directory, workdir=workdir, harness="Droid", owns_sandbox=owns_sandbox
            ),
        )
        try:
            async with asyncio.timeout(self.config.sandbox_install_timeout_seconds):
                if owns_sandbox:
                    await sandbox.start(spec)
                    self._check_setup(await sandbox.exec(f"mkdir -p -- {quote(workdir)}", cwd="/", timeout_s=30))
                prepared = await sandbox.exec(
                    _BASH_BOOTSTRAP + f"test -d {quote(workdir)} && umask 077 && mkdir -p {quote(directory)}/home",
                    cwd="/",
                    timeout_s=self.config.sandbox_install_timeout_seconds,
                )
                if prepared.return_code == _UNSERVABLE_IMAGE and not prepared.error_type:
                    raise AgentSeedRejectedError(
                        (prepared.stderr or "").strip()[-2000:] or "Droid cannot run in this sandbox image"
                    )
                self._check_setup(prepared)
                await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
                installer = "install_droid_runtime.sh"
                await sandbox.upload(Path(__file__).with_name(installer), f"{directory}/{installer}")
                await install_sandbox_runtime(
                    sandbox,
                    version=self.config.droid_version,
                    directory=directory,
                    runtime=state.runtime,
                    timeout=self.config.sandbox_install_timeout_seconds,
                )
            return state
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise

    @staticmethod
    def _check_setup(result: SandboxExecResult) -> None:
        if result.return_code != 0 or result.error_type:
            details = "\n".join(part for part in (result.stderr, result.stdout) if part)
            raise RuntimeError(
                f"Droid sandbox setup failed (exit={result.return_code}, error={result.error_type}): {details[-16000:]}"
            )

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Confirm cleanup and return a bundle even when no activation ran."""
        assert isinstance(state, DroidSandboxSession)
        await state.close(self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.session.cleanup_confirmed,
            agent_observations=state.observations
            or AgentObservationBundle(source="droid", gaps=[ObservationGap(code="agent_activation_interrupted")]),
        )

    def _validate_request(self, body: NeMoGymResponseCreateParamsNonStreaming) -> None:
        """Reject request controls Droid cannot apply instead of dropping them."""
        for name, field in type(body).model_fields.items():
            value = getattr(body, name)
            if name in {"input", "model"} or (name in {"stream", "background"} and value is False):
                continue
            if name == "metadata" and value is not None and set(value) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"Droid sandbox execution does not support request field {name}")
        if body.model not in {None, "", self.config.model, self.config.model_server.name} or body.model_extra:
            raise HTTPException(422, "Droid's model is selected by agent configuration")
        if not isinstance(body.input, str):
            roles = [getattr(item, "role", None) for item in body.input]
            if roles not in (["user"], ["system", "user"]):
                raise HTTPException(422, "Droid accepts one user message with an optional leading system message")
            for item in body.input:
                if not isinstance(item.content, str) and any(
                    (part.get("type") if isinstance(part, dict) else part.type) != "input_text"
                    for part in item.content
                ):
                    raise HTTPException(422, "Droid input requires text messages")

    def _conversation_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, Optional[str]]:
        """Return the prompt and the system text appended to Droid's own system prompt."""
        if isinstance(body.input, str):
            return body.input, self.config.system_prompt

        def text(item: Any) -> str:
            if isinstance(item.content, str):
                return item.content
            return "".join(part.get("text", "") if isinstance(part, dict) else part.text for part in item.content)

        system = text(body.input[0]) if len(body.input) == 2 else None
        return text(body.input[-1]), "\n\n".join(part for part in (self.config.system_prompt, system) if part) or None

    def _settings(self, invocation_id: str) -> dict[str, Any]:
        """Droid's runtime settings: one BYOK model that points at this rollout's Model Server route."""
        model: dict[str, Any] = {
            "model": self.config.model,
            "displayName": "Gym",
            "baseUrl": self.resolve_model_base_url(self.config.model_server.name, invocation_id),
            # The Model Server holds the provider credential; Droid needs a non-empty key to send.
            "apiKey": "gym",
            "provider": "generic-chat-completion-api",
            "maxOutputTokens": self.config.max_output_tokens,
            # Droid sends these on every call, its subagents' and compaction's included, so capture can
            # assign each call, a failed one too, to this invocation.
            "extraHeaders": {"x-session-id": invocation_id},
        }
        if self.config.context_window is not None:
            model["maxContextLimit"] = self.config.context_window
        return {"cloudSessionSync": False, "customModels": [model]}

    def _command(self, state: DroidSandboxSession, *, system: bool) -> list[str]:
        directory = state.session.session_dir
        command = [
            f"{state.runtime}/droid",
            "exec",
            "--settings",
            f"{directory}/settings.json",
            "--model",
            f"custom:{self.config.model}",
            "--output-format",
            "stream-json",
            # The task sandbox is the isolation boundary, as for Codex's danger-full-access.
            "--skip-permissions-unsafe",
            "--cwd",
            state.session.workdir,
        ]
        if system:
            command += ["--append-system-prompt-file", f"{directory}/system.txt"]
        return command + ["--file", f"{directory}/prompt.txt"]

    @staticmethod
    def _usage(run: DroidRun) -> Optional[NeMoGymResponseUsage]:
        # The completion event totals every call of the run, compaction and subagents included. Like Anthropic's
        # usage, Droid's input_tokens leaves out cache reads and writes, which a Responses input total includes.
        if not run.completed or not run.usage:
            return None
        keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        counts = {key: run.usage.get(key, 0 if key == "cache_creation_input_tokens" else None) for key in keys}
        if any(type(value) is not int or value < 0 for value in counts.values()):
            return None
        input_tokens = (
            counts["input_tokens"] + counts["cache_read_input_tokens"] + counts["cache_creation_input_tokens"]
        )
        return NeMoGymResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=counts["cache_read_input_tokens"]),
            output_tokens=counts["output_tokens"],
            # Droid 1.1.0 reports thinking_tokens 0 for a Chat Completions model that reasons, so the count is
            # unknown; see the gap below.
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=None),
            total_tokens=input_tokens + counts["output_tokens"],
        )

    async def _sandbox_response(
        self, state: DroidSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        prompt, system = self._conversation_input(body)
        invocation_id = state.request.episode_id.capture_key
        timeout = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
        artifacts = failure = None
        try:
            # Close can fence the staging phase just as it fences the execution phase.
            async with self.sem:
                artifacts = await state.execute(
                    {
                        "settings": self._settings(invocation_id),
                        "prompt": prompt,
                        "system": system,
                        "command": self._command(state, system=system is not None),
                        "env": {"FACTORY_HOME_OVERRIDE": state.factory_home, "FACTORY_AIRGAP_ENABLED": "1"},
                    },
                    timeout=timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            failure = error
            artifacts = state.session.artifacts
        run = await asyncio.to_thread(parse_events, artifacts.events if artifacts else "", invocation_id=invocation_id)
        cleanup = state.session.cleanup
        return_code = cleanup["return_code"] if cleanup else None
        if cleanup and cleanup["timed_out"]:
            status, reason = "incomplete", "timeout"
        elif isinstance(failure, asyncio.CancelledError):
            status, reason = "incomplete", "cancelled"
        elif failure is not None:
            status, reason = "failed", "sandbox_error"
        elif run.error is not None and self.classify_outcome({"error": run.error}).failure_kind == MODEL_CONTEXT_LIMIT:
            # The conversation outgrew the model's context: the partial work is verified, not retried.
            status, reason = "incomplete", "context_limit"
        elif run.error is not None:
            status, reason = "failed", "agent_error"
        elif return_code not in (0, None):
            status, reason = "failed", f"process_exit_{return_code}"
        elif run.completed:
            status, reason = "completed", None
        else:
            # Exit 0 without a completion event: the stream was cut or lost, so there is no result to grade.
            status, reason = "failed", "result_missing"

        conversation = [NeMoGymEasyInputMessage(role="system", content=system)] if system else []
        conversation.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        usage = self._usage(run)
        gaps = list(run.gaps)
        if artifacts is None:
            gaps.append(ObservationGap(code="agent_artifact_unavailable", invocation_id=invocation_id))
        elif usage is None:
            gaps.append(ObservationGap(code="model_call_usage_unavailable", invocation_id=invocation_id))
        else:
            gaps.append(
                ObservationGap(
                    code="reasoning_token_usage_unavailable",
                    invocation_id=invocation_id,
                    detail="Droid does not count reasoning tokens of Chat Completions models",
                )
            )
        if state.runtime_info is None:
            gaps.append(ObservationGap(code="runtime_info_unavailable"))
        if status == "failed" and run.error is None and artifacts is not None and artifacts.stderr.strip():
            # Droid reports a failure that ends it before an error event (a crash, rejected settings) only on stderr.
            LOG.warning("Droid run %s failed (%s): %s", invocation_id, reason, artifacts.stderr[-2000:])
            if not artifacts.events.strip():
                gaps.append(
                    ObservationGap(
                        code="agent_startup_failed", invocation_id=invocation_id, detail=artifacts.stderr[-4000:]
                    )
                )
        sandbox_id = state.session.sandbox.sandbox_id
        for tool in run.tools:
            tool.sandbox_id = sandbox_id
        records: list[Any] = [
            AgentInvocation(
                invocation_id=invocation_id,
                harness_session_id=run.session_id,
                status=status,
                # A rollout stop also ends the run as at its deadline; its response carries ng_rollout_stop instead.
                stop_reason="context_limit"
                if reason == "context_limit"
                else "timeout"
                if reason == "timeout" and not state.session.expired
                else None,
                duration_ms=run.duration_ms,
                error_type=reason,
                conversation=[*conversation, *run.conversation],
            ),
            *run.tools,
        ]
        if cleanup:
            access = state.request.sandbox_access
            records.append(
                SandboxObservation(
                    role="agent",
                    provider=access.connection.provider_config_ref if access else self.config.sandbox_provider,
                    sandbox_id=sandbox_id,
                    outcome={"timeout": "timeout", "cancelled": "cancelled", None: "completed"}.get(reason, "failed"),
                    exit_code=return_code,
                    error_type=reason,
                )
            )
        state.observations = AgentObservationBundle(source="droid", records=records, gaps=gaps)
        if failure is not None:
            raise failure
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=run.items,
            status=status,
            # Droid's own error message, or what ended the run without one. Droid holds no credential (its key is a
            # placeholder), so the message is bounded, not redacted.
            error={"code": "server_error", "message": bounded_text(run.error or reason, _FAILURE_REASON_CHARS)}
            if status == "failed"
            else None,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
            metadata={"harness_execution": "sandbox", AGENT_TIMEOUT_METADATA_KEY: str(timeout)},
        )
        response = self.report_rollout_stop(response, state.session)
        if reason == "context_limit":
            # report_outcome describes failed responses only; the stop still tells the environment its cause.
            outcome = AgentOutcome(failure_kind=MODEL_CONTEXT_LIMIT).model_dump_json(exclude_none=True)
            return response.model_copy(update={"metadata": {**response.metadata, AGENT_OUTCOME_METADATA_KEY: outcome}})
        return self.report_outcome(response, {"error": run.error})

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "Droid runs only in an agent session seeded by an Environment Server")
        state = self._require_agent_session(session_id)
        assert isinstance(state, DroidSandboxSession)
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Droid activation does not match the seeded session and rollout")
        if state.session.closing:
            raise HTTPException(409, "Droid session is closing")
        self._validate_request(body)
        if state.task is None:
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._sandbox_response(state, state.activation_request))
        elif body != state.activation_request:
            raise HTTPException(409, "Droid sandbox sessions support one activation; retry the same request")
        return (await asyncio.shield(state.task)).model_copy(deep=True)


if __name__ == "__main__":
    DroidAgent.run_webserver()
