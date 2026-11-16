# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the OpenHands 0.62 CodeAct controller in a borrowed task sandbox, one activation per agent session."""

import asyncio
import json
import logging
import shlex
from asyncio import Semaphore
from collections.abc import Mapping
from pathlib import PurePosixPath
from time import time
from typing import Any, ClassVar, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, field_validator

from nemo_gym.agents import (
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
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, DirectSandboxConnection, create_provider, resolve_provider_config
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse, MCPToolAccess
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    SandboxObservation,
)
from responses_api_agents.openhands_codeact_agent.conversion import (
    SUMMARY_CALL_KIND,
    RunOutcome,
    classify_error,
    compaction_records,
    load_calls,
    output_items,
    read_jsonl,
    response_usage,
    run_outcome,
)
from responses_api_agents.openhands_codeact_agent.sandbox import (
    SESSION_ROOT,
    CodeActArtifacts,
    CodeActSandboxSession,
    check_exec,
)
from responses_api_agents.openhands_codeact_agent.sandbox_runtime import (
    DEFAULT_PYTHON,
    INSTALL_PREFIX,
    install_runtime,
)


LOG = logging.getLogger(__name__)
OPENHANDS_REVISION = "54d2a95edc666551b854a7e5d4205ec69e60ff23"
# Request fields the activation maps onto the controller; every other non-default field is refused.
_MAPPED_REQUEST_FIELDS = {"input", "instructions", "model", "temperature", "top_p", "max_output_tokens"}


class OpenHandsCodeActAgentConfig(BaseResponsesAPIAgentConfig):
    # Agent sessions take the benchmark from the Environment Server; there is no /run path.
    resources_server: Optional[ResourcesServerRef] = None
    model_server: ModelServerRef
    model: str = Field(description="Model name OpenHands sends to the Gym model server.")
    concurrency: int = Field(default=8, gt=0)
    timeout: int = Field(default=10800, gt=0, description="Seconds one activation may run.")
    max_iterations: Optional[int] = Field(
        default=200, ge=1, description="CodeAct's iteration limit; reaching it ends the run as incomplete."
    )
    max_context: int = Field(default=196608, ge=1)
    max_output_tokens: int = Field(default=32768, ge=1, description="Per model call, unless the request sets it.")
    temperature: float = Field(default=1.0, ge=0, description="Unless the request sets it.")
    top_p: float = Field(default=1.0, gt=0, le=1, description="Unless the request sets it.")
    model_read_timeout_s: int = Field(default=1800, ge=1)
    # Re-send a model call whose response timed out or was cut this many times (the request is stateless).
    model_read_retries: int = Field(default=0, ge=0)
    # When the last reply's prompt plus completion tokens reach this fraction of max_context, the condenser
    # summarizes everything after the task prompt in one model call and the agent continues from
    # system prompt + task + summary. Off by default.
    context_summarization_cutoff: Optional[float] = Field(default=None, gt=0, le=1)
    context_summarization_max_event_chars: int = Field(default=10_000, ge=100)
    mcp_tool_timeout_seconds: int = Field(default=900, ge=1, description="Per MCP tool call.")
    finish_tool_names: list[str] = Field(
        default_factory=list,
        description=(
            "Granted task tools (by the server's own name) whose successful call ends the run. When set, CodeAct's "
            "own finish tool is not offered, so the run cannot end without submitting through one of them."
        ),
    )
    remote_openhands_install_script_path: str = Field(
        description="Absolute sandbox path to the staged runtime installer."
    )
    remote_openhands_archive_path: str = Field(
        description="Absolute sandbox path to the runtime archive and its .sha256 sidecar."
    )
    sandbox_python: str = DEFAULT_PYTHON
    harness_install_timeout: int = Field(default=900, ge=31)

    @field_validator("remote_openhands_install_script_path", "remote_openhands_archive_path", "sandbox_python")
    @classmethod
    def require_absolute_runtime_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("Runtime paths must be absolute sandbox paths")
        return value


class OpenHandsCodeActAgent(SimpleResponsesAPIAgent):
    """OpenHands CodeAct runs in a task sandbox borrowed from the Resources Server, one activation per session."""

    config: OpenHandsCodeActAgentConfig
    # The runner calls the model server from inside the task sandbox.
    calls_model_from_sandbox: ClassVar[bool] = True
    sem: Optional[Semaphore] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.sem = Semaphore(self.config.concurrency)

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Classify a failed run from the last model-request record or the controller's last error."""
        kind, status_code = classify_error(
            dict(native_result.get("result") or {}), native_result.get("requests") or []
        )
        return AgentOutcome(failure_kind=kind, status_code=status_code)

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, CodeActSandboxSession) else None

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> CodeActSandboxSession:
        unsupported = sorted(
            access.name
            for access in self.effective_tool_accesses(body)
            if access.required and not isinstance(access, MCPToolAccess)
        )
        if unsupported:
            raise AgentSeedRejectedError(
                "OpenHands CodeAct consumes tool grants only over MCP; required grants it cannot use: "
                + ", ".join(unsupported)
            )
        access = body.sandbox_access
        if access is None:
            raise AgentSeedRejectedError("OpenHands CodeAct runs only in a task sandbox the Resources Server lends")
        if not isinstance(access.connection, DirectSandboxConnection):
            raise AgentSeedRejectedError("OpenHands CodeAct supports only direct sandbox connections")
        workdir = access.workdir
        path = PurePosixPath(workdir if isinstance(workdir, str) else "")
        if (
            not path.is_absolute()
            or ".." in path.parts
            or path in (PurePosixPath("/"), PurePosixPath("/tmp"))
            or str(path).startswith((INSTALL_PREFIX, SESSION_ROOT))
        ):
            raise AgentSeedRejectedError(
                "OpenHands CodeAct sandbox workdir must be absolute and separate from its own files"
            )
        global_config = self.server_client.global_config_dict
        try:
            model_base_url = self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key)
            provider_config = resolve_provider_config(access.connection.provider_config_ref, global_config)
            # The runner calls the model server from inside the sandbox, where loopback is the sandbox itself.
            assert_model_url_reachable_from_sandbox(model_base_url, provider_name=next(iter(provider_config), None))
            provider = create_provider(provider_config)
        except (TypeError, ValueError) as error:
            raise AgentSeedRejectedError(str(error)) from error
        try:
            sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
        except BaseException:
            try:
                await provider.aclose()
            except BaseException:
                LOG.exception("OpenHands CodeAct provider cleanup failed before the sandbox was ready")
            raise
        directory = f"{SESSION_ROOT}/{uuid4().hex}"
        state = CodeActSandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=directory,
                workdir=workdir,
                harness="OpenHands CodeAct",
                owns_sandbox=False,
            ),
            interpreter=self.config.sandbox_python,
        )
        try:
            async with asyncio.timeout(self.config.harness_install_timeout * 4 + 300):
                check_exec(
                    await sandbox.exec(
                        f"test -d {shlex.quote(workdir)} && umask 077 && mkdir -p {shlex.quote(directory)}/trace",
                        cwd="/",
                        timeout_s=60,
                    ),
                    "session directory",
                )
                # Install only the harness runtime; the Resources Server has already prepared the task.
                await install_runtime(
                    sandbox,
                    script_path=self.config.remote_openhands_install_script_path,
                    archive_path=self.config.remote_openhands_archive_path,
                    timeout_s=self.config.harness_install_timeout,
                )
                await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
                await state.upload_runner()
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                LOG.exception("OpenHands CodeAct seed cleanup failed; retaining session %s", body.agent_session_id)
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Stop the runner before verification; the borrowed sandbox stays with the Resources Server."""
        assert isinstance(state, CodeActSandboxSession)
        await state.close(self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.session.cleanup_confirmed,
            agent_observations=state.observations
            or AgentObservationBundle(
                source="openhands_codeact", gaps=[ObservationGap(code="agent_activation_interrupted")]
            ),
        )

    def _validate_request(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, dict[str, Any]]:
        """Return the prompt and the model settings; refuse request fields the controller cannot honour."""
        for name, field in type(body).model_fields.items():
            value = getattr(body, name)
            if name in _MAPPED_REQUEST_FIELDS or (name in ("stream", "background") and value is False):
                continue
            if name == "metadata" and value is not None and set(value) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"OpenHands CodeAct does not support request field {name}")
        if body.model not in (None, "", self.config.model) or body.model_extra:
            raise HTTPException(422, "OpenHands CodeAct model is selected by the agent configuration")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"], ["developer", "user"]):
            raise HTTPException(
                422, "OpenHands CodeAct accepts one user message with an optional leading system message"
            )
        texts = []
        for item in items:
            if isinstance(item.content, str):
                texts.append(item.content)
                continue
            if any(
                (part.get("type") if isinstance(part, dict) else part.type) != "input_text" for part in item.content
            ):
                raise HTTPException(422, "OpenHands CodeAct accepts only text input")
            texts.append("".join(part["text"] if isinstance(part, dict) else part.text for part in item.content))
        # CodeAct keeps its own system prompt; request instructions and a system message lead the task message.
        prompt = "\n\n".join(part for part in (body.instructions, *texts) if part)
        settings = {
            "temperature": self.config.temperature if body.temperature is None else body.temperature,
            "top_p": self.config.top_p if body.top_p is None else body.top_p,
            # A per-call limit, as the controller applies it, not a budget for the whole run.
            "max_output_tokens": body.max_output_tokens or self.config.max_output_tokens,
        }
        return prompt, settings

    def _runner_input(self, state: CodeActSandboxSession, *, prompt: str, settings: dict[str, Any]) -> dict[str, Any]:
        invocation_id = state.request.episode_id.capture_key
        endpoint = self.model_endpoint(model_server=self.config.model_server, rollout_id=invocation_id)
        cfg = self.config
        return {
            "workdir": state.session.workdir,
            "prompt": prompt,
            "session_id": invocation_id,
            "model": {
                "name": cfg.model,
                "base_url": endpoint.base_url,
                # Every model call, including the condenser's summaries, names the invocation that owns it.
                "headers": {"x-session-id": invocation_id},
                "max_context": cfg.max_context,
                "read_timeout_s": cfg.model_read_timeout_s,
                "read_retries": cfg.model_read_retries,
                **settings,
            },
            "max_iterations": cfg.max_iterations,
            "finish_tool_names": cfg.finish_tool_names,
            "context_summarization": (
                {
                    "cutoff": cfg.context_summarization_cutoff,
                    "max_event_chars": cfg.context_summarization_max_event_chars,
                }
                if cfg.context_summarization_cutoff is not None
                else None
            ),
            "mcp_servers": [
                {
                    "name": access.name,
                    "url": str(access.connection.url),
                    "headers": access.connection.headers,
                    "required": access.required,
                    "timeout_s": cfg.mcp_tool_timeout_seconds,
                }
                for access in self.effective_tool_accesses(state.request)
                if isinstance(access, MCPToolAccess)
            ],
            "trace_dir": state.path("trace"),
            "result_path": state.path("result.json"),
        }

    async def _activate(
        self,
        state: CodeActSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        prompt: str,
        settings: dict[str, Any],
    ) -> NeMoGymResponse:
        timeout = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
        failure: BaseException | None = None
        artifacts: CodeActArtifacts | None = None
        try:
            async with self.sem:
                artifacts = await state.execute(
                    self._runner_input(state, prompt=prompt, settings=settings),
                    timeout=timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            failure = error
            artifacts = state.session.artifacts
        try:
            return self._response(state, body, artifacts, failure, prompt=prompt, timeout=timeout)
        finally:
            if artifacts is not None:
                artifacts.cleanup()

    def _response(
        self,
        state: CodeActSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        artifacts: CodeActArtifacts | None,
        failure: BaseException | None,
        *,
        prompt: str,
        timeout: float,
    ) -> NeMoGymResponse:
        cleanup = state.session.cleanup
        timed_out = bool(cleanup and cleanup["timed_out"])
        if isinstance(failure, TimeoutError) and timed_out and artifacts is not None:
            # The runner's own deadline ended it and its logs were collected: a timed-out run, judged as such.
            failure = None
        result = None
        if artifacts is not None and artifacts.result is not None:
            try:
                parsed = json.loads(artifacts.result)
                result = parsed if isinstance(parsed, dict) else None
            except ValueError:
                result = None
        calls: list[dict[str, Any]] = []
        requests: list[dict[str, Any]] = []
        compactions: list[dict[str, Any]] = []
        malformed_logs = False
        if artifacts is not None and artifacts.trace_dir is not None:
            requests = read_jsonl(artifacts.trace_dir / "model-requests.jsonl")
            compactions = read_jsonl(artifacts.trace_dir / "compactions.jsonl")
            try:
                calls = load_calls(artifacts.trace_dir / "completions")
            except (OSError, ValueError, KeyError, TypeError):
                LOG.warning("OpenHands CodeAct completion logs are malformed", exc_info=True)
                malformed_logs = True
        outcome = run_outcome(result, timed_out=timed_out, rollout_stop=state.session.expired)
        if failure is not None:
            outcome = RunOutcome(
                "incomplete" if isinstance(failure, asyncio.CancelledError) else "failed",
                "cancelled" if isinstance(failure, asyncio.CancelledError) else "sandbox_error",
                f"{type(failure).__name__}: {failure}",
            )
        try:
            items, tools = output_items(calls)
        except (ValueError, KeyError, TypeError):
            LOG.warning("Could not convert the OpenHands CodeAct trajectory", exc_info=True)
            items, tools, malformed_logs = [], [], True
        state.observations = self._observations(
            state, calls, compactions, outcome, prompt=prompt, items=items, malformed_logs=malformed_logs
        )
        if artifacts is None:
            state.observations.gaps.append(ObservationGap(code="agent_artifact_unavailable"))
        if failure is not None:
            # Termination, transport and capture failures are not results; the episode can be retried.
            raise failure
        if outcome.stop_reason == "runner_error":
            raise RuntimeError(f"OpenHands CodeAct runner failed before writing a result: {artifacts.stderr[-4000:]}")
        if outcome.status == "failed" and artifacts.stderr:
            LOG.warning("%s; the runner's stderr ends with:\n%s", outcome.message, artifacts.stderr[-4000:])
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=items,
            status=outcome.status,
            error={"code": "server_error", "message": outcome.message[:4000]} if outcome.status == "failed" else None,
            tool_choice="auto",
            tools=tools,
            parallel_tool_calls=True,
            usage=response_usage(requests),
            metadata={
                "harness_execution": "sandbox",
                "openhands_revision": OPENHANDS_REVISION,
                "stop_reason": outcome.stop_reason,
                AGENT_TIMEOUT_METADATA_KEY: str(timeout),
            },
        )
        response = self.report_rollout_stop(response, state.session)
        # Only a failed response gets an outcome; after the deadline the cause is unknown.
        return self.report_outcome(response, {} if timed_out else {"result": result or {}, "requests": requests})

    def _observations(
        self,
        state: CodeActSandboxSession,
        calls: list[dict[str, Any]],
        compactions: list[dict[str, Any]],
        outcome: RunOutcome,
        *,
        prompt: str,
        items: list[Any],
        malformed_logs: bool,
    ) -> AgentObservationBundle:
        invocation_id = state.request.episode_id.capture_key
        model_ref = self.config.model_server
        bundle = AgentObservationBundle(source="openhands_codeact")
        try:
            bundle.records.append(
                AgentInvocation(
                    invocation_id=invocation_id,
                    status=outcome.status,
                    stop_reason=outcome.stop_reason,
                    error_type=outcome.stop_reason if outcome.status != "completed" else None,
                    model_calls=[
                        ModelCallRef(model_ref=model_ref, response_id=call["response_id"])
                        for call in calls
                        if call["kind"] != SUMMARY_CALL_KIND and call["response_id"]
                    ],
                    conversation=[NeMoGymEasyInputMessage(role="user", content=prompt), *items],
                )
            )
            bundle.records.extend(
                compaction_records(calls, compactions, invocation_id=invocation_id, model_ref=model_ref)
            )
        except Exception:
            # Observation parsing must not turn a gradable response into a failed episode.
            LOG.exception("Could not build OpenHands CodeAct observations")
            bundle = AgentObservationBundle(
                source="openhands_codeact", gaps=[ObservationGap(code="observation_parse_failed")]
            )
        if malformed_logs:
            bundle.gaps.append(ObservationGap(code="agent_event_malformed", invocation_id=invocation_id))
        # Tool calls are carried by the response items; the controller's tool timings are not reported.
        bundle.gaps.append(ObservationGap(code="tool_calls_unobserved", invocation_id=invocation_id))
        cleanup = state.session.cleanup
        if cleanup:
            access = state.request.sandbox_access
            bundle.records.append(
                SandboxObservation(
                    role="agent",
                    provider=access.connection.provider_config_ref if access else None,
                    sandbox_id=state.session.sandbox.sandbox_id,
                    outcome="timeout"
                    if cleanup["timed_out"]
                    else "cancelled"
                    if outcome.stop_reason == "cancelled"
                    else "failed"
                    if outcome.status == "failed"
                    else "completed",
                    exit_code=cleanup["return_code"],
                    error_type=None if outcome.status == "completed" else outcome.stop_reason,
                )
            )
        return bundle

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(
                422, "OpenHands CodeAct runs only in an agent session managed by an Environment Server"
            )
        state = self._require_agent_session(session_id)
        assert isinstance(state, CodeActSandboxSession)
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "OpenHands CodeAct activation does not match the seeded session and rollout")
        if state.session.closing:
            raise HTTPException(409, "OpenHands CodeAct session is closing")
        prompt, settings = self._validate_request(body)
        if state.task is None:
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(
                self._activate(state, state.activation_request, prompt=prompt, settings=settings)
            )
        elif body != state.activation_request:
            raise HTTPException(409, "OpenHands CodeAct sessions support one activation; retry the same request")
        # A disconnected client must not cancel the shared activation; only close does.
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        """OpenHands CodeAct has no /run path; an Environment Server owns the episode."""
        raise HTTPException(422, "Run OpenHands CodeAct through an Environment Server such as single_agent_turn")


if __name__ == "__main__":
    OpenHandsCodeActAgent.run_webserver()
