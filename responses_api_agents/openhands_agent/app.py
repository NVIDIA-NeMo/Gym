# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the OpenHands Software Agent SDK headless inside the task sandbox of an agent session."""

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
    SandboxNetworkAccess,
    SandboxSession,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
    resolve_agent_timeout,
    restrict_sandbox_network,
)
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
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
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap, SandboxObservation
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.openhands_agent.conversion import (
    RunOutcome,
    classify_error,
    error_chain,
    observation_records,
    output_items,
    parse_events,
    response_usage,
    run_outcome,
)
from responses_api_agents.openhands_agent.sandbox import (
    OPENHANDS_VERSION,
    RUNNER_NAME,
    RUNNER_SOURCE,
    RUNTIME_PREFIX,
    SESSION_ROOT,
    OpenHandsArtifacts,
    OpenHandsSandboxSession,
    check_exec,
    prepare_runtime,
)


LOG = logging.getLogger(__name__)
# OpenHands SDK 1.59.0 LLM settings that change only how the harness calls the model. The adapter owns the
# connection (model, base_url, api_key, usage_id, extra_headers) and turns OpenHands' own model retries off, so its
# retry settings are not offered. Settings that write files (completion logs, tokenizer downloads), call other models
# (fallbacks), or stream (Gym's Chat Completions route does not) are not offered either. OpenHands silently ignores
# unknown names, so every name is checked here.
SUPPORTED_LLM_SETTINGS = frozenset(
    {
        "api_mode",
        "caching_prompt",
        "capability_overrides",
        "disable_stop_word",
        "disable_vision",
        "drop_params",
        "enable_encrypted_reasoning",
        "extended_thinking_budget",
        "force_string_serializer",
        "inline_image_urls",
        "input_cost_per_token",
        "litellm_extra_body",
        "max_input_tokens",
        "max_message_chars",
        "max_output_tokens",
        "model_canonical_name",
        "native_tool_calling",
        "output_cost_per_token",
        "prompt_cache_retention",
        "reasoning_effort",
        "reasoning_summary",
        "seed",
        "temperature",
        "timeout",
        "top_k",
        "top_p",
    }
)


class OpenHandsAgentConfig(BaseResponsesAPIAgentConfig):
    # Agent sessions take the benchmark from the Environment Server; there is no /run path.
    resources_server: Optional[ResourcesServerRef] = None
    model_server: ModelServerRef
    model: str = Field(description="Model name OpenHands sends to the Gym Model Server.")
    # Per worker: with num_workers > 1, each worker runs this many turns at once.
    concurrency: int = Field(default=8, gt=0)
    timeout: int = Field(default=1800, gt=0, description="Seconds one activation may run.")
    max_iterations: int = Field(
        default=500, gt=0, description="OpenHands' max_iteration_per_run; reaching it ends the run as incomplete."
    )
    condenser: bool = Field(
        default=True,
        description="Use OpenHands' default LLMSummarizingCondenser, which summarizes through the same Model Server.",
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Text appended to OpenHands' own system prompt, before the request's instructions."
    )
    llm_settings: dict[str, Any] = Field(
        default_factory=dict, description="OpenHands SDK LLM settings; see SUPPORTED_LLM_SETTINGS."
    )
    # Only for an agent-owned sandbox, when Resources supplies no sandbox_access.
    sandbox_provider: Optional[str] = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    network_access: SandboxNetworkAccess = Field(
        default="inherit",
        description=(
            "Egress of a sandbox OpenHands creates: `model_only` allows only the model server's host. OpenHands keeps "
            "no tool grants, so `model_and_tools` allows the same. A borrowed task sandbox keeps its owner's network, "
            "so a session that borrows one rejects any other value than `inherit`. The Agent Server builds the runtime "
            "and uploads it, so a restricted sandbox needs no network for setup."
        ),
    )
    sandbox_install_timeout_seconds: float = Field(default=1200, gt=0, allow_inf_nan=False)

    @field_validator("llm_settings")
    @classmethod
    def require_supported_llm_settings(cls, settings: dict[str, Any]) -> dict[str, Any]:
        unsupported = sorted(set(settings) - SUPPORTED_LLM_SETTINGS)
        if unsupported:
            raise ValueError(f"OpenHands llm_settings does not support {unsupported}")
        return settings


class OpenHandsAgent(SimpleResponsesAPIAgent):
    """OpenHands SDK conversations in task sandboxes, one activation per agent session."""

    config: OpenHandsAgentConfig
    # The runner calls the model server from inside the task sandbox.
    calls_model_from_sandbox: ClassVar[bool] = True
    sem: Optional[Semaphore] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.sem = Semaphore(self.config.concurrency)

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Classify the error that ended an OpenHands 1.59.0 run, from the runner's recorded exception chain."""
        kind, status_code = classify_error(error_chain(dict(native_result)))
        return AgentOutcome(failure_kind=kind, status_code=status_code)

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, OpenHandsSandboxSession) else None

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> OpenHandsSandboxSession:
        required = sorted(access.name for access in self.effective_tool_accesses(body) if access.required)
        if required:
            raise AgentSeedRejectedError(
                "OpenHands runs its own terminal, file editor and task tracker and cannot use required tool grants: "
                + ", ".join(required)
            )
        global_config = self.server_client.global_config_dict
        access = body.sandbox_access
        owns_sandbox = access is None
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise AgentSeedRejectedError("OpenHands requires sandbox_access or a configured sandbox_provider")
            provider_ref = self.config.sandbox_provider
            settings = {"workdir": "/app", **self.config.sandbox_config}
            # Covers installation, the activation and cleanup, so the provider reclaims a sandbox a crash leaked.
            settings.setdefault("ttl_s", self.config.sandbox_install_timeout_seconds + self.config.timeout + 600)
            try:
                settings["metadata"] = {
                    **resolve_provider_metadata(provider_ref, global_config),
                    **(settings.get("metadata") or {}),
                    "gym-agent": "openhands",
                    "gym-session": body.agent_session_id[:63],
                    "gym-rollout": body.episode_id.capture_key[:63],
                }
                spec = SandboxSpec(**settings)
            except (TypeError, ValueError) as error:
                raise AgentSeedRejectedError(f"Invalid OpenHands sandbox configuration: {error}") from error
            workdir = spec.workdir
        else:
            if not isinstance(access.connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("OpenHands supports only direct sandbox connections")
            if self.config.network_access != "inherit":
                raise AgentSeedRejectedError(
                    "OpenHands network_access applies only to a sandbox it creates, not a borrowed one"
                )
            workdir, provider_ref = access.workdir, access.connection.provider_config_ref
        # A workdir that is not a string (sandbox_config is untyped) is refused like an empty one.
        path = PurePosixPath(workdir if isinstance(workdir, str) else "")
        if (
            not path.is_absolute()
            or ".." in path.parts
            or path in (PurePosixPath("/"), PurePosixPath("/tmp"))
            or str(path).startswith((RUNTIME_PREFIX, SESSION_ROOT))
        ):
            raise AgentSeedRejectedError("OpenHands sandbox workdir must be absolute and separate from its own files")
        try:
            model_base_url = self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key)
            provider_config = resolve_provider_config(provider_ref, global_config)
            # The runner calls the Model Server from inside the sandbox, where loopback is the sandbox itself.
            assert_model_url_reachable_from_sandbox(model_base_url, provider_name=next(iter(provider_config), None))
            # An unknown provider name or malformed provider settings fail here, before any sandbox exists.
            provider = create_provider(provider_config)
        except (TypeError, ValueError) as error:
            raise AgentSeedRejectedError(str(error)) from error

        try:
            if owns_sandbox:
                # OpenHands keeps no tool grants (required ones were refused above), so only the model's host opens.
                spec = restrict_sandbox_network(
                    spec, network_access=self.config.network_access, provider=provider, model_base_url=model_base_url
                )
                sandbox = AsyncSandbox(provider)
            else:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
        except BaseException:
            try:
                await provider.aclose()
            except BaseException as cleanup_error:
                LOG.exception("OpenHands provider cleanup failed before the sandbox was ready")
                if isinstance(cleanup_error, NgFailureEvent):
                    observe_failure(cleanup_error)
                else:
                    cleanup_failure = NgFailureEvent(
                        boundary="sandbox",
                        failure_kind="session_release_failed",
                        failure_reason=f"OpenHands provider cleanup failed: {type(cleanup_error).__name__}",
                        operation="aclose",
                        stage="cleanup",
                    )
                    cleanup_failure.__cause__ = cleanup_error
                    observe_failure(cleanup_failure)
            raise
        directory = f"{SESSION_ROOT}/{uuid4().hex}"
        state = OpenHandsSandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox, session_dir=directory, workdir=workdir, harness="OpenHands", owns_sandbox=owns_sandbox
            ),
        )
        try:
            async with asyncio.timeout(self.config.sandbox_install_timeout_seconds):
                if owns_sandbox:
                    await sandbox.start(spec)
                    # Providers need not create SandboxSpec.workdir. Never prepare a borrowed task here.
                    check_exec(
                        await sandbox.exec(f"mkdir -p -- {shlex.quote(workdir)}", cwd="/", timeout_s=30), "workdir"
                    )
                check_exec(
                    await sandbox.exec(
                        f"test -d {shlex.quote(workdir)} && umask 077 && mkdir -p {shlex.quote(directory)}/state",
                        cwd="/",
                        timeout_s=60,
                    ),
                    "session directory",
                )
                # Install only the harness runtime; Resources has already prepared the task.
                state.interpreter = await prepare_runtime(sandbox, timeout=self.config.sandbox_install_timeout_seconds)
                await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
                await sandbox.upload(RUNNER_SOURCE, f"{directory}/{RUNNER_NAME}")
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                LOG.exception("OpenHands seed cleanup failed; retaining session %s", body.agent_session_id)
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Stop OpenHands before verification; a borrowed sandbox stays with Resources."""
        assert isinstance(state, OpenHandsSandboxSession)
        if state.provider_wait_failure is not None:
            observe_failure(state.provider_wait_failure)
        await state.close(self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.session.cleanup_confirmed,
            agent_observations=state.observations
            or AgentObservationBundle(source="openhands", gaps=[ObservationGap(code="agent_activation_interrupted")]),
        )

    def _validate_request(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
        """Accept one text prompt with optional system text; return (prompt, system)."""
        supported = {"input", "instructions", "model"}
        for name, field in type(body).model_fields.items():
            value = getattr(body, name)
            if name in supported or (name in ("stream", "background") and value is False):
                continue
            if name == "metadata" and value is not None and set(value) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"OpenHands does not support request field {name}")
        if body.model not in (None, "", self.config.model) or body.model_extra:
            raise HTTPException(422, "OpenHands model is selected by the agent configuration")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"]):
            raise HTTPException(422, "OpenHands accepts one user message with an optional leading system message")
        texts = []
        for item in items:
            if isinstance(item.content, str):
                texts.append(item.content)
                continue
            if any(
                (part.get("type") if isinstance(part, dict) else part.type) != "input_text" for part in item.content
            ):
                raise HTTPException(422, "OpenHands accepts only text input")
            texts.append("".join(part["text"] if isinstance(part, dict) else part.text for part in item.content))
        system = "\n\n".join(part for part in (self.config.system_prompt, body.instructions, *texts[:-1]) if part)
        return texts[-1], system

    def _runner_input(self, state: OpenHandsSandboxSession, *, prompt: str, system: str) -> dict[str, Any]:
        invocation_id = state.request.episode_id.capture_key
        endpoint = self.model_endpoint(model_server=self.config.model_server, rollout_id=invocation_id)
        return {
            "workdir": state.session.workdir,
            "prompt": prompt,
            "system_message_suffix": system or None,
            "llm": {
                # With "auto", OpenHands sends models its metadata marks as Responses API models (such as gpt-5)
                # to /responses; the adapter's model connection is Chat Completions.
                "api_mode": "chat",
                **self.config.llm_settings,
                # LiteLLM's openai provider sends Chat Completions to the base URL as given.
                "model": f"openai/{self.config.model}",
                "base_url": endpoint.base_url,
                # The Model Server holds the provider credentials; this placeholder never leaves Gym.
                "api_key": "gym",
                "usage_id": "agent",
                # OpenHands sends a failed call again on its own, but through LiteLLM it cannot tell a 4xx from a
                # transient error: LiteLLM 1.105.7 raises any error whose text mentions a rate limit as a
                # RateLimitError, and OpenHands also retries 409. The Model Server retries transient failures.
                "num_retries": 0,
                # Every model call, including the condenser's, names the invocation that owns it.
                "extra_headers": {"x-session-id": invocation_id},
            },
            "condenser": self.config.condenser,
            "max_iterations": self.config.max_iterations,
            "state_dir": state.path("state"),
            "events_path": state.path("events.jsonl"),
            "result_path": state.path("result.json"),
        }

    async def _activate(
        self,
        state: OpenHandsSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        prompt: str,
        system: str,
    ) -> NeMoGymResponse:
        timeout = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
        failure: BaseException | None = None
        artifacts: OpenHandsArtifacts | None = None
        try:
            # Close can fence the staging phase just as it fences execution.
            async with self.sem:
                artifacts = await state.execute(
                    self._runner_input(state, prompt=prompt, system=system),
                    timeout=timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            failure = error
            artifacts = state.session.artifacts
        cleanup = state.session.cleanup
        timed_out = bool(cleanup and cleanup["timed_out"])
        if isinstance(failure, TimeoutError) and timed_out and artifacts is not None:
            # The provider wait and the confirmed harness deadline are separate observed facts.
            state.provider_wait_failure = NgFailureEvent(
                boundary="sandbox",
                failure_kind="transport_timeout",
                failure_reason="Sandbox provider wait expired after OpenHands confirmed its deadline",
                operation="exec_wait",
                stage="agent",
                foreign_error=sdk_error_details(failure, message=str(failure)),
            )
            state.provider_wait_failure.__cause__ = failure
            failure = None
        if state.provider_wait_failure is not None:
            observe_failure(state.provider_wait_failure)
        events, malformed = parse_events(artifacts.events) if artifacts is not None else ([], 0)
        result = None
        if artifacts is not None and artifacts.result is not None:
            try:
                parsed = json.loads(artifacts.result)
                result = parsed if isinstance(parsed, dict) else None
            except ValueError:
                result = None
        outcome = run_outcome(events, result, timed_out=timed_out, rollout_stop=state.session.expired)
        if failure is not None:
            outcome = RunOutcome(
                "incomplete" if isinstance(failure, asyncio.CancelledError) else "failed",
                "cancelled" if isinstance(failure, asyncio.CancelledError) else "sandbox_error",
                f"{type(failure).__name__}: {failure}",
            )
        items = output_items(events)
        conversation = [NeMoGymEasyInputMessage(role="system", content=system)] if system else []
        conversation.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        state.observations = self._observations(
            state, events, result, outcome, conversation=[*conversation, *items], malformed=malformed
        )
        if artifacts is None:
            state.observations.gaps.append(ObservationGap(code="agent_artifact_unavailable"))
        if failure is not None:
            # Termination, transport and capture failures are not results; the episode can be retried.
            raise failure
        if outcome.stop_reason == "runner_error":
            raise RuntimeError(f"OpenHands runner failed before writing a result: {artifacts.stderr[-4000:]}")
        if outcome.stop_reason == "error" and artifacts.stderr:
            # The response names only the error classes; the runner wrote the traceback to its stderr.
            LOG.warning("%s; the runner's stderr ends with:\n%s", outcome.message, artifacts.stderr[-4000:])
        runtime = state.runtime_info
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=items,
            status=outcome.status,
            error={"code": "server_error", "message": outcome.message[:4000]} if outcome.status == "failed" else None,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=response_usage(events, result),
            metadata={
                "harness_execution": "sandbox",
                **({"harness_hostname": runtime.hostname, "harness_pid": str(runtime.pid)} if runtime else {}),
                "openhands_version": OPENHANDS_VERSION,
                "stop_reason": outcome.stop_reason,
                AGENT_TIMEOUT_METADATA_KEY: str(timeout),
            },
        )
        response = self.report_rollout_stop(response, state.session)
        # Only a failed response gets an outcome. After the deadline that is one without a model reply, whose
        # cause is unknown, so no error the runner may have recorded classifies it.
        return self.report_outcome(response, {} if timed_out else result or {})

    def _observations(
        self,
        state: OpenHandsSandboxSession,
        events: list[tuple[float, dict[str, Any]]],
        result: Optional[dict[str, Any]],
        outcome: RunOutcome,
        *,
        conversation: list[Any],
        malformed: int,
    ) -> AgentObservationBundle:
        try:
            evidence = observation_records(
                events,
                result,
                invocation_id=state.request.episode_id.capture_key,
                model_ref=self.config.model_server,
                conversation=conversation,
                outcome=outcome,
                sandbox_id=state.session.sandbox.sandbox_id,
            )
            bundle = AgentObservationBundle(source="openhands", records=evidence.records, gaps=evidence.gaps)
        except Exception:
            # Observation parsing must not turn a gradable response into a failed episode.
            LOG.exception("Could not build OpenHands observations")
            bundle = AgentObservationBundle(source="openhands", gaps=[ObservationGap(code="observation_parse_failed")])
        if malformed:
            bundle.gaps.append(ObservationGap(code="agent_event_malformed", detail=f"{malformed} lines"))
        if state.runtime_info is None:
            bundle.gaps.append(ObservationGap(code="runtime_info_unavailable"))
        cleanup = state.session.cleanup
        if cleanup:
            access = state.request.sandbox_access
            bundle.records.append(
                SandboxObservation(
                    role="agent",
                    provider=access.connection.provider_config_ref if access else self.config.sandbox_provider,
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
            raise HTTPException(422, "OpenHands runs only in an agent session managed by an Environment Server")
        state = self._require_agent_session(session_id)
        assert isinstance(state, OpenHandsSandboxSession)
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "OpenHands activation does not match the seeded session and rollout")
        if state.session.closing:
            raise HTTPException(409, "OpenHands session is closing")
        prompt, system = self._validate_request(body)
        if state.task is None:
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(
                self._activate(state, state.activation_request, prompt=prompt, system=system)
            )
        elif body != state.activation_request:
            raise HTTPException(409, "OpenHands sessions support one activation; retry the same request")
        # A disconnected client must not cancel the shared activation; only close does.
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        """OpenHands has no compatibility /run path; an Environment Server owns the episode."""
        raise HTTPException(422, "Run OpenHands through an Environment Server such as single_agent_turn")


if __name__ == "__main__":
    OpenHandsAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenHandsAgent.run_webserver()  # noqa: F401
