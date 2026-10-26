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
"""GitHub Copilot CLI harness: runs ``copilot -p`` inside the task sandbox in agent sessions.

Copilot reaches the model through its bring-your-own-key provider support, pointed at the
rollout-prefixed Gym Model Server with ``COPILOT_OFFLINE=true``, so it never contacts GitHub.
"""

import asyncio
import json
import logging
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, ClassVar, Literal, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, JsonValue

from nemo_gym.agent_outcome import AgentOutcome
from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.agent_utils.task_timeout import AGENT_TIMEOUT_METADATA_KEY, resolve_agent_timeout
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.failure_kinds import MODEL_REQUEST_ERROR, MODEL_TRANSPORT_ERROR
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    SandboxObservation,
    ToolCallObservation,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import get_global_config_dict
from responses_api_agents.copilot_agent.sandbox import CopilotArtifacts, CopilotRuntime, CopilotSandboxSession


LOG = logging.getLogger(__name__)
COPILOT_VERSION = "1.0.103"
SOURCE = "copilot"
_EXACT_VERSION = re.compile(r"\d+\.\d+\.\d+")
# Copilot 1.0.103 reports an unreachable BYOK provider with this message and no status code.
_TRANSPORT_MESSAGES = ("Could not connect to", "fetch failed", "ECONNREFUSED", "ECONNRESET", "ETIMEDOUT", "timed out")


def _sandbox_prepare_command(directory: str) -> str:
    """POSIX sh: the supervisor and the installer need python3 before anything else runs."""
    return f"""set -eu
if ! command -v python3 >/dev/null 2>&1; then
    [ "$(id -u)" = 0 ] || {{ echo "Copilot in the task sandbox requires python3: preinstall it or use a root image." >&2; exit 1; }}
    if command -v apk >/dev/null 2>&1; then
        apk add --no-cache python3
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 ca-certificates
    else
        echo "Copilot in the task sandbox requires python3 (automatic installation requires apt-get or apk)." >&2
        exit 1
    fi
fi
mkdir -p {shlex.quote(directory)}
"""


def _timestamp(value: Any) -> Optional[float]:
    """Parse Copilot's ISO-8601 event timestamps into epoch seconds."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else json.dumps(value)


@dataclass
class CopilotTranscript:
    """What one Copilot run reported in its JSONL events."""

    output: list[NeMoGymResponseOutputItem] = field(default_factory=list)
    response_ids: list[str] = field(default_factory=list)
    tools: dict[str, ToolCallObservation] = field(default_factory=dict)
    error: Optional[dict[str, JsonValue]] = None
    result: Optional[dict[str, JsonValue]] = None
    session_id: Optional[str] = None
    gaps: list[ObservationGap] = field(default_factory=list)


def parse_copilot_events(events: list[tuple[float, dict[str, JsonValue]]], invocation_id: str) -> CopilotTranscript:
    """Convert Copilot 1.0.103 JSONL events into Responses output items and tool observations."""
    transcript = CopilotTranscript()
    message_start: Optional[int] = None
    starts: dict[str, tuple[Optional[float], Optional[str]]] = {}
    for _, event in events:
        kind = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if kind == "assistant.message":
            message_start = len(transcript.output)
            content = data.get("content")
            if isinstance(content, str) and content.strip():
                transcript.output.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(transcript.output)}",
                        content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            for request in data.get("toolRequests") or []:
                if not isinstance(request, dict) or not isinstance(request.get("toolCallId"), str):
                    continue
                arguments = request.get("arguments")
                transcript.output.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=json.dumps(arguments) if isinstance(arguments, (dict, list)) else _text(arguments),
                        call_id=request["toolCallId"],
                        name=str(request.get("name") or ""),
                        type="function_call",
                        id=request["toolCallId"],
                        status="completed",
                    )
                )
            response_id = data.get("apiCallId")
            if isinstance(response_id, str) and response_id:
                transcript.response_ids.append(response_id)
        elif kind == "assistant.reasoning":
            text = data.get("content")
            if isinstance(text, str) and text.strip():
                item = NeMoGymResponseReasoningItem(
                    id=f"reasoning-{len(transcript.output)}", summary=[{"type": "summary_text", "text": text}]
                )
                # Copilot reports the reasoning after the message it preceded.
                if message_start is None:
                    transcript.output.append(item)
                else:
                    transcript.output.insert(message_start, item)
                    message_start += 1
        elif kind == "tool.execution_start" and isinstance(data.get("toolCallId"), str):
            name = data.get("toolName")
            starts[data["toolCallId"]] = (_timestamp(event.get("timestamp")), name if isinstance(name, str) else None)
        elif kind == "tool.execution_complete" and isinstance(data.get("toolCallId"), str):
            call_id = data["toolCallId"]
            result = data.get("result") if isinstance(data.get("result"), dict) else {}
            error = data.get("error") if isinstance(data.get("error"), dict) else {}
            output = result.get("content")
            if not isinstance(output, str):
                output = _text(error.get("message") if error else result)
            transcript.output.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output", call_id=call_id, output=output, status="completed"
                )
            )
            started_at, name = starts.pop(call_id, (None, None))
            completed_at = _timestamp(event.get("timestamp"))
            valid = started_at is not None and completed_at is not None and completed_at >= started_at
            success = data.get("success")
            shell = data.get("shellExecution") if isinstance(data.get("shellExecution"), dict) else {}
            exit_code = shell.get("exitCode")
            # Copilot reports a shell command that exits nonzero as a successful tool call.
            if type(exit_code) is int and exit_code != 0:
                success = False
            transcript.tools[call_id] = ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=call_id,
                tool_name=name,
                started_at=started_at if valid else None,
                completed_at=completed_at,
                duration_ms=(completed_at - started_at) * 1000 if valid else None,
                timing_source="harness",
                status="completed" if success is True else "failed" if success is False else "unknown",
                error_type=f"exit_code_{exit_code}" if type(exit_code) is int and exit_code != 0 else None,
            )
            if not valid:
                transcript.gaps.append(ObservationGap(code="tool_timing_unavailable", detail=call_id))
            if not isinstance(success, bool):
                transcript.gaps.append(ObservationGap(code="tool_outcome_unavailable", detail=call_id))
        elif kind == "session.error":
            transcript.error = data
        elif kind == "result":
            transcript.result = event
            session_id = event.get("sessionId")
            transcript.session_id = session_id if isinstance(session_id, str) else None
    for call_id, (started_at, name) in starts.items():
        transcript.tools[call_id] = ToolCallObservation(
            invocation_id=invocation_id,
            tool_call_id=call_id,
            tool_name=name,
            started_at=started_at,
            timing_source="harness",
            status="incomplete",
        )
        transcript.gaps.append(ObservationGap(code="tool_timing_unavailable", detail=call_id))
    return transcript


@dataclass
class CopilotUsage:
    """Token counts summed over successful model calls, from Copilot's OpenTelemetry chat spans."""

    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cached_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    response_ids: list[str]
    compaction_usage_unavailable: bool = False
    compaction_state_unavailable: bool = False


def parse_copilot_usage(otel: Optional[str], *, compaction_state: bool | None = False) -> Optional[CopilotUsage]:
    """Sum usage over successful chat spans; a missing count stays unknown instead of becoming zero.

    Copilot's own usage file reports zero when the provider omits usage, so it is not used for counts.
    Returns ``None`` when no chat spans were observed. Rejected calls contribute zero
    completed-response tokens; missing usage on a successful call remains unknown.
    Copilot 1.0.103 omits compaction calls from its spans. Persisted session events
    also detect late compaction with no following chat span. Observed compaction
    or an unavailable native event store makes aggregate usage unknown. Per-call
    counts remain in Gym model capture.
    """
    if otel is None:
        return None
    totals: dict[str, Optional[int]] = {"input": 0, "output": 0, "cached": 0, "reasoning": 0}
    response_ids: list[str] = []
    observed_calls = 0
    compacted = compaction_state is True
    for line in otel.split("\n"):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if not isinstance(record, dict) or record.get("type") != "span":
            continue
        attributes = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}
        if attributes.get("gen_ai.operation.name") != "chat":
            continue
        observed_calls += 1
        compacted |= attributes.get("gen_ai.conversation.compacted") is True
        status = record.get("status") if isinstance(record.get("status"), dict) else {}
        if status.get("code") == 2:
            continue  # A failed attempt returned no tokens to count.
        response_id = attributes.get("gen_ai.response.id")
        if isinstance(response_id, str) and response_id:
            response_ids.append(response_id)
        for key, attribute in (
            ("input", "gen_ai.usage.input_tokens"),
            ("output", "gen_ai.usage.output_tokens"),
            ("cached", "gen_ai.usage.cache_read.input_tokens"),
            ("reasoning", "gen_ai.usage.reasoning.output_tokens"),
        ):
            value = attributes.get(attribute)
            if totals[key] is not None and type(value) is int and value >= 0:
                totals[key] += value
            else:
                totals[key] = None
    if not observed_calls:
        return None
    return CopilotUsage(
        input_tokens=None if compacted or compaction_state is None else totals["input"],
        output_tokens=None if compacted or compaction_state is None else totals["output"],
        cached_tokens=None if compacted or compaction_state is None else totals["cached"],
        reasoning_tokens=None if compacted or compaction_state is None else totals["reasoning"],
        response_ids=response_ids,
        compaction_usage_unavailable=compacted,
        compaction_state_unavailable=compaction_state is None,
    )


class CopilotAgentConfig(BaseResponsesAPIAgentConfig):
    """Pinned Copilot runtime and sandbox/model settings, independent of the benchmark."""

    # Agent sessions get the benchmark from the Environment Server; Copilot has no compatibility /run.
    resources_server: Optional[ResourcesServerRef] = None
    model_server: Optional[ModelServerRef] = None
    model: str
    copilot_version: str = COPILOT_VERSION
    concurrency: int = Field(default=8, gt=0)
    timeout: int = Field(default=900, gt=0)
    context_window: Optional[int] = Field(
        default=None, gt=0, description="Prompt-token limit for the BYOK model; Copilot's default when unset."
    )
    max_output_tokens: Optional[int] = Field(
        default=None, gt=0, description="Per-model-call output limit, sent as max_tokens; unset sends none."
    )
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]] = None
    npm_registry: str = "https://registry.npmjs.org"
    sandbox_provider: Optional[str] = None
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)
    sandbox_install_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)


class CopilotAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class CopilotAgent(SimpleResponsesAPIAgent):
    """Runs ``copilot -p`` (GitHub Copilot CLI) in agent sessions only."""

    config: CopilotAgentConfig
    # The CLI runs inside the task sandbox and calls the model server from there.
    calls_model_from_sandbox: ClassVar[bool] = True
    sem: asyncio.Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = asyncio.Semaphore(self.config.concurrency)

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Read Copilot 1.0.103's ``session.error`` event; tested on that pinned version."""
        status = native_result.get("statusCode")
        if type(status) is int and 400 <= status <= 599:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=status)
        message = str(native_result.get("message") or "")
        if any(marker in message for marker in _TRANSPORT_MESSAGES):
            return AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
        return AgentOutcome(failure_kind="unknown")

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> CopilotSandboxSession:
        owns_sandbox = body.sandbox_access is None
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise AgentSeedRejectedError("Copilot requires sandbox_access or a configured sandbox_provider")
            sandbox_config = dict(self.config.sandbox_config)
            sandbox_config.setdefault("ttl_s", self.config.sandbox_install_timeout_seconds + self.config.timeout + 600)
            sandbox_config["metadata"] = {
                **resolve_provider_metadata(self.config.sandbox_provider, get_global_config_dict()),
                **sandbox_config.get("metadata", {}),
                "harness": "copilot",
                "agent_session_id": body.agent_session_id[:63],
            }
            spec = SandboxSpec(**{"workdir": "/app", **sandbox_config})
            workdir = spec.workdir
            provider_ref = self.config.sandbox_provider
        else:
            if not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("Copilot requires direct SandboxAccess")
            workdir = body.sandbox_access.workdir
            provider_ref = body.sandbox_access.connection.provider_config_ref
        if not isinstance(workdir, str) or not workdir.startswith("/"):
            raise AgentSeedRejectedError("Copilot sandbox workdir must be absolute")
        required = sorted(access.name for access in self.effective_tool_accesses(body) if access.required)
        if required:
            raise AgentSeedRejectedError(
                "Copilot agent sessions do not use Resources Server tool grants; required grants: "
                + ", ".join(required)
            )
        if self.config.model_server is None:
            raise AgentSeedRejectedError("Copilot requires a sandbox-reachable Gym model_server")
        if not _EXACT_VERSION.fullmatch(self.config.copilot_version):
            raise AgentSeedRejectedError(f"Copilot requires an exact copilot_version, for example {COPILOT_VERSION}")

        provider = create_provider(resolve_provider_config(provider_ref, get_global_config_dict()))
        if owns_sandbox:
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(body.sandbox_access.connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        directory = f"/tmp/nemo-gym-copilot-sessions/{uuid4().hex}"
        state = CopilotSandboxSession(
            request=body,
            sandbox_id=sandbox.sandbox_id,
            session=SandboxSession(
                sandbox=sandbox, session_dir=directory, workdir=workdir, owns_sandbox=owns_sandbox, harness="Copilot"
            ),
        )
        try:
            if owns_sandbox:
                await sandbox.start(spec)
                state.sandbox_id = sandbox.sandbox_id
                # Providers need not create SandboxSpec.workdir. Never prepare a borrowed task here.
                workspace = await sandbox.exec(f"mkdir -p -- {shlex.quote(workdir)}", cwd="/", timeout_s=30)
                if workspace.return_code != 0 or workspace.error_type:
                    raise RuntimeError(
                        f"Cannot create Copilot sandbox workdir {workdir}: {workspace.stderr or workspace.stdout}"
                    )
            prepared = await sandbox.exec(
                _sandbox_prepare_command(directory), timeout_s=self.config.sandbox_install_timeout_seconds
            )
            if prepared.return_code != 0 or prepared.error_type:
                details = "\n".join(part for part in (prepared.stderr, prepared.stdout) if part)
                raise RuntimeError(
                    f"Cannot prepare Copilot session (exit {prepared.return_code}, "
                    f"error_type={prepared.error_type}): {details[-16000:]}"
                )
            # Install only the agent runtime; Resources has already prepared the task.
            installer = "install_copilot_runtime.py"
            await sandbox.upload(Path(__file__).with_name(installer), f"{directory}/{installer}")
            runtime_dir = f"/tmp/nemo-gym-copilot-{self.config.copilot_version}"
            installed = await sandbox.exec(
                " ".join(
                    shlex.quote(part)
                    for part in (
                        "python3",
                        "-I",
                        f"{directory}/{installer}",
                        runtime_dir,
                        self.config.copilot_version,
                        self.config.npm_registry,
                    )
                ),
                cwd=workdir,
                timeout_s=self.config.sandbox_install_timeout_seconds,
            )
            if installed.return_code != 0 or installed.error_type:
                details = "\n".join(part for part in (installed.stderr, installed.stdout) if part)
                raise RuntimeError(
                    f"Copilot sandbox installation failed (exit {installed.return_code}, "
                    f"error_type={installed.error_type}): {details[-16000:]}"
                )
            state.runtime = CopilotRuntime.model_validate_json(installed.stdout.strip().splitlines()[-1])
            await sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), f"{directory}/sandbox_runner.py")
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                LOG.exception("Copilot seed cleanup failed; retaining session %s", body.agent_session_id)
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Stop Copilot before verification, retaining a borrowed sandbox."""
        assert isinstance(state, CopilotSandboxSession)
        await state.close(self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=state.observations
            or AgentObservationBundle(source=SOURCE, gaps=[ObservationGap(code="observation_capture_failed")]),
        )

    def _validate_request(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> tuple[NeMoGymResponseCreateParamsNonStreaming, str]:
        """Accept one text user prompt; reject every control Copilot cannot apply."""
        if body.model is not None and body.model != self.config.model:
            raise HTTPException(422, "Copilot model must match the configured model")
        if body.max_output_tokens is not None:
            raise HTTPException(
                422,
                "Copilot does not support the total max_output_tokens budget; "
                "configure the agent's max_output_tokens for a per-model-call limit instead",
            )
        if body.instructions:
            raise HTTPException(422, "Copilot has no system-prompt option; put instructions in the user prompt")
        for name, model_field in type(body).model_fields.items():
            if name in ("input", "model", "instructions"):
                continue
            value = getattr(body, name)
            if name in ("stream", "background") and value is False:
                continue
            if name == "metadata" and value and set(value) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            if value != model_field.get_default(call_default_factory=True):
                raise HTTPException(422, f"Copilot does not support request field {name}")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        if [getattr(item, "role", None) for item in items] != ["user"]:
            raise HTTPException(422, "Copilot accepts one text user prompt and no system message")
        content = items[0].content
        if not isinstance(content, str):
            parts = [part if isinstance(part, dict) else part.model_dump() for part in content]
            if any(part.get("type") != "input_text" for part in parts):
                raise HTTPException(422, "Copilot only supports text input")
            content = "".join(part.get("text", "") for part in parts)
        return body.model_copy(update={"input": items}).model_copy(deep=True), content

    def _command(self, state: CopilotSandboxSession, prompt: str) -> list[str]:
        assert state.runtime is not None
        directory = state.session.session_dir
        command = [
            state.runtime.binary,
            "-p",
            prompt,
            "--model",
            self.config.model,
            # The task sandbox is the isolation boundary, as for Codex's danger-full-access.
            "--allow-all-tools",
            "--allow-all-paths",
            "--no-ask-user",
            "--stream",
            "off",
            "--output-format",
            "json",
            "--usage-output-file",
            f"{directory}/usage.json",
            "--log-dir",
            f"{directory}/logs",
            "--disable-builtin-mcps",
            "--disable-client-telemetry",
            "--no-auto-update",
        ]
        if self.config.reasoning_effort:
            command += ["--reasoning-effort", self.config.reasoning_effort]
        return command

    def _env(self, state: CopilotSandboxSession) -> dict[str, str]:
        assert state.runtime is not None
        directory = state.session.session_dir
        env = {
            # Copilot's state stays in the session; tools keep the task's HOME and XDG directories.
            "COPILOT_HOME": f"{directory}/copilot-home",
            "COPILOT_CLI_DIST_DIR": state.runtime.dist_dir,
            "COPILOT_OFFLINE": "true",
            "COPILOT_AUTO_UPDATE": "false",
            "COPILOT_PROVIDER_TYPE": "openai",
            "COPILOT_PROVIDER_WIRE_API": "completions",
            "COPILOT_PROVIDER_BASE_URL": self.resolve_model_base_url(
                self.config.model_server.name, state.request.episode_id.capture_key
            ),
            "COPILOT_PROVIDER_API_KEY": "EMPTY",  # pragma: allowlist secret
            # Model-call capture assigns every attempt, including failed ones, to this invocation.
            "COPILOT_PROVIDER_HEADERS": f"x-session-id: {state.request.episode_id.capture_key}",
            "COPILOT_MODEL": self.config.model,
            "COPILOT_OTEL_EXPORTER_TYPE": "file",
            "COPILOT_OTEL_FILE_EXPORTER_PATH": f"{directory}/otel.jsonl",
            "NO_COLOR": "1",
        }
        if self.config.context_window is not None:
            env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] = str(self.config.context_window)
        if self.config.max_output_tokens is not None:
            env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] = str(self.config.max_output_tokens)
        return env

    def _observations(
        self,
        state: CopilotSandboxSession,
        transcript: CopilotTranscript,
        usage: Optional[CopilotUsage],
        conversation: list[Any],
        status: Literal["completed", "failed", "incomplete", "unknown"],
        *,
        failure: BaseException | None = None,
    ) -> AgentObservationBundle:
        invocation_id = state.request.episode_id.capture_key
        response_ids = usage.response_ids if usage is not None and usage.response_ids else transcript.response_ids
        gaps = [
            ObservationGap(code="subagent_hierarchy_unavailable", invocation_id=invocation_id),
            ObservationGap(code="compaction_observations_unavailable", invocation_id=invocation_id),
            *transcript.gaps,
        ]
        if usage is None:
            gaps.append(ObservationGap(code="model_usage_unavailable", detail="Copilot wrote no OpenTelemetry spans"))
        if usage is not None and usage.compaction_usage_unavailable:
            gaps.append(
                ObservationGap(
                    code="compaction_model_usage_unavailable",
                    detail="Copilot omits compaction model calls from native usage spans; aggregate usage is unknown",
                )
            )
        if usage is not None and usage.compaction_state_unavailable:
            gaps.append(ObservationGap(code="native_compaction_state_unavailable"))
        cleanup = state.session.cleanup
        access = state.request.sandbox_access
        if cleanup is None:
            gaps.append(ObservationGap(code="sandbox_cleanup_unavailable"))
        if state.sandbox_id is None:
            gaps.append(ObservationGap(code="sandbox_identity_unavailable"))
        for tool in transcript.tools.values():
            tool.sandbox_id = state.sandbox_id
        sandbox = SandboxObservation(
            role="agent",
            provider=access.connection.provider_config_ref if access else self.config.sandbox_provider,
            sandbox_id=state.sandbox_id,
            outcome=(
                "timeout"
                if cleanup and cleanup["timed_out"]
                else "cancelled"
                if isinstance(failure, asyncio.CancelledError)
                else "sandbox_error"
                if failure is not None
                else "failed"
                if status == "failed"
                else "completed"
                if status == "completed"
                else "unknown"
            ),
            exit_code=cleanup["return_code"] if cleanup else None,
            error_type=type(failure).__name__
            if failure is not None
            else "copilot_error"
            if status == "failed"
            else None,
        )
        return AgentObservationBundle(
            source=SOURCE,
            records=[
                AgentInvocation(
                    invocation_id=invocation_id,
                    harness_session_id=transcript.session_id,
                    status=status,
                    model_calls=[
                        ModelCallRef(model_ref=self.config.model_server, response_id=response_id)
                        for response_id in response_ids
                    ],
                    conversation=conversation,
                ),
                *transcript.tools.values(),
                sandbox,
            ],
            gaps=gaps,
        )

    async def _sandbox_response(
        self, state: CopilotSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming, *, prompt: str
    ) -> NeMoGymResponse:
        invocation_id = state.request.episode_id.capture_key
        timeout = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
        payload = {
            "directory": state.session.session_dir,
            "command": self._command(state, prompt),
            "cwd": state.session.workdir,
            "env": self._env(state),
        }
        conversation = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        try:
            async with self.sem:
                artifacts = await state.execute(
                    payload, timeout=timeout, close_timeout=self.config.session_close_timeout_seconds
                )
        except BaseException as failure:
            # Common finalization captures before close removes the session directory.
            partial = state.session.artifacts
            transcript = parse_copilot_events(_events(partial.events if partial else ""), invocation_id)
            try:
                state.observations = self._observations(
                    state,
                    transcript,
                    parse_copilot_usage(
                        partial.otel if partial else None,
                        compaction_state=partial.compaction_state if partial else None,
                    ),
                    [*conversation, *transcript.output],
                    "incomplete",
                    failure=failure,
                )
            except Exception:
                LOG.exception("failed to build interrupted Copilot observations")
                state.observations = AgentObservationBundle(
                    source=SOURCE, gaps=[ObservationGap(code="observation_parse_failed")]
                )
            state.observations.gaps.append(
                ObservationGap(code="agent_activation_interrupted", detail=type(failure).__name__)
            )
            if state.runtime_info is None:
                state.observations.gaps.append(ObservationGap(code="runtime_info_unavailable"))
            raise
        return self._response(state, body, artifacts, conversation, timeout=timeout)

    def _response(
        self,
        state: CopilotSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        artifacts: CopilotArtifacts,
        conversation: list[Any],
        *,
        timeout: float,
    ) -> NeMoGymResponse:
        invocation_id = state.request.episode_id.capture_key
        transcript = parse_copilot_events(_events(artifacts.events), invocation_id)
        usage = parse_copilot_usage(artifacts.otel, compaction_state=artifacts.compaction_state)
        cleanup = state.session.cleanup
        assert cleanup is not None
        reported_error = transcript.error is not None and not cleanup["timed_out"] and cleanup["error"] is None
        execution_error = cleanup["error"]
        if not reported_error and not cleanup["timed_out"]:
            if cleanup["return_code"] not in (None, 0):
                execution_error = execution_error or f"Copilot exited with code {cleanup['return_code']}"
            elif transcript.result is None:
                execution_error = execution_error or "Copilot ended without a result event"
        status = (
            "failed" if reported_error or execution_error else "incomplete" if cleanup["timed_out"] else "completed"
        )
        if not any(getattr(item, "type", None) == "message" for item in transcript.output) and status != "failed":
            transcript.output.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-{len(transcript.output)}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
        known = usage is not None and usage.input_tokens is not None and usage.output_tokens is not None
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=transcript.output,
            status=status,
            error=(
                {"code": "server_error", "message": str((transcript.error or {}).get("message") or execution_error)}
                if status == "failed"
                else None
            ),
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=(
                NeMoGymResponseUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.input_tokens + usage.output_tokens,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=usage.cached_tokens),
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=usage.reasoning_tokens),
                )
                if known
                else None
            ),
            metadata={
                "harness_execution": "sandbox",
                **(
                    {"harness_hostname": state.runtime_info.hostname, "harness_pid": str(state.runtime_info.pid)}
                    if state.runtime_info
                    else {}
                ),
                "copilot_version": self.config.copilot_version,
                AGENT_TIMEOUT_METADATA_KEY: str(timeout),
            },
        )
        try:
            state.observations = self._observations(
                state, transcript, usage, [*conversation, *transcript.output], status
            )
        except Exception:
            # Observation parsing must not turn a gradable result into a failed episode.
            LOG.exception("failed to build Copilot observations")
            state.observations = AgentObservationBundle(
                source=SOURCE, gaps=[ObservationGap(code="observation_parse_failed")]
            )
        gaps = state.observations.gaps
        if usage is not None and not known:
            gaps.append(
                ObservationGap(code="partial_model_usage_unavailable", detail="a model call reported no usage")
            )
        for count, code in (
            (usage.cached_tokens if usage else None, "cached_token_usage_unavailable"),
            (usage.reasoning_tokens if usage else None, "reasoning_token_usage_unavailable"),
        ):
            if count is None:
                gaps.append(ObservationGap(code=code))
        if state.runtime_info is None:
            gaps.append(ObservationGap(code="runtime_info_unavailable"))
        if cleanup["return_code"] is None:
            gaps.append(ObservationGap(code="worker_exit_code_unavailable"))
        if not artifacts.events.strip():
            gaps.append(ObservationGap(code="agent_transcript_unavailable"))
        if execution_error:
            raise RuntimeError(f"Copilot agent failed: {execution_error}")
        if reported_error:
            return self.report_outcome(response, transcript.error)
        return response

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(
                422, "Copilot runs only in agent sessions; submit the episode to an Environment Server /run"
            )
        body, prompt = self._validate_request(body)
        state = self._require_agent_session(session_id)
        assert isinstance(state, CopilotSandboxSession)
        if state.request.episode_id.capture_key != request.path_params.get("rollout_id"):
            raise HTTPException(409, "Copilot activation does not match the seeded session and rollout route")
        if state.session.closing:
            raise HTTPException(409, "Copilot session is closing")
        if state.task is None:
            # Bind the request before the first await so retries join one invocation.
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._sandbox_response(state, body, prompt=prompt))
        elif body != state.activation_request:
            raise HTTPException(409, "Copilot sandbox sessions support one activation; retry the same request")
        # Session close owns cancellation. Losing an HTTP waiter must not stop the harness.
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def run(self, request: Request, body: CopilotAgentRunRequest) -> BaseVerifyResponse:
        raise HTTPException(422, "Copilot has no compatibility /run; submit the episode to an Environment Server /run")


def _events(raw: str) -> list[tuple[float, dict[str, Any]]]:
    """Read the runner's ``[observed_at, event]`` lines, skipping malformed ones."""
    events = []
    # JSONL: split on "\n" only; str.splitlines() would also split inside strings on U+2028/U+0085.
    for line in raw.split("\n"):
        if not line.strip():
            continue
        try:
            observed_at, event = json.loads(line)
            if isinstance(event, dict):
                events.append((float(observed_at), event))
        except (ValueError, TypeError):
            LOG.warning("Skipping malformed Copilot event")
    return events


if __name__ == "__main__":
    CopilotAgent.run_webserver()
