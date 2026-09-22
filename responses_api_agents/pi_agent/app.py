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
import copy
import json
import logging
import os
import re
import shlex
import shutil
from asyncio import Semaphore
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
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
)
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import get_global_config_dict, get_response_json, raise_for_status
from responses_api_agents.pi_agent.sandbox import PiSandboxSession
from responses_api_agents.pi_agent.setup_pi import ensure_pi


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"
_SANDBOX_SESSION_KEY = "nemo_gym_pi_sandbox_session"


def parse_pi_events(stdout: str | bytes) -> tuple[list[Any], dict[str, int]]:
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") != "message_end":
            continue
        message = event.get("message") or {}
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, list):
            continue

        if role == "assistant":
            usage = message.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            input_tokens += int(usage.get("input") or 0) + int(usage.get("cacheRead") or 0)
            output_tokens += int(usage.get("output") or 0)
            texts = [b["text"] for b in content if isinstance(b, dict) and (b.get("text") or "").strip()]
            if texts:
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text="\n".join(texts), annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                args = block.get("arguments")
                arguments = json.dumps(args) if isinstance(args, (dict, list)) else str(args or "")
                call_id = block.get("id") or f"call-{uuid4().hex[:8]}"
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=arguments,
                        call_id=call_id,
                        name=block.get("name", ""),
                        type="function_call",
                        id=call_id,
                        status="completed",
                    )
                )

        elif role == "toolResult":
            call_id = message.get("toolCallId", "")
            result_text = "".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=result_text,
                    status="completed",
                )
            )

    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


async def _read_pi_stdout(stream: asyncio.StreamReader) -> tuple[str, list[tuple[float, dict[str, Any]]]]:
    lines: list[str] = []
    events: list[tuple[float, dict[str, Any]]] = []

    def consume(line: bytes) -> None:
        observed_at = time()
        text = line.decode(errors="replace")
        lines.append(text)
        try:
            event = json.loads(text)
        except (json.JSONDecodeError, RecursionError):
            return
        if isinstance(event, dict):
            events.append((observed_at, event))

    pending = bytearray()
    while chunk := await stream.read(64 * 1024):
        pending.extend(chunk)
        while (newline := pending.find(b"\n")) >= 0:
            consume(bytes(pending[: newline + 1]))
            del pending[: newline + 1]
    if pending:
        consume(bytes(pending))
    return "".join(lines), events


def _build_pi_observations(
    events: list[tuple[float, dict[str, Any]]],
    invocation_id: str,
    model_ref: Optional[ModelServerRef],
    conversation: list[Any],
    *,
    transcript_available: bool = True,
) -> AgentObservationBundle:
    def gap(code: str, detail: Optional[str] = None) -> ObservationGap:
        return ObservationGap(code=code, invocation_id=invocation_id, detail=detail)

    gaps = [gap("subagent_hierarchy_unavailable")]
    if not transcript_available:
        gaps.append(gap("agent_transcript_unavailable"))
    model_calls: list[ModelCallRef] = []
    model_call_join_missing = False
    starts: dict[str, tuple[float, Optional[str]]] = {}
    tools: dict[str, ToolCallObservation] = {}
    compaction_start: Optional[tuple[float, Optional[str], Optional[ModelCallRef]]] = None
    compactions: list[ContextCompactionObservation] = []
    compactions_waiting_for_call: list[ContextCompactionObservation] = []
    last_model_call: Optional[ModelCallRef] = None
    invocation_status = "unknown"

    for observed_at, event in events:
        event_type = event.get("type")
        message = event.get("message")
        call_id = event.get("toolCallId")
        tool_name = event.get("toolName")
        tool_name = tool_name if isinstance(tool_name, str) and tool_name else None

        if event_type == "message_end" and isinstance(message, dict) and message.get("role") == "assistant":
            response_id = message.get("responseId")
            if model_ref is not None and isinstance(response_id, str) and response_id:
                last_model_call = ModelCallRef(model_ref=model_ref, response_id=response_id)
                model_calls.append(last_model_call)
            else:
                last_model_call = None
                model_call_join_missing = True
            for compaction in compactions_waiting_for_call:
                compaction.after_model_call = last_model_call
                if last_model_call is None:
                    gaps.append(gap("compaction_after_model_call_unavailable"))
            compactions_waiting_for_call.clear()
        elif event_type == "agent_end":
            terminal_messages = event.get("messages")
            if isinstance(terminal_messages, list):
                stop_reason = next(
                    (
                        item.get("stopReason")
                        for item in reversed(terminal_messages)
                        if isinstance(item, dict) and item.get("role") == "assistant"
                    ),
                    None,
                )
                invocation_status = {
                    "stop": "completed",
                    "error": "failed",
                    "aborted": "incomplete",
                    "length": "incomplete",
                }.get(stop_reason, "unknown")
        elif event_type == "tool_execution_start" and isinstance(call_id, str):
            starts[call_id] = (observed_at, tool_name)
        elif event_type == "tool_execution_end" and isinstance(call_id, str):
            completed_at = observed_at
            started_at, started_name = starts.pop(call_id, (None, None))
            valid_interval = started_at is not None and completed_at >= started_at
            duration_ms = (completed_at - started_at) * 1000 if started_at is not None and valid_interval else None
            tools[call_id] = ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=call_id,
                tool_name=tool_name or started_name,
                started_at=started_at if valid_interval else None,
                completed_at=completed_at,
                duration_ms=duration_ms,
                timing_source="harness",
                status=(
                    "failed"
                    if event.get("isError") is True
                    else "completed"
                    if event.get("isError") is False
                    else "unknown"
                ),
            )
            if not valid_interval:
                gaps.append(gap("tool_timing_unavailable", call_id))
            if not isinstance(event.get("isError"), bool):
                gaps.append(gap("tool_outcome_unavailable", call_id))
        elif event_type == "compaction_start":
            reason = event.get("reason")
            compaction_start = (observed_at, reason if isinstance(reason, str) else None, last_model_call)
        elif event_type == "compaction_end":
            reason = event.get("reason")
            started_at, started_reason, before_model_call = compaction_start or (observed_at, None, None)
            raw_result = event.get("result")
            result: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
            before = result.get("tokensBefore")
            after = result.get("estimatedTokensAfter")
            summary = result.get("summary")
            first_kept_item_id = result.get("firstKeptEntryId")
            outcome = (
                "aborted"
                if event.get("aborted") is True
                else "completed"
                if result
                else "failed"
                if isinstance(event.get("errorMessage"), str)
                else "unknown"
            )
            compaction = ContextCompactionObservation(
                invocation_id=invocation_id,
                observed_at=started_at,
                trigger=reason if isinstance(reason, str) else started_reason,
                tokens_before=before if type(before) is int and before >= 0 else None,
                tokens_after=after if type(after) is int and after >= 0 else None,
                outcome=outcome,
                summary=summary if isinstance(summary, str) else None,
                first_kept_item_id=first_kept_item_id if isinstance(first_kept_item_id, str) else None,
                before_model_call=before_model_call,
            )
            compactions.append(compaction)
            compactions_waiting_for_call.append(compaction)
            if compaction_start is None:
                gaps.append(gap("compaction_start_unavailable"))
            if not result:
                gaps.append(gap("compaction_result_unavailable"))
            else:
                if type(before) is not int or before < 0:
                    gaps.append(gap("compaction_tokens_before_unavailable"))
                if not isinstance(summary, str):
                    gaps.append(gap("compaction_summary_unavailable"))
                if not isinstance(first_kept_item_id, str):
                    gaps.append(gap("compaction_boundary_unavailable"))
                if type(after) is not int or after < 0:
                    gaps.append(gap("compaction_tokens_after_unavailable"))
            if outcome == "unknown":
                gaps.append(gap("compaction_outcome_unavailable"))
            compaction_start = None
    if not model_calls or model_call_join_missing:
        gaps.append(gap("model_call_ownership_unavailable"))
    if invocation_status == "unknown":
        gaps.append(gap("invocation_outcome_unavailable"))

    for call_id, (started_at, tool_name) in starts.items():
        tools[call_id] = ToolCallObservation(
            invocation_id=invocation_id,
            tool_call_id=call_id,
            tool_name=tool_name,
            started_at=started_at,
            timing_source="harness",
            status="incomplete",
        )
        gaps.append(gap("tool_timing_unavailable", call_id))
    if compaction_start is not None:
        started_at, reason, before_model_call = compaction_start
        compactions.append(
            ContextCompactionObservation(
                invocation_id=invocation_id,
                observed_at=started_at,
                trigger=reason,
                before_model_call=before_model_call,
            )
        )
        gaps.append(gap("compaction_result_unavailable"))
        gaps.append(gap("compaction_outcome_unavailable"))
    for _ in compactions_waiting_for_call:
        gaps.append(gap("compaction_after_model_call_unavailable"))

    def field(item: Any, name: str) -> Any:
        return item.get(name) if isinstance(item, dict) else getattr(item, name, None)

    result_ids = {
        field(item, "call_id")
        for item in conversation
        if field(item, "type") == "function_call_output" and isinstance(field(item, "call_id"), str)
    }
    for item in conversation:
        if field(item, "type") != "function_call":
            continue
        call_id = field(item, "call_id")
        if not isinstance(call_id, str) or not call_id or call_id in tools:
            continue
        tools[call_id] = ToolCallObservation(
            invocation_id=invocation_id,
            tool_call_id=call_id,
            tool_name=field(item, "name"),
            status="unknown" if call_id in result_ids else "incomplete",
        )
        gaps.append(gap("tool_timing_unavailable", call_id))

    return AgentObservationBundle(
        source="pi",
        records=[
            AgentInvocation(
                invocation_id=invocation_id,
                status=invocation_status,
                model_calls=model_calls,
                conversation=conversation,
            ),
            *tools.values(),
            *compactions,
        ],
        gaps=gaps,
    )


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    """Return (user_message, system_message) from a responses body input list."""
    items = list(body_input)
    system_message: Optional[str] = None

    if items:
        first = items[0]
        role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        if role == "system":
            content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            system_message = content or ""
            items = items[1:]

    user_message = ""
    for item in reversed(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role == "user":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            user_message = content or ""
            break

    return user_message, system_message


class PiAgentConfig(BaseResponsesAPIAgentConfig):
    # Only the direct /run compatibility path calls Resources. Native sessions
    # receive SandboxAccess from EnvironmentServer instead.
    resources_server: Optional[ResourcesServerRef] = None
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "pi"
    model: str = "nvinf/nvidia/qwen/qwen3-next-80b-a3b-instruct"
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/pi_agent/workspaces"
    thinking: Optional[str] = None
    system_prompt: Optional[str] = None
    timeout: int = 900
    extra_args: list[str] = []
    models_config: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 262144
    max_output_tokens: int = 131072
    pi_version: Optional[str] = None
    sandbox_install_timeout_seconds: float = Field(default=600, gt=0)
    session_close_timeout_seconds: float = Field(default=60, gt=0)

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class PiAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class PiAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )


class PiAgent(SimpleResponsesAPIAgent):
    """Runs the pi CLI (pi --print --mode json --no-session)"""

    config: PiAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _sandbox_sessions: dict[str, PiSandboxSession] = PrivateAttr(default_factory=dict)
    _closed_sandbox_sessions: OrderedDict[str, tuple[EpisodeId, AgentCloseSessionResponse]] = PrivateAttr(
        default_factory=OrderedDict
    )
    _local_setup_task: asyncio.Task[None] | None = PrivateAttr(default=None)

    async def seed_agent_session(self, request: Request, body: AgentSeedSessionRequest) -> AgentSeedSessionResponse:
        """Borrow the Resources-owned task sandbox and install a pinned Pi runtime inside it."""
        if request.session.get(_SANDBOX_SESSION_KEY) in self._sandbox_sessions:
            raise HTTPException(409, "Pi session already exists")
        session_id = f"pi-{uuid4().hex}"
        state = await self._initialize_agent_session_state(session_id, body)
        self._sandbox_sessions[session_id] = state
        request.session[_SANDBOX_SESSION_KEY] = session_id
        return AgentSeedSessionResponse(agent_session_id=session_id)

    async def _initialize_agent_session_state(
        self, agent_session_id: str, body: AgentSeedSessionRequest
    ) -> PiSandboxSession:
        # Match Hermes: session initialization owns the sandbox runtime setup,
        # with the same AgentSeedSessionRequest/SandboxAccess wire contracts.
        if self.config.num_workers not in (None, 1):
            raise HTTPException(422, "Native Pi sessions require num_workers=1")
        if body.sandbox_access is None or not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
            raise HTTPException(422, "Native Pi requires direct, Resources-owned SandboxAccess")
        if not body.sandbox_access.workdir.startswith("/"):
            raise HTTPException(422, "Pi sandbox workdir must be absolute")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise HTTPException(422, "Native Pi supports its own sandbox tools, not required HTTP/MCP tools")
        if self.config.model_server is None:
            raise HTTPException(422, "Native Pi requires a sandbox-reachable Gym model_server")
        if not self.config.pi_version or not re.fullmatch(r"\d+\.\d+\.\d+", self.config.pi_version):
            raise HTTPException(422, "Native Pi requires an exact pi_version, for example 0.80.2")
        if self.config.command != "pi" or self.config.extra_args or self.config.env:
            raise HTTPException(422, "Native Pi does not support command, extra_args, or env overrides")

        connection = body.sandbox_access.connection
        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        directory = f"/tmp/nemo-gym-pi-sessions/{agent_session_id}"
        runtime = f"/tmp/nemo-gym-pi-node-22.19.0-{self.config.pi_version}"
        try:
            prepared = await sandbox.exec(f"mkdir -p {shlex.quote(directory + '/home/.pi/agent')}", timeout_s=30)
            if prepared.return_code != 0:
                raise RuntimeError(prepared.stderr or "Cannot create Pi sandbox session directory")
            # Install only the agent runtime in the existing task sandbox.
            # Resources has already prepared the task repository and its dependencies.
            installer = "install_pi_runtime.sh"
            await sandbox.upload(Path(__file__).with_name(installer), f"{directory}/{installer}")
            installed = await sandbox.exec(
                f"bash {shlex.quote(directory + '/' + installer)} {shlex.quote(runtime)} "
                f"{shlex.quote(self.config.pi_version)}",
                cwd=body.sandbox_access.workdir,
                timeout_s=self.config.sandbox_install_timeout_seconds,
            )
            if installed.return_code != 0:
                raise RuntimeError(installed.stderr or installed.stdout or "Pi sandbox installation failed")
            await sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), f"{directory}/sandbox_runner.py")
        except BaseException:
            await sandbox.disconnect()
            raise
        return PiSandboxSession(body, sandbox, directory, runtime)

    async def close_agent_session(self, request: Request, body: AgentCloseSessionRequest) -> AgentCloseSessionResponse:
        """Confirm Pi teardown before allowing verification; never destroy the borrowed sandbox."""
        session_id = request.session.get(_SANDBOX_SESSION_KEY)
        closed = self._closed_sandbox_sessions.get(session_id)
        if closed is not None and body.agent_session_id == session_id and body.episode_id == closed[0]:
            return closed[1]
        state = self._sandbox_sessions.get(session_id)
        if state is None or body.agent_session_id != session_id or body.episode_id != state.seed.episode_id:
            raise HTTPException(409, "Pi close does not match the seeded session and episode")
        await state.close(self.config.session_close_timeout_seconds)
        observations = state.observations or AgentObservationBundle(
            source="pi", gaps=[ObservationGap(code="agent_activation_interrupted")]
        )
        self._sandbox_sessions.pop(session_id, None)
        result = AgentCloseSessionResponse(agent_session_id=session_id, agent_observations=observations)
        # Bound retry receipts in memory; keep the cookie so stale activations
        # cannot silently fall back to host execution after a successful close.
        self._closed_sandbox_sessions[session_id] = (body.episode_id, result)
        while len(self._closed_sandbox_sessions) > 64:
            self._closed_sandbox_sessions.popitem(last=False)
        return result

    async def _sandbox_response(
        self, state: PiSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        unsupported = (
            "temperature",
            "top_p",
            "reasoning",
            "max_tool_calls",
            "previous_response_id",
            "prompt",
            "text",
            "context_management",
            "conversation",
            "moderation",
            "top_logprobs",
            "truncation",
        )
        values = body.model_dump(mode="json")
        for name in unsupported:
            if values.get(name) is not None:
                raise HTTPException(422, f"Native Pi does not support request field {name}")
        if body.tools or body.tool_choice != "auto" or not body.parallel_tool_calls or body.background:
            raise HTTPException(422, "Pi owns tool selection and execution policy")
        if (body.metadata or {}).get("chat_template_kwargs") is not None:
            raise HTTPException(422, "Configure chat_template_kwargs on the Gym model server for Pi")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"]):
            raise HTTPException(422, "Native Pi accepts one text user prompt with an optional system message")
        for item in items:
            if not isinstance(item.content, str) and any(
                (part.get("type") if isinstance(part, dict) else getattr(part, "type", None)) != "input_text"
                for part in item.content
            ):
                raise HTTPException(422, "Native Pi only supports text input")
        prompt, input_system = _extract_instruction(items)
        system = "\n\n".join(part for part in (self.config.system_prompt, body.instructions, input_system) if part)
        # Native sessions use only Gym's provider; do not copy credentials for
        # unrelated direct providers from the host configuration into the sandbox.
        models = {
            "providers": {"nemo": self._build_models_config(state.seed.episode_id.capture_key)["providers"]["nemo"]}
        }
        if body.max_output_tokens is not None:
            models["providers"]["nemo"]["models"][0]["maxTokens"] = body.max_output_tokens
        await state.upload_json("home/.pi/agent/models.json", models)
        command = [
            f"{state.runtime}/node/bin/node",
            f"{state.runtime}/pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js",
            "--print",
            "--mode",
            "json",
            "--no-session",
            "--provider",
            "nemo",
            "--model",
            self.config.model,
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
        ]
        if self.config.thinking:
            command += ["--thinking", self.config.thinking]
        if system:
            command += ["--append-system-prompt", system]
        payload = {
            "directory": state.directory,
            "command": command,
            "prompt": prompt,
            "cwd": state.seed.sandbox_access.workdir,
            "env": {
                "HOME": f"{state.directory}/home",
                "PI_CODING_AGENT_DIR": f"{state.directory}/home/.pi/agent",
                "PI_SKIP_VERSION_CHECK": "1",
                "PI_TELEMETRY": "0",
            },
            "timeout": self.config.timeout,
            "cleanup_timeout": self.config.session_close_timeout_seconds / 3,
        }
        async with self.sem:
            raw = await state.execute(
                payload, timeout=self.config.timeout, close_timeout=self.config.session_close_timeout_seconds
            )
        events = [tuple(json.loads(line)) for line in raw.splitlines() if line.strip()]
        output = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        cached_tokens = 0
        errors = []
        stop_reasons = []
        for _, event in events:
            message = event.get("message") or {}
            if event.get("type") == "message_end" and message.get("role") == "assistant":
                for part in message.get("content") or []:
                    if part.get("type") == "thinking" and part.get("thinking"):
                        output.append(
                            NeMoGymResponseReasoningItem(
                                id=f"reasoning-{len(output)}",
                                summary=[{"type": "summary_text", "text": part["thinking"]}],
                            )
                        )
                cached_tokens += int((message.get("usage") or {}).get("cacheRead") or 0)
                stop_reasons.append(message.get("stopReason"))
                if message.get("stopReason") in ("error", "aborted"):
                    errors.append(message.get("errorMessage") or message["stopReason"])
            parsed, tokens = parse_pi_events(json.dumps(event))
            output.extend(parsed)
            for key in usage:
                usage[key] += tokens[key]
        result = state.result
        error = result.error or ("; ".join(errors) if errors else None)
        if result.return_code != 0 and not result.timed_out:
            error = error or f"Pi exited with code {result.return_code}"
        if not stop_reasons:
            error = error or "Pi produced no assistant result"
        elif stop_reasons[-1] not in ("stop", "length", "error", "aborted") and not result.timed_out:
            error = error or "Pi ended without a terminal assistant result"
        incomplete = result.timed_out or (stop_reasons and stop_reasons[-1] == "length")
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=output,
            status="failed" if error else "incomplete" if incomplete else "completed",
            error={"code": "server_error", "message": error} if error else None,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["input_tokens"] + usage["output_tokens"],
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            ),
            metadata={
                "harness_execution": "sandbox",
                "harness_hostname": result.hostname,
                "harness_pid": str(result.pid),
                "pi_version": self.config.pi_version,
            },
        )
        conversation_input = [NeMoGymEasyInputMessage(role="system", content=system)] if system else []
        conversation_input.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        try:
            state.observations = _build_pi_observations(
                events,
                state.seed.episode_id.capture_key,
                self.config.model_server,
                [*conversation_input, *output],
                transcript_available=bool(output),
            )
        except Exception:
            # Observation parsing must not turn a valid response/patch into a
            # failed episode (the existing local Pi path has the same policy).
            LOG.exception("failed to build sandbox Pi observations")
            state.observations = AgentObservationBundle(
                source="pi", gaps=[ObservationGap(code="observation_parse_failed")]
            )
        return response

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"pi_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _env(self, home: Path) -> dict[str, str]:
        env = {**os.environ, "HOME": str(home), "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}
        env.update({k: v for k, v in self.config.env.items() if v})
        return env

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None) -> str:
        if self.config.model_server is None:
            return ""
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    def _effective_model(self) -> str:
        return f"nemo/{self.config.model}" if self.config.model_server else self.config.model

    def _build_models_config(self, rollout_id: Optional[str] = None) -> dict[str, Any]:
        config = copy.deepcopy(self.config.models_config)
        if self.config.model_server is None:
            return config
        providers = config.setdefault("providers", {})
        providers["nemo"] = {
            "baseUrl": self._resolve_model_base_url(rollout_id),
            "api": "openai-completions",
            "apiKey": "EMPTY",  # pragma: allowlist secret
            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
            "models": [
                {
                    "id": self.config.model,
                    "reasoning": True,
                    "input": ["text"],
                    "contextWindow": self.config.context_window,
                    "maxTokens": self.config.max_output_tokens,
                }
            ],
        }
        return config

    async def _run_pi(
        self,
        instruction: str,
        system_prompt: Optional[str],
        *,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
    ) -> tuple[list[Any], dict[str, int], str, list[tuple[float, dict[str, Any]]]]:
        # Local callers still get automatic installation. Native sessions never
        # enter this path, so their host does not need Pi, Node, or npm.
        if self._local_setup_task is None:
            self._local_setup_task = asyncio.create_task(asyncio.to_thread(ensure_pi, self.config.pi_version))
        setup_task = self._local_setup_task
        try:
            # Share one installation across concurrent local calls. Cancelling
            # a caller must not release another caller to start a second install.
            await asyncio.shield(setup_task)
        except Exception:
            if self._local_setup_task is setup_task:
                self._local_setup_task = None
            raise
        effective_model = self._effective_model()
        provider, _, model_id = effective_model.partition("/")
        work_dir = self._workspace_root()
        home = work_dir / ".pi-home"
        (home / ".pi" / "agent").mkdir(parents=True, exist_ok=True)
        models_config = self._build_models_config(rollout_id)
        if models_config:
            (home / ".pi" / "agent" / "models.json").write_text(json.dumps(models_config, indent=2))
        env = self._env(home)

        cmd = [*self.config.command_parts, "--print", "--mode", "json", "--no-session"]
        if provider:
            cmd += ["--provider", provider, "--model", model_id]
        else:
            cmd += ["--model", self.config.model]
        if self.config.thinking:
            cmd += ["--thinking", self.config.thinking]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        cmd += self.config.extra_args
        cmd.append(instruction)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            assert proc.stdout is not None and proc.stderr is not None
            events: list[tuple[float, dict[str, Any]]] = []
            if collect_observations:
                stdout_task = asyncio.create_task(_read_pi_stdout(proc.stdout))
                stderr_task = asyncio.create_task(proc.stderr.read())
                output_task = asyncio.gather(stdout_task, stderr_task, proc.wait())
                try:
                    (stdout, events), stderr, _ = await asyncio.wait_for(
                        asyncio.shield(output_task), timeout=self.config.timeout
                    )
                except asyncio.TimeoutError:
                    if proc.returncode is None:
                        proc.kill()
                    (_, events), _, _ = await output_task
                    LOG.warning("pi timed out after %ds", self.config.timeout)
                    return [], {"input_tokens": 0, "output_tokens": 0}, self.config.model, events
            else:
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    LOG.warning("pi timed out after %ds", self.config.timeout)
                    return [], {"input_tokens": 0, "output_tokens": 0}, self.config.model, events

            if proc.returncode not in (0, None):
                LOG.warning("pi exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])
            output_items, usage = parse_pi_events(stdout)
            return output_items, usage, self.config.model, events
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
    ) -> AgentEpisode:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        conversation_input = (
            [NeMoGymEasyInputMessage(role="system", content=system_prompt)] if system_prompt is not None else []
        )
        conversation_input.append(NeMoGymEasyInputMessage(role="user", content=user_message))

        output_items, usage, model_name, events = await self._run_pi(
            user_message,
            system_prompt,
            rollout_id=rollout_id,
            collect_observations=collect_observations,
        )
        observed_output_items = list(output_items)
        if not observed_output_items and events:
            observed_output_items, _ = parse_pi_events("\n".join(json.dumps(event) for _, event in events))

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("pi produced no assistant message; padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )
        observations = AgentObservationBundle(source="pi")
        if collect_observations:
            invocation_id = rollout_id or response.id
            try:
                observations = _build_pi_observations(
                    events,
                    invocation_id,
                    self.config.model_server,
                    [*conversation_input, *observed_output_items],
                    transcript_available=bool(observed_output_items),
                )
            except Exception:
                LOG.exception("failed to build Pi observations")
                observations = AgentObservationBundle(
                    source="pi", gaps=[ObservationGap(code="observation_parse_failed")]
                )
            observations.gaps.append(ObservationGap(code="no_sandbox_runtime"))
        return AgentEpisode(response=response, observations=observations)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        try:
            session_id = request.session.get(_SANDBOX_SESSION_KEY)
        except (AssertionError, AttributeError):
            session_id = None
        if isinstance(session_id, str):
            state = self._sandbox_sessions.get(session_id)
            rollout_id = request.path_params.get("rollout_id")
            if state is None or state.seed.episode_id.capture_key != rollout_id:
                raise HTTPException(409, "Pi activation does not match the seeded session and rollout route")
            if state.activated or state.closing:
                raise HTTPException(409, "Pi sandbox sessions support one activation")
            state.activated = True
            state.task = asyncio.create_task(self._sandbox_response(state, body))
            try:
                return await asyncio.shield(state.task)
            except asyncio.CancelledError:
                if not state.task.done() and not state.task.cancelling():
                    state.task.cancel()
                raise
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        episode = await self._create_episode(
            body,
            rollout_id=rollout_id,
            collect_observations=isinstance(rollout_id, str),
        )
        if not isinstance(rollout_id, str):
            return episode.response
        return episode.response.model_copy(
            update={_INTERNAL_OBSERVATIONS_KEY: episode.observations.model_dump(mode="json")}
        )

    async def run(self, request: Request, body: PiAgentRunRequest) -> PiAgentVerifyResponse:
        if self.config.resources_server is None:
            raise HTTPException(
                422, "Pi /run requires resources_server; use EnvironmentServer /run for native sessions"
            )
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

            return PiAgentVerifyResponse.model_validate(
                verify_json
                | {"turns_used": turns, "finished_naturally": naturally}
                | ({"ng_agent_observations": observations} if observations is not None else {})
            )


if __name__ == "__main__":
    PiAgent.run_webserver()
