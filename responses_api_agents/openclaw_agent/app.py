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
import contextlib
import copy
import json
import logging
import os
import re
import shlex
import shutil
import signal
from asyncio import Semaphore
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic, time
from typing import Any, Callable, ClassVar, Optional
from uuid import uuid4

import psutil
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
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentObservationBundle,
    ObservationGap,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import get_global_config_dict, get_response_json, raise_for_status
from responses_api_agents.openclaw_agent.observability import (
    OPENCLAW_OBSERVATION_SOURCE,
    OpenClawSessionTree,
    build_openclaw_observation_tree,
    build_openclaw_observations,
    discover_openclaw_session_tree,
)
from responses_api_agents.openclaw_agent.sandbox import _SANDBOX_PATH_CHECK, OpenClawSandboxSession
from responses_api_agents.openclaw_agent.setup_openclaw import ensure_openclaw


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"
_SANDBOX_SESSION_KEY = "nemo_gym_openclaw_sandbox_session"


def _decode_last_json_dict_suffix(raw: str) -> Optional[dict[str, Any]]:
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            obj, consumed = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and not text[start + consumed :].strip():
            return obj
    return None


def _text_from_openclaw_payloads(envelope: dict[str, Any]) -> str:
    payloads = envelope.get("payloads")
    if not isinstance(payloads, list):
        payloads = []
    parts = [p["text"].strip() for p in payloads if isinstance(p, dict) and (p.get("text") or "").strip()]
    if parts:
        return "\n\n".join(parts)
    final = (envelope.get("meta") or {}).get("finalAssistantVisibleText")
    return final.strip() if isinstance(final, str) else ""


def _openclaw_output_items(envelope: dict[str, Any]) -> list[Any]:
    """Read envelope text independently of optional numeric usage metadata."""
    text = _text_from_openclaw_payloads(envelope)
    output_items: list[Any] = []
    if text:
        output_items.append(
            NeMoGymResponseOutputMessage(
                id="msg-0",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                role="assistant",
                status="completed",
                type="message",
            )
        )
    return output_items


def parse_openclaw_output(stdout: str) -> tuple[list[Any], dict[str, int]]:
    envelope = _decode_last_json_dict_suffix(stdout)
    if not envelope:
        return [], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    output_items = _openclaw_output_items(envelope)

    meta = envelope.get("meta") if isinstance(envelope.get("meta"), dict) else {}
    agent_meta = meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
    usage = agent_meta.get("usage") if isinstance(agent_meta.get("usage"), dict) else {}
    cache_read = int(usage.get("cacheRead") or 0)
    input_tokens = int(usage.get("input") or 0) + cache_read
    output_tokens = int(usage.get("output") or 0)
    return output_items, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cache_read,
    }


def _unique_usage_messages(
    assistants: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[ObservationGap]]:
    # OpenClaw rewrites transcript branches by appending the original messages
    # again. Entry IDs change, but the provider response ID and message do not.
    identified: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    unidentified = []
    gaps = []
    for message in assistants:
        response_id = message.get("responseId")
        if not isinstance(response_id, str) or not response_id.strip():
            unidentified.append(message)
            gaps.append(
                ObservationGap(
                    code="model_call_usage_identity_unavailable",
                    detail="Usage counted per transcript record; missing response ID prevents rewrite deduplication",
                )
            )
        elif response_id not in identified:
            identified[response_id] = message
        elif message != identified[response_id]:
            ambiguous.add(response_id)
    for response_id in sorted(ambiguous):
        gaps.append(
            ObservationGap(
                code="model_call_usage_identity_ambiguous",
                detail=f"Conflicting transcript messages for response ID {response_id}; usage excluded from totals",
            )
        )
    return [message for key, message in identified.items() if key not in ambiguous] + unidentified, gaps


def parse_openclaw_session_items(events: list[dict[str, Any]], *, include_input: bool = False) -> list[Any]:
    """Convert OpenClaw session events into Gym conversation items."""
    output_items: list[Any] = []
    for event in events:
        event_id = event.get("id")
        if event.get("type") != "message":
            continue
        message = event.get("message") or {}
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if include_input and role in {"user", "system", "developer"}:
            text = content if isinstance(content, str) else ""
            if isinstance(content, list):
                text = "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"]
                )
            if text:
                output_items.append(NeMoGymEasyInputMessage(role=role, content=text))
            continue
        if not include_input and not isinstance(content, list):
            continue

        if role == "assistant":
            reasoning = []
            for key in ("reasoning_content", "reasoning_text", "thinking"):
                value = message.get(key)
                if isinstance(value, str) and value:
                    reasoning.append(value)
            message_reasoning = message.get("reasoning")
            if isinstance(message_reasoning, str) and message_reasoning:
                reasoning.append(message_reasoning)
            elif isinstance(message_reasoning, dict):
                for key in ("content", "text", "summary"):
                    value = message_reasoning.get(key)
                    if isinstance(value, str) and value:
                        reasoning.append(value)
            if isinstance(content, list):
                reasoning.extend(
                    text
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") in {"thinking", "reasoning"}
                    and isinstance((text := block.get("thinking") or block.get("text") or block.get("reasoning")), str)
                    and text
                )
            if include_input and reasoning:
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{event_id or len(output_items)}",
                        summary=[NeMoGymSummary(text="\n".join(reasoning), type="summary_text")],
                    )
                )

            texts = [content] if include_input and isinstance(content, str) and content else []
            if isinstance(content, list):
                texts = [
                    block["text"] for block in content if isinstance(block, dict) and (block.get("text") or "").strip()
                ]
                if include_input:
                    texts = [
                        block["text"]
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") not in {"thinking", "reasoning", "toolCall"}
                        and isinstance(block.get("text"), str)
                        and block["text"].strip()
                    ]
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
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                args = block.get("arguments")
                if include_input and args is None:
                    args = block.get("partialArgs")
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
            call_id = message.get("toolCallId") or (message.get("tool_call_id") if include_input else "") or ""
            result_text = content if include_input and isinstance(content, str) else ""
            if isinstance(content, list):
                result_text = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if include_input and not result_text and message.get("details") is not None:
                result_text = json.dumps(message["details"], ensure_ascii=False)
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=result_text,
                    status="completed",
                )
            )

    return output_items


def openclaw_session_conversation(
    events: list[dict[str, Any]],
    *,
    input_items: list[Any] | None = None,
    fallback_output: list[Any] | None = None,
) -> list[Any]:
    """Prefer retained transcript items and fill only evidence missing from the artifact."""
    conversation = parse_openclaw_session_items(events, include_input=True)
    inputs = input_items or []
    fallback = fallback_output or []
    if not conversation:
        return [*inputs, *fallback]
    retained_roles = {
        role for item in conversation if (role := getattr(item, "role", None)) in {"user", "system", "developer"}
    }
    missing_inputs = (
        [item for item in inputs if getattr(item, "role", None) not in retained_roles] if retained_roles else inputs
    )
    if missing_inputs:
        conversation = [*missing_inputs, *conversation]
    if fallback and not any(
        getattr(item, "role", None) == "assistant"
        or getattr(item, "type", None) in {"reasoning", "function_call", "function_call_output"}
        for item in conversation
    ):
        conversation.extend(fallback)
    return conversation


def parse_openclaw_session(session_text: str) -> list[Any]:
    """Convert an OpenClaw session .jsonl into Gym output items, including tool calls."""
    return parse_openclaw_session_items(parse_openclaw_session_events(session_text))


def parse_openclaw_session_events(session_text: str) -> list[dict[str, Any]]:
    events = []
    for line in session_text.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            event = {"raw": line}
        events.append(event if isinstance(event, dict) else {"raw": line})
    return events


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


class OpenClawAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: Optional[ResourcesServerRef] = None
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    command: str = "openclaw"
    model: str = "nvinf/nvidia/meta/llama-3.3-70b-instruct"
    node_bin_dir: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/openclaw_agent/workspaces"
    openclaw_agent_id: str = "main"
    thinking: str = "off"
    system_prompt: Optional[str] = None
    setup_timeout: int = 900
    timeout: int = 900
    extra_args: list[str] = []
    openclaw_config: dict[str, Any] = Field(default_factory=dict)
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    # required: every config must pin an explicit version so runs are reproducible and cannot silently drift
    openclaw_version: str
    sandbox_install_timeout_seconds: float = Field(default=600, gt=0)
    session_close_timeout_seconds: float = Field(default=60, gt=0)
    session_lifetime_seconds: float = Field(default=21600, gt=0, allow_inf_nan=False)
    session_close_retry_window_seconds: float = Field(
        default=300.0,
        gt=0,
        allow_inf_nan=False,
        description="Keep successful close receipts for this many seconds; cover the caller's retry horizon.",
    )

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class OpenClawAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenClawAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: AgentObservationBundle | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class OpenClawAgent(SimpleResponsesAPIAgent):
    """Runs the OpenClaw CLI (openclaw agent --local --json)"""

    config: OpenClawAgentConfig
    sem: Semaphore = None
    sigterm_events: set = Field(default_factory=set)
    sigterm_handler_installed: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # deny the interactive "message" channel so the headless agent finishes
    _HEADLESS_TOOL_DENY: ClassVar[tuple[str, ...]] = ("message",)

    _sandbox_sessions: dict[str, OpenClawSandboxSession] = PrivateAttr(default_factory=dict)
    _closed_sandbox_sessions: OrderedDict[str, tuple[EpisodeId, AgentCloseSessionResponse, float]] = PrivateAttr(
        default_factory=OrderedDict
    )
    _local_setup_task: asyncio.Task[None] | None = PrivateAttr(default=None)
    _session_locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _session_lock_users: dict[str, int] = PrivateAttr(default_factory=dict)
    _session_expiry_tasks: dict[str, asyncio.Task[None]] = PrivateAttr(default_factory=dict)
    _closed_session_ids: dict[str, tuple[EpisodeId, float]] = PrivateAttr(default_factory=dict)

    @asynccontextmanager
    async def _session_lock(self, session_id: str) -> AsyncIterator[None]:
        """Count holders and waiters so pruning cannot replace an in-flight ID lock."""
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        self._session_lock_users[session_id] = self._session_lock_users.get(session_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            remaining = self._session_lock_users[session_id] - 1
            if remaining:
                self._session_lock_users[session_id] = remaining
            else:
                self._session_lock_users.pop(session_id)
                if session_id not in self._sandbox_sessions and session_id not in self._closed_session_ids:
                    self._session_locks.pop(session_id, None)

    async def seed_agent_session(self, request: Request, body: AgentSeedSessionRequest) -> AgentSeedSessionResponse:
        """Borrow the Resources-owned task sandbox and install a pinned OpenClaw runtime inside it."""
        self._expire_closed_agent_sessions()
        session_id = body.agent_session_id
        previous = request.session.get(_SANDBOX_SESSION_KEY)
        if previous != session_id and previous in self._sandbox_sessions:
            raise HTTPException(409, "OpenClaw cookie belongs to another active session")
        async with self._session_lock(session_id):
            if session_id in self._closed_session_ids:
                raise HTTPException(409, "OpenClaw session is already closed")
            state = self._sandbox_sessions.get(session_id)
            if state is not None:
                if state.seed != body:
                    raise HTTPException(409, "OpenClaw session ID is bound to a different seed request")
                if state.closing:
                    raise HTTPException(409, "OpenClaw session is closing")
            else:
                try:
                    state = await self._initialize_agent_session_state(session_id, body)
                except BaseException:
                    if session_id not in self._sandbox_sessions:
                        # Initialization either never connected or confirmed its cleanup.
                        self._closed_sandbox_sessions[session_id] = (
                            body.episode_id,
                            AgentCloseSessionResponse(agent_session_id=session_id),
                            monotonic() + self.config.session_close_retry_window_seconds,
                        )
                        self._remember_closed_session(session_id, body.episode_id)
                    else:
                        self._session_expiry_tasks[session_id] = asyncio.create_task(
                            self._expire_agent_session(session_id, body.episode_id)
                        )
                    raise
                self._sandbox_sessions[session_id] = state
                self._session_expiry_tasks[session_id] = asyncio.create_task(
                    self._expire_agent_session(session_id, body.episode_id)
                )
            request.session[_SANDBOX_SESSION_KEY] = session_id
            return AgentSeedSessionResponse(agent_session_id=session_id)

    async def _expire_agent_session(self, session_id: str, episode_id: EpisodeId) -> None:
        """Bound abandoned sessions through the same fail-closed teardown as explicit close."""
        try:
            await asyncio.sleep(self.config.session_lifetime_seconds)
            await self.close_agent_session(
                Request({"type": "http", "session": {}}),
                AgentCloseSessionRequest(agent_session_id=session_id, episode_id=episode_id),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("Could not clean up expired OpenClaw session %s; retaining failed state", session_id)
        finally:
            self._session_expiry_tasks.pop(session_id, None)

    async def _initialize_agent_session_state(
        self, agent_session_id: str, body: AgentSeedSessionRequest
    ) -> OpenClawSandboxSession:
        # Match Hermes: session initialization owns the sandbox runtime setup,
        # with the same AgentSeedSessionRequest/SandboxAccess wire contracts.
        if self.config.num_workers not in (None, 1):
            raise HTTPException(422, "Native OpenClaw sessions require num_workers=1")
        if body.sandbox_access is None or not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
            raise HTTPException(422, "Native OpenClaw requires direct, Resources-owned SandboxAccess")
        if not body.sandbox_access.workdir.startswith("/") or body.sandbox_access.workdir in ("/", "/tmp"):
            raise HTTPException(422, "OpenClaw workdir must be absolute and separate from /tmp runtime storage")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise HTTPException(422, "Native OpenClaw supports its own sandbox tools, not required HTTP/MCP tools")
        if self.config.model_server is None:
            raise HTTPException(422, "Native OpenClaw requires a sandbox-reachable Gym model_server")
        if not self.config.openclaw_version or not re.fullmatch(
            r"\d+\.\d+\.\d+(?:-\d+)?", self.config.openclaw_version
        ):
            raise HTTPException(422, "Native OpenClaw requires an exact openclaw_version, for example 2026.6.11")
        if self.config.openclaw_config or self.config.max_output_tokens is not None or self.config.node_bin_dir:
            raise HTTPException(
                422, "Native OpenClaw does not support openclaw_config, max_output_tokens, or node_bin_dir overrides"
            )
        if self.config.openclaw_agent_id != "main":
            raise HTTPException(422, "Native OpenClaw currently requires openclaw_agent_id=main")
        if self.config.command != "openclaw" or self.config.extra_args or self.config.env:
            raise HTTPException(422, "Native OpenClaw does not support command, extra_args, or env overrides")

        connection = body.sandbox_access.connection
        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        directory = f"/tmp/nemo-gym-openclaw-sessions/{uuid4().hex}"
        runtime = f"/tmp/nemo-gym-openclaw-node-22.19.0-{self.config.openclaw_version}"
        state = OpenClawSandboxSession(body, sandbox, directory, runtime)
        prepared_directory = False
        try:
            check_paths = shlex.join(
                [
                    "python3",
                    "-c",
                    _SANDBOX_PATH_CHECK,
                    body.sandbox_access.workdir,
                    "/tmp/nemo-gym-openclaw-sessions",
                    runtime,
                ]
            )
            prepared = await sandbox.exec(
                "command -v python3 >/dev/null 2>&1 || "
                "{ echo 'Native OpenClaw requires Python >=3.9 in the task image' >&2; exit 1; }; "
                f"{check_paths} && mkdir -p {shlex.quote(directory + '/home/.openclaw')}",
                timeout_s=30,
            )
            if prepared.return_code != 0 or getattr(prepared, "error_type", None):
                raise RuntimeError(prepared.stderr or "Cannot create OpenClaw sandbox session directory")
            prepared_directory = True
            # Install only the agent runtime in the existing task sandbox.
            # Resources has already prepared the task repository and its dependencies.
            installer = "install_openclaw_runtime.sh"
            await sandbox.upload(Path(__file__).with_name(installer), f"{directory}/{installer}")
            installed = await sandbox.exec(
                f"bash {shlex.quote(directory + '/' + installer)} {shlex.quote(runtime)} "
                f"{shlex.quote(self.config.openclaw_version)}",
                cwd=body.sandbox_access.workdir,
                timeout_s=self.config.sandbox_install_timeout_seconds,
            )
            if installed.return_code != 0 or getattr(installed, "error_type", None):
                raise RuntimeError(
                    f"OpenClaw runtime setup failed (exit {installed.return_code}): "
                    f"bash {directory}/{installer} {runtime} {self.config.openclaw_version}\n"
                    f"{installed.stderr or installed.stdout}"
                )
            await sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), f"{directory}/sandbox_runner.py")
        except BaseException:
            try:
                if prepared_directory:
                    await state.close(self.config.session_close_timeout_seconds)
                else:
                    # The path check may have rejected overlap with task-owned storage.
                    state.closing = True
                    await sandbox.disconnect()
                    state.closed = True
            except BaseException:
                self._sandbox_sessions[agent_session_id] = state
                LOG.exception("OpenClaw seed cleanup failed; retaining session %s", agent_session_id)
            raise
        return state

    async def close_agent_session(self, request: Request, body: AgentCloseSessionRequest) -> AgentCloseSessionResponse:
        """Confirm OpenClaw teardown before allowing verification; never destroy the borrowed sandbox."""
        self._expire_closed_agent_sessions()
        session_id = body.agent_session_id
        cookie = request.session.get(_SANDBOX_SESSION_KEY)
        if cookie is not None and cookie != session_id:
            raise HTTPException(409, "OpenClaw close cookie does not match the requested session")
        async with self._session_lock(session_id):
            closed = self._closed_sandbox_sessions.get(session_id)
            if closed is not None:
                if body.episode_id != closed[0]:
                    raise HTTPException(409, "OpenClaw close does not match the seeded episode")
                request.session[_SANDBOX_SESSION_KEY] = session_id
                return closed[1]
            if session_id in self._closed_session_ids:
                raise HTTPException(409, "OpenClaw close receipt has expired")
            state = self._sandbox_sessions.get(session_id)
            if state is None:
                result = AgentCloseSessionResponse(agent_session_id=session_id)
            else:
                if body.episode_id != state.seed.episode_id:
                    raise HTTPException(409, "OpenClaw close does not match the seeded episode")
                await state.close(self.config.session_close_timeout_seconds)
                observations = state.observations or AgentObservationBundle(
                    source="openclaw", gaps=[ObservationGap(code="agent_activation_interrupted")]
                )
                result = AgentCloseSessionResponse(agent_session_id=session_id, agent_observations=observations)
                self._sandbox_sessions.pop(session_id, None)
                expiry = self._session_expiry_tasks.pop(session_id, None)
                if expiry is not None and expiry is not asyncio.current_task():
                    expiry.cancel()
            request.session[_SANDBOX_SESSION_KEY] = session_id
            self._closed_sandbox_sessions[session_id] = (
                body.episode_id,
                result,
                monotonic() + self.config.session_close_retry_window_seconds,
            )
            self._remember_closed_session(session_id, body.episode_id)
            return result

    def _remember_closed_session(self, session_id: str, episode_id: EpisodeId) -> None:
        """Prevent delayed seeds through the session lifetime and close retry horizon."""
        retention = max(self.config.session_lifetime_seconds, self.config.session_close_retry_window_seconds)
        self._closed_session_ids[session_id] = (episode_id, monotonic() + retention)
        asyncio.get_running_loop().call_later(retention, self._expire_closed_agent_sessions)

    def _expire_closed_agent_sessions(self) -> None:
        """Prune receipts and tombstones by elapsed time, never by traffic or retry order."""
        now = monotonic()
        while self._closed_sandbox_sessions:
            if next(iter(self._closed_sandbox_sessions.values()))[2] > now:
                break
            self._closed_sandbox_sessions.popitem(last=False)
        for session_id, (_, deadline) in tuple(self._closed_session_ids.items()):
            if deadline <= now:
                self._closed_session_ids.pop(session_id)
                if not self._session_lock_users.get(session_id, 0):
                    self._session_locks.pop(session_id, None)

    def _sandbox_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
        """Validate and normalize input before consuming the session's activation."""
        unsupported = (
            "include",
            "store",
            "service_tier",
            "prompt_cache_key",
            "prompt_cache_retention",
            "safety_identifier",
            "stream_options",
            "user",
            "max_output_tokens",
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
        if body.model is not None and body.model != self.config.model:
            raise HTTPException(422, "Native OpenClaw model must match the configured model")
        extras = set(body.model_extra or {})
        if extras:
            raise HTTPException(422, f"Native OpenClaw does not support extra request fields: {sorted(extras)}")
        values = body.model_dump(mode="json")
        for name in unsupported:
            if values.get(name) is not None:
                raise HTTPException(422, f"Native OpenClaw does not support request field {name}")
        if body.tools or body.tool_choice != "auto" or not body.parallel_tool_calls or body.background:
            raise HTTPException(422, "OpenClaw owns tool selection and execution policy")
        if (body.metadata or {}).get("chat_template_kwargs") is not None:
            raise HTTPException(422, "Configure chat_template_kwargs on the Gym model server for OpenClaw")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"]):
            raise HTTPException(422, "Native OpenClaw accepts one text user prompt with an optional system message")
        for item in items:
            if not isinstance(item.content, str) and any(
                (part.get("type") if isinstance(part, dict) else getattr(part, "type", None)) != "input_text"
                for part in item.content
            ):
                raise HTTPException(422, "Native OpenClaw only supports text input")
        prompt, input_system = _extract_instruction(items)
        system = "\n\n".join(part for part in (self.config.system_prompt, body.instructions, input_system) if part)
        if not prompt.strip():
            raise HTTPException(422, "Native OpenClaw requires a nonempty user prompt")
        return prompt, system

    async def _sandbox_response(
        self,
        state: OpenClawSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        prompt: str,
        system: str,
    ) -> NeMoGymResponse:
        # The pinned CLI reads configuration directly; onboarding would create
        # workspace bootstrap files and unrelated provider/channel state.
        native_config = self._build_openclaw_config({}, state.seed.episode_id.capture_key)
        # OpenClaw conservatively disables stream usage for custom origins. Gym
        # supports it; request the final usage chunk instead of retaining zero counters.
        native_config["models"]["providers"]["nemo"]["models"][0]["compat"] = {
            "supportsUsageInStreaming": True,
        }
        native_config.update(
            {
                "agents": {
                    "defaults": {
                        "workspace": state.seed.sandbox_access.workdir,
                        "skipBootstrap": True,
                        "skills": [],
                        "model": {"primary": self._effective_model()},
                        "sandbox": {"mode": "off"},
                        "memorySearch": {"enabled": False},
                    }
                },
                "tools": {
                    "allow": ["read", "write", "edit", "exec", "process"],
                    "exec": {"host": "gateway", "security": "full", "ask": "off"},
                },
                "plugins": {"enabled": False},
            }
        )
        await state.upload_json("home/.openclaw/openclaw.json", native_config)
        command = [
            f"{state.runtime}/node/bin/node",
            f"{state.runtime}/openclaw/node_modules/openclaw/openclaw.mjs",
            "agent",
            "--local",
            "--json",
            "--agent",
            "main",
            "--session-id",
            state.directory.rsplit("/", 1)[-1],
            "--thinking",
            self.config.thinking,
            "--model",
            self._effective_model(),
            "--message-file",
            f"{state.directory}/prompt.txt",
            "--timeout",
            str(self.config.timeout),
        ]
        payload = {
            "directory": state.directory,
            "command": command,
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "cwd": state.seed.sandbox_access.workdir,
            "env": {
                "HOME": f"{state.directory}/home",
                "XDG_CACHE_HOME": f"{state.directory}/home/.cache",
                "OPENCLAW_STATE_DIR": f"{state.directory}/home/.openclaw",
                "OPENCLAW_CONFIG_PATH": f"{state.directory}/home/.openclaw/openclaw.json",
                "OPENCLAW_TELEMETRY": "0",
                "CLAWHUB_DISABLE_TELEMETRY": "1",
                "OPENCLAW_EXEC_SHELL_SNAPSHOT": "0",
            },
            "timeout": self.config.timeout,
            "cleanup_timeout": self.config.session_close_timeout_seconds / 3,
        }
        try:
            async with self.sem:
                stdout = await state.execute(
                    payload, timeout=self.config.timeout, close_timeout=self.config.session_close_timeout_seconds
                )
        except BaseException:
            # Cancellation/cleanup failure still leaves useful transcript evidence
            # for the subsequent close. Never convert uncertain cleanup to success.
            try:
                await self._collect_sandbox_response(state, body, prompt=prompt, system=system)
            except Exception:
                LOG.exception("Could not salvage interrupted OpenClaw observations")
            raise
        return await self._collect_sandbox_response(state, body, prompt=prompt, system=system, stdout=stdout)

    async def _collect_sandbox_response(
        self,
        state: OpenClawSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        prompt: str,
        system: str,
        stdout: str = "",
    ) -> NeMoGymResponse:
        gaps = []
        if not stdout:
            try:
                stdout = await state.read_text("stdout.log")
            except Exception:
                gaps.append(ObservationGap(code="agent_stdout_unavailable"))
        session_id = state.directory.rsplit("/", 1)[-1]
        events = []
        try:
            transcript = await state.read_text(f"home/.openclaw/agents/main/sessions/{session_id}.jsonl")
            events = parse_openclaw_session_events(transcript)
        except Exception:
            gaps.append(ObservationGap(code="agent_transcript_unavailable"))
        # include_input=True preserves reasoning fields; only remove the input
        # messages when projecting the transcript into Responses output.
        output = [
            item
            for item in parse_openclaw_session_items(events, include_input=True)
            if getattr(item, "role", None) not in {"user", "system", "developer"}
        ]
        try:
            envelope = _decode_last_json_dict_suffix(stdout) or {}
            fallback = _openclaw_output_items(envelope)
        except (TypeError, ValueError, AttributeError):
            fallback = []
            envelope = {}
            gaps.append(ObservationGap(code="agent_stdout_unparseable"))
        meta = envelope.get("meta")
        agent_meta = meta.get("agentMeta") if isinstance(meta, dict) else None
        raw_usage = agent_meta.get("usage") if isinstance(agent_meta, dict) else None
        raw_usage = raw_usage if isinstance(raw_usage, dict) else {}
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in (raw_usage.get(name) for name in ("input", "output", "cacheRead"))
        ):
            gaps.append(ObservationGap(code="agent_stdout_unparseable"))
        if not output:
            output = fallback
        assistants = [
            event["message"]
            for event in events
            if event.get("type") == "message"
            and isinstance(event.get("message"), dict)
            and event["message"].get("role") == "assistant"
            # The pinned CLI can append a synthetic final-text mirror with
            # aggregate run usage and stopReason="stop". It is neither another
            # model call nor evidence that the underlying call succeeded.
            and event["message"].get("api") != "cli"
        ]
        usage_messages, identity_gaps = _unique_usage_messages(assistants)
        gaps.extend(identity_gaps)
        # Auxiliary summarization can consume tokens and then abort before a
        # compaction event is written, so absent events do not prove coverage.
        gaps.append(
            ObservationGap(
                code="auxiliary_model_usage_unavailable",
                detail="Transcript has no auxiliary-call usage counters; totals cover observed assistant calls only",
            )
        )
        input_tokens = output_tokens = 0
        cached_tokens: int | None = 0 if usage_messages else None
        for message in usage_messages:
            usage = message.get("usage")
            if not isinstance(usage, dict):
                gaps.append(ObservationGap(code="model_call_usage_unavailable"))
                gaps.append(ObservationGap(code="cached_token_usage_unavailable"))
                cached_tokens = None
                continue
            if any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("input", "output")):
                gaps.append(ObservationGap(code="model_call_usage_unavailable", detail="Incomplete usage counters"))
            if not any(usage.get(key, 0) for key in ("input", "output", "cacheRead", "cacheWrite")):
                gaps.append(
                    ObservationGap(code="model_call_usage_unavailable", detail="Harness reported only zero counters")
                )
            cache_read = usage.get("cacheRead")
            # The pinned provider defaults unavailable backend cache details to
            # zero, so only positive counts retain measurement provenance.
            if type(cache_read) is not int or cache_read <= 0:
                gaps.append(ObservationGap(code="cached_token_usage_unavailable"))
                cached_tokens = None
            elif cached_tokens is not None:
                cached_tokens += cache_read

            def count(name: str) -> int:
                value = usage.get(name)
                return value if type(value) is int and value >= 0 else 0

            # OpenClaw/Pi input excludes cache reads and writes. Count distinct
            # observed calls, including failed calls, without adding CLI aggregate mirrors.
            input_tokens += count("input") + count("cacheRead") + count("cacheWrite")
            output_tokens += count("output")
        if not assistants:
            # The legacy envelope parser defaults missing counters to zero.
            # Native totals retain valid subtotals without coercing malformed cache values.
            cache_read = raw_usage.get("cacheRead")
            cached_tokens = cache_read if type(cache_read) is int and cache_read > 0 else None
            input_count, output_count = raw_usage.get("input"), raw_usage.get("output")
            input_tokens = input_count if type(input_count) is int and input_count >= 0 else 0
            output_tokens = output_count if type(output_count) is int and output_count >= 0 else 0
            input_tokens += cached_tokens or 0
            if cached_tokens is None:
                gaps.append(ObservationGap(code="cached_token_usage_unavailable"))
            gaps.append(
                ObservationGap(code="model_call_usage_unavailable", detail="Only CLI envelope totals available")
            )
        gaps.append(ObservationGap(code="reasoning_token_usage_unavailable"))
        result = state.result
        error = result.error if result else "OpenClaw activation ended without a cleanup receipt"
        last = assistants[-1] if assistants else {}
        if last.get("stopReason") in {"error", "aborted"}:
            error = error or last.get("errorMessage") or f"OpenClaw model call {last['stopReason']}"
        if result and result.return_code and not result.timed_out:
            try:
                stderr = (await state.read_text("stderr.log"))[-4000:]
            except Exception:
                stderr = "stderr unavailable"
            error = error or f"OpenClaw exited {result.return_code}: {stderr}"
        incomplete = bool(result and result.timed_out) or last.get("stopReason") == "length"
        if not incomplete and not error and last.get("stopReason") != "stop":
            error = "OpenClaw produced no terminal assistant result"
        status = "failed" if error else "incomplete" if incomplete else "completed"
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=output,
            status=status,
            error={"code": "server_error", "message": error} if error else None,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=None),
            ),
            metadata={
                "harness_execution": "sandbox",
                "harness_hostname": result.hostname if result else "unknown",
                "harness_pid": str(result.pid) if result else "unknown",
                "openclaw_version": self.config.openclaw_version,
            },
        )
        input_items = [NeMoGymEasyInputMessage(role="system", content=system)] if system else []
        input_items.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        try:
            state.observations = build_openclaw_observations(
                session_id,
                openclaw_session_conversation(events, input_items=input_items, fallback_output=output),
                events,
                transcript_available=bool(events),
                model_ref=self.config.model_server,
            )
            # Native sessions enable only local file/process tools, so there is
            # exactly one invocation; child sessions cannot be spawned.
            state.observations.gaps = [
                gap for gap in state.observations.gaps if gap.code != "subagent_hierarchy_unavailable"
            ]
            state.observations.records[0].status = status
            state.observations.gaps.extend(gaps)
        except Exception:
            LOG.exception("Could not build native OpenClaw observations")
            state.observations = AgentObservationBundle(
                source=OPENCLAW_OBSERVATION_SOURCE,
                gaps=[ObservationGap(code="observation_capture_failed"), *gaps],
            )
        return response

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenClawAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _merge_headless_tool_denies(self, cfg: dict[str, Any]) -> None:
        tools = cfg.setdefault("tools", {})
        deny = tools.get("deny")
        if not isinstance(deny, list):
            deny = []
        merged = list(dict.fromkeys([item for item in deny if isinstance(item, str)] + list(self._HEADLESS_TOOL_DENY)))
        tools["deny"] = merged

    def _build_openclaw_config(self, base: dict[str, Any], rollout_id: Optional[str] = None) -> dict[str, Any]:
        cfg = copy.deepcopy(base)
        self._deep_merge(cfg, copy.deepcopy(self.config.openclaw_config))
        if self.config.model_server:
            providers = cfg.setdefault("models", {}).setdefault("providers", {})
            nemo = providers.setdefault("nemo", {})
            model_entry = {
                "id": self.config.model,
                "name": self.config.model,
                "api": "openai-completions",
                "reasoning": True,
                "input": ["text"],
            }
            if self.config.context_window is not None:
                model_entry["contextWindow"] = self.config.context_window
            if self.config.max_output_tokens is not None:
                model_entry["maxTokens"] = self.config.max_output_tokens
            nemo.update(
                {
                    "api": "openai-completions",
                    "baseUrl": self._resolve_model_base_url(rollout_id),
                    "apiKey": "EMPTY",  # pragma: allowlist secret
                    "models": [model_entry],
                }
            )
        self._merge_headless_tool_denies(cfg)
        return cfg

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None) -> str:
        if self.config.model_server is None:
            return ""
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    def _effective_model(self) -> str:
        return f"nemo/{self.config.model}" if self.config.model_server else self.config.model

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"openclaw_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _env(self, home: Path) -> dict[str, str]:
        env = {
            **os.environ,
            "HOME": str(home),
            "OPENCLAW_TELEMETRY": "0",
            "CLAWHUB_DISABLE_TELEMETRY": "1",
        }
        if self.config.node_bin_dir:
            env["PATH"] = f"{self.config.node_bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env.update({k: v for k, v in self.config.env.items() if v})
        return env

    async def _run_exec(
        self, args: list[str], *, cwd: Optional[str], env: dict[str, str], timeout: int
    ) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            self._kill_process_tree(proc.pid)
            await proc.communicate()
            raise TimeoutError(f"Timed out after {timeout}s: {shlex.join(args)}") from None
        except asyncio.CancelledError:
            # Cancellation (e.g. SIGTERM salvage) only stops us from awaiting the process; it does
            # not stop the process itself. Kill it here so we never leak an orphaned process tree.
            self._kill_process_tree(proc.pid)
            with contextlib.suppress(Exception):
                await proc.communicate()
            raise
        return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    @staticmethod
    def _kill_process_tree(pid: int) -> None:
        """Kill a subprocess and every descendant it has already started."""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            return
        for child in reversed(children):
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        with contextlib.suppress(psutil.NoSuchProcess):
            parent.kill()

    @staticmethod
    def _session_file(envelope: Optional[dict[str, Any]]) -> Optional[Path]:
        meta = (envelope or {}).get("meta") if isinstance(envelope, dict) else None
        agent_meta = meta.get("agentMeta") if isinstance(meta, dict) else None
        session_file = agent_meta.get("sessionFile") if isinstance(agent_meta, dict) else None
        return Path(session_file) if isinstance(session_file, str) and session_file else None

    @staticmethod
    def _find_partial_session(home: Path) -> Optional[Path]:
        """Locate OpenClaw's session file on disk when there is no completion envelope to point at
        it, i.e. the run was cut short by a timeout. OpenClaw writes the session incrementally, so
        the transcript up to the last completed turn is already on disk; return the most recently
        written .jsonl under the OpenClaw home that parses to at least one message."""
        try:
            candidates = sorted(
                (p for p in home.rglob("*.jsonl") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return None
        for path in candidates:
            try:
                if parse_openclaw_session(path.read_text(errors="replace")):
                    return path
            except OSError:
                continue
        return None

    def _install_sigterm_handler(self) -> None:
        """Install one process-level wrapper that fans SIGTERM out to active runs.

        Keep the event loop's existing handler registered so uvicorn still receives the signal
        through Python's wakeup fd and performs its normal graceful shutdown.
        """
        if self.sigterm_handler_installed:
            return
        previous = signal.getsignal(signal.SIGTERM)

        def _on_sigterm(signum, frame) -> None:
            for event in self.sigterm_events:
                event.set()
            if callable(previous):
                previous(signum, frame)

        try:
            signal.signal(signal.SIGTERM, _on_sigterm)
            self.sigterm_handler_installed = True
        except ValueError:
            pass  # signal handlers need the main thread; fall back to timeout-only salvage

    async def _run_openclaw(
        self,
        instruction: str,
        system_prompt: Optional[str],
        rollout_id: Optional[str] = None,
        observation_collector: Optional[
            Callable[[str, list[dict[str, Any]], OpenClawSessionTree, list[ObservationGap]], None]
        ] = None,
    ) -> tuple[list[Any], dict[str, int], str]:
        """setup and run agent. returns (output_items, usage, model_name)."""
        # Local callers still get automatic installation. Native sessions never
        # enter this path, so their host does not need OpenClaw, Node, or npm.
        if self._local_setup_task is None:
            self._local_setup_task = asyncio.create_task(
                asyncio.to_thread(ensure_openclaw, self.config.openclaw_version)
            )
        setup_task = self._local_setup_task
        try:
            # Share one installation across concurrent local calls. Cancelling
            # a caller must not release another caller to start a second install.
            await asyncio.shield(setup_task)
        except Exception:
            if self._local_setup_task is setup_task:
                self._local_setup_task = None
            raise
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        home = work_dir / ".openclaw-home"
        home.mkdir(parents=True, exist_ok=True)
        env = self._env(home)

        try:
            code, _, stderr = await self._run_exec(
                [*self.config.command_parts, "onboard", "--non-interactive", "--accept-risk", "--skip-health"],
                cwd=str(work_dir),
                env=env,
                timeout=self.config.setup_timeout,
            )
            if code:
                raise RuntimeError(f"openclaw onboard exited {code}: {stderr}")

            config_path = home / ".openclaw" / "openclaw.json"
            if not config_path.is_file():
                raise RuntimeError(f"openclaw onboard did not produce a config at {config_path}: {stderr}")
            base_cfg = json.loads(config_path.read_text())
            config_path.write_text(json.dumps(self._build_openclaw_config(base_cfg, rollout_id), indent=2) + "\n")

            cmd = [
                *self.config.command_parts,
                "agent",
                "--local",
                "--json",
                "--agent",
                self.config.openclaw_agent_id,
                "--thinking",
                self.config.thinking,
                "--model",
                self._effective_model(),
                "--message",
                prompt,
                *self.config.extra_args,
            ]
            # Run OpenClaw, salvaging a partial transcript if the run is cut short. It can be cut
            # short two ways: our own self.config.timeout (raises TimeoutError), or an outer
            # harness/sandbox timeout that SIGTERMs this whole process during its grace window before
            # SIGKILL. In the SIGTERM case a plain `finally` would not run in time and the workdir
            # would be lost, so we stop waiting on the signal, read OpenClaw's incrementally-written
            # session off disk, and return it. Returning quickly lets the harness still write the
            # response before the SIGKILL, so no harness change is needed.
            code, stdout, stderr = None, "", ""
            self._install_sigterm_handler()
            sigterm_hit = asyncio.Event()
            self.sigterm_events.add(sigterm_hit)
            run_task = asyncio.ensure_future(
                self._run_exec(cmd, cwd=str(work_dir), env=env, timeout=self.config.timeout)
            )
            try:
                term_task = asyncio.ensure_future(sigterm_hit.wait())
                done, _ = await asyncio.wait({run_task, term_task}, return_when=asyncio.FIRST_COMPLETED)
                term_task.cancel()
                if run_task in done:
                    code, stdout, stderr = run_task.result()
                else:
                    LOG.warning("openclaw received SIGTERM; salvaging partial session")
                    run_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await run_task
            except TimeoutError:
                LOG.warning("openclaw timed out after %ds; salvaging partial session", self.config.timeout)
            finally:
                self.sigterm_events.discard(sigterm_hit)

            if code:
                LOG.warning("openclaw exited %d: %s", code, stderr)
            if stdout:
                LOG.debug("openclaw stdout (%d chars): %s", len(stdout), stdout[:2000])

            fallback_items, usage = parse_openclaw_output(stdout)
            envelope = _decode_last_json_dict_suffix(stdout)

            # On a normal finish the envelope points at the session file; on a cut-short run there is
            # no envelope, so fall back to locating the partial session that OpenClaw wrote to disk.
            output_items: list[Any] = []
            session_path = self._session_file(envelope) or self._find_partial_session(home)
            if session_path and session_path.is_file():
                session_text = session_path.read_text(errors="replace")
                output_items = parse_openclaw_session(session_text)
                if observation_collector is not None:
                    try:
                        session_events = parse_openclaw_session_events(session_text)
                        native_session_id = next(
                            (
                                event.get("id")
                                for event in session_events
                                if event.get("type") == "session" and isinstance(event.get("id"), str)
                            ),
                            session_path.stem,
                        )
                        session_tree, tree_gaps = discover_openclaw_session_tree(
                            home / ".openclaw" / "agents",
                            native_session_id,
                        )
                        observation_collector(native_session_id, session_events, session_tree, tree_gaps)
                    except Exception:
                        LOG.exception("failed to record OpenClaw session artifact")
            if not output_items:
                output_items = fallback_items
            return output_items, usage, self.config.model
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        rollout_id: Optional[str] = None,
        observation_collector: Optional[
            Callable[[str, list[dict[str, Any]], OpenClawSessionTree, list[ObservationGap]], None]
        ] = None,
        output_collector: Optional[Callable[[list[Any]], None]] = None,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        try:
            output_items, usage, model_name = await self._run_openclaw(
                user_message,
                system_prompt,
                rollout_id=rollout_id,
                observation_collector=observation_collector,
            )
        except TimeoutError:
            LOG.warning("OpenClaw timed out, padding empty output so the rollout scores instead of erroring")
            output_items, usage, model_name = [], {"input_tokens": 0, "output_tokens": 0}, self.config.model

        if output_collector is not None:
            output_collector(list(output_items))
        if not output_items:
            LOG.warning("OpenClaw produced no assistant message. Padding empty output")
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
        cached_tokens = usage.get("cached_tokens", 0)

        return NeMoGymResponse(
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
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        try:
            session_id = request.session.get(_SANDBOX_SESSION_KEY)
            has_session = isinstance(request.session, Mapping) and _SANDBOX_SESSION_KEY in request.session
        except (AssertionError, AttributeError):
            session_id, has_session = None, False
        if has_session:
            if not isinstance(session_id, str):
                raise HTTPException(409, "Invalid OpenClaw session marker")
            state = self._sandbox_sessions.get(session_id)
            rollout_id = request.path_params.get("rollout_id")
            if state is None or state.seed.episode_id.capture_key != rollout_id:
                raise HTTPException(409, "OpenClaw activation does not match the seeded session and rollout route")
            if state.activated or state.closing:
                raise HTTPException(409, "OpenClaw sandbox sessions support one activation")
            prompt, system = self._sandbox_input(body)
            state.activated = True
            state.task = asyncio.create_task(self._sandbox_response(state, body, prompt=prompt, system=system))
            try:
                return await asyncio.shield(state.task)
            except asyncio.CancelledError:
                if not state.task.done() and not state.task.cancelling():
                    state.task.cancel()
                raise
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
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
        session_id: Optional[str] = None
        session_events: list[dict[str, Any]] = []
        session_tree: OpenClawSessionTree = []
        tree_gaps: list[ObservationGap] = []
        input_items: list[Any] = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)]
            if isinstance(body.input, str)
            else list(body.input)
        )
        observed_output: list[Any] = []

        def collect(
            value: str,
            events: list[dict[str, Any]],
            tree: OpenClawSessionTree,
            gaps: list[ObservationGap],
        ) -> None:
            nonlocal session_id, session_events, session_tree, tree_gaps
            session_id = value
            session_events = events
            session_tree = tree
            tree_gaps = gaps

        def collect_output(value: list[Any]) -> None:
            observed_output.extend(value)

        response = await self._create_response(
            body,
            rollout_id=rollout_id,
            observation_collector=collect,
            output_collector=collect_output,
        )
        try:
            if session_tree:
                tree_inputs = []
                for invocation_id, parent_id, events in session_tree:
                    conversation = openclaw_session_conversation(
                        events,
                        input_items=input_items if parent_id is None else None,
                        fallback_output=observed_output if parent_id is None else None,
                    )
                    tree_inputs.append((invocation_id, parent_id, conversation, events))
                observations = build_openclaw_observation_tree(
                    tree_inputs,
                    model_ref=self.config.model_server,
                )
                observations.gaps.extend(tree_gaps)
            else:
                transcript_available = any(event.get("type") == "message" for event in session_events)
                observations = build_openclaw_observations(
                    session_id or response.id,
                    openclaw_session_conversation(
                        session_events,
                        input_items=input_items,
                        fallback_output=observed_output,
                    ),
                    session_events,
                    transcript_available=transcript_available,
                    model_ref=self.config.model_server,
                )
                if any(gap.code == "subagent_hierarchy_unavailable" for gap in tree_gaps):
                    observations.gaps = [
                        gap for gap in observations.gaps if gap.code != "subagent_hierarchy_unavailable"
                    ]
                observations.gaps.extend(tree_gaps)
        except Exception:
            LOG.exception("failed to build OpenClaw observations")
            observations = AgentObservationBundle(
                source=OPENCLAW_OBSERVATION_SOURCE,
                gaps=[ObservationGap(code="observation_capture_failed")],
            )
        return AgentEpisode(response=response, observations=observations)

    async def run(self, request: Request, body: OpenClawAgentRunRequest) -> OpenClawAgentVerifyResponse:
        try:
            if isinstance(request.session, Mapping) and _SANDBOX_SESSION_KEY in request.session:
                raise HTTPException(409, "Native OpenClaw sessions must use EnvironmentServer /run")
        except (AssertionError, AttributeError):
            pass
        if self.config.resources_server is None:
            raise HTTPException(
                422, "Direct OpenClaw /run requires resources_server; use EnvironmentServer /run for native sessions"
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

            result = verify_json | {"turns_used": turns, "finished_naturally": naturally}
            if observations is not None:
                result["ng_agent_observations"] = observations.model_dump(mode="json")
            return OpenClawAgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    OpenClawAgent.run_webserver()
