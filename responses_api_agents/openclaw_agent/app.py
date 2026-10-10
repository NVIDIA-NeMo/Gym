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
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, Callable, ClassVar, Optional
from uuid import uuid4

import psutil
from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr

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
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider
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


def _sandbox_prepare_command(workdir: str, directory: str, runtime: str) -> str:
    # Bootstrap with POSIX sh; the runtime installer itself requires Bash.
    bootstrap = """set -eu
set --
command -v python3 >/dev/null 2>&1 || set -- "$@" python3
command -v bash >/dev/null 2>&1 || set -- "$@" bash
if [ "$#" -gt 0 ]; then
    [ "$(id -u)" = 0 ] || { echo "OpenClaw sandbox execution requires $*: preinstall these tools or use a root image." >&2; exit 1; }
    if command -v apk >/dev/null 2>&1; then
        apk add --no-cache "$@"
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
    else
        echo "OpenClaw sandbox execution requires $*: preinstall these tools (automatic installation requires apt-get or apk)." >&2
        exit 1
    fi
fi
"""
    check_paths = shlex.join(
        ["python3", "-I", "-c", _SANDBOX_PATH_CHECK, workdir, str(Path(directory).parent), runtime]
    )
    return bootstrap + f"{check_paths} && mkdir -p {shlex.quote(directory + '/home/.openclaw')}"


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
    timeout: int = Field(default=900, gt=0)
    model_timeout_seconds: int = Field(default=600, gt=0)
    extra_args: list[str] = []
    openclaw_config: dict[str, Any] = Field(default_factory=dict)
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    # required: every config must pin an explicit version so runs are reproducible and cannot silently drift
    openclaw_version: str
    sandbox_provider: str | None = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_install_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)

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

    ray_enabled = False

    config: OpenClawAgentConfig
    sem: Semaphore = None
    sigterm_events: set = Field(default_factory=set)
    sigterm_handler_installed: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # deny the interactive "message" channel so the headless agent finishes
    _HEADLESS_TOOL_DENY: ClassVar[tuple[str, ...]] = ("message",)

    _local_setup_task: asyncio.Task[None] | None = PrivateAttr(default=None)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> OpenClawSandboxSession:
        """Install a pinned runtime in the borrowed or agent-owned session sandbox."""
        owns_sandbox = body.sandbox_access is None
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise HTTPException(422, "OpenClaw requires sandbox_access or a configured sandbox_provider")
            spec = SandboxSpec(**{"workdir": "/app", **self.config.sandbox_config})
            workdir = spec.workdir
            provider_ref = self.config.sandbox_provider
        else:
            if not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
                raise HTTPException(422, "OpenClaw requires direct SandboxAccess")
            workdir = body.sandbox_access.workdir
            provider_ref = body.sandbox_access.connection.provider_config_ref
        if not isinstance(workdir, str) or not workdir.startswith("/") or workdir in ("/", "/tmp"):
            raise HTTPException(422, "OpenClaw workdir must be absolute and separate from /tmp runtime storage")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise HTTPException(
                422, "OpenClaw sandbox execution supports its own sandbox tools, not required HTTP/MCP tools"
            )
        if self.config.model_server is None:
            raise HTTPException(422, "OpenClaw sandbox execution requires a sandbox-reachable Gym model_server")
        if not self.config.openclaw_version or not re.fullmatch(
            r"\d+\.\d+\.\d+(?:-\d+)?", self.config.openclaw_version
        ):
            raise HTTPException(
                422, "OpenClaw sandbox execution requires an exact openclaw_version, for example 2026.6.11"
            )
        if self.config.openclaw_config or self.config.max_output_tokens is not None or self.config.node_bin_dir:
            raise HTTPException(
                422,
                "OpenClaw sandbox execution does not support openclaw_config, max_output_tokens, or node_bin_dir overrides",
            )
        if self.config.openclaw_agent_id != "main":
            raise HTTPException(422, "OpenClaw sandbox execution currently requires openclaw_agent_id=main")
        if self.config.command != "openclaw" or self.config.extra_args or self.config.env:
            raise HTTPException(
                422, "OpenClaw sandbox execution does not support command, extra_args, or env overrides"
            )

        provider = create_provider(resolve_provider_config(provider_ref, get_global_config_dict()))
        if owns_sandbox:
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(body.sandbox_access.connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        directory = f"/tmp/nemo-gym-openclaw-sessions/{uuid4().hex}"
        runtime = f"/tmp/nemo-gym-openclaw-node-22.19.0-{self.config.openclaw_version}"
        state = OpenClawSandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox, session_dir=directory, workdir=workdir, harness="OpenClaw", owns_sandbox=owns_sandbox
            ),
            runtime=runtime,
        )
        prepared_directory = False
        try:
            if owns_sandbox:
                await sandbox.start(spec)
                # Providers need not create SandboxSpec.workdir. Never prepare a borrowed task here.
                workspace = await sandbox.exec(f"mkdir -p -- {shlex.quote(workdir)}", cwd="/", timeout_s=30)
                if workspace.return_code != 0 or workspace.error_type:
                    raise RuntimeError(
                        f"Cannot create OpenClaw sandbox workdir {workdir}: {workspace.stderr or workspace.stdout}"
                    )
            prepared = await sandbox.exec(
                _sandbox_prepare_command(workdir, directory, runtime),
                timeout_s=self.config.sandbox_install_timeout_seconds,
            )
            if prepared.return_code != 0 or prepared.error_type:
                raise RuntimeError(
                    f"OpenClaw sandbox preparation failed (exit {prepared.return_code}, error={prepared.error_type}): "
                    f"stderr={(prepared.stderr or '')[-16000:]}; stdout={(prepared.stdout or '')[-16000:]}"
                )
            prepared_directory = True
            # Install only the agent runtime in the existing task sandbox.
            # Resources has already prepared the task repository and its dependencies.
            installer = "install_openclaw_runtime.sh"
            await sandbox.upload(Path(__file__).with_name(installer), f"{directory}/{installer}")
            installed = await sandbox.exec(
                f"bash {shlex.quote(directory + '/' + installer)} {shlex.quote(runtime)} "
                f"{shlex.quote(self.config.openclaw_version)}",
                cwd=workdir,
                timeout_s=self.config.sandbox_install_timeout_seconds,
            )
            if installed.return_code != 0 or installed.error_type:
                raise RuntimeError(
                    f"OpenClaw runtime setup failed (exit {installed.return_code}, error={installed.error_type}): "
                    f"bash {directory}/{installer} {runtime} {self.config.openclaw_version}\n"
                    f"stderr={(installed.stderr or '')[-16000:]}; stdout={(installed.stdout or '')[-16000:]}"
                )
            await sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), f"{directory}/sandbox_runner.py")
        except BaseException as error:
            try:
                if prepared_directory or owns_sandbox:
                    await state.close(self.config.session_close_timeout_seconds)
                else:
                    # The path check may have rejected overlap with task-owned storage.
                    await sandbox.disconnect()
            except BaseException:
                LOG.exception("OpenClaw seed cleanup failed; retaining session %s", body.agent_session_id)
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Confirm teardown and return observations without destroying the borrowed sandbox."""
        assert isinstance(state, OpenClawSandboxSession)
        await state.close(self.config.session_close_timeout_seconds)
        observations = state.observations or AgentObservationBundle(
            source="openclaw", gaps=[ObservationGap(code="agent_activation_interrupted")]
        )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, agent_observations=observations
        )

    def _prompt_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
        """Normalize prompts identically for local and borrowed-sandbox execution."""
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        prompt, input_system = _extract_instruction(items)
        system = "\n\n".join(part for part in (self.config.system_prompt, body.instructions, input_system) if part)
        return prompt, system

    def _sandbox_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
        """Validate and normalize input before consuming the session's activation."""
        if body.model is not None and body.model != self.config.model:
            raise HTTPException(422, "OpenClaw sandbox execution model must match the configured model")
        extras = set(body.model_extra or {})
        if extras:
            raise HTTPException(
                422, f"OpenClaw sandbox execution does not support extra request fields: {sorted(extras)}"
            )
        if "chat_template_kwargs" in (body.metadata or {}):
            raise HTTPException(422, "Configure chat_template_kwargs on the Gym model server for OpenClaw")
        supported = {"input", "instructions", "model"}
        for name, field in type(body).model_fields.items():
            if name in supported:
                continue
            value = getattr(body, name)
            if name in ("stream", "background") and value is False:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"OpenClaw sandbox execution does not support request field {name}")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"]):
            raise HTTPException(
                422, "OpenClaw sandbox execution accepts one text user prompt with an optional system message"
            )
        for item in items:
            if not isinstance(item.content, str) and any(
                (part.get("type") if isinstance(part, dict) else getattr(part, "type", None)) != "input_text"
                for part in item.content
            ):
                raise HTTPException(422, "OpenClaw sandbox execution only supports text input")
        prompt, system = self._prompt_input(body)
        if not prompt.strip():
            raise HTTPException(422, "OpenClaw sandbox execution requires a nonempty user prompt")
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
        session_config = self._build_openclaw_config({}, state.request.episode_id.capture_key)
        # OpenClaw conservatively disables stream usage for custom origins. Gym
        # supports it; request the final usage chunk instead of retaining zero counters.
        session_config["models"]["providers"]["nemo"]["models"][0]["compat"] = {
            "supportsUsageInStreaming": True,
        }
        session_config.update(
            {
                "agents": {
                    "defaults": {
                        "workspace": state.session.workdir,
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
        command = [
            f"{state.runtime}/node/bin/node",
            f"{state.runtime}/openclaw/node_modules/openclaw/openclaw.mjs",
            "agent",
            "--local",
            "--json",
            "--agent",
            "main",
            "--session-id",
            state.session.session_dir.rsplit("/", 1)[-1],
            "--thinking",
            self.config.thinking,
            "--model",
            self._effective_model(),
            "--message-file",
            f"{state.session.session_dir}/prompt.txt",
            "--timeout",
            str(self.config.timeout),
        ]
        payload = {
            "config": session_config,
            "directory": state.session.session_dir,
            "command": command,
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "cwd": state.session.workdir,
            "env": {
                "HOME": f"{state.session.session_dir}/home",
                "XDG_CACHE_HOME": f"{state.session.session_dir}/home/.cache",
                "OPENCLAW_STATE_DIR": f"{state.session.session_dir}/home/.openclaw",
                "OPENCLAW_CONFIG_PATH": f"{state.session.session_dir}/home/.openclaw/openclaw.json",
                "OPENCLAW_TELEMETRY": "0",
                "CLAWHUB_DISABLE_TELEMETRY": "1",
                "OPENCLAW_EXEC_SHELL_SNAPSHOT": "0",
            },
        }
        try:
            async with self.sem:
                await state.execute(
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
        response = await self._collect_sandbox_response(state, body, prompt=prompt, system=system)
        if response.error is not None:
            # Preserve the transcript in session observations, but let the environment
            # classify runtime/provider failures before it invokes the verifier.
            raise HTTPException(502, response.error.message)
        return response

    async def _collect_sandbox_response(
        self,
        state: OpenClawSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        prompt: str,
        system: str,
    ) -> NeMoGymResponse:
        gaps = []
        artifacts = state.session.artifacts or {}
        stdout = artifacts.get("stdout.log", "")
        if "stdout.log" not in artifacts:
            gaps.append(ObservationGap(code="agent_stdout_unavailable"))
        session_id = state.session.session_dir.rsplit("/", 1)[-1]
        transcript_name = f"home/.openclaw/agents/main/sessions/{session_id}.jsonl"
        events = parse_openclaw_session_events(artifacts.get(transcript_name, ""))
        if transcript_name not in artifacts:
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
            # Harness totals retain valid subtotals without coercing malformed cache values.
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
        result = state.session.cleanup
        runtime = state.runtime_info
        if runtime is None:
            gaps.append(ObservationGap(code="runtime_info_unavailable"))
        if result is not None and result["return_code"] is None:
            gaps.append(ObservationGap(code="worker_exit_code_unavailable"))
        error = result["error"] if result else "OpenClaw activation ended without a cleanup receipt"
        last = assistants[-1] if assistants else {}
        if last.get("stopReason") == "error" or (
            last.get("stopReason") == "aborted" and not (result and result["timed_out"])
        ):
            error = error or last.get("errorMessage") or f"OpenClaw model call {last['stopReason']}"
        if result and result["return_code"] and not result["timed_out"]:
            stderr = artifacts.get("stderr.log", "stderr unavailable")[-4000:]
            error = error or f"OpenClaw exited {result['return_code']}: {stderr}"
        incomplete = bool(result and result["timed_out"]) or last.get("stopReason") == "length"
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
                **({"harness_hostname": runtime.hostname, "harness_pid": str(runtime.pid)} if runtime else {}),
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
            # Sandbox sessions enable only local file/process tools, so there is
            # exactly one invocation; child sessions cannot be spawned.
            state.observations.gaps = [
                gap for gap in state.observations.gaps if gap.code != "subagent_hierarchy_unavailable"
            ]
            state.observations.records[0].status = status
            state.observations.gaps.extend(gaps)
        except Exception:
            LOG.exception("Could not build OpenClaw observations")
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
            # OpenClaw uses this provider request deadline for its stream-idle
            # watchdog too. Gym may buffer a long reasoning call before SSE delivery.
            # Preserve explicit provider settings on the legacy local CLI path.
            nemo.setdefault("timeoutSeconds", self.config.model_timeout_seconds)
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

    def _clear_failed_local_setup(self, task: asyncio.Task[None]) -> None:
        # All shielded waiters may have gone away before the installer fails.
        failed = task.cancelled() or task.exception() is not None
        if failed and self._local_setup_task is task:
            self._local_setup_task = None

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
        # Local callers still get automatic installation. Sandbox sessions never
        # enter this path, so their host does not need OpenClaw, Node, or npm.
        if self._local_setup_task is None:
            self._local_setup_task = asyncio.create_task(
                asyncio.to_thread(ensure_openclaw, self.config.openclaw_version, node_bin_dir=self.config.node_bin_dir)
            )
            self._local_setup_task.add_done_callback(self._clear_failed_local_setup)
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

        user_message, system_prompt = self._prompt_input(body)

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
        session_id = self._agent_session_id_from_request(request)
        try:
            legacy_marker = _SANDBOX_SESSION_KEY in request.session
        except (AssertionError, AttributeError):
            legacy_marker = False
        if legacy_marker:
            raise HTTPException(409, "Obsolete OpenClaw session cookie; seed a new agent session")
        if session_id is not None:
            state = self._require_agent_session(session_id)
            assert isinstance(state, OpenClawSandboxSession)
            if state.request.episode_id.capture_key != request.path_params.get("rollout_id"):
                raise HTTPException(409, "OpenClaw activation does not match the seeded session and rollout route")
            prompt, system = self._sandbox_input(body)
            if state.task is None:
                # Bind the immutable request before yielding, so identical retries
                # share one invocation and different requests cannot replace it.
                state.activation_request = body.model_copy(deep=True)
                state.task = asyncio.create_task(
                    self._sandbox_response(state, state.activation_request, prompt=prompt, system=system)
                )
            elif body != state.activation_request:
                raise HTTPException(409, "OpenClaw sandbox sessions support one activation; retry the same request")
            # Session close owns cancellation, not an individual HTTP waiter.
            return (await asyncio.shield(state.task)).model_copy(deep=True)
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
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "OpenClaw sandbox sessions must use EnvironmentServer /run")
        try:
            if isinstance(request.session, Mapping) and _SANDBOX_SESSION_KEY in request.session:
                raise HTTPException(409, "OpenClaw sandbox sessions must use EnvironmentServer /run")
        except (AssertionError, AttributeError):
            pass
        if self.config.resources_server is None:
            raise HTTPException(
                422, "Direct OpenClaw /run requires resources_server; use EnvironmentServer /run for sandbox sessions"
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
