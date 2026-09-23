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
import json
import logging
import os
import re
import shlex
import shutil
import signal
import tempfile
from asyncio import Semaphore
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from time import monotonic, time
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.base_resources_server import NEMO_GYM_MCP_METADATA_KEY, BaseRunRequest, BaseVerifyResponse
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
from nemo_gym.global_config import SKILLS_REF_KEY_NAME, get_first_server_config_dict
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
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap, ToolCallObservation
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import get_global_config_dict, get_response_json, raise_for_status
from nemo_gym.skills import stage_skills
from responses_api_agents.codex_agent.sandbox import CodexSandboxSession
from responses_api_agents.codex_agent.setup_codex import ensure_codex


LOG = logging.getLogger(__name__)
_SANDBOX_SESSION_KEY = "nemo_gym_codex_sandbox_session"
_COMPACTION_ADVISORY = (
    "Heads up: Long threads and multiple compactions can cause the model to be less accurate. "
    "Start a new thread when possible to keep threads small and targeted."
)


def _sandbox_prepare_command(workdir: str, directory: str, runtime: str) -> str:
    validate = """
from pathlib import Path
import sys
workdir = Path(sys.argv[1]).resolve(strict=True)
if not workdir.is_dir():
    raise ValueError("Codex workdir must be an existing directory")
for value in sys.argv[2:]:
    owned = Path(value).resolve()
    if workdir == owned or workdir in owned.parents or owned in workdir.parents:
        raise ValueError("Codex task workdir and adapter paths must be disjoint after resolving symlinks")
"""
    return (
        f"python3 -I -c {shlex.quote(validate)} {shlex.quote(workdir)} "
        f"{shlex.quote(directory)} {shlex.quote(runtime)} && mkdir -p {shlex.quote(directory + '/home/.codex')}"
    )


def _toml_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # JSON string escaping is a valid TOML basic string.
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def toml_dumps(data: dict[str, Any], _prefix: str = "") -> str:
    """Serialize a nested dict of scalars/lists/dicts to TOML (the subset Codex config uses)."""
    lines: list[str] = []
    tables: list[tuple[str, dict]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
    chunks = ["\n".join(lines)] if lines else []
    for key, value in tables:
        full_key = f"{_prefix}.{_toml_key(key)}" if _prefix else _toml_key(key)
        body = toml_dumps(value, full_key)
        chunks.append(f"[{full_key}]" + (f"\n{body}" if body else ""))
    return "\n\n".join(chunks)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _mcp_result_text(item: dict[str, Any]) -> str:
    if item.get("error"):
        return f"error: {item['error']}"
    result = item.get("result")
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if any(texts):
                return "".join(texts)
        return json.dumps(result)
    return "" if result is None else str(result)


def parse_exec_jsonl(
    stdout: str,
    *,
    structured_reasoning: bool = False,
    include_partial: bool = False,
    conservative_usage_details: bool = False,
) -> tuple[list[Any], dict]:
    """Convert ``codex exec --json`` JSONL stdout into (output_items, metadata).

    Codex emits ``item.completed`` events for each unit of work (assistant messages, reasoning,
    shell commands, MCP tool calls, file changes, ...) and a terminal ``turn.completed`` carrying
    its available aggregate token usage; failed stream attempts can be omitted even after a
    successful retry. Tool-shaped items are mapped to a
    ``function_call`` + ``function_call_output`` pair so verifiers see one uniform trajectory
    shape across agent harnesses; reasoning is buffered and prepended to the next assistant
    message inside ``<think>`` tags (mirroring the Claude Code agent). Native callers use
    ``conservative_usage_details`` because the pinned CLI defaults absent backend cache/reasoning
    counters to zero: only positive integers establish measured details in this artifact.
    """
    output_items: list[Any] = []
    buffered_think: Optional[str] = None
    metadata: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0}
    if conservative_usage_details:
        metadata.update(cached_input_tokens=None, reasoning_tokens=None)
    usage_seen = False
    errors: list[str] = []
    unfinished: dict[str, dict[str, Any]] = {}

    def _add_tool_pair(item: dict[str, Any], name: str, arguments: dict[str, Any], output: str) -> None:
        call_id = str(item.get("id") or f"call-{uuid4().hex[:8]}")
        status = "completed" if item.get("status") != "failed" else "incomplete"
        output_items.append(
            NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(arguments),
                call_id=call_id,
                name=name,
                type="function_call",
                id=call_id,
                status=status,
            )
        )
        output_items.append(
            NeMoGymFunctionCallOutput(type="function_call_output", call_id=call_id, output=output, status="completed")
        )

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        etype = event.get("type")

        if etype == "turn.completed":
            usage = event.get("usage") or {}
            metadata["input_tokens"] += int(usage.get("input_tokens") or 0)
            metadata["output_tokens"] += int(usage.get("output_tokens") or 0)
            for source, target in (
                ("cached_input_tokens", "cached_input_tokens"),
                ("reasoning_output_tokens", "reasoning_tokens"),
            ):
                value = usage.get(source)
                if conservative_usage_details:
                    # Zero is ambiguous in Codex JSONL, not a known backend measurement.
                    if type(value) is not int or value <= 0 or (usage_seen and metadata[target] is None):
                        metadata[target] = None
                    else:
                        metadata[target] = (metadata[target] or 0) + value
                else:
                    metadata[target] += int(value or 0)
            usage_seen = True
            continue

        if etype == "turn.failed":
            message = (event.get("error") or {}).get("message") or "turn failed"
            errors.append(message)
            continue

        item = event.get("item")
        if include_partial and isinstance(item, dict) and item.get("id"):
            if etype in ("item.started", "item.updated"):
                unfinished[item["id"]] = unfinished.get(item["id"], {}) | item
            elif etype == "item.completed":
                unfinished.pop(item["id"], None)
        if etype != "item.completed":
            continue

        item = event.get("item")
        if not isinstance(item, dict):
            continue
        itype = item.get("type")

        if itype == "agent_message":
            text = item.get("text") or ""
            if buffered_think:
                text = f"<think>\n{buffered_think}\n</think>\n\n{text}"
                buffered_think = None
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=str(item.get("id") or f"msg-{len(output_items)}"),
                    content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
        elif itype == "reasoning":
            think = item.get("text") or ""
            if think and structured_reasoning:
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=str(item.get("id") or f"reasoning-{len(output_items)}"),
                        summary=[{"type": "summary_text", "text": think}],
                    )
                )
            elif think:
                buffered_think = (buffered_think + "\n" + think) if buffered_think else think
        elif itype == "command_execution":
            output = item.get("aggregated_output") or ""
            exit_code = item.get("exit_code")
            if exit_code not in (None, 0):
                output = f"{output}\n[exit code: {exit_code}]"
            _add_tool_pair(item, "exec_command", {"cmd": item.get("command") or ""}, output)
        elif itype == "mcp_tool_call":
            _add_tool_pair(item, str(item.get("tool") or ""), item.get("arguments") or {}, _mcp_result_text(item))
        elif itype == "file_change":
            _add_tool_pair(item, "apply_patch", {"changes": item.get("changes")}, item.get("status") or "completed")
        elif itype == "web_search":
            _add_tool_pair(item, "web_search", {"query": item.get("query") or ""}, "")
        elif itype == "todo_list":
            _add_tool_pair(item, "update_plan", {"items": item.get("items") or []}, "")
        elif itype == "error":
            errors.append(item.get("message") or "unknown error")

    # Native cancellation can leave useful command output in an item.updated event.
    # Only synthesize items that never completed, retaining their latest snapshot by ID.
    for item in unfinished.values():
        partial, _ = parse_exec_jsonl(
            json.dumps({"type": "item.completed", "item": item}), structured_reasoning=structured_reasoning
        )
        output_items.extend(
            entry.model_copy(update={"status": "incomplete"}) if hasattr(entry, "status") else entry
            for entry in partial
        )

    # Some backends route the final answer through the reasoning channel (e.g. a vLLM reasoning
    # parser labeling the closing message as reasoning). If the run ends on buffered reasoning with
    # no assistant message after it, surface it as a think-tagged message rather than dropping it.
    if buffered_think:
        output_items.append(
            NeMoGymResponseOutputMessage(
                id=f"msg-{len(output_items)}",
                content=[
                    NeMoGymResponseOutputText(
                        type="output_text", text=f"<think>\n{buffered_think}\n</think>", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        )

    if errors:
        metadata["errors"] = errors
    return output_items, metadata


def _kill_process_group(proc: Any) -> None:
    """Kill the codex subprocess and every child in its process group.

    Killing only the direct child leaves the npm shim's vendored-binary child alive, holding the
    stdout pipe open — the post-kill ``communicate()`` would then block until the orphan exits.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()


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


class CodexAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: Optional[ResourcesServerRef] = None
    # When model_server is set, the Codex model provider's base_url is resolved from the Gym model
    # server's URL (every Gym model server speaks the streaming Responses dialect on /v1/responses).
    # When None, openai_base_url is used directly (default: the real OpenAI API).
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    # None -> omit `model` from the generated config and use the Codex CLI's own default. Gym model
    # servers substitute their configured model anyway; set explicitly for direct endpoints.
    model: Optional[str] = None
    model_context_window: Optional[int] = Field(default=None, gt=0, strict=True)
    model_auto_compact_token_limit: Optional[int] = Field(default=None, gt=0, strict=True)
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    sandbox_mode: Literal["read-only", "workspace-write", "danger-full-access"] = "danger-full-access"
    timeout: int = Field(default=600, gt=0)
    system_prompt: Optional[str] = None
    reasoning_effort: Optional[str] = None
    # Required: every config pins an explicit npm version so auto-install is reproducible and cannot
    # silently drift as new Codex releases land. Version bumps are then explicit, tested changes.
    codex_version: str
    # Working root handed to `codex exec --cd`. None -> a fresh temp dir per request, removed
    # afterwards, so rollouts cannot see each other's files.
    cwd: Optional[str] = None
    # Provider stream idle timeout. Gym model servers emit the synthesized SSE only once the full
    # response is computed, so the idle budget must cover an entire generation; None -> timeout * 1000.
    stream_idle_timeout_ms: Optional[int] = None
    # Extra config.toml content deep-merged over the generated base config (mcp_servers, features,
    # tools, model_verbosity, ...). Per-rollout Gym MCP entries take precedence on name collisions.
    extra_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_install_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    session_close_retry_window_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_context_budget(self) -> "CodexAgentConfig":
        """Check the explicit config pair against the pinned CLI's 90% compaction threshold."""
        if (
            self.model_context_window is not None
            and self.model_auto_compact_token_limit is not None
            and self.model_auto_compact_token_limit > self.model_context_window * 9 // 10
        ):
            raise ValueError("model_auto_compact_token_limit must not exceed 90% of model_context_window")
        return self


class CodexAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class CodexAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class CodexAgent(SimpleResponsesAPIAgent):
    config: CodexAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sandbox_sessions: dict[str, CodexSandboxSession] = PrivateAttr(default_factory=dict)
    _closed_sandbox_sessions: OrderedDict[str, tuple[EpisodeId, AgentCloseSessionResponse, float]] = PrivateAttr(
        default_factory=OrderedDict
    )
    _local_setup_task: asyncio.Task[None] | None = PrivateAttr(default=None)

    def _session_marker(self, request: Request) -> Optional[str]:
        try:
            session = request.session
        except (AssertionError, AttributeError):
            return None
        if not isinstance(session, Mapping) or _SANDBOX_SESSION_KEY not in session:
            return None
        marker = session[_SANDBOX_SESSION_KEY]
        if not isinstance(marker, str) or not marker:
            raise HTTPException(409, "Invalid Codex sandbox session marker")
        return marker

    async def seed_agent_session(self, request: Request, body: AgentSeedSessionRequest) -> AgentSeedSessionResponse:
        """Borrow the Resources-owned task sandbox and install a pinned Codex runtime inside it."""
        self._expire_closed_agent_sessions()
        if self._session_marker(request) is not None:
            raise HTTPException(409, "Codex session already exists")
        session_id = f"codex-{uuid4().hex}"
        state = await self._initialize_agent_session_state(session_id, body)
        self._sandbox_sessions[session_id] = state
        request.session[_SANDBOX_SESSION_KEY] = session_id
        return AgentSeedSessionResponse(agent_session_id=session_id)

    async def _initialize_agent_session_state(
        self, agent_session_id: str, body: AgentSeedSessionRequest
    ) -> CodexSandboxSession:
        # Match Hermes: session initialization owns the sandbox runtime setup,
        # with the same AgentSeedSessionRequest/SandboxAccess wire contracts.
        if self.config.num_workers not in (None, 1):
            raise HTTPException(422, "Native Codex sessions require num_workers=1")
        if body.sandbox_access is None or not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
            raise HTTPException(422, "Native Codex requires direct, Resources-owned SandboxAccess")
        if not body.sandbox_access.workdir.startswith("/") or ".." in PurePosixPath(body.sandbox_access.workdir).parts:
            raise HTTPException(422, "Codex sandbox workdir must be absolute")
        workdir = PurePosixPath(body.sandbox_access.workdir)
        if workdir in (PurePosixPath("/"), PurePosixPath("/tmp")) or str(workdir).startswith("/tmp/nemo-gym-codex"):
            raise HTTPException(422, "Codex runtime/session files must be outside SandboxAccess.workdir")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise HTTPException(422, "Native Codex supports its own sandbox tools, not required HTTP/MCP tools")
        if self.config.model_server is None:
            raise HTTPException(422, "Native Codex requires a sandbox-reachable Gym model_server")
        if not self.config.codex_version or not re.fullmatch(r"\d+\.\d+\.\d+", self.config.codex_version):
            raise HTTPException(422, "Native Codex requires an exact codex_version, for example 0.144.4")
        if self.config.cwd is not None or self.config.extra_config or self.config.openai_base_url is not None:
            raise HTTPException(
                422,
                "Native Codex uses SandboxAccess.workdir and Gym routing; cwd, extra_config, and openai_base_url overrides are unsupported",
            )
        if self.config.reasoning_effort is not None:
            raise HTTPException(422, "Native Codex cannot guarantee reasoning_effort for custom Gym models")
        if self.config.sandbox_mode != "danger-full-access":
            raise HTTPException(422, "Native Codex requires danger-full-access inside the Resources-owned sandbox")

        connection = body.sandbox_access.connection
        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        directory = f"/tmp/nemo-gym-codex-sessions/{agent_session_id}"
        runtime = f"/tmp/nemo-gym-codex-node-22.19.0-{self.config.codex_version}"
        prepared_ok = False
        try:
            prepare = _sandbox_prepare_command(body.sandbox_access.workdir, directory, runtime)
            prepared = await sandbox.exec(prepare, timeout_s=30)
            if prepared.return_code != 0 or prepared.error_type:
                raise RuntimeError(
                    f"Cannot prepare Codex session: {prepare}; exit={prepared.return_code}, "
                    f"error_type={prepared.error_type}; {prepared.stderr or prepared.stdout}"
                )
            prepared_ok = True
            # Install only the agent runtime in the existing task sandbox.
            # Resources has already prepared the task repository and its dependencies.
            installer = "install_codex_runtime.sh"
            await sandbox.upload(Path(__file__).with_name(installer), f"{directory}/{installer}")
            installed = await sandbox.exec(
                f"bash {shlex.quote(directory + '/' + installer)} {shlex.quote(runtime)} "
                f"{shlex.quote(self.config.codex_version)}",
                cwd=body.sandbox_access.workdir,
                timeout_s=self.config.sandbox_install_timeout_seconds,
            )
            if installed.return_code != 0 or installed.error_type:
                raise RuntimeError(
                    f"Codex installer exited {installed.return_code}, error_type={installed.error_type}: {installed.stderr or installed.stdout}"
                )
            await sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), f"{directory}/sandbox_runner.py")
        except BaseException:
            try:
                if prepared_ok:
                    cleanup = await sandbox.exec(f"rm -rf -- {shlex.quote(directory)}", timeout_s=30)
                    if cleanup.return_code != 0 or cleanup.error_type:
                        LOG.error("Could not remove failed Codex setup files: %s", cleanup)
            finally:
                await sandbox.disconnect()
            raise
        return CodexSandboxSession(body, sandbox, directory, runtime)

    async def close_agent_session(self, request: Request, body: AgentCloseSessionRequest) -> AgentCloseSessionResponse:
        """Confirm Codex teardown before allowing verification; never destroy the borrowed sandbox."""
        self._expire_closed_agent_sessions()
        session_id = self._session_marker(request)
        closed = self._closed_sandbox_sessions.get(session_id)
        if closed is not None and body.agent_session_id == session_id and body.episode_id == closed[0]:
            return closed[1]
        state = self._sandbox_sessions.get(session_id)
        if state is None or body.agent_session_id != session_id or body.episode_id != state.seed.episode_id:
            raise HTTPException(409, "Codex close does not match the seeded session and episode")
        await state.close(self.config.session_close_timeout_seconds)
        # Another close may have completed while this caller waited for the
        # session's cleanup lock. Reuse its receipt without renewing its expiry.
        self._expire_closed_agent_sessions()
        closed = self._closed_sandbox_sessions.get(session_id)
        if closed is not None:
            return closed[1]
        if self._sandbox_sessions.get(session_id) is not state:
            raise HTTPException(409, "Codex close receipt has expired")
        observations = state.observations or AgentObservationBundle(
            source="codex", gaps=[ObservationGap(code="agent_activation_interrupted")]
        )
        self._sandbox_sessions.pop(session_id, None)
        result = AgentCloseSessionResponse(agent_session_id=session_id, agent_observations=observations)
        # Keep the cookie as a tombstone so later activations cannot fall back
        # to host execution. Other sessions must not shorten the retry window.
        self._closed_sandbox_sessions[session_id] = (
            body.episode_id,
            result,
            monotonic() + self.config.session_close_retry_window_seconds,
        )
        return result

    def _expire_closed_agent_sessions(self) -> None:
        """Prune receipts by close time, never by traffic or retry access order."""
        now = monotonic()
        while self._closed_sandbox_sessions:
            if next(iter(self._closed_sandbox_sessions.values()))[2] > now:
                break
            self._closed_sandbox_sessions.popitem(last=False)

    def _sandbox_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
        """Validate and normalize input before consuming the session's activation."""
        unsupported = (
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
            "include",
            "service_tier",
            "safety_identifier",
            "user",
            "store",
            "stream_options",
            "prompt_cache_key",
            "prompt_cache_retention",
        )
        values = body.model_dump(mode="json")
        for name in unsupported:
            if values.get(name) is not None:
                raise HTTPException(422, f"Native Codex does not support request field {name}")
        if body.model is not None and body.model != self._effective_model():
            raise HTTPException(422, "Native Codex model is selected by agent and model-server configuration")
        unknown = set(body.model_extra or {})
        if unknown:
            raise HTTPException(422, f"Native Codex does not support extra request fields: {sorted(unknown)}")
        if body.tools or body.tool_choice != "auto" or not body.parallel_tool_calls or body.background:
            raise HTTPException(422, "Codex owns tool selection and execution policy")
        if (body.metadata or {}).get("chat_template_kwargs") is not None:
            raise HTTPException(422, "Configure chat_template_kwargs on the Gym model server for Codex")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"], ["developer", "user"]):
            raise HTTPException(422, "Native Codex accepts one text user prompt with an optional system message")
        for item in items:
            if not isinstance(item.content, str) and any(
                (part.get("type") if isinstance(part, dict) else getattr(part, "type", None)) != "input_text"
                for part in item.content
            ):
                raise HTTPException(422, "Native Codex only supports text input")

        def text(item):
            return (
                item.content
                if isinstance(item.content, str)
                else "".join(part["text"] if isinstance(part, dict) else part.text for part in item.content)
            )

        prompt = text(items[-1])
        input_system = text(items[0]) if len(items) == 2 else None
        system = "\n\n".join(part for part in (self.config.system_prompt, body.instructions, input_system) if part)
        return prompt, system

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    async def _ensure_local_runtime(self) -> None:
        if self._local_setup_task is None:
            self._local_setup_task = asyncio.create_task(asyncio.to_thread(ensure_codex, self.config.codex_version))
        setup = self._local_setup_task
        try:
            await asyncio.shield(setup)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._local_setup_task is setup:
                self._local_setup_task = None
            raise

    async def _sandbox_response(
        self,
        state: CodexSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        prompt: str,
        system: str,
    ) -> NeMoGymResponse:
        config = self._build_config(
            self._resolve_call_base_url(state.seed.episode_id.capture_key), developer_instructions=system
        )
        await state.upload_text("home/.codex/config.toml", toml_dumps(config))
        command = self._build_command("-", state.seed.sandbox_access.workdir)
        command[0:1] = [
            f"{state.runtime}/node/bin/node",
            f"{state.runtime}/codex/node_modules/@openai/codex/bin/codex.js",
        ]
        payload = {
            "directory": state.directory,
            "command": command,
            "prompt": prompt,
            "cwd": state.seed.sandbox_access.workdir,
            "env": {
                "HOME": f"{state.directory}/home",
                "CODEX_HOME": f"{state.directory}/home/.codex",
                "XDG_CACHE_HOME": f"{state.directory}/home/.cache",
                # Gym receives the calls; never copy the direct OpenAI credential into this path.
                "OPENAI_API_KEY": "gym",  # pragma: allowlist secret
            },
            "timeout": self.config.timeout,
            "cleanup_timeout": self.config.session_close_timeout_seconds / 3,
        }
        raw = ""
        failure = None
        try:
            async with self.sem:
                raw = await state.execute(
                    payload, timeout=self.config.timeout, close_timeout=self.config.session_close_timeout_seconds
                )
        except BaseException as exc:
            failure = exc
            try:
                raw = await state.read_text("events.jsonl")
            except Exception:
                LOG.warning("Codex event transcript unavailable after interrupted activation", exc_info=True)
        events = []
        for line in raw.splitlines():
            try:
                observed_at, event = json.loads(line)
                if isinstance(event, dict):
                    events.append((float(observed_at), event))
            except (ValueError, TypeError):
                LOG.warning("Skipping malformed Codex event record")
        result = state.result
        terminal_events = [
            event.get("type") for _, event in events if event.get("type") in ("turn.completed", "turn.failed")
        ]
        parse_events = []
        startup_warnings = []
        compaction_warnings = []
        started = False
        successful_exit = (
            result is not None
            and result.return_code == 0
            and not result.timed_out
            and not result.error
            and failure is None
            and terminal_events[-1:] == ["turn.completed"]
            and any(event.get("type") == "turn.started" for _, event in events)
        )
        for _, event in events:
            started = started or event.get("type") == "turn.started"
            item = event.get("item") or {}
            message = item.get("message") or ""
            # Codex 0.144.4 serializes these known warnings as error items.
            # Its compaction warning can occur during a successful turn. Keep
            # exact advisories visible without masking other errors or failed exits.
            if successful_exit and event.get("type") == "item.completed" and item.get("type") == "error":
                if message == _COMPACTION_ADVISORY:
                    compaction_warnings.append(message)
                    continue
                if (
                    not started
                    and message.startswith("Model metadata for `")
                    and message.endswith(
                        "` not found. Defaulting to fallback metadata; this can degrade performance and cause issues."
                    )
                ):
                    startup_warnings.append(message)
                    continue
            parse_events.append(event)
        output, usage = parse_exec_jsonl(
            "\n".join(json.dumps(event) for event in parse_events),
            structured_reasoning=True,
            include_partial=True,
            conservative_usage_details=True,
        )
        if (startup_warnings or compaction_warnings) and usage.get("errors"):
            usage["errors"] = [*startup_warnings, *compaction_warnings, *usage["errors"]]
            startup_warnings = []
            compaction_warnings = []
        error = result.error if result else "Codex runner result unavailable"
        errors = usage.get("errors") or []
        if errors:
            error = error or "; ".join(errors)
        if result and result.return_code != 0 and not result.timed_out:
            error = error or f"Codex exited with code {result.return_code}"
        completed = any(event.get("type") == "turn.completed" for _, event in events)
        if result and not completed and not result.timed_out:
            error = error or "Codex ended without turn.completed"
        status = "failed" if error else "incomplete" if result.timed_out else "completed"
        conversation = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        if system:
            conversation.insert(0, NeMoGymEasyInputMessage(role="system", content=system))
        gaps = [
            ObservationGap(code="model_call_join_key_unavailable", detail="CLI events omit model response IDs"),
            ObservationGap(code="subagent_hierarchy_unavailable"),
            ObservationGap(code="compaction_observations_unavailable"),
            *(ObservationGap(code="model_metadata_fallback", detail=warning) for warning in startup_warnings),
            *(ObservationGap(code="compaction_accuracy_advisory", detail=warning) for warning in compaction_warnings),
        ]
        for field, code in (
            ("cached_input_tokens", "cached_token_usage_unavailable"),
            ("reasoning_tokens", "reasoning_token_usage_unavailable"),
        ):
            if usage[field] is None:
                gaps.append(ObservationGap(code=code, detail="CLI detail counters are absent or defaulted to zero"))
        stream_errors = [event for _, event in events if event.get("type") == "error"]
        if not completed or stream_errors:
            gaps.append(
                ObservationGap(
                    code="partial_model_usage_unavailable",
                    detail=(
                        "CLI totals may omit model calls interrupted by stream errors, including recovered retries"
                        if stream_errors
                        else "CLI emitted no turn.completed usage"
                    ),
                )
            )
        if failure is not None:
            gaps.append(ObservationGap(code="agent_activation_interrupted", detail=type(failure).__name__))
        if not events:
            gaps.append(ObservationGap(code="agent_transcript_unavailable"))
        tools = []
        starts = {}
        for observed_at, event in events:
            item = event.get("item") or {}
            if item.get("type") not in (
                "command_execution",
                "mcp_tool_call",
                "file_change",
                "web_search",
                "todo_list",
            ):
                continue
            item_id = item.get("id")
            if not item_id:
                continue
            if event.get("type") == "item.started":
                starts[item_id] = observed_at
            if event.get("type") == "item.completed":
                start = starts.pop(item_id, None)
                tools.append(
                    ToolCallObservation(
                        invocation_id=state.seed.episode_id.capture_key,
                        tool_call_id=item_id,
                        tool_name={"command_execution": "exec_command", "file_change": "apply_patch"}.get(
                            item["type"], item["type"]
                        ),
                        started_at=start,
                        completed_at=observed_at,
                        status="failed" if item.get("status") == "failed" else "completed",
                        timing_source="artifact",
                    )
                )
                if start is None:
                    gaps.append(ObservationGap(code="tool_start_unavailable", detail=item_id))
        for item_id, start in starts.items():
            tools.append(
                ToolCallObservation(
                    invocation_id=state.seed.episode_id.capture_key,
                    tool_call_id=item_id,
                    started_at=start,
                    status="incomplete",
                    timing_source="artifact",
                )
            )
        state.observations = AgentObservationBundle(
            source="codex",
            records=[
                AgentInvocation(
                    invocation_id=state.seed.episode_id.capture_key,
                    status="incomplete" if failure else status,
                    conversation=[*conversation, *output],
                ),
                *tools,
            ],
            gaps=gaps,
        )
        if failure is not None:
            raise failure
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self._effective_model(),
            object="response",
            output=output,
            status=status,
            error={"code": "server_error", "message": error} if error else None,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["input_tokens"] + usage["output_tokens"],
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=usage["cached_input_tokens"]),
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=usage["reasoning_tokens"]),
            ),
            metadata={
                "harness_execution": "sandbox",
                "harness_hostname": result.hostname,
                "harness_pid": str(result.pid),
                "codex_version": self.config.codex_version,
            },
        )

    def _resolve_call_base_url(self, rollout_id: Optional[str]) -> str:
        """Provider base_url for the CLI's model calls (Codex appends ``/responses`` to it).

        A Gym model server gets the per-rollout capture prefix plus the ``/v1`` suffix; a direct
        endpoint (``model_server`` unset) is used verbatim and never prefixed — it has no
        prefix-stripping middleware, so a prefix would 404 every call.
        """
        if self.config.model_server:
            return self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        # Mirrors claude_code_agent's null anthropic_base_url: null means the provider's real API.
        return self.config.openai_base_url or "https://api.openai.com/v1"

    def _effective_model(self) -> Optional[str]:
        """The model name written into the generated config (and reported on the response).

        An explicit (even unknown) model name keeps Codex from applying model-family feature
        gating: for models it recognizes, Codex may switch tools into code-mode carriers that
        models served through a Gym model server cannot drive. Gym model servers substitute their
        own configured model anyway, so a placeholder never reaches the backend. Returns None only
        for a direct endpoint with no configured model, where Codex uses its own default.
        """
        if self.config.model:
            return self.config.model
        if self.config.model_server:
            return "gym-policy-model"
        return None

    def _build_config(
        self,
        base_url: str,
        developer_instructions: Optional[str] = None,
        mcp_servers: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Assemble the per-run CODEX_HOME/config.toml content.

        The base config pins a Gym-owned model provider (bypassing Codex's login flow), disables
        everything that would make a rollout depend on ambient host state or phone home (analytics,
        update checks, on-disk history), and turns off the tools a Gym-served model cannot execute
        (server-side web search, multi-agent). ``extra_config`` is deep-merged on top; the
        per-rollout Gym MCP entries are applied last so they win name collisions.
        """
        config: dict[str, Any] = {
            "model_provider": "gym",
            "approval_policy": "never",
            "sandbox_mode": self.config.sandbox_mode,
            "web_search": "disabled",
            "check_for_update_on_startup": False,
            "analytics": {"enabled": False},
            "history": {"persistence": "none"},
            # multi_agent and code_mode add tool shapes (namespace fan-out, custom JS-exec tools)
            # that models served through a Gym model server cannot execute or express.
            "features": {"multi_agent": False, "code_mode": False},
            "model_providers": {
                "gym": {
                    "name": "gym",
                    "base_url": base_url,
                    # A custom provider reads its API key only from the env var named here; the
                    # agent sets it on the codex subprocess from `openai_api_key` (see _run_codex),
                    # so no `codex login` is ever needed.
                    "env_key": "OPENAI_API_KEY",
                    "wire_api": "responses",
                    "stream_idle_timeout_ms": self.config.stream_idle_timeout_ms or self.config.timeout * 1000,
                }
            },
        }
        model = self._effective_model()
        if model:
            config["model"] = model
        for name in ("model_context_window", "model_auto_compact_token_limit"):
            value = getattr(self.config, name)
            if value is not None:
                config[name] = value
        if self.config.reasoning_effort:
            config["model_reasoning_effort"] = self.config.reasoning_effort
        if developer_instructions:
            config["developer_instructions"] = developer_instructions
        if self.config.extra_config:
            config = _deep_merge(config, deepcopy(self.config.extra_config))
        if mcp_servers:
            config["mcp_servers"] = {**config.get("mcp_servers", {}), **mcp_servers}
        return config

    def _setup_codex_home(self, config: dict[str, Any], skills_path: Optional[str] = None) -> Path:
        """Create a per-run CODEX_HOME and stage config.toml (and optionally skills) into it.

        The directory lives for the duration of a single ``_run_codex`` call. When ``skills_path``
        is provided, the directory of skills is copied into ``<home>/skills/`` where Codex's native
        skill discovery picks them up. Each request gets its own ephemeral copy, so concurrent
        requests with different skills do not contaminate one another. If setup fails partway
        (e.g. a bad ``skills_path``), the partially-created dir is removed before re-raising.
        """
        codex_home = Path.home() / ".codex_agent" / uuid4().hex
        codex_home.mkdir(parents=True)
        try:
            (codex_home / "config.toml").write_text(toml_dumps(config))
            if skills_path:
                stage_skills(skills_path, codex_home / "skills")
        except Exception:
            shutil.rmtree(codex_home, ignore_errors=True)
            raise
        return codex_home

    def _build_command(self, instruction: str, cwd: str) -> list[str]:
        """Construct the ``codex exec`` argv.

        ``--json`` emits machine-readable JSONL events; ``--ephemeral`` skips session persistence;
        ``--skip-git-repo-check`` allows running in the per-rollout scratch dir. Sandboxing and
        approvals are pinned in the generated config.toml (``approval_policy = "never"``), not argv.
        The ``--`` separator keeps prompts from being parsed as flags or subcommands.
        """
        return [
            "codex",
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--cd",
            cwd,
            "--",
            instruction,
        ]

    async def _run_codex(
        self,
        instruction: str,
        system_prompt: Optional[str] = None,
        mcp_servers: Optional[dict[str, Any]] = None,
        skills_path: Optional[str] = None,
        rollout_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Run ``codex exec --json`` and return (stdout, model_name).

        When ``rollout_id`` is set and a model server is configured, the per-rollout capture prefix
        is applied to the provider base_url so the CLI's streaming /v1/responses calls correlate to
        this rollout.
        """
        await self._ensure_local_runtime()
        base_url = self._resolve_call_base_url(rollout_id)
        # Report the name the config actually pins (so response.model matches what Codex was told);
        # falls back to a sentinel only for a direct endpoint that lets Codex pick its own default.
        model = self._effective_model() or "codex-default"

        config = self._build_config(base_url, developer_instructions=system_prompt, mcp_servers=mcp_servers)

        codex_home: Optional[Path] = None
        scratch_cwd: Optional[str] = None
        try:
            # Inside the try so a bad skills_path (raising in stage_skills) still cleans up the
            # partially-created home in the finally rather than leaking it per failing request.
            codex_home = self._setup_codex_home(config, skills_path=skills_path)
            cwd = self.config.cwd
            if cwd is None:
                cwd = scratch_cwd = tempfile.mkdtemp(prefix="nemo_gym_codex_ws_")

            env = {
                **os.environ,
                "CODEX_HOME": str(codex_home),
                # The provider's `env_key` in the generated config.toml; always set from config so
                # a key inherited from the server's environment can never leak into a rollout.
                "OPENAI_API_KEY": self.config.openai_api_key or "local",  # pragma: allowlist secret
            }

            proc = await asyncio.create_subprocess_exec(
                *self._build_command(instruction, cwd),
                stdin=asyncio.subprocess.DEVNULL,  # codex appends piped stdin to the prompt and blocks on it
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # Own process group: `codex` on PATH is an npm shim whose child (the vendored
                # binary) must die with it, or it keeps the stdout pipe open past the kill below.
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                _kill_process_group(proc)
                await proc.communicate()
                LOG.warning("codex timed out after %ds", self.config.timeout)
                return "", model

            if proc.returncode not in (0, None):
                LOG.warning("codex exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            LOG.debug("codex stdout (%d chars): %s", len(stdout), stdout[:2000].decode(errors="replace"))
            return stdout.decode(errors="replace"), model
        finally:
            if codex_home is not None:
                shutil.rmtree(codex_home, ignore_errors=True)
            if scratch_cwd is not None:
                shutil.rmtree(scratch_cwd, ignore_errors=True)

    def _resources_server_base_url(self) -> str:
        cfg = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.resources_server.name,
        )
        return self.server_client._build_server_base_url(cfg)

    def _rollout_mcp_servers(self, seed_response_json: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Per-rollout ``mcp_servers`` config.toml entries from /seed_session MCP metadata.

        Codex reaches Gym MCP tools over streamable HTTP; the per-rollout session token rides on a
        custom header via ``http_headers``.
        """
        metadata = seed_response_json.get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, dict):
            return None

        server_name = str(metadata.get("server_name") or self.config.resources_server.name)
        url_path = str(metadata.get("url_path") or "/mcp")
        entry: dict[str, Any] = {
            "url": f"{self._resources_server_base_url().rstrip('/')}/{url_path.lstrip('/')}",
        }
        headers = metadata.get("headers")
        if isinstance(headers, dict) and headers:
            entry["http_headers"] = {str(key): str(value) for key, value in headers.items()}
        else:
            LOG.warning(
                "MCP seed metadata for %r has no headers; the tool endpoint will be called without a "
                "session token and will reject the calls.",
                server_name,
            )
        return {server_name: entry}

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        mcp_servers: Optional[dict[str, Any]] = None,
        skills_path: Optional[str] = None,
        rollout_id: Optional[str] = None,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        stdout, model_name = await self._run_codex(
            user_message,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            skills_path=skills_path,
            rollout_id=rollout_id,
        )
        output_items, usage = parse_exec_jsonl(stdout)

        if usage.get("errors"):
            LOG.warning("codex reported errors: %s", usage["errors"])

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("codex produced no assistant message; padding empty output")
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
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cached_tokens=usage.get("cached_input_tokens", 0)
                ),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(
                    reasoning_tokens=usage.get("reasoning_tokens", 0)
                ),
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = self._session_marker(request)
        if session_id is not None:
            state = self._sandbox_sessions.get(session_id)
            rollout_id = request.path_params.get("rollout_id")
            if state is None or state.seed.episode_id.capture_key != rollout_id:
                raise HTTPException(409, "Codex activation does not match the seeded session and rollout route")
            if state.activated or state.closing:
                raise HTTPException(409, "Codex sandbox sessions support one activation")
            prompt, system = self._sandbox_input(body)
            state.activated = True
            state.task = asyncio.create_task(self._sandbox_response(state, body, prompt=prompt, system=system))
            try:
                return await asyncio.shield(state.task)
            except asyncio.CancelledError:
                if not state.task.done() and not state.task.cancelling():
                    state.task.cancel()
                raise
        return await self._create_response(body)

    async def run(self, request: Request, body: CodexAgentRunRequest) -> CodexAgentVerifyResponse:
        if self._session_marker(request) is not None:
            raise HTTPException(409, "Use the native session responses and close routes")
        if self.config.resources_server is None:
            raise HTTPException(422, "Legacy /run requires resources_server")
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
            seed_resp_json = await get_response_json(seed_resp)

            # The run-level skills_ref (stamped by rollout collection) rides on the request body
            # (extra="allow"). Pass its path straight into _create_response so the CLI invocation
            # can stage the skills into its per-request CODEX_HOME.
            skills_path = ((body.model_extra or {}).get(SKILLS_REF_KEY_NAME) or {}).get("path")
            rollout_id = self.rollout_id_from_run(body)

            agent_resp = await self._create_response(
                body.responses_create_params,
                mcp_servers=self._rollout_mcp_servers(seed_resp_json),
                skills_path=skills_path,
                rollout_id=rollout_id,
            )
            agent_resp_json = agent_resp.model_dump(mode="json")

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

            return CodexAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    CodexAgent.run_webserver()
