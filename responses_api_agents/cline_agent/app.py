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
from asyncio import Semaphore
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath
from time import time
from typing import Any, ClassVar, Literal, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agents import (
    AGENT_TIMEOUT_METADATA_KEY,
    HARNESS_COMMAND_NOT_STARTED,
    HARNESS_RUNNER_FAILED,
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
    resolve_agent_timeout,
    restricted_network_policy,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    AGENT_INVALID_OUTPUT,
    MODEL_CONTEXT_LIMIT,
    MODEL_TRANSPORT_ERROR,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
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
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap, SandboxObservation
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.cline_agent.observability import build_cline_observations
from responses_api_agents.cline_agent.sandbox import CLINE_RUNTIME_SHA512, ClineArtifacts, ClineSandboxSession
from responses_api_agents.cline_agent.setup_cline import ensure_cline


LOG = logging.getLogger(__name__)

# Provider id the agent authenticates when a Gym model server is configured. `cline auth` only
# accepts a base URL for the OpenAI and OpenAI-compatible providers, and every Gym model server
# speaks the OpenAI Chat Completions dialect, so this is the provider that can be pointed at one.
OPENAI_COMPATIBLE_PROVIDER = "openai-compatible"
# Linux limits one argv string to 128 KiB (MAX_ARG_STRLEN), and Cline 3.0.55 reads a prompt only from argv or a pipe.
MAX_PROMPT_BYTES = 128 * 1024 - 1
SESSION_ROOT = "/tmp/nemo-gym-cline-sessions"
RUNTIME_PREFIX = "/tmp/nemo-gym-cline-runtime-"
# Transport errors reported by Cline 3.0.55's runtime. Its request timeout is not a wall-clock
# bound on Gym's streaming route; the process supervisor owns the run deadline.
_TRANSPORT_ERRORS = ("Cannot connect to API", "The operation timed out.")
# Cline 3.0.55 ends the run with this error when a model call stops at its output-token cap.
_OUTPUT_LIMIT_ERROR = "Model reached the maximum output token limit"
# Cline 3.0.55 ends the run with this error when the model answers with no text, reasoning or tool call.
_EMPTY_RESPONSE_ERROR = "Model returned empty response"


def _cline_state_env(data_dir: str) -> dict[str, str]:
    """Pin every Cline state path into one run's data dir; see ClineAgent._env."""
    return {
        "CLINE_DATA_DIR": data_dir,
        "CLINE_PROVIDER_SETTINGS_PATH": f"{data_dir}/settings/providers.json",
        "CLINE_GLOBAL_SETTINGS_PATH": f"{data_dir}/settings/global-settings.json",
        "CLINE_MCP_SETTINGS_PATH": f"{data_dir}/settings/cline_mcp_settings.json",
        "CLINE_SESSION_DATA_DIR": f"{data_dir}/sessions",
        "CLINE_DB_DATA_DIR": f"{data_dir}/db",
        "CLINE_TEAM_DATA_DIR": f"{data_dir}/teams",
        "CLINE_HOOKS_LOG_PATH": f"{data_dir}/logs/hooks.jsonl",
        # Local session backend: a shared hub daemon would hand concurrent rollouts the same
        # process and state, which is exactly what the per-run data dir is avoiding.
        "CLINE_SESSION_BACKEND_MODE": "local",
        # Cline otherwise ends every call by spawning a detached `npm update -g cline`, which
        # replaces the pinned cline_version mid-collection and briefly removes the binary from
        # under concurrent spawns. Only the exact value "1" disables it.
        "CLINE_NO_AUTO_UPDATE": "1",
    }


def _session_usage(stdout: str) -> Optional[NeMoGymResponseUsage]:
    """Usage from Cline's final ``run_result`` in a sandbox session, or None without one.

    A run stopped before its result has only running totals, which can lag calls the Model Server
    already answered, so its usage stays unknown. The sandbox path always uses the
    ``openai-compatible`` provider, whose ``inputTokens`` already include the prompt's cache reads
    (checked against Gym's captured calls: inputTokens is the prompt total and cacheReadTokens its
    cached part), so cache reads are a detail, not an addition. Cline 3.0.55 reports no reasoning count
    for this provider, so that stays unknown unless it does.
    """
    totals: Optional[tuple[Any, Any, Any, Optional[int]]] = None
    for line in stdout.split("\n"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("type") == "run_result":
            usage = record.get("aggregateUsage") or record.get("usage")
            if isinstance(usage, dict) and usage:
                reasoning = usage.get("reasoningTokens") if "reasoningTokens" in usage else None
                totals = (usage.get("inputTokens"), usage.get("cacheReadTokens"), usage.get("outputTokens"), reasoning)
    if totals is None or not all(type(value) is int and value >= 0 for value in (totals[0], totals[2])):
        return None
    input_tokens, cached_tokens, output_tokens, reasoning = totals
    return NeMoGymResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=NeMoGymResponseInputTokensDetails(
            cached_tokens=cached_tokens if type(cached_tokens) is int and cached_tokens >= 0 else None
        ),
        output_tokens=output_tokens,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(
            reasoning_tokens=reasoning if type(reasoning) is int and reasoning >= 0 else None
        ),
        total_tokens=input_tokens + output_tokens,
    )


def _run_result_finish(stdout: str) -> Optional[str]:
    """``finishReason`` of Cline's final ``run_result``, or None without one; a ``done`` event alone is not a result."""
    finish = None
    for line in stdout.split("\n"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("type") == "run_result":
            finish = record.get("finishReason")
    return finish


def _terminal_error(stdout: str) -> dict[str, Any]:
    """The last `error` agent event of a `cline --json` stream: its error and Cline's error class."""
    terminal: dict[str, Any] = {}
    for line in stdout.split("\n"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        event = record.get("event")
        if record.get("type") == "agent_event" and isinstance(event, dict) and event.get("type") == "error":
            error = event.get("error")
            terminal = {
                "error": error if isinstance(error, dict) else {"message": str(error)},
                "errorClass": event.get("errorClass"),
            }
    return terminal


def _message(index: int, text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=f"msg-{index}",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
        role="assistant",
        status="completed",
        type="message",
    )


def _stringify(value: Any) -> str:
    """Render a tool output/error payload as text for a function_call_output item."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _reported_count(value: Any) -> Optional[int]:
    """A token count Cline reported, or None when it reported none."""
    return value if type(value) is int and value >= 0 else None


def _reasoning_token_count(usage: dict[str, Any], *, running_total: bool = False) -> int:
    """Return Cline's reasoning-token count across its supported usage shapes."""
    direct_keys = (
        ("totalReasoningTokens", "reasoningTokens")
        if running_total
        else (
            "reasoningTokens",
            "totalReasoningTokens",
        )
    )
    for key in direct_keys:
        value = usage.get(key)
        if value is not None:
            return int(value or 0)

    details = usage.get("outputTokenDetails") or usage.get("outputTokensDetails") or {}
    if isinstance(details, dict):
        return int(details.get("reasoningTokens") or details.get("reasoning_tokens") or 0)
    return 0


def parse_cline_events(stdout: str) -> tuple[list[Any], dict[str, Any]]:
    """Convert ``cline --json`` stdout into (output_items, metadata).

    ``cline --json`` writes one JSON object per line. Two record types carry the trajectory:

    - ``{"type": "agent_event", "event": {...}}`` — the agent loop's own events. ``content_start``
      with ``contentType: "text"`` streams assistant text in chunks; the matching ``content_end``
      carries the final text for the turn, so the message is taken from ``content_end`` and the
      chunks only serve as a fallback for a truncated stream. ``contentType: "tool"`` brackets one
      tool call: ``content_start`` has ``toolName``/``toolCallId``/``input``, ``content_end`` has
      ``output`` or ``error``. ``usage`` events carry running token totals, ``done`` the finish
      reason.
    - ``{"type": "run_result", ...}`` — the final summary (``finishReason``, ``iterations``,
      aggregate ``usage``, final ``text``, resolved ``model``).

    ``hook_event`` records duplicate tool-call boundaries the agent events already cover and are
    ignored. Reasoning arrives as ``contentType: "reasoning"`` and becomes a ``<think>`` block on
    the assistant message of the turn that produced it, matching the other CLI agents. Verified
    against cline 3.0.55.
    """
    output_items: list[Any] = []
    # Cache reads stay unknown unless Cline reports a count.
    metadata: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_tokens": None}
    # Text/reasoning chunks for the turn being streamed; the matching content_end supersedes them.
    text_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    # True once the current turn's reasoning has been folded into a message, so the trailing
    # content_end for that same reasoning is not attached a second time (see below).
    reasoning_consumed = False
    # toolCallId -> tool name for calls whose content_start was seen. Cline emits start/end pairs,
    # but a stream cut short can leave a start without an end and a malformed one an end without a
    # start, so both halves are tracked rather than assumed.
    open_tool_calls: dict[str, str] = {}
    saw_usage = False

    def flush_text() -> None:
        """Emit the buffered assistant text for the current turn, with any reasoning attached.

        Cline closes the turn's reasoning *after* its text (``content_end`` for text, then
        ``content_end`` for reasoning), so the think block is taken from the reasoning chunks
        streamed so far rather than from that trailing event, which would otherwise land on the
        following turn.
        """
        nonlocal reasoning_consumed
        text = "".join(text_chunks)
        text_chunks.clear()
        think = "".join(reasoning_chunks).strip()
        if not text.strip():
            return
        if think:
            text = f"<think>\n{think}\n</think>\n\n{text}"
            reasoning_chunks.clear()
            reasoning_consumed = True
        output_items.append(_message(len(output_items), text))

    for line in stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue

        rtype = record.get("type")

        if rtype == "run_result":
            metadata["finish_reason"] = record.get("finishReason")
            metadata["iterations"] = record.get("iterations")
            usage = record.get("aggregateUsage") or record.get("usage") or {}
            if isinstance(usage, dict) and usage:
                # The aggregate is authoritative: Cline computes it from the session rather than
                # summing per-turn events, which a truncated stream can drop.
                metadata["input_tokens"] = int(usage.get("inputTokens") or 0) + int(usage.get("cacheReadTokens") or 0)
                metadata["output_tokens"] = int(usage.get("outputTokens") or 0)
                metadata["reasoning_tokens"] = _reasoning_token_count(usage)
                metadata["cached_tokens"] = _reported_count(usage.get("cacheReadTokens"))
                saw_usage = True
            model = record.get("model")
            if isinstance(model, dict) and model.get("id"):
                metadata["model"] = model["id"]
            elif isinstance(model, str) and model:
                metadata["model"] = model
            continue

        if rtype == "error":
            message = record.get("message")
            if message:
                metadata.setdefault("error", str(message))
                LOG.warning("cline reported an error: %s", str(message)[:500])
            continue

        if rtype != "agent_event":
            # hook_event records mirror tool boundaries the agent events already carry.
            continue

        event = record.get("event")
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        content_type = event.get("contentType")

        if etype == "content_start" and content_type == "text":
            text_chunks.append(event.get("text") or "")

        elif etype == "content_end" and content_type == "text":
            # The turn's final text; it supersedes the streamed chunks.
            final = event.get("text")
            if final is not None:
                text_chunks.clear()
                text_chunks.append(final)
            flush_text()

        elif etype == "content_start" and content_type == "reasoning":
            if not event.get("redacted"):
                reasoning_chunks.append(event.get("reasoning") or "")
                reasoning_consumed = False

        elif etype == "content_end" and content_type == "reasoning":
            # Closes reasoning the turn's message already carries (flush_text ran first), so this
            # only matters when there was no text to attach it to: keep the final text for a
            # reasoning-only turn, which some vLLM parsers produce by routing the whole answer
            # through the reasoning channel.
            if reasoning_consumed:
                reasoning_chunks.clear()
                reasoning_consumed = False
            else:
                final = event.get("reasoning")
                if final:
                    reasoning_chunks.clear()
                    reasoning_chunks.append(final)

        elif etype == "content_start" and content_type == "tool":
            # Text streamed before a tool call belongs to the turn that made it, so it is emitted
            # ahead of the call to keep the trajectory ordered.
            flush_text()
            call_id = str(event.get("toolCallId") or f"call-{uuid4().hex[:8]}")
            name = event.get("toolName") or ""
            tool_input = event.get("input")
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else _stringify(tool_input)
            open_tool_calls[call_id] = name
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=name,
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )

        elif etype == "content_end" and content_type == "tool":
            call_id = str(event.get("toolCallId") or "")
            if call_id and call_id not in open_tool_calls:
                # A result with no recorded call: emit the call so the output is not orphaned.
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments="{}",
                        call_id=call_id,
                        name=event.get("toolName") or "",
                        type="function_call",
                        id=call_id,
                        status="completed",
                    )
                )
            open_tool_calls.pop(call_id, None)
            error = event.get("error")
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id or f"call-{uuid4().hex[:8]}",
                    output=_stringify(error if error else event.get("output")),
                    status="completed",
                )
            )

        elif etype == "usage":
            # Running totals for the session so far; run_result's aggregate wins when the run ends
            # normally, and these are what remains when it does not.
            metadata["input_tokens"] = int(event.get("totalInputTokens") or 0) + int(
                event.get("totalCacheReadTokens") or 0
            )
            metadata["output_tokens"] = int(event.get("totalOutputTokens") or 0)
            metadata["reasoning_tokens"] = _reasoning_token_count(event, running_total=True)
            metadata["cached_tokens"] = _reported_count(event.get("totalCacheReadTokens"))
            saw_usage = True

        elif etype == "done":
            metadata.setdefault("finish_reason", event.get("reason"))
            metadata.setdefault("iterations", event.get("iterations"))

        elif etype == "error":
            error = event.get("error")
            text = error.get("message") if isinstance(error, dict) else _stringify(error)
            if text:
                metadata.setdefault("error", text)
                LOG.warning("cline agent error event: %s", str(text)[:500])

    # A stream cut off mid-turn leaves text with no content_end; surface it rather than drop it.
    flush_text()
    trailing_think = "".join(reasoning_chunks).strip()
    if trailing_think:
        # Reasoning with no message to attach it to (a reasoning-only turn, or a truncated
        # stream) is surfaced on its own so the trajectory does not silently lose it.
        output_items.append(_message(len(output_items), f"<think>\n{trailing_think}\n</think>"))

    if not saw_usage:
        LOG.debug("cline stream carried no usage events; token counts reported as 0")

    return output_items, metadata


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


def quote_prompt(prompt: str) -> str:
    """Return a prompt Cline's CLI accepts as a prompt rather than as a command.

    Shells strip quotes before argv, so Cline decides a bare argument was quoted by looking for
    whitespace in it (``promptArgLooksQuoted`` in the CLI). A single-word prompt like ``hello`` is
    rejected with "Unknown command or unquoted prompt" — including after ``--``, which does not
    exempt it. Padding with a trailing space is the documented shape's observable property and
    leaves the prompt text itself unchanged. Verified against cline 3.0.55.
    """
    return prompt if any(ch.isspace() for ch in prompt) else f"{prompt} "


class ClineAgentConfig(BaseResponsesAPIAgentConfig):
    # Only the host /run path calls the Resources Server. Agent sessions receive the task sandbox from
    # the Environment Server instead.
    resources_server: Optional[ResourcesServerRef] = None
    # When set, Cline's model calls go through this Gym model server instead of straight to a
    # provider, so they are captured. The agent then configures Cline's `openai-compatible`
    # provider with the server's URL, and `model` is the bare model name that server serves.
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    # The Cline command, split on spaces so a multi-word launcher works (e.g. `npx cline`).
    command: str = "cline"
    # The model id handed to Cline. Required with a model server, because `cline auth` refuses to
    # write provider settings without one; None otherwise leaves whatever the configured provider
    # already selects (`-m` is omitted).
    model: Optional[str] = None
    # Provider id passed as `-P`. With a model server this is forced to `openai-compatible`, the
    # only provider besides OpenAI whose base URL `cline auth` accepts.
    provider: str = OPENAI_COMPATIBLE_PROVIDER
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/cline_agent/workspaces"
    # Optional persistent project dir to run in; None -> an ephemeral per-request dir.
    repo_dir: Optional[str] = None
    # Prepended to the user message, like the other CLI agent harnesses. This adds task framing
    # without disturbing Cline's own operating instructions.
    system_prompt: Optional[str] = None
    # Passed as `-s`, which *replaces* Cline's built-in system prompt outright rather than adding
    # to it (buildClineSystemPrompt returns the override alone). That drops the instructions its
    # tools are described by, so prefer `system_prompt` unless a full replacement is the point.
    system_prompt_override: Optional[str] = None
    # Reasoning effort: none|low|medium|high|xhigh. None leaves the provider default (the flag is
    # omitted entirely).
    thinking: Optional[str] = None
    # Context compaction mode: agentic|basic|off. None leaves Cline's default (agentic).
    compaction: Optional[str] = None
    # Max consecutive mistakes before Cline halts, passed as --retries. None leaves its default.
    retries: Optional[int] = None
    # Seconds for one Cline run. The host path kills the subprocess then (and passes --timeout, which
    # Cline 3.0.55 ignores for --json); in a sandbox the process supervisor enforces it.
    timeout: int = 900
    # Seconds for the one-off `cline auth` call that writes the provider settings.
    setup_timeout: int = 300
    extra_args: list[str] = []
    # JSON policy restricting the shell commands Cline may run, passed as CLINE_COMMAND_PERMISSIONS
    # (e.g. {"allow": ["python3 *"], "deny": ["sudo *"]}). Empty -> no restriction.
    command_permissions: dict[str, Any] = Field(default_factory=dict)
    # npm version of the `cline` package installed on a clean machine (the parser was validated
    # against the pinned version, so treat a bump as a deliberate, retested change). None installs
    # @latest on the host path; agent sessions require an exact version with pinned checksums.
    cline_version: Optional[str] = None
    # Agent sessions only: the provider block for an agent-owned sandbox, used when the Resources
    # Server supplies no sandbox_access, and the SandboxSpec fields for it.
    sandbox_provider: Optional[str] = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    network_access: Literal["inherit", "model_only"] = Field(
        default="inherit",
        description=(
            "Egress of an agent-owned sandbox: inherit keeps the provider policy; model_only permits only the "
            "Model Server's host. Borrowed sandboxes require inherit. Restricted setup uploads the pinned runtime "
            "from the Agent Server and needs Bash and coreutils in the image."
        ),
    )
    # Seconds for preparing the session and installing (or reusing) Cline inside the task sandbox.
    sandbox_install_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class ClineAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ClineAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class ClineAgent(SimpleResponsesAPIAgent):
    """Runs the Cline CLI headlessly (``cline --json``).

    Cline runs its own tools internally; its newline-delimited JSON event stream is parsed into Gym
    format and the resources server verifies the result. An agent session runs Cline inside the task
    sandbox the Environment Server hands it; a call without a session runs Cline on this host. Cline's
    model calls go through ``model_server`` when one is set — see that field. Eval-only either way:
    token IDs and logprobs are not wired up.
    """

    config: ClineAgentConfig
    # Sandbox sessions run the CLI inside the task sandbox, which calls the model server from there.
    calls_model_from_sandbox: ClassVar[bool] = True
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _local_setup_task: asyncio.Task[None] | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    async def _ensure_local_cline(self) -> None:
        """Install Cline on this host on the first host call; agent sessions never need it here."""
        if self._local_setup_task is None:
            self._local_setup_task = asyncio.create_task(asyncio.to_thread(ensure_cline, self.config.cline_version))
        setup_task = self._local_setup_task
        try:
            # Concurrent calls share one installation; a cancelled caller must not start a second one.
            await asyncio.shield(setup_task)
        except Exception:
            # A failed installation surfaces to its callers, and the next call retries it.
            if self._local_setup_task is setup_task:
                self._local_setup_task = None
            raise
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("cline command %r is not on PATH", self.config.command)

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"cline_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _repo_dir(self, fallback: Path) -> Path:
        """Return the configured persistent repository or the temporary fallback."""
        if not self.config.repo_dir:
            return fallback
        root = Path(self.config.repo_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None) -> str:
        """The Gym model server's ``/v1`` URL with the per-rollout capture prefix, "" if unset."""
        if self.config.model_server is None:
            return ""
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    def _effective_provider(self) -> str:
        """The provider id for `-P`: forced to openai-compatible on the model-server path."""
        return OPENAI_COMPATIBLE_PROVIDER if self.config.model_server else self.config.provider

    def _env(self, data_dir: Path, model_base_url: str = "") -> dict[str, str]:
        """Environment for the cline subprocesses, isolated to this run's data dir.

        ``--data-dir`` alone is not enough: the CLI resolves several state paths from
        ``CLINE_DATA_DIR`` and friends, and an ambient value (set by an outer Cline session, for
        instance) would otherwise leak a shared provider settings file and session db across
        concurrent rollouts. Pinning them here keeps each run's provider config, sessions, and
        logs inside its own directory, and never touches the user's ``~/.cline``.
        """
        env = {**os.environ, **_cline_state_env(str(data_dir))}
        # A Gym model server wins over openai_base_url, so a config carrying both cannot point the
        # subprocess at the provider the model server is meant to replace.
        base_url = model_base_url or self.config.openai_base_url
        api_key = "EMPTY" if model_base_url else self.config.openai_api_key  # pragma: allowlist secret
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        if self.config.command_permissions:
            env["CLINE_COMMAND_PERMISSIONS"] = json.dumps(self.config.command_permissions)
        env.update({k: v for k, v in self.config.env.items() if v})
        return env

    def _build_auth_command(self, data_dir: Path, model_base_url: str) -> Optional[list[str]]:
        """``cline auth`` argv that writes this run's provider settings, or None when not needed.

        Cline reads its provider, key, model, and base URL from the provider settings file rather
        than from run flags (``-k`` overrides only the key, and no flag sets a base URL), so the
        model-server path has to write that file once per run before the task runs.
        """
        if not model_base_url:
            return None
        if not self.config.model:
            raise ValueError("cline_agent requires `model` to be set when `model_server` is configured")
        return [
            *self.config.command_parts,
            "auth",
            OPENAI_COMPATIBLE_PROVIDER,
            # `cline auth` requires a key; the Gym model server does not check it, and the real
            # credential (if any) lives on the model server, not here.
            "--apikey",
            "EMPTY",  # pragma: allowlist secret
            "--modelid",
            self.config.model,
            "--baseurl",
            model_base_url,
            "--data-dir",
            str(data_dir),
        ]

    def _build_command(self, project_dir: Path, data_dir: Path, prompt: str) -> list[str]:
        """``cline`` argv for one headless run.

        ``--json`` selects the newline-delimited JSON event stream (and implies headless);
        ``--auto-approve true`` runs the tools without prompting; ``--cwd`` scopes the run to the
        per-request project dir; ``--timeout`` is passed, but Cline 3.0.55 does not bound a ``--json``
        run with it, so the subprocess deadline ends the run. The prompt goes last, after ``--``, so
        one starting with ``-`` is not parsed as a flag.
        """
        cmd = [
            *self.config.command_parts,
            "--json",
            "--auto-approve",
            "true",
            "--cwd",
            str(project_dir),
            "--data-dir",
            str(data_dir),
            "--timeout",
            str(self.config.timeout),
            *self._run_flags(),
            *self.config.extra_args,
            "--",
            quote_prompt(prompt),
        ]
        return cmd

    def _run_flags(self) -> list[str]:
        """Model, provider and loop settings shared by the host and sandbox command lines."""
        flags = []
        if self.config.model:
            flags += ["-m", self.config.model]
        flags += ["-P", self._effective_provider()]
        if self.config.thinking:
            flags += ["--thinking", self.config.thinking]
        if self.config.compaction:
            flags += ["--compaction", self.config.compaction]
        if self.config.retries is not None:
            flags += ["--retries", str(self.config.retries)]
        if self.config.system_prompt_override:
            flags += ["-s", self.config.system_prompt_override]
        return flags

    @staticmethod
    def _kill_process_group(proc: "asyncio.subprocess.Process") -> None:
        # `cline` is an npm shim whose child is the real binary; killing only the shim orphans the
        # child holding the stdout pipe, so kill the whole process group (start_new_session=True).
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()

    async def _spawn(self, cmd: list[str], cwd: Path, env: dict[str, str], timeout: int) -> tuple[str, str, int]:
        """Run one subprocess to completion; on timeout kill its group and report what it wrote."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.CancelledError:
            # A cancelled rollout must not leave Cline's npm shim or native child running and
            # holding the sandbox's stdout pipe. Drain the process after killing its group before
            # preserving cancellation for the caller.
            self._kill_process_group(proc)
            await proc.communicate()
            raise
        except asyncio.TimeoutError:
            self._kill_process_group(proc)
            stdout, stderr = await proc.communicate()
            return stdout.decode(errors="replace"), stderr.decode(errors="replace"), -1
        return stdout.decode(errors="replace"), stderr.decode(errors="replace"), proc.returncode or 0

    async def _run_cline(
        self, instruction: str, system_prompt: Optional[str], rollout_id: Optional[str] = None
    ) -> tuple[list[Any], dict[str, Any], str]:
        """Run one headless cline task. Returns (output_items, metadata, model_name)."""
        await self._ensure_local_cline()
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        project_dir = self._repo_dir(work_dir)
        data_dir = work_dir / ".cline-data"
        data_dir.mkdir(parents=True, exist_ok=True)
        model_base_url = self._resolve_model_base_url(rollout_id)
        env = self._env(data_dir, model_base_url)

        try:
            auth_cmd = self._build_auth_command(data_dir, model_base_url)
            if auth_cmd:
                _, auth_err, auth_rc = await self._spawn(auth_cmd, project_dir, env, self.config.setup_timeout)
                if auth_rc != 0:
                    LOG.warning("cline auth exited %d: %s", auth_rc, auth_err[:500])

            cmd = self._build_command(project_dir, data_dir, prompt)
            stdout, stderr, returncode = await self._spawn(cmd, project_dir, env, self.config.timeout)
            if returncode == -1:
                LOG.warning("cline timed out after %ds", self.config.timeout)
            elif returncode != 0:
                LOG.warning("cline exited %d: %s", returncode, stderr[:500])

            output_items, metadata = parse_cline_events(stdout)
            if returncode == -1:
                metadata["timed_out"] = True
            model_name = metadata.get("model") or self.config.model or ""
            return output_items, metadata, model_name
        finally:
            # The run dir holds the ephemeral data dir (and the workspace, unless repo_dir moved
            # the project elsewhere), so nothing survives to leak into the next rollout.
            shutil.rmtree(work_dir, ignore_errors=True)

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Read the terminal error event of a Cline 3.0.55 run.

        Cline names a context overflow itself (``ContextWindowOverflowError``, error class
        ``context_window_exceeded``). Its other model errors carry no HTTP status, only the provider's
        message, so they stay ``unknown`` apart from a request that never got a response.
        """
        error = native_result.get("error") or {}
        if error.get("name") == "ContextWindowOverflowError" or native_result.get("errorClass") == (
            "context_window_exceeded"
        ):
            return AgentOutcome(failure_kind=MODEL_CONTEXT_LIMIT)
        if str(error.get("message") or "").startswith(_TRANSPORT_ERRORS):
            return AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
        if str(error.get("message") or "").startswith(_EMPTY_RESPONSE_ERROR):
            return AgentOutcome(failure_kind=AGENT_INVALID_OUTPUT)
        return AgentOutcome(failure_kind="unknown")

    def _reached_model_limit(self, terminal: Mapping[str, Any]) -> bool:
        """Whether Cline's terminal error is the model's context window or its output-token cap."""
        message = str((terminal.get("error") or {}).get("message") or "")
        return self.classify_outcome(terminal).failure_kind == MODEL_CONTEXT_LIMIT or message.startswith(
            _OUTPUT_LIMIT_ERROR
        )

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> ClineSandboxSession:
        """Install or reuse the pinned Cline runtime in the task sandbox before the activation."""
        required = sorted(access.name for access in self.effective_tool_accesses(body) if access.required)
        if required:
            raise AgentSeedRejectedError(
                "Cline sandbox sessions use Cline's built-in tools; required tool grants are unsupported: "
                + ", ".join(required)
            )
        version = self.config.cline_version
        if not version or not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise AgentSeedRejectedError("Cline sandbox sessions require an exact cline_version, for example 3.0.55")
        if version not in CLINE_RUNTIME_SHA512:
            raise AgentSeedRejectedError(
                f"Cline {version} has no pinned runtime checksums; supported: {', '.join(sorted(CLINE_RUNTIME_SHA512))}"
            )
        if self.config.model_server is None or not self.config.model:
            raise AgentSeedRejectedError("Cline sandbox sessions require a Gym model_server and its model name")
        host_only = [
            name
            for name in ("env", "extra_args", "repo_dir", "openai_api_key", "openai_base_url")
            if getattr(self.config, name)
        ]
        if self.config.command != "cline":
            host_only.append("command")
        if host_only:
            raise AgentSeedRejectedError(
                "Cline sandbox sessions do not support these host settings: " + ", ".join(host_only)
            )
        access = body.sandbox_access
        owns_sandbox = access is None
        try:
            if owns_sandbox:
                if not self.config.sandbox_provider:
                    raise AgentSeedRejectedError("Cline requires sandbox_access or a configured sandbox_provider")
                settings = {"workdir": "/app", **self.config.sandbox_config}
                settings["metadata"] = {
                    **resolve_provider_metadata(self.config.sandbox_provider, self.server_client.global_config_dict),
                    **(settings.get("metadata") or {}),
                    "gym-agent": "cline",
                    "gym-session": body.agent_session_id[:63],
                    "gym-rollout": body.episode_id.capture_key[:63],
                }
                # Provider TTLs outlive this server: cover installation and the run, plus ten minutes for close.
                settings.setdefault("ttl_s", self.config.sandbox_install_timeout_seconds + self.config.timeout + 600)
                spec = SandboxSpec(**settings)
                workdir, provider_ref = spec.workdir, self.config.sandbox_provider
            else:
                if not isinstance(access.connection, DirectSandboxConnection):
                    raise AgentSeedRejectedError("Cline supports only direct SandboxAccess")
                if self.config.network_access != "inherit":
                    raise AgentSeedRejectedError(
                        "Cline network_access applies only to a sandbox it starts, not a borrowed one"
                    )
                workdir, provider_ref = access.workdir, access.connection.provider_config_ref
            if not isinstance(workdir, str):
                raise AgentSeedRejectedError("Cline sandbox workdir must be an absolute path string")
            path = PurePosixPath(workdir)
            if (
                not path.is_absolute()
                or ".." in path.parts
                or path in (PurePosixPath("/"), PurePosixPath("/tmp"))
                or str(path).startswith((SESSION_ROOT, RUNTIME_PREFIX))
            ):
                raise AgentSeedRejectedError(
                    "Cline sandbox workdir must be absolute and separate from its session storage"
                )

            provider = create_provider(resolve_provider_config(provider_ref, self.server_client.global_config_dict))
        except AgentSeedRejectedError:
            raise
        except (TypeError, ValueError) as error:
            raise AgentSeedRejectedError(str(error)) from error
        try:
            if owns_sandbox:
                if self.config.network_access == "model_only":
                    options = spec.provider_options or {}
                    if not isinstance(options, Mapping):
                        raise AgentSeedRejectedError("Cline sandbox_config.provider_options must be a mapping")
                    if "network_policy" in options:
                        raise AgentSeedRejectedError(
                            "Set network_access or sandbox_config.provider_options.network_policy, not both"
                        )
                    try:
                        policy = restricted_network_policy(
                            provider.name,
                            [self.resolve_model_base_url(self.config.model_server.name, trace_context=False)],
                        )
                    except ValueError as error:
                        raise AgentSeedRejectedError(str(error)) from error
                    spec = replace(spec, provider_options={**options, "network_policy": policy})
                sandbox = AsyncSandbox(provider)
            else:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
        except BaseException:
            try:
                await provider.aclose()
            except BaseException as cleanup_error:
                LOG.exception("Cline provider cleanup failed after sandbox setup failed")
                if isinstance(cleanup_error, NgFailureEvent):
                    observe_failure(cleanup_error)
                else:
                    cleanup_failure = NgFailureEvent(
                        boundary="sandbox",
                        failure_kind="session_release_failed",
                        failure_reason=f"Cline provider cleanup failed: {type(cleanup_error).__name__}",
                        operation="aclose",
                        stage="cleanup",
                    )
                    cleanup_failure.__cause__ = cleanup_error
                    observe_failure(cleanup_failure)
            raise
        state = ClineSandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=f"{SESSION_ROOT}/{uuid4().hex}",
                workdir=str(path),
                owns_sandbox=owns_sandbox,
                harness="Cline",
            ),
            runtime=f"{RUNTIME_PREFIX}{version}",
        )
        try:
            async with asyncio.timeout(self.config.sandbox_install_timeout_seconds):
                restricted = self.config.network_access != "inherit"
                if owns_sandbox:
                    await sandbox.start(spec)
                    if not restricted:
                        # Restricted setup checks its image prerequisites before creating the workdir.
                        created = await sandbox.exec(f"mkdir -p -- {shlex.quote(str(path))}", cwd="/", timeout_s=30)
                        if created.return_code != 0 or created.error_type:
                            raise RuntimeError(f"Cannot create the Cline sandbox workdir {path}: {created.stderr}")
                await state.prepare(
                    timeout=self.config.sandbox_install_timeout_seconds,
                    allow_package_install=not restricted,
                    create_workdir=owns_sandbox and restricted,
                )
                await state.install_runtime(
                    version=version,
                    timeout=self.config.sandbox_install_timeout_seconds,
                    download_on_host=restricted,
                )
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                LOG.exception("Cline seed cleanup failed; retaining session %s", body.agent_session_id)
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Stop Cline before verification; a borrowed sandbox stays with its Resources Server."""
        assert isinstance(state, ClineSandboxSession)
        if state.provider_wait_failure is not None:
            observe_failure(state.provider_wait_failure)
        await state.close(self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=state.observations
            or AgentObservationBundle(source="cline", gaps=[ObservationGap(code="observation_capture_failed")]),
        )

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, ClineSandboxSession) else None

    @staticmethod
    def _validate_output_budget(body: NeMoGymResponseCreateParamsNonStreaming) -> None:
        """Refuse a response-wide budget before starting host or sandbox work."""
        if body.max_output_tokens is not None:
            raise HTTPException(
                422,
                "Cline cannot enforce request max_output_tokens across model calls; "
                "omit it and configure a per-call limit on the Model Server instead",
            )

    def _validate_session_request(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        """Accept only the controls a sandbox session applies; Cline has no per-request sampling flags."""
        self._validate_output_budget(body)
        supported = {"input", "instructions", "model"}
        for name, field in type(body).model_fields.items():
            value = getattr(body, name)
            if name in supported or (name in ("stream", "background") and value is False):
                continue
            if name == "metadata" and set(value or {}) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"Cline sandbox sessions do not support request field {name}")
        models = {None, "", self.config.model, self.config.model_server.name if self.config.model_server else None}
        if body.model not in models:
            raise HTTPException(422, "Cline's model is selected by the agent configuration")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        # A developer message is the system message; the instruction extractor recognizes only "system".
        items = [
            item.model_copy(update={"role": "system"}) if getattr(item, "role", None) == "developer" else item
            for item in items
        ]
        if [getattr(item, "role", None) for item in items] not in (["user"], ["system", "user"]):
            raise HTTPException(422, "Cline accepts one text user prompt with an optional system message")
        for item in items:
            if not isinstance(item.content, str) and any(
                (part.get("type") if isinstance(part, dict) else getattr(part, "type", None)) != "input_text"
                for part in item.content
            ):
                raise HTTPException(422, "Cline only supports text input")
        return body.model_copy(update={"input": items}).model_copy(deep=True)

    def _session_prompt(self, body: NeMoGymResponseCreateParamsNonStreaming) -> str:
        """The one user prompt Cline receives, built as on the host path; Cline keeps its own system prompt."""
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        user_message, input_system = _extract_instruction(items)
        system = "\n\n".join(part for part in (self.config.system_prompt, body.instructions, input_system) if part)
        prompt = f"{system}\n\n{user_message}" if system else user_message
        if len(quote_prompt(prompt).encode()) > MAX_PROMPT_BYTES:
            raise HTTPException(422, f"Cline's prompt exceeds the {MAX_PROMPT_BYTES}-byte argv limit")
        return prompt

    async def _sandbox_response(
        self, state: ClineSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        prompt = self._session_prompt(body)
        invocation_id = state.request.episode_id.capture_key
        timeout = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
        providers = {
            "version": 1,
            "lastUsedProvider": OPENAI_COMPATIBLE_PROVIDER,
            "providers": {
                OPENAI_COMPATIBLE_PROVIDER: {
                    "settings": {
                        "provider": OPENAI_COMPATIBLE_PROVIDER,
                        # The Gym model server does not check the key; the real credential stays there.
                        "apiKey": "EMPTY",  # pragma: allowlist secret
                        "model": self.config.model,
                        "baseUrl": self.resolve_model_base_url(self.config.model_server.name, invocation_id),
                        # Owns every attempt, including ones that fail without a response ID.
                        "headers": {"x-session-id": invocation_id},
                    },
                    "updatedAt": "1970-01-01T00:00:00.000Z",
                    "tokenSource": "manual",
                }
            },
        }
        env = _cline_state_env(state.data_dir)
        if self.config.command_permissions:
            env["CLINE_COMMAND_PERMISSIONS"] = json.dumps(self.config.command_permissions)
        # No --timeout: Cline 3.0.55 does not bound a --json run with it. The supervisor enforces the deadline.
        command = [
            state.executable,
            "--json",
            "--auto-approve",
            "true",
            "--cwd",
            state.session.workdir,
            "--config",
            state.config_dir,
            "--data-dir",
            state.data_dir,
            *self._run_flags(),
            "--",
            quote_prompt(prompt),
        ]
        failure: BaseException | None = None
        try:
            async with self.sem:
                await state.session.execute(
                    stage_activation=lambda: state.stage_activation(
                        command=command,
                        env=env,
                        providers=providers,
                        # Keep telemetry and update checks out of evaluation sandboxes.
                        global_settings={"telemetryOptOut": True, "autoUpdateEnabled": False},
                    ),
                    collect=state.collect_artifacts,
                    timeout=timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            failure = error
        artifacts: ClineArtifacts | None = state.session.artifacts
        cleanup = state.session.cleanup
        timed_out = cleanup is not None and cleanup["timed_out"]
        if isinstance(failure, TimeoutError) and timed_out and artifacts is not None:
            # The provider stopped waiting just as the supervisor stopped Cline at its deadline: still a timeout.
            LOG.warning("The sandbox provider stopped waiting for Cline's supervisor after Cline's deadline")
            state.provider_wait_failure = NgFailureEvent(
                boundary="sandbox",
                failure_kind="transport_timeout",
                failure_reason="Sandbox provider wait expired after Cline's supervisor confirmed its deadline",
                operation="exec_wait",
                stage="agent",
                foreign_error=sdk_error_details(failure, message=str(failure)),
            )
            state.provider_wait_failure.__cause__ = failure
            failure = None
        if state.provider_wait_failure is not None:
            observe_failure(state.provider_wait_failure)
        stdout = artifacts.stdout if artifacts is not None else ""
        output_items, metadata = parse_cline_events(stdout)
        terminal = _terminal_error(stdout)
        finish = _run_result_finish(stdout)
        exit_code = cleanup["return_code"] if cleanup is not None and not timed_out else None
        # Cline 3.0.55 exits 0 for finishReason "completed" and also for an abort, so the final run_result decides
        # the status; its error event names what ended any other run.
        crash: str | None = None
        error_message: str | None = None
        if failure is not None:
            status = "incomplete" if isinstance(failure, asyncio.CancelledError) else "failed"
        elif timed_out:
            # A supervisor deadline says nothing about whether a model request was ever made.
            # Without any recorded model output, there is no partial work to grade as a model failure.
            status = "incomplete" if output_items or state.session.expired else "failed"
            if status == "failed":
                error_message = "Cline's deadline passed before any model output was recorded"
        elif exit_code == HARNESS_RUNNER_FAILED or exit_code in HARNESS_COMMAND_NOT_STARTED or finish is None:
            status, crash = "failed", f"Cline produced no run result (exit {exit_code})"
        elif finish == "completed":
            status = "completed"
        elif finish == "error" and not self._reached_model_limit(terminal):
            status = "failed"
            error_message = str((terminal.get("error") or {}).get("message") or metadata.get("error") or "Cline error")
        else:
            # A context or output-token limit, an abort or another harness limit: the partial work stays gradable.
            status = "incomplete"
        try:
            state.observations = build_cline_observations(
                stdout,
                invocation_id,
                status=status,
                error_type=finish if status != "completed" else None,
                sandbox_id=state.session.sandbox.sandbox_id,
                conversation=[NeMoGymEasyInputMessage(role="user", content=prompt), *output_items],
                unfinished_tool_status="timeout"
                if timed_out
                else "cancelled"
                if isinstance(failure, asyncio.CancelledError)
                else "incomplete",
            )
        except Exception:
            LOG.exception("failed to build Cline observations")
            state.observations = AgentObservationBundle(
                source="cline", gaps=[ObservationGap(code="observation_parse_failed")]
            )
        if artifacts is None:
            state.observations.gaps.append(ObservationGap(code="agent_artifact_unavailable"))
        if state.runtime_info is None:
            state.observations.gaps.append(ObservationGap(code="runtime_info_unavailable"))
        if timed_out and not state.session.expired:
            for observation in state.observations.records:
                if isinstance(observation, AgentInvocation):
                    observation.stop_reason = "timeout"
        if cleanup is not None:
            access = state.request.sandbox_access
            state.observations.records.append(
                SandboxObservation(
                    role="agent",
                    provider=access.connection.provider_config_ref if access else self.config.sandbox_provider,
                    sandbox_id=state.session.sandbox.sandbox_id,
                    outcome="timeout" if timed_out else "completed" if exit_code == 0 else "failed",
                    exit_code=cleanup["return_code"],
                    wall_time_s=artifacts.wall_time_s if artifacts is not None else None,
                )
            )
        if failure is not None:
            raise failure
        if crash:
            stderr = artifacts.stderr if artifacts is not None else ""
            raise RuntimeError(f"{crash}: {stderr[-4000:]}")
        usage = _session_usage(stdout)
        if usage is None:
            state.observations.gaps.append(
                ObservationGap(code="model_call_usage_unavailable", detail="Cline reported no final run result")
            )
        elif usage.output_tokens_details.reasoning_tokens is None:
            state.observations.gaps.append(ObservationGap(code="reasoning_token_usage_unavailable"))
        runtime = state.runtime_info
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=output_items,
            status=status,
            error={"code": "server_error", "message": error_message} if error_message else None,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
            metadata={
                "harness_execution": "sandbox",
                "cline_version": self.config.cline_version,
                **({"harness_hostname": runtime.hostname, "harness_pid": str(runtime.pid)} if runtime else {}),
                **({"finish_reason": finish} if finish else {}),
                AGENT_TIMEOUT_METADATA_KEY: str(timeout),
            },
        )
        return self.report_rollout_stop(self.report_outcome(response, terminal), state.session)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is not None:
            state = self._require_agent_session(session_id)
            assert isinstance(state, ClineSandboxSession)
            if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
                raise HTTPException(409, "Cline activation does not match the seeded session and rollout route")
            if state.session.closing:
                raise HTTPException(409, "Cline session is closing")
            body = self._validate_session_request(body)
            if state.task is None:
                # Bind the request before the first await so retries join one invocation.
                state.activation_request = body.model_copy(deep=True)
                state.task = asyncio.create_task(self._sandbox_response(state, state.activation_request))
            elif body != state.activation_request:
                raise HTTPException(409, "Cline sandbox sessions support one activation; retry the same request")
            # Session close owns cancellation. Losing an HTTP waiter must not stop the harness.
            return (await asyncio.shield(state.task)).model_copy(deep=True)

        self._validate_output_budget(body)
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        # run() reaches this handler through the /ng-rollout/<id> self-call route, which carries
        # the correlation id Cline's model calls are tagged with. Absent on a direct
        # /v1/responses call.
        rollout_id = request.path_params.get("rollout_id")
        output_items, metadata, model_name = await self._run_cline(user_message, system_prompt, rollout_id)

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("Cline produced no assistant message. Padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = metadata.get("input_tokens", 0)
        output_tokens = metadata.get("output_tokens", 0)
        reasoning_tokens = metadata.get("reasoning_tokens", 0)

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
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=metadata.get("cached_tokens")),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def run(self, request: Request, body: ClineAgentRunRequest) -> ClineAgentVerifyResponse:
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "Cline sandbox sessions run through an Environment Server's /run")
        if self.config.resources_server is None:
            raise HTTPException(
                422, "Cline /run requires resources_server; sandbox sessions use an Environment Server"
            )
        self._validate_output_budget(body.responses_create_params)
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

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)

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

            return ClineAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    ClineAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = ClineAgent.run_webserver()  # noqa: F401
