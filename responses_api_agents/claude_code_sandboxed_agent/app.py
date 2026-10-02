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
"""Runs the Claude Code CLI inside the task's own sandbox.

Same shape as ``opencode_sandboxed_agent``: the resources server seeds the task and hands back its
sandbox, this agent attaches to it, runs one command under ``sandbox_timeout``, reads the
transcript the CLI left behind, calls ``/verify`` and stops the sandbox. Gym stays on the host;
nothing but the Claude Code binary is needed inside the sandbox. The command:

1. installs Claude Code. A staged binary is copied out of its read-only mount and made executable,
   since S3 mounts keep no file modes. Without a staged binary the official installer runs, which
   needs network access from the sandbox.
2. checks that the binary reports ``claude_code_version``.
3. writes a settings file and runs ``claude -p`` in the task's working directory. The stream-json
   output goes to a file outside the task repository, so a verifier that diffs the repository
   never sees it, and stdin is closed so the CLI does not wait for piped input.

Model calls go to the Gym model server's ``/v1/messages`` route under this rollout's URL prefix, so
training-token capture records every call. The transcript built here is what the verifier and the
rollout record see; training rebuilds ``response.output`` from the captured tokens.
"""

import json
import sys
import tarfile
import tempfile
from asyncio import Semaphore
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from time import time
from traceback import format_exc
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
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
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    SandboxObservation,
    ToolCallObservation,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.agent_tools import sandbox_server_url, verify_agent_response
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint, raise_for_status
from responses_api_agents.claude_code_agent.observability import extract_claude_code_observations


STREAM_FNAME = "stream.jsonl"
TRANSCRIPTS_TAR_FNAME = "transcripts.tar"
FINISHED_MARKER = "Claude Code run finished"
OFFICIAL_INSTALLER_URL = "https://claude.ai/install.sh"
# The Gym model server does not authenticate; Claude Code only needs some credential to start.
PLACEHOLDER_API_KEY = "nemo-gym"  # pragma: allowlist secret
# Claude Code's model name for messages it makes up itself, such as API error notices.
SYNTHETIC_MODEL = "<synthetic>"
# The tools that start a subagent (Task is the pre-2.1 name Claude Code still accepts).
SUBAGENT_TOOLS = frozenset({"Agent", "Task"})
# How Claude Code reports a tool call it refused to run, and the refusal for a tool name that does
# not exist (for example a lowercase ``read`` from a model used to another harness).
TOOL_ERROR_TAG = "<tool_use_error>"
UNKNOWN_TOOL_MESSAGE = "No such tool available"
# The error text of a ``result`` event whose last request no longer fit the model's context.
CONTEXT_OVERFLOW_MESSAGE = "Prompt is too long"


class ClaudeCodeSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    claude_code_version: str
    # Binaries staged inside the sandbox, e.g. on a read-only bucket mount. The musl build is only
    # needed for musl-based task images. With no staged binary the official installer runs.
    remote_claude_code_binary_path: Optional[str] = None
    remote_claude_code_musl_binary_path: Optional[str] = None
    # Each rollout's binary, settings and transcripts go in a fresh directory under this one,
    # which must be writable, allow executables, and lie outside the task repository.
    remote_work_dir: str = "/tmp"

    # The model name Claude Code sends, for every role (main loop, subagents, background tasks). A
    # Gym model server serves its own configured model whatever the name.
    model: str = "policy"
    # Claude Code's automatic context compaction. Compaction rewrites the conversation, which
    # training-token capture cannot follow, so training turns it off.
    auto_compact: bool = True
    # Auto-memory keeps notes across sessions; every rollout starts from an empty config directory,
    # and notes written mid-session would change the context of later calls.
    auto_memory: bool = False
    # `--bare` is Claude Code's minimal mode (CLAUDE_CODE_SIMPLE): a one-line system prompt and only
    # the Bash, Read and Edit tools. Off, the agent runs the full harness.
    bare: bool = False
    # Settings sources Claude Code loads besides --settings (user, project, local). "user" alone is
    # this rollout's empty config directory, so a task repository's .claude/ settings and hooks
    # cannot change the run. Null loads Claude Code's default sources.
    setting_sources: Optional[str] = "user"
    allowed_tools: Optional[str] = None
    disallowed_tools: Optional[str] = None
    max_turns: Optional[int] = None
    append_system_prompt: Optional[str] = None
    # Per-call output cap (CLAUDE_CODE_MAX_OUTPUT_TOKENS). Claude Code clamps it to its own limit.
    max_output_tokens: Optional[int] = None
    # Claude Code's request timeout (API_TIMEOUT_MS) and how long it waits for response headers
    # (CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS) before it aborts and retries the call. A Gym model
    # server sends the headers of a streamed reply only after generation finishes.
    api_timeout_ms: Optional[int] = None
    stream_first_byte_timeout_ms: Optional[int] = None
    # The CLI is stopped this many seconds before sandbox_timeout so that it cannot keep editing
    # the repository or calling the model after the agent gives up on it.
    harness_timeout_margin_s: float = 120

    # Layered over the settings and environment derived from the fields above.
    claude_code_settings: Dict[str, Any] = Field(default_factory=dict)
    claude_code_env: Dict[str, str] = Field(default_factory=dict)

    concurrency: int = Field(default=64, gt=0)
    # Grade a failed run (exec error, nonzero exit, missing transcript, error result) as an empty
    # response instead of whatever patch it left behind. The verifier must score an empty
    # response as an unmasked zero.
    execution_failure_reward_zero: bool = False
    # Where each rollout's transcript and generation receipt are kept; this agent's ``results``
    # directory by default.
    artifacts_dir: Optional[str] = None

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]
    sandbox_timeout: float

    debug: bool = False


class ClaudeCodeSandboxedAgentRunRequest(BaseRunRequest):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")


class ClaudeCodeSandboxedAgentVerifyRequest(BaseVerifyRequest):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")


class ClaudeCodeSandboxedAgentVerifyResponse(BaseVerifyResponse):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")

    claude_code_results_fpath: str
    claude_code_run_stdout: str
    claude_code_run_stderr: str
    claude_code_finished: bool
    claude_code_export_found: bool
    claude_code_exit_code: Optional[int] = None
    claude_code_error_type: Optional[str] = None
    claude_code_failed: bool = False
    # How the session ended, from its final ``result`` event: finished on its own, ended with an
    # error (``claude_code_context_overflow`` names the one that matters most), or never wrote one
    # (the deadline or a crash cut it off).
    claude_code_result_success: bool = False
    claude_code_result_error: bool = False
    claude_code_result_missing: bool = True
    claude_code_context_overflow: bool = False
    claude_code_duration_s: Optional[float] = None
    # What the session did: model turns of the main conversation and of subagents, subagent
    # launches, tool calls, and tool calls Claude Code refused, of which the ones naming a tool that
    # does not exist.
    claude_code_main_turns: int = 0
    claude_code_subagent_turns: int = 0
    claude_code_subagent_calls: int = 0
    claude_code_tool_calls: int = 0
    claude_code_tool_errors: int = 0
    claude_code_unknown_tool_calls: int = 0
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


@dataclass(frozen=True)
class ClaudeCodeRunPaths:
    """Where one rollout's files live inside the sandbox."""

    root: str

    @property
    def binary(self) -> str:
        return f"{self.root}/bin/claude"

    @property
    def config_dir(self) -> str:
        return f"{self.root}/config"

    @property
    def settings(self) -> str:
        return f"{self.root}/settings.json"

    @property
    def stream(self) -> str:
        return f"{self.root}/{STREAM_FNAME}"

    @property
    def transcripts_tar(self) -> str:
        return f"{self.root}/{TRANSCRIPTS_TAR_FNAME}"


@dataclass
class ClaudeCodeStream:
    """What the stream-json output of one ``claude -p`` run says about the run."""

    output: List[NeMoGymResponseOutputItem]
    usage: Optional[NeMoGymResponseUsage]
    result: Optional[Dict[str, Any]]
    compaction_attempts: List[Dict[str, str]]
    compaction_boundaries: int
    invalid_lines: int
    # Distinct assistant message ids, in the main conversation and under a parent_tool_use_id.
    main_turns: int = 0
    subagent_turns: int = 0
    # Tool calls of the main conversation that started a subagent.
    subagent_calls: int = 0
    # Tool calls, main conversation and subagents alike, and the ones Claude Code refused to run.
    tool_calls: int = 0
    tool_errors: int = 0
    unknown_tool_calls: int = 0


def _tool_result_text(content: Any) -> str:
    """Flatten a tool_result's content the way the Gym Messages converter does."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else json.dumps(content)
    return "\n".join(
        block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


def _token_count(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else 0


def _usage_from_messages_usage(usages: Iterable[Dict[str, Any]]) -> Optional[NeMoGymResponseUsage]:
    """Sum Anthropic Messages usage blocks into one Responses usage.

    Anthropic counts cache reads and writes outside ``input_tokens``; they are prompt tokens too.
    """
    found = False
    input_tokens = cached_tokens = output_tokens = 0
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        found = True
        cache_read = _token_count(usage.get("cache_read_input_tokens"))
        cache_creation = _token_count(usage.get("cache_creation_input_tokens"))
        input_tokens += _token_count(usage.get("input_tokens")) + cache_read + cache_creation
        cached_tokens += cache_read
        output_tokens += _token_count(usage.get("output_tokens"))
    if not found:
        return None
    return NeMoGymResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
        output_tokens=output_tokens,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
        total_tokens=input_tokens + output_tokens,
    )


def _content_blocks(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def parse_claude_code_stream(lines: Iterable[str]) -> ClaudeCodeStream:
    """Convert ``claude -p --output-format stream-json --verbose`` output into Responses items.

    Only the main conversation is converted. Subagent traffic carries a ``parent_tool_use_id``
    and stays out of the transcript; the main conversation sees a subagent only through its tool
    call and result. Messages Claude Code makes up itself (model ``<synthetic>``, e.g. API error
    notices) are not model output and are left out too. Claude Code emits one ``assistant`` event
    per content block, all with the same message id, so usage is taken once per message id
    unless the final ``result`` event reports the session total. The counters cover the main
    conversation and subagents alike.
    """
    output: List[NeMoGymResponseOutputItem] = []
    usage_by_message: Dict[str, Dict[str, Any]] = {}
    result: Optional[Dict[str, Any]] = None
    compacting: set[str] = set()
    compaction_attempts: List[Dict[str, str]] = []
    compaction_boundaries = 0
    invalid_lines = 0
    main_messages: set[str] = set()
    subagent_messages: set[str] = set()
    subagent_calls = tool_calls = tool_errors = unknown_tool_calls = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            invalid_lines += 1
            continue
        if not isinstance(event, dict):
            invalid_lines += 1
            continue

        event_type = event.get("type")
        if event_type == "result":
            result = event
            continue
        if event_type == "system":
            session_id = event.get("session_id")
            if event.get("subtype") == "compact_boundary":
                compaction_boundaries += 1
            elif event.get("subtype") == "status" and isinstance(session_id, str) and session_id:
                if event.get("status") == "compacting":
                    compacting.add(session_id)
                elif event.get("compact_result") in {"failed", "success"}:
                    if event["compact_result"] == "failed":
                        compaction_attempts.append({"invocation_id": session_id, "outcome": "failed"})
                    compacting.discard(session_id)
            continue

        message = event.get("message")
        if not isinstance(message, dict):
            continue
        in_subagent = bool(event.get("parent_tool_use_id"))
        blocks = _content_blocks(message.get("content"))
        if event_type == "assistant":
            if message.get("model") == SYNTHETIC_MODEL:
                continue
            message_id = message.get("id")
            if isinstance(message_id, str):
                (subagent_messages if in_subagent else main_messages).add(message_id)
                if isinstance(message.get("usage"), dict):
                    usage_by_message[message_id] = message["usage"]
            for block in blocks:
                if block.get("type") == "tool_use":
                    tool_calls += 1
                    if not in_subagent and block.get("name") in SUBAGENT_TOOLS:
                        subagent_calls += 1
        elif event_type == "user":
            for block in blocks:
                if block.get("type") != "tool_result":
                    continue
                text = _tool_result_text(block.get("content"))
                if block.get("is_error") is True or TOOL_ERROR_TAG in text:
                    tool_errors += 1
                    if UNKNOWN_TOOL_MESSAGE in text:
                        unknown_tool_calls += 1
        if in_subagent:
            continue

        if event_type == "assistant":
            for block in blocks:
                block_type = block.get("type")
                if block_type == "thinking" and block.get("thinking"):
                    output.append(
                        NeMoGymResponseReasoningItem(
                            id=f"rs_{uuid4().hex}",
                            summary=[NeMoGymSummary(text=block["thinking"], type="summary_text")],
                            type="reasoning",
                        )
                    )
                elif block_type == "text" and block.get("text"):
                    text = NeMoGymResponseOutputText(text=block["text"], annotations=[], type="output_text")
                    output.append(
                        NeMoGymResponseOutputMessage(
                            id=f"msg_{uuid4().hex}",
                            content=[text],
                            role="assistant",
                            status="completed",
                            type="message",
                        )
                    )
                elif block_type == "tool_use" and isinstance(block.get("id"), str):
                    output.append(
                        NeMoGymResponseFunctionToolCall(
                            # Serialized as the Gym Messages converter does for the next request.
                            arguments=json.dumps(block.get("input", {})),
                            call_id=block["id"],
                            name=str(block.get("name") or ""),
                            type="function_call",
                            status="completed",
                        )
                    )
        elif event_type == "user":
            for block in blocks:
                if block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                    output.append(
                        NeMoGymFunctionCallOutput(
                            call_id=block["tool_use_id"],
                            output=_tool_result_text(block.get("content")),
                            type="function_call_output",
                        )
                    )

    compaction_attempts.extend({"invocation_id": session_id, "outcome": "unknown"} for session_id in compacting)
    result_usage = result.get("usage") if result is not None else None
    usage = _usage_from_messages_usage([result_usage] if isinstance(result_usage, dict) else usage_by_message.values())
    return ClaudeCodeStream(
        output=output,
        usage=usage,
        result=result,
        compaction_attempts=compaction_attempts,
        compaction_boundaries=compaction_boundaries,
        invalid_lines=invalid_lines,
        main_turns=len(main_messages),
        subagent_turns=len(subagent_messages),
        subagent_calls=subagent_calls,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        unknown_tool_calls=unknown_tool_calls,
    )


def context_overflow(result: Optional[Dict[str, Any]]) -> bool:
    """Whether the session ended because a request no longer fit the model's context."""
    if result is None or result.get("is_error") is not True:
        return False
    text = result.get("result")
    return isinstance(text, str) and text.lstrip().startswith(CONTEXT_OVERFLOW_MESSAGE)


def invocation_outcome(result: Optional[Dict[str, Any]]) -> tuple[str, Optional[str]]:
    """Map the final ``result`` event to an invocation status and error type.

    Claude Code reports an error either as an ``error_*`` subtype or, for an API error such as a
    context overflow, as subtype ``success`` with ``is_error`` set and the error text in ``result``.
    """
    if result is None:
        return "incomplete", "result_missing"
    subtype = result.get("subtype")
    if subtype == "error_max_turns":
        return "incomplete", subtype
    if context_overflow(result):
        return "failed", "context_overflow"
    if isinstance(subtype, str) and subtype.startswith("error"):
        return "failed", subtype
    if result.get("is_error") is True:
        return "failed", "agent_error"
    if subtype == "success":
        return "completed", None
    return "incomplete", "result_unrecognized"


class ClaudeCodeSandboxedAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: ClaudeCodeSandboxedAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._sem = Semaphore(self.config.concurrency)
        self._sandbox_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._sandbox_id_to_run_result: Dict[str, Dict[str, Any]] = dict()

    async def _start_sandbox(self, sandbox_id: Optional[str] = None, workdir: Optional[str] = None) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = create_provider(
            resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        )
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        if sandbox_id:
            return await AsyncSandbox.connect(
                {"sandbox_id": sandbox_id, "workdir": workdir}, provider=resolved_sandbox_provider
            )

        if self.config.debug:
            print("Creating new sandbox since one wasn't provided", file=sys.stderr)

        resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        env = cpu_cap_env(resources.cpu) if self.config.sandbox_config.get("derive_cpu_env", True) else {}
        env |= dict(self.config.sandbox_config.get("env", {}))  # explicit keys win over the derived caps

        sandbox_spec = SandboxSpec(
            image=self.config.sandbox_config.get("image", "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"),
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=self.config.sandbox_config.get("workdir"),
            env=env,
            files=dict(self.config.sandbox_config.get("files", {})),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
            },
            resources=resources,
            entrypoint=self.config.sandbox_config.get("entrypoint"),
            provider_options=deepcopy(self.config.sandbox_config.get("provider_options", {})),
        )
        sandbox = AsyncSandbox(resolved_sandbox_provider)
        await sandbox.start(sandbox_spec)

        return sandbox

    def _agent_sandbox_observation(
        self,
        *,
        sandbox: AsyncSandbox,
        return_code: Any,
        error_type: Any,
        finished: bool,
    ) -> SandboxObservation:
        handle = getattr(sandbox, "_handle", None)
        handle_provider = getattr(handle, "provider_name", None)
        handle_sandbox_id = getattr(handle, "sandbox_id", None)
        normalized_error = error_type.lower() if isinstance(error_type, str) else ""
        if "timeout" in normalized_error or (not normalized_error and return_code == 124):
            outcome = "timeout"
        elif normalized_error:
            outcome = "sandbox_error"
        elif return_code == 0 and finished:
            outcome = "completed"
        elif isinstance(return_code, int):
            outcome = "failed" if return_code != 0 else "unknown"
        else:
            outcome = "unknown"
        return SandboxObservation(
            role="agent",
            provider=handle_provider if isinstance(handle_provider, str) else None,
            sandbox_id=handle_sandbox_id if isinstance(handle_sandbox_id, str) else None,
            outcome=outcome,
            exit_code=return_code if not normalized_error and isinstance(return_code, int) else None,
            error_type=error_type if isinstance(error_type, str) else None,
        )

    def _new_run_paths(self) -> ClaudeCodeRunPaths:
        work_dir = self.config.remote_work_dir.rstrip("/")
        return ClaudeCodeRunPaths(root=f"{work_dir}/nemo-gym-claude-code-{uuid4().hex}")

    def _results_dir(self, request: Request) -> Path:
        results_root = (
            Path(self.config.artifacts_dir) if self.config.artifacts_dir else Path(__file__).parent / "results"
        )
        results_dir = results_root / request.session[SESSION_ID_KEY]
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def _settings(self) -> Dict[str, Any]:
        settings: Dict[str, Any] = {}
        if not self.config.auto_compact:
            settings["autoCompactEnabled"] = False
            # Background summaries for a compaction that will never happen.
            settings["precomputeCompactionEnabled"] = False
        return settings | self.config.claude_code_settings

    def _env(self, base_url: str, paths: ClaudeCodeRunPaths) -> Dict[str, str]:
        model = self.config.model
        env = {
            "CLAUDE_CONFIG_DIR": paths.config_dir,
            # Claude Code appends /v1/messages.
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_API_KEY": PLACEHOLDER_API_KEY,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            # As root, Claude Code accepts --dangerously-skip-permissions only in a declared sandbox.
            "IS_SANDBOX": "1",
            # No update checks, telemetry, error reports or feature-flag fetches.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            # Plain Messages requests: no beta parameters or headers the Gym converter would drop.
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            # Otherwise every request's system prompt carries a per-request attribution line, and a
            # system prompt that changes between calls breaks the capture's call chain.
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        }
        if self.config.max_output_tokens is not None:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.config.max_output_tokens)
        if self.config.api_timeout_ms is not None:
            env["API_TIMEOUT_MS"] = str(self.config.api_timeout_ms)
        if self.config.stream_first_byte_timeout_ms is not None:
            env["CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS"] = str(self.config.stream_first_byte_timeout_ms)
        if not self.config.auto_compact:
            # DISABLE_COMPACT also removes manual /compact and Claude Code's own context cap, so
            # requests run until the model server rejects one, as with compaction-free harnesses.
            env["DISABLE_COMPACT"] = "1"
            env["DISABLE_AUTO_COMPACT"] = "1"
        if not self.config.auto_memory:
            env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        return env | self.config.claude_code_env

    def _install_command(self, paths: ClaudeCodeRunPaths) -> str:
        version = self.config.claude_code_version
        binary = quote(paths.binary)
        if self.config.remote_claude_code_binary_path:
            glibc = quote(self.config.remote_claude_code_binary_path)
            if self.config.remote_claude_code_musl_binary_path:
                musl_branch = f"src={quote(self.config.remote_claude_code_musl_binary_path)}"
            else:
                musl_branch = (
                    'echo "Claude Code: musl-based sandbox, but remote_claude_code_musl_binary_path is not set" >&2'
                    " && false"
                )
            install = (
                "if [ -f /lib/libc.musl-x86_64.so.1 ] || [ -f /lib/libc.musl-aarch64.so.1 ]; "
                f"then {musl_branch}; else src={glibc}; fi"
                f' && cp "$src" {binary} && chmod 755 {binary}'
            )
        else:
            print(
                "Installing Claude Code from the official installer inside the sandbox. "
                "Consider staging the binary and setting remote_claude_code_binary_path instead!",
                file=sys.stderr,
            )
            installer = quote(f"{paths.root}/install.sh")
            install = (
                f"curl -fsSL {quote(OFFICIAL_INSTALLER_URL)} -o {installer}"
                f" && bash {installer} {quote(version)}"
                f' && cp "$HOME/.local/bin/claude" {binary} && chmod 755 {binary}'
            )
        # `claude --version` prints "<version> (Claude Code)".
        version_check = (
            f'case " $({binary} --version) " in *{quote(" " + version + " ")}*) ;; '
            f'*) echo "Claude Code: {binary} does not report version {version}" >&2 && false ;; esac'
        )
        return f"{install} && {version_check}"

    def _claude_args(self, paths: ClaudeCodeRunPaths, query: str) -> List[str]:
        args = [
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--settings",
            paths.settings,
            "--model",
            self.config.model,
        ]
        if self.config.bare:
            args.append("--bare")
        if self.config.setting_sources is not None:
            args += ["--setting-sources", self.config.setting_sources]
        if self.config.allowed_tools:
            args += ["--allowedTools", self.config.allowed_tools]
        if self.config.disallowed_tools:
            args += ["--disallowedTools", self.config.disallowed_tools]
        if self.config.max_turns is not None:
            args += ["--max-turns", str(self.config.max_turns)]
        if self.config.append_system_prompt:
            args += ["--append-system-prompt", self.config.append_system_prompt]
        # `--` ends the options, so a variadic option above cannot swallow the prompt.
        return args + ["--", query]

    def build_command(self, base_url: str, query: str, paths: ClaudeCodeRunPaths) -> str:
        env = " ".join(f"{name}={quote(value)}" for name, value in self._env(base_url, paths).items())
        args = " ".join(quote(arg) for arg in self._claude_args(paths, query))
        deadline_s = int(self.config.sandbox_timeout - self.config.harness_timeout_margin_s)
        limit = (
            f"if command -v timeout >/dev/null 2>&1; then limit='timeout -k 30 {deadline_s}'; else limit=''; fi && "
            if deadline_s > 0
            else "limit='' && "
        )
        return (
            'echo "Shell: $SHELL"'
            f" && mkdir -p {quote(paths.root + '/bin')} {quote(paths.config_dir)}"
            f" && {self._install_command(paths)}"
            ' && echo "Installed Claude Code"'
            f" && printf '%s' {quote(json.dumps(self._settings()))} > {quote(paths.settings)}"
            f" && {limit}{env} $limit {quote(paths.binary)} {args} < /dev/null > {quote(paths.stream)}"
            f' && echo "{FINISHED_MARKER}"'
        )

    async def _model_base_url(self, request: Request) -> str:
        """The model server URL for this rollout; Claude Code appends ``/v1/messages`` itself."""
        return self.base_url_for_run(
            base_url=sandbox_server_url(self.config.model_server.name),
            body=await request.json(),
        )

    async def _download_stream(
        self, sandbox: AsyncSandbox, paths: ClaudeCodeRunPaths, local_path: Path
    ) -> Optional[ClaudeCodeStream]:
        try:
            await sandbox.download(paths.stream, local_path)
        except Exception:
            print(f"Failed to download the Claude Code stream {paths.stream}", format_exc(), file=sys.stderr)
            return None
        if not local_path.is_file():
            return None
        with local_path.open(encoding="utf-8", errors="replace") as lines:
            return parse_claude_code_stream(lines)

    async def _collect_observations(
        self,
        sandbox: AsyncSandbox,
        paths: ClaudeCodeRunPaths,
        stream: Optional[ClaudeCodeStream],
        observation_invocation_id: Optional[str],
    ) -> AgentObservationBundle:
        result = stream.result if stream is not None else None
        status, error_type = invocation_outcome(result)
        duration_ms = result.get("duration_ms") if result is not None else None
        try:
            archived = await sandbox.exec(
                command=f"tar -C {quote(paths.config_dir)} -cf {quote(paths.transcripts_tar)} projects"
            )
            if archived.return_code != 0 or archived.error_type is not None:
                raise RuntimeError(f"archiving the Claude Code transcripts failed: {archived.stderr}")
            with tempfile.TemporaryDirectory(prefix="nemo-gym-claude-code-") as local_dir:
                local_tar = Path(local_dir) / TRANSCRIPTS_TAR_FNAME
                config_dir = Path(local_dir) / "config"
                await sandbox.download(paths.transcripts_tar, local_tar)
                with tarfile.open(local_tar) as archive:
                    archive.extractall(config_dir, filter="data")
                return extract_claude_code_observations(
                    config_dir,
                    model_ref=self.config.model_server,
                    root_status=status,
                    root_duration_ms=duration_ms if isinstance(duration_ms, (int, float)) else None,
                    root_error_type=error_type,
                    compaction_attempts=stream.compaction_attempts if stream is not None else None,
                )
        except Exception:
            print("Failed to capture Claude Code observations", format_exc(), file=sys.stderr)
            return AgentObservationBundle(
                source="claude_code",
                records=(
                    [AgentInvocation(invocation_id=observation_invocation_id)] if observation_invocation_id else []
                ),
                gaps=[
                    ObservationGap(code="agent_artifact_unavailable"),
                    ObservationGap(code="agent_transcript_unavailable"),
                    ObservationGap(code="observation_capture_failed"),
                ],
            )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        if "sandbox_id" not in request.cookies:
            raise ValueError("Use /run: Claude Code needs the sandbox the resources server seeded")
        sandbox = self._sandbox_id_to_sandbox[request.cookies["sandbox_id"]]

        query = None
        # This can be modified to handle system/developer prompts too.
        for input_item in body.input:
            if input_item.role == "user":
                assert not query, body.input
                if isinstance(input_item.content, str):
                    query = input_item.content
                elif isinstance(input_item.content, list):
                    assert len(input_item.content) == 1, body.input
                    query = input_item.content[0]["text"]

        assert query, body.input

        paths = self._new_run_paths()
        command = self.build_command(await self._model_base_url(request), query, paths)
        if self.config.debug:
            print(f"Running command:\n```bash\n{command}\n```\n", file=sys.stderr)

        run_error_type = None
        try:
            result = await sandbox.exec(
                command=command,
                timeout_s=self.config.sandbox_timeout,
            )
        except Exception as exc:
            result = None
            run_error_type = type(exc).__name__
            print("Claude Code exec hit error.", format_exc(), file=sys.stderr)

        if self.config.debug and result:
            print("Claude Code install and run stdout:\n", result.stdout, file=sys.stderr)
            print("Claude Code install and run stderr:\n", result.stderr, file=sys.stderr)

        results_dir = self._results_dir(request)
        local_stream = results_dir / STREAM_FNAME
        local_stream.unlink(missing_ok=True)
        stream = await self._download_stream(sandbox, paths, local_stream)
        if stream is not None:
            if stream.compaction_boundaries and not self.config.auto_compact:
                print(
                    f"Claude Code compacted {stream.compaction_boundaries} time(s) although auto_compact is off",
                    file=sys.stderr,
                )
            if stream.invalid_lines:
                print(f"Claude Code stream had {stream.invalid_lines} unparseable line(s)", file=sys.stderr)

        observation_invocation_id = getattr(request.state, "_ng_observation_invocation_id", None)
        observation_invocation_id = observation_invocation_id if isinstance(observation_invocation_id, str) else None
        collect_observations = observation_invocation_id is not None
        observations = None
        if collect_observations:
            observations = await self._collect_observations(sandbox, paths, stream, observation_invocation_id)

        result_stdout = (result.stdout if result else "") or ""
        result_stderr = (result.stderr if result else "") or ""
        finished = False
        std_out_split = result_stdout.rsplit("Shell: ", maxsplit=1)
        if len(std_out_split) > 1:
            finished = FINISHED_MARKER in std_out_split[1]
        return_code = getattr(result, "return_code", None)
        error_type = getattr(result, "error_type", None) or run_error_type

        if collect_observations and observations is not None:
            agent_sandbox_observation = self._agent_sandbox_observation(
                sandbox=sandbox,
                return_code=return_code,
                error_type=error_type,
                finished=finished,
            )
            for record in observations.records:
                if isinstance(record, ToolCallObservation):
                    record.sandbox_id = agent_sandbox_observation.sandbox_id
                elif isinstance(record, AgentInvocation) and record.parent_invocation_id is None:
                    status = {
                        "completed": "completed",
                        "failed": "failed",
                        "sandbox_error": "failed",
                        "timeout": "incomplete",
                        "cancelled": "incomplete",
                    }.get(agent_sandbox_observation.outcome)
                    if status is not None:
                        record.status = status
            observations.records.append(agent_sandbox_observation)
            observations.gaps.append(ObservationGap(code="sandbox_lifecycle_timing_unavailable"))

        export_found = stream is not None
        stream_result = stream.result if stream is not None else None
        outcome, outcome_error = invocation_outcome(stream_result)
        duration_ms = stream_result.get("duration_ms") if stream_result is not None else None
        run_result: Dict[str, Any] = {
            "claude_code_failed": bool(error_type)
            or return_code != 0
            or not finished
            or not export_found
            or outcome == "failed",
            "claude_code_exit_code": return_code,
            "claude_code_error_type": error_type or (outcome_error if outcome == "failed" else None),
            "claude_code_results_fpath": str(local_stream) if export_found else "",
            "claude_code_run_stdout": result_stdout,
            "claude_code_run_stderr": result_stderr,
            "claude_code_export_found": export_found,
            "claude_code_finished": finished,
            "claude_code_result_success": outcome == "completed",
            "claude_code_result_error": outcome == "failed",
            "claude_code_result_missing": stream_result is None,
            "claude_code_context_overflow": context_overflow(stream_result),
            "claude_code_duration_s": (
                duration_ms / 1000
                if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool)
                else None
            ),
            "claude_code_main_turns": stream.main_turns if stream is not None else 0,
            "claude_code_subagent_turns": stream.subagent_turns if stream is not None else 0,
            "claude_code_subagent_calls": stream.subagent_calls if stream is not None else 0,
            "claude_code_tool_calls": stream.tool_calls if stream is not None else 0,
            "claude_code_tool_errors": stream.tool_errors if stream is not None else 0,
            "claude_code_unknown_tool_calls": stream.unknown_tool_calls if stream is not None else 0,
        }
        if collect_observations:
            run_result["_ng_agent_observations"] = observations
        self._sandbox_id_to_run_result[request.cookies["sandbox_id"]] = run_result

        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.model_server.name,
            object="response",
            output=stream.output if stream is not None else [],
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=stream.usage if stream is not None else None,
        )
        # Written before grading, so a failed or interrupted verification still leaves the
        # generation and how it ended on disk.
        receipt = {
            "response": response.model_dump(mode="json"),
            "execution": {key: value for key, value in run_result.items() if not key.startswith("_ng_")},
        }
        pending = results_dir / "generation.json.partial"
        pending.write_text(json.dumps(receipt))
        pending.replace(results_dir / "generation.json")
        return response

    async def run(
        self, request: Request, body: ClaudeCodeSandboxedAgentRunRequest
    ) -> ClaudeCodeSandboxedAgentVerifyResponse:
        async with self._sem:
            return await self._run(request, body)

    async def _run(
        self, request: Request, body: ClaudeCodeSandboxedAgentRunRequest
    ) -> ClaudeCodeSandboxedAgentVerifyResponse:
        cookies = request.cookies
        session_key = request.session[SESSION_ID_KEY]
        rollout_id = self.rollout_id_from_run(body)

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = cookies | seed_session_response.cookies

        # "sandbox_handle" and "workdir" come from the SWE resources servers' seed_session responses.
        seed_session_result = await seed_session_response.json()
        sandbox = await self._start_sandbox(
            sandbox_id=seed_session_result.get("sandbox_handle"),
            workdir=seed_session_result.get("workdir"),
        )
        self._sandbox_id_to_sandbox[session_key] = sandbox

        # Propagating the sandbox handle
        cookies["sandbox_id"] = session_key

        request._cookies = cookies
        request.state._ng_observation_invocation_id = rollout_id
        observations = None
        try:
            response = await self.responses(request, body.responses_create_params)
            run_result = self._sandbox_id_to_run_result.get(session_key, {}).copy()
            observations = run_result.pop("_ng_agent_observations", None)
            response_dict = await verify_agent_response(
                self.server_client,
                self.config.resources_server,
                body,
                response,
                cookies,
                force_zero_reward=self.config.execution_failure_reward_zero
                and run_result.get("claude_code_failed", False),
            )
        finally:
            del request.state._ng_observation_invocation_id
            try:
                await sandbox.stop()
            except Exception:
                print("Failed to stop sandbox", format_exc(), file=sys.stderr)
            finally:
                self._sandbox_id_to_sandbox.pop(session_key, None)
                self._sandbox_id_to_run_result.pop(session_key, None)

        response_dict |= run_result
        raw_verifier_sandbox_observation = response_dict.pop("verifier_sandbox_observation", None)
        if rollout_id is not None:
            if observations is None:
                observations = AgentObservationBundle(
                    source="claude_code",
                    records=[AgentInvocation(invocation_id=rollout_id)],
                    gaps=[ObservationGap(code="observation_capture_failed")],
                )
            if raw_verifier_sandbox_observation is not None:
                try:
                    verifier_observation = SandboxObservation.model_validate(raw_verifier_sandbox_observation)
                    if verifier_observation.role != "verifier":
                        raise ValueError("resources server returned a non-verifier sandbox observation")
                    observations.records.append(verifier_observation)
                except Exception:
                    observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_invalid"))
            else:
                observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_unavailable"))
            response_dict["ng_agent_observations"] = observations.model_dump(mode="json")
        return ClaudeCodeSandboxedAgentVerifyResponse.model_validate(response_dict)


if __name__ == "__main__":
    ClaudeCodeSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ClaudeCodeSandboxedAgent.run_webserver()  # noqa: F401
