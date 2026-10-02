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

``SandboxedHarnessAgent`` attaches to the sandbox the resources server seeded, runs this agent's
command once under ``sandbox_timeout`` and then calls ``_harness_collect``. The command:

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
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from traceback import format_exc
from typing import Any, ClassVar, Dict, Iterable, List, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import Field

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
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
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_utils import SESSION_ID_KEY, get_server_url, is_nemo_gym_fastapi_entrypoint
from responses_api_agents.claude_code_agent.observability import extract_claude_code_observations
from responses_api_agents.sandboxed_harness_agent.app import (
    HarnessTranscript,
    SandboxedHarnessAgent,
    SandboxedHarnessAgentConfig,
    SandboxedHarnessAgentRunRequest,
    SandboxedHarnessAgentVerifyRequest,
    SandboxedHarnessAgentVerifyResponse,
)


STREAM_FNAME = "stream.jsonl"
TRANSCRIPTS_TAR_FNAME = "transcripts.tar"
OFFICIAL_INSTALLER_URL = "https://claude.ai/install.sh"
# The Gym model server does not authenticate; Claude Code only needs some credential to start.
PLACEHOLDER_API_KEY = "nemo-gym"  # pragma: allowlist secret
# Claude Code's model name for messages it makes up itself, such as API error notices.
SYNTHETIC_MODEL = "<synthetic>"


class ClaudeCodeSandboxedAgentConfig(SandboxedHarnessAgentConfig):
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


ClaudeCodeSandboxedAgentRunRequest = SandboxedHarnessAgentRunRequest
ClaudeCodeSandboxedAgentVerifyRequest = SandboxedHarnessAgentVerifyRequest


class ClaudeCodeSandboxedAgentVerifyResponse(SandboxedHarnessAgentVerifyResponse):
    claude_code_results_fpath: str
    claude_code_run_stdout: str
    claude_code_run_stderr: str
    claude_code_finished: bool
    claude_code_export_found: bool


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


def parse_claude_code_stream(lines: Iterable[str]) -> ClaudeCodeStream:
    """Convert ``claude -p --output-format stream-json --verbose`` output into Responses items.

    Only the main conversation is converted. Subagent traffic carries a ``parent_tool_use_id``
    and stays out of the transcript; the main conversation sees a subagent only through its tool
    call and result. Messages Claude Code makes up itself (model ``<synthetic>``, e.g. API error
    notices) are not model output and are left out too. Claude Code emits one ``assistant`` event
    per content block, all with the same message id, so usage is taken once per message id
    unless the final ``result`` event reports the session total.
    """
    output: List[NeMoGymResponseOutputItem] = []
    usage_by_message: Dict[str, Dict[str, Any]] = {}
    result: Optional[Dict[str, Any]] = None
    compacting: set[str] = set()
    compaction_attempts: List[Dict[str, str]] = []
    compaction_boundaries = 0
    invalid_lines = 0

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
        if event_type == "assistant":
            if message.get("model") == SYNTHETIC_MODEL:
                continue
            message_id = message.get("id")
            if isinstance(message_id, str) and isinstance(message.get("usage"), dict):
                usage_by_message[message_id] = message["usage"]
        if event.get("parent_tool_use_id"):
            continue

        content = message.get("content")
        if event_type == "assistant":
            if isinstance(content, str):
                blocks: List[Any] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                blocks = content
            else:
                blocks = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
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
        elif event_type == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if not isinstance(block.get("tool_use_id"), str):
                        continue
                    output.append(
                        NeMoGymFunctionCallOutput(
                            call_id=block["tool_use_id"],
                            output=_tool_result_text(block.get("content")),
                            type="function_call_output",
                        )
                    )

    compaction_attempts.extend({"invocation_id": session_id, "outcome": "unknown"} for session_id in compacting)
    result_usage = result.get("usage") if result is not None else None
    usage = _usage_from_messages_usage(
        [result_usage] if isinstance(result_usage, dict) else usage_by_message.values()
    )
    return ClaudeCodeStream(
        output=output,
        usage=usage,
        result=result,
        compaction_attempts=compaction_attempts,
        compaction_boundaries=compaction_boundaries,
        invalid_lines=invalid_lines,
    )


def invocation_outcome(result: Optional[Dict[str, Any]]) -> tuple[str, Optional[str]]:
    """Map the final ``result`` event to an invocation status and error type."""
    if result is None:
        return "incomplete", "result_missing"
    subtype = result.get("subtype")
    if subtype == "error_max_turns":
        return "incomplete", subtype
    if result.get("is_error") is True or (isinstance(subtype, str) and subtype.startswith("error")):
        return "failed", subtype if isinstance(subtype, str) else "agent_error"
    if subtype == "success":
        return "completed", None
    return "incomplete", "result_unrecognized"


class ClaudeCodeSandboxedAgent(SandboxedHarnessAgent):
    config: ClaudeCodeSandboxedAgentConfig

    harness_name: ClassVar[str] = "Claude Code"
    harness_id: ClassVar[str] = "claude_code"
    finished_marker: ClassVar[str] = "Claude Code run finished"
    verify_response_class: ClassVar[type[SandboxedHarnessAgentVerifyResponse]] = (
        ClaudeCodeSandboxedAgentVerifyResponse
    )

    def _new_run_paths(self) -> ClaudeCodeRunPaths:
        work_dir = self.config.remote_work_dir.rstrip("/")
        return ClaudeCodeRunPaths(root=f"{work_dir}/nemo-gym-claude-code-{uuid4().hex}")

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
            f' && echo "{self.finished_marker}"'
        )

    async def _harness_command(
        self, request: Request, query: str, collect_observations: bool
    ) -> tuple[str, ClaudeCodeRunPaths]:
        base_url = self.base_url_for_run(
            base_url=get_server_url(self.config.model_server.name),
            body=await request.json(),
        )
        paths = self._new_run_paths()
        command = self.build_command(base_url, query, paths)
        if self.config.debug:
            print(f"Running command:\n```bash\n{command}\n```\n", file=sys.stderr)
        return command, paths

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

    async def _harness_collect(
        self,
        request: Request,
        sandbox: AsyncSandbox,
        state: ClaudeCodeRunPaths,
        collect_observations: bool,
        observation_invocation_id: Optional[str],
    ) -> HarnessTranscript:
        results_dir = Path(__file__).parent / "results" / request.session[SESSION_ID_KEY]
        results_dir.mkdir(parents=True, exist_ok=True)
        local_stream = results_dir / STREAM_FNAME
        local_stream.unlink(missing_ok=True)

        stream = await self._download_stream(sandbox, state, local_stream)
        if stream is not None:
            if stream.compaction_boundaries and not self.config.auto_compact:
                print(
                    f"Claude Code compacted {stream.compaction_boundaries} time(s) although auto_compact is off",
                    file=sys.stderr,
                )
            if stream.invalid_lines:
                print(f"Claude Code stream had {stream.invalid_lines} unparseable line(s)", file=sys.stderr)

        observations = None
        if collect_observations:
            observations = await self._collect_observations(sandbox, state, stream, observation_invocation_id)

        return HarnessTranscript(
            output=stream.output if stream is not None else [],
            usage=stream.usage if stream is not None else None,
            results_fpath=local_stream if stream is not None else None,
            observations=observations,
        )

    async def run(
        self, request: Request, body: ClaudeCodeSandboxedAgentRunRequest
    ) -> ClaudeCodeSandboxedAgentVerifyResponse:
        # FastAPI builds the /run request and response models from these annotations.
        return await super().run(request, body)


if __name__ == "__main__":
    ClaudeCodeSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ClaudeCodeSandboxedAgent.run_webserver()  # noqa: F401
