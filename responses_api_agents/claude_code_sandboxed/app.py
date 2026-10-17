# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-activation Claude Code adapter borrowing an owner-managed sandbox.

The adapter owns its pinned runtime and process lifecycle, and never creates,
verifies, or destroys the task sandbox. It is usable with any native benchmark.
"""

import asyncio
import json
import logging
import math
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from time import time
from uuid import uuid4

from fastapi import Body, HTTPException, Request
from pydantic import Field

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap
from nemo_gym.sandbox import create_provider
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.utils import upload_text
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.claude_code_agent.app import _extract_instruction, _invocation_outcome, parse_stream_json
from responses_api_agents.claude_code_agent.observability import extract_claude_code_observations


LOG = logging.getLogger(__name__)


class ClaudeCodeSandboxedConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    model: str = "claude-opus-4-6"
    claude_code_version: str = "2.1.108"
    max_turns: int = Field(default=50, gt=0)
    timeout: float = Field(default=1200, gt=0)
    close_timeout: float = Field(default=90, gt=0)
    executable_path: str = "/opt/gym-runtime/claude"
    executable_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    python_executable: str = "/opt/gym-runtime/python/bin/python3"
    debug_log: bool = False


def extract_prompt(
    body: NeMoGymResponseCreateParamsNonStreaming, *, model: str | None = None
) -> tuple[str, str | None]:
    """Validate the single-turn CLI input before using the shared converter."""
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
            raise ValueError(
                f"Sandboxed Claude does not support request field {name}; configure the native agent instead"
            )
    if body.model is not None and body.model != model:
        raise ValueError("Sandboxed Claude model is selected by agent and model-server configuration")
    if body.model_extra:
        raise ValueError(f"Sandboxed Claude does not support extra request fields: {sorted(body.model_extra)}")
    if body.tools or body.tool_choice != "auto" or not body.parallel_tool_calls or body.background:
        raise ValueError("Sandboxed Claude owns tool selection and execution policy")
    raw = body.model_dump(mode="json")["input"]
    if isinstance(raw, str):
        return raw, body.instructions
    users = []
    systems = []
    for item in raw:
        role = item.get("role")
        if item.get("type", "message") != "message" or role not in {"user", "system", "developer"}:
            raise ValueError("Sandboxed Claude requires a single user message with optional system instructions")
        content = item.get("content")
        if isinstance(content, list):
            if any(part.get("type") not in {"input_text", "text"} for part in content):
                raise ValueError("Sandboxed Claude accepts text input only")
            content = "".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("Sandboxed Claude accepts text input only")
        if role == "user":
            users.append(content)
        elif users:
            raise ValueError("System instructions must precede the user turn")
        else:
            systems.append(content)
    if len(users) != 1:
        raise ValueError("Sandboxed Claude requires exactly one user message")
    system = "\n\n".join(systems + ([body.instructions] if body.instructions else [])) or None
    return _extract_instruction(
        ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": users[0]}]
    )


def remaining_seconds(deadline: float | None) -> float:
    remaining = deadline - time() if deadline is not None else float("inf")
    if remaining <= 0:
        raise TimeoutError("Claude activation deadline expired")
    return remaining


@dataclass
class ClaudeCodeSandboxedState(AgentSessionState):
    session: SandboxSession[str]
    executable: str = ""
    version: str = ""
    python_executable: str = "python3"
    body: NeMoGymResponseCreateParamsNonStreaming | None = None
    execution: asyncio.Task[NeMoGymResponse] | None = None
    observations: AgentObservationBundle | None = None


class ClaudeCodeSandboxedAgent(SimpleResponsesAPIAgent):
    config: ClaudeCodeSandboxedConfig
    ray_enabled = False

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> ClaudeCodeSandboxedState:
        if body.sandbox_access is None or body.tool_accesses:
            raise ValueError("Sandboxed Claude Code requires SandboxAccess and no remote tools")
        access = body.sandbox_access
        provider = resolve_provider_config(
            access.connection.provider_config_ref, self.server_client.global_config_dict
        )
        sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=create_provider(provider))
        directory = "/tmp/nemo-gym-claude-" + uuid4().hex
        session = SandboxSession(
            sandbox=sandbox, session_dir=directory, workdir=access.workdir, harness="claude_code", owns_sandbox=False
        )
        state = ClaudeCodeSandboxedState(request=body, session=session)
        try:
            state.python_executable = self.config.python_executable
            async with asyncio.timeout(remaining_seconds(body.setup_deadline_unix_seconds)):
                result = await sandbox.exec(
                    f"umask 077; mkdir -p {quote(directory)}/home {quote(directory)}/runtime; "
                    + quote(state.python_executable)
                    + " --version",
                    timeout_s=30,
                )
                if result.return_code or result.error_type:
                    raise RuntimeError("Cannot initialize Claude session with provisioned Python")
                executable = directory + "/runtime/claude"
                command = (
                    "cp -- "
                    + quote(self.config.executable_path)
                    + " "
                    + quote(executable)
                    + " && chmod 700 "
                    + quote(executable)
                )
                if self.config.executable_sha256:
                    command = (
                        "printf '%s  %s\n' "
                        + quote(self.config.executable_sha256)
                        + " "
                        + quote(self.config.executable_path)
                        + " | sha256sum -c - && "
                        + command
                    )
                result = await sandbox.exec(command, timeout_s=120)
                if result.return_code or result.error_type:
                    raise RuntimeError("Provisioned Claude executable is unavailable or failed SHA-256 verification")
            async with asyncio.timeout(remaining_seconds(body.setup_deadline_unix_seconds)):
                result = await sandbox.exec(quote(executable) + " --version", timeout_s=30)
            state.version = (result.stdout or "").strip()
            if result.return_code or not state.version.startswith(self.config.claude_code_version + " "):
                raise RuntimeError(
                    f"Claude version mismatch: expected {self.config.claude_code_version}, got {state.version}"
                )
            state.executable = executable
            return state
        except BaseException as error:
            try:
                await session.close(timeout=self.config.close_timeout)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise

    async def _close_agent_session_state(self, state: ClaudeCodeSandboxedState) -> AgentCloseSessionResponse:
        await state.session.close(timeout=self.config.close_timeout)
        if state.execution and not state.execution.done():
            state.execution.cancel()
            try:
                await state.execution
            except asyncio.CancelledError:
                pass
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=True,
            agent_observations=state.observations,
        )

    def _execution_input(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> tuple[str, str | None, float, float | None]:
        prompt, system = extract_prompt(body, model=self.config.model)
        metadata = body.metadata or {}
        timeout = float(metadata.get("timeout_seconds", self.config.timeout))
        deadline = float(metadata["deadline_unix_seconds"]) if "deadline_unix_seconds" in metadata else None
        if not math.isfinite(timeout) or not 0 < timeout <= self.config.timeout:
            raise ValueError("Invalid activation timeout")
        if deadline is not None and (not math.isfinite(deadline) or deadline <= 0):
            raise ValueError("Invalid activation deadline")
        return prompt, system, timeout, deadline

    async def _execute(
        self,
        state: ClaudeCodeSandboxedState,
        body: NeMoGymResponseCreateParamsNonStreaming,
        execution_input: tuple[str, str | None, float, float | None],
    ) -> NeMoGymResponse:
        session = state.session
        directory = session.session_dir
        prompt, system, timeout, deadline = execution_input

        async def stage() -> SandboxCommand:
            env = {
                "HOME": directory + "/home",
                "CLAUDE_CONFIG_DIR": directory + "/home/.claude",
                "IS_SANDBOX": "1",
                "ANTHROPIC_API_KEY": "EMPTY",
                "ANTHROPIC_MODEL": self.config.model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": self.config.model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": self.config.model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": self.config.model,
                "ANTHROPIC_AUTH_TOKEN": "EMPTY",
                "ANTHROPIC_BASE_URL": self.resolve_model_base_url(
                    self.config.model_server.name, state.request.episode_id.capture_key
                ).removesuffix("/v1"),
                "DISABLE_AUTOUPDATER": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
            argv = [
                state.executable,
                "--print",
                "--max-turns",
                str(self.config.max_turns),
                "--model",
                self.config.model,
                "--dangerously-skip-permissions",
                "--setting-sources",
                "user",
                "--output-format",
                "stream-json",
                "--verbose",
            ]
            if self.config.debug_log:
                argv += ["--debug-file", directory + "/debug.log"]
            if system:
                argv += ["--append-system-prompt", system]
            argv += ["--", prompt]
            await upload_text(
                session.sandbox, path=directory + "/launch.json", text=json.dumps({"argv": argv, "env": env})
            )
            await upload_text(
                session.sandbox,
                path=directory + "/launch.py",
                text='import json,os,sys\nr=json.load(open(sys.argv[1]))\nos.environ.update(r["env"])\nos.execvpe(r["argv"][0],r["argv"],os.environ)\n',
            )
            return SandboxCommand(
                argv=[state.python_executable, directory + "/launch.py", directory + "/launch.json"],
                python=state.python_executable,
            )

        async def collect() -> str:
            raw = await session.read_output_log()
            _, run_metadata = parse_stream_json(raw)
            status, reason = _invocation_outcome(run_metadata, (session.cleanup or {}).get("return_code"))
            if session.cleanup and session.cleanup["timed_out"]:
                status, reason = "incomplete", "timeout"
            try:
                result = await session.sandbox.exec(
                    "tar -czf "
                    + quote(directory + "/transcripts.tar.gz")
                    + " -C "
                    + quote(directory + "/home/.claude")
                    + " projects",
                    timeout_s=30,
                )
                if result.return_code or result.error_type:
                    raise RuntimeError("Claude native transcripts are unavailable")
                with tempfile.TemporaryDirectory(prefix="gym-claude-observations-") as temporary:
                    local = Path(temporary)
                    await session.sandbox.download(directory + "/transcripts.tar.gz", local / "transcripts.tar.gz")
                    with tarfile.open(local / "transcripts.tar.gz") as archive:
                        archive.extractall(local, filter="data")
                    state.observations = extract_claude_code_observations(
                        local,
                        model_ref=self.config.model_server,
                        root_status=status,
                        root_duration_ms=run_metadata.get("duration_ms"),
                        root_error_type=reason,
                        compaction_attempts=run_metadata.get("compaction_attempts"),
                    )
            except Exception:
                LOG.warning("Could not collect Claude native observations", exc_info=True)
                state.observations = AgentObservationBundle(
                    source="claude_code", gaps=[ObservationGap(code="agent_transcript_unavailable")]
                )
            return raw

        raw = await session.execute(
            stage_activation=stage,
            collect=collect,
            timeout=timeout,
            close_timeout=self.config.close_timeout,
            timeout_resolver=lambda: remaining_seconds(deadline),
        )
        items, usage = parse_stream_json(raw)
        metadata = {
            "runtime_version": state.version,
            "raw_log": raw,
            "timed_out": str(bool(session.cleanup and session.cleanup["timed_out"])),
        }
        for line in raw.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                # Terminal usage includes the whole invocation; do not count it
                # again alongside the per-message usage in the stream parser.
                if isinstance(event.get("usage"), dict):
                    usage.update(event["usage"])
        status, reason = _invocation_outcome(usage, (session.cleanup or {}).get("return_code"))
        if metadata["timed_out"] == "True":
            status, reason = "incomplete", "timeout"
        if reason:
            metadata["stop_reason"] = reason
        if self.config.debug_log:
            debug = await session.sandbox.exec("cat " + quote(directory + "/debug.log"), timeout_s=30)
            metadata["debug_log"] = debug.stdout or ""
        return NeMoGymResponse(
            id="resp_" + uuid4().hex,
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=items,
            tools=body.tools,
            tool_choice=body.tool_choice,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            ),
            status=status,
            metadata=metadata,
        )

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if not session_id:
            raise HTTPException(400, "Seed a sandboxed agent session first")
        async with self._locked_agent_session(session_id) as record:
            if record.closing or not isinstance(record.state, ClaudeCodeSandboxedState):
                raise HTTPException(409, "Agent session is closed")
            state = record.state
            if state.body is not None and state.body != body:
                raise HTTPException(409, "A different activation was already submitted")
            if state.execution is None:
                try:
                    execution_input = self._execution_input(body)
                except (ValueError, TypeError) as error:
                    raise HTTPException(422, str(error)) from error
                state.body = body.model_copy(deep=True)
                state.execution = asyncio.create_task(self._execute(state, body, execution_input))
            execution = state.execution
        return await asyncio.shield(execution)

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        raise HTTPException(400, "Use an Environment Server and agent sessions")


if __name__ == "__main__":
    ClaudeCodeSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    from nemo_gym.server_utils import create_nemo_gym_fastapi_app

    app = create_nemo_gym_fastapi_app(ClaudeCodeSandboxedAgent)
