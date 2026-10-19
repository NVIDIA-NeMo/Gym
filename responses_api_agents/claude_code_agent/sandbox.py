# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claude Code sessions with owned or borrowed sandboxes and native tools."""

import asyncio
import json
import logging
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from time import monotonic, time
from uuid import uuid4

from fastapi import Body, HTTPException, Request

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
)
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
from responses_api_agents.claude_code_agent.observability import extract_claude_code_observations
from responses_api_agents.claude_code_agent.parsing import _extract_instruction, _invocation_outcome, parse_stream_json


LOG = logging.getLogger(__name__)


def extract_prompt(
    body: NeMoGymResponseCreateParamsNonStreaming, *, model: str | None = None
) -> tuple[str, str | None]:
    """Validate the single-turn CLI input before using the shared converter."""
    allowed = {"input", "instructions", "model", "metadata", "timeout_seconds"}
    for name, info in type(body).model_fields.items():
        if name not in allowed and getattr(body, name) != info.get_default(call_default_factory=True):
            raise ValueError(f"Sandboxed Claude does not support request field {name}")
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


@dataclass
class ClaudeCodeSandboxedState(AgentSessionState):
    session: SandboxSession[str]
    executable: str = ""
    version: str = ""
    python_executable: str = "python3"
    body: NeMoGymResponseCreateParamsNonStreaming | None = None
    execution: asyncio.Task[NeMoGymResponse] | None = None
    observations: AgentObservationBundle | None = None


class ClaudeCodeSandboxMixin:
    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> ClaudeCodeSandboxedState:
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise ValueError("Claude Code sandbox sessions use native tools; remote tools are unsupported")
        version = self.config.claude_code_version
        if not version or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
            raise ValueError("Sandboxed Claude Code requires a pinned claude_code_version")
        access = body.sandbox_access
        owns = access is None
        if owns and (self.config.sandbox_provider is None or self.config.sandbox_spec is None):
            raise ValueError("Provide SandboxAccess or configure sandbox_provider and sandbox_spec")
        provider = create_provider(
            resolve_provider_config(
                access.connection.provider_config_ref if access else self.config.sandbox_provider,
                self.server_client.global_config_dict,
            )
        )
        deadline = monotonic() + min(self.config.sandbox_install_timeout_seconds, body.timeout_seconds or float("inf"))
        if owns:
            sandbox = AsyncSandbox(provider)
        else:
            try:
                async with asyncio.timeout_at(deadline):
                    sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        directory = "/tmp/nemo-gym-claude-" + uuid4().hex
        workdir = access.workdir if access else self.config.sandbox_spec.workdir
        session = SandboxSession(
            sandbox=sandbox, session_dir=directory, workdir=workdir, harness="claude_code", owns_sandbox=owns
        )
        state = ClaudeCodeSandboxedState(
            request=body, session=session, python_executable=self.config.python_executable
        )
        try:
            async with asyncio.timeout_at(deadline):
                if owns:
                    await sandbox.start(self.config.sandbox_spec)
                    if workdir:
                        await sandbox.exec("mkdir -p -- " + quote(workdir), timeout_s=30)
                await sandbox.exec(
                    "umask 077; mkdir -p " + quote(directory) + "/home/.claude " + quote(directory) + "/runtime",
                    timeout_s=30,
                )
                await sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), directory + "/sandbox_runner.py")
                await sandbox.upload(Path(__file__).with_name("install_claude_runtime.py"), directory + "/install.py")
                source = self.config.executable_path or ""
                if self.config.local_executable_path:
                    source = directory + "/claude-upload"
                    await sandbox.upload(Path(self.config.local_executable_path), source)
                executable = directory + "/runtime/claude"
                command = " ".join(
                    quote(value)
                    for value in [
                        state.python_executable,
                        directory + "/install.py",
                        version,
                        executable,
                        source,
                        self.config.executable_sha256 or "",
                    ]
                )
                result = await sandbox.exec(command, timeout_s=max(0.001, deadline - monotonic()))
                if result.return_code or result.error_type:
                    raise RuntimeError(
                        f"Claude runtime installation failed: command={command!r}, exit={result.return_code}, "
                        f"error={result.error_type}; stderr={result.stderr}; stdout={result.stdout}"
                    )
                state.version = (result.stdout or "").strip().splitlines()[-1]
                if not state.version.startswith(version + " "):
                    raise RuntimeError(f"Claude version mismatch: expected {version}, got {state.version}")
                state.executable = executable
            return state
        except BaseException as error:
            try:
                await session.close(timeout=self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise

    async def _close_agent_session_state(self, state: ClaudeCodeSandboxedState) -> AgentCloseSessionResponse:
        await state.session.close(timeout=self.config.session_close_timeout_seconds)
        if state.execution and not state.execution.done():
            state.execution.cancel()
            try:
                await state.execution
            except asyncio.CancelledError:
                pass
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.session.cleanup_confirmed,
            agent_observations=state.observations,
        )

    def _execution_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str | None, float]:
        prompt, system = extract_prompt(body, model=self.config.model)
        system = "\n\n".join(part for part in (self.config.system_prompt, system) if part) or None
        timeout = min(self.config.timeout, body.timeout_seconds or self.config.timeout)
        return prompt, system, monotonic() + timeout

    async def _execute(
        self,
        state: ClaudeCodeSandboxedState,
        body: NeMoGymResponseCreateParamsNonStreaming,
        execution_input: tuple[str, str | None, float],
    ) -> NeMoGymResponse:
        session = state.session
        directory = session.session_dir
        prompt, system, deadline = execution_input

        async def stage() -> SandboxCommand:
            env = {
                "HOME": directory + "/home",
                "CLAUDE_CONFIG_DIR": directory + "/home/.claude",
                "IS_SANDBOX": "1",
                "ANTHROPIC_API_KEY": self.config.anthropic_api_key or "EMPTY",
                "ANTHROPIC_MODEL": self.config.model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": self.config.model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": self.config.model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": self.config.model,
                "ANTHROPIC_AUTH_TOKEN": self.config.anthropic_api_key or "EMPTY",
                "ANTHROPIC_BASE_URL": self._resolve_call_base_url(state.request.episode_id.capture_key),
                "DISABLE_AUTOUPDATER": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
            await upload_text(
                session.sandbox,
                path=directory + "/home/.claude/settings.json",
                text=json.dumps(self._build_settings()),
            )
            mcp_config = None
            if self.config.mcp_config:
                mcp_config = directory + "/mcp.json"
                await upload_text(session.sandbox, path=mcp_config, text=json.dumps(self._get_static_mcp_config()))
            argv = self._build_command(self.config.model, prompt, system, mcp_config=mcp_config)
            argv[0] = state.executable
            if self.config.debug_log:
                argv[1:1] = ["--debug-file", directory + "/debug.log"]
            await upload_text(
                session.sandbox, path=directory + "/launch.json", text=json.dumps({"argv": argv, "env": env})
            )
            return SandboxCommand(
                argv=[state.python_executable, directory + "/sandbox_runner.py", directory + "/launch.json"],
                python=state.python_executable,
            )

        parsed = None
        outcome = None

        def classify(metadata):
            if session.cleanup and session.cleanup["timed_out"]:
                return "incomplete", "timeout"
            return _invocation_outcome(metadata, (session.cleanup or {}).get("return_code"))

        async def collect() -> str:
            nonlocal parsed, outcome
            raw = await session.read_output_log()
            parsed = parse_stream_json(raw)
            _, run_metadata = parsed
            outcome = classify(run_metadata)
            status, reason = outcome
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

        async with asyncio.timeout_at(deadline):
            await self.sem.acquire()
        try:
            raw = await session.execute(
                stage_activation=stage,
                collect=collect,
                timeout=max(0.001, deadline - monotonic()),
                close_timeout=self.config.session_close_timeout_seconds,
                deadline=deadline,
            )
        finally:
            self.sem.release()
        items, usage = parsed if parsed is not None else parse_stream_json(raw)
        metadata = {
            "runtime_version": state.version,
            "raw_log": raw,
            "timed_out": "true" if session.cleanup and session.cleanup["timed_out"] else "false",
        }
        status, reason = outcome or classify(usage)
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

    async def _sandbox_responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if not session_id:
            raise HTTPException(400, "Seed a sandboxed agent session first")
        async with self._locked_agent_session(session_id) as record:
            if record.closing or not isinstance(record.state, ClaudeCodeSandboxedState):
                raise HTTPException(409, "Agent session is closed")
            state = record.state
            rollout_id = getattr(request, "path_params", {}).get("rollout_id")
            if rollout_id is not None and rollout_id != state.request.episode_id.capture_key:
                raise HTTPException(409, "Claude activation rollout does not match its session")
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
