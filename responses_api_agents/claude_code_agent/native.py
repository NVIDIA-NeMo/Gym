# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-activation Claude Code adapter borrowing an owner-managed sandbox.

The adapter owns its pinned runtime and process lifecycle, and never creates,
verifies, or destroys the task sandbox. It is usable with any native benchmark.
"""

import asyncio
import json
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
from nemo_gym.sandbox import create_provider
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.python_runtime import ensure_python
from nemo_gym.sandbox.utils import upload_text
from responses_api_agents.claude_code_agent.app import _invocation_outcome, parse_stream_json


class NativeClaudeCodeConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    model: str = "claude-opus-4-6"
    claude_code_version: str = "2.1.108"
    max_turns: int = Field(default=50, gt=0)
    timeout: float = Field(default=1200, gt=0)
    close_timeout: float = Field(default=90, gt=0)
    runtime_archive: str | None = None
    runtime_executable: str | None = None
    install_runtime: bool = True
    debug_log: bool = False
    python_runtime_url: str | None = None
    python_runtime_sha256: str | None = None


@dataclass
class NativeClaudeState(AgentSessionState):
    session: SandboxSession[str]
    executable: str = ""
    version: str = ""
    python_executable: str = "python3"
    body: NeMoGymResponseCreateParamsNonStreaming | None = None
    execution: asyncio.Task | None = None


class NativeClaudeCodeAgent(SimpleResponsesAPIAgent):
    config: NativeClaudeCodeConfig
    ray_enabled = False

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> NativeClaudeState:
        if body.sandbox_access is None or body.tool_accesses:
            raise ValueError("Native Claude Code requires SandboxAccess and no remote tools")
        access = body.sandbox_access
        provider = resolve_provider_config(
            access.connection.provider_config_ref, self.server_client.global_config_dict
        )
        sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=create_provider(provider))
        directory = "/tmp/nemo-gym-claude-" + uuid4().hex
        session = SandboxSession(
            sandbox=sandbox, session_dir=directory, workdir=access.workdir, harness="claude_code", owns_sandbox=False
        )
        state = NativeClaudeState(request=body, session=session)
        try:
            state.python_executable = await ensure_python(
                sandbox, runtime_url=self.config.python_runtime_url, runtime_sha256=self.config.python_runtime_sha256
            )
            result = await sandbox.exec(
                f"umask 077; mkdir -p {quote(directory)}/home {quote(directory)}/runtime", timeout_s=30
            )
            if result.return_code:
                raise RuntimeError("Cannot initialize private Claude runtime")
            executable = self.config.runtime_executable
            if not executable:
                result = await sandbox.exec(
                    'export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"; command -v claude', timeout_s=15
                )
                existing = (result.stdout or "").strip()
                if existing:
                    checked = await sandbox.exec(quote(existing) + " --version", timeout_s=30)
                    if checked.return_code == 0 and (checked.stdout or "").strip().startswith(
                        self.config.claude_code_version + " "
                    ):
                        executable = existing
            if not executable and self.config.runtime_archive:
                await sandbox.upload(Path(self.config.runtime_archive), directory + "/runtime.tar.gz")
                result = await sandbox.exec(
                    f"tar -xzf {quote(directory)}/runtime.tar.gz -C {quote(directory)}/runtime", timeout_s=120
                )
                if result.return_code:
                    raise RuntimeError("Cannot unpack pinned Claude runtime")
                executable = directory + "/runtime/claude"
            if not executable and self.config.install_runtime:
                result = await sandbox.exec(
                    f"HOME={quote(directory)}/home bash -c "
                    + quote(
                        "curl -fsSL https://claude.ai/install.sh | bash -s -- "
                        + quote(self.config.claude_code_version)
                    ),
                    timeout_s=240,
                )
                if result.return_code:
                    raise RuntimeError("Claude runtime installation failed")
                executable = directory + "/home/.local/bin/claude"
            if not executable:
                raise RuntimeError("Pinned Claude runtime is unavailable")
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

    async def _close_agent_session_state(self, state: NativeClaudeState) -> AgentCloseSessionResponse:
        await state.session.close(timeout=self.config.close_timeout)
        if state.execution and not state.execution.done():
            state.execution.cancel()
            try:
                await state.execution
            except asyncio.CancelledError:
                pass
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id, cleanup_confirmed=True)

    async def _execute(
        self, state: NativeClaudeState, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        session = state.session
        directory = session.session_dir
        messages = body.model_dump(mode="json")["input"]
        if not isinstance(messages, list):
            raise ValueError("Native Claude requires message input")
        prompt = "\n\n".join(
            item["content"]
            if isinstance(item.get("content"), str)
            else "\n".join(part.get("text", "") for part in item.get("content", []) if isinstance(part, dict))
            for item in messages
        )
        timeout = float((body.metadata or {}).get("timeout_seconds", self.config.timeout))
        if not 0 < timeout <= self.config.timeout:
            raise ValueError("Invalid activation timeout")

        async def stage():
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
            if body.instructions:
                argv += ["--append-system-prompt", body.instructions]
            argv += [prompt]
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

        async def collect():
            return await session.read_output_log()

        raw = await session.execute(
            stage_activation=stage, collect=collect, timeout=timeout, close_timeout=self.config.close_timeout
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
        status, reason = _invocation_outcome(usage, None)
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
            raise HTTPException(400, "Seed a native agent session first")
        async with self._locked_agent_session(session_id) as record:
            if record.closing or not isinstance(record.state, NativeClaudeState):
                raise HTTPException(409, "Agent session is closed")
            state = record.state
            if state.body is not None and state.body != body:
                raise HTTPException(409, "A different activation was already submitted")
            if state.execution is None:
                state.body = body.model_copy(deep=True)
                state.execution = asyncio.create_task(self._execute(state, body))
            execution = state.execution
        return await asyncio.shield(execution)

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        raise HTTPException(400, "Use a native Environment Server and agent sessions")


if __name__ == "__main__":
    NativeClaudeCodeAgent.run_webserver()
