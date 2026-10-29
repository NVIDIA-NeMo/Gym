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
import shutil
import tempfile
from asyncio import Semaphore
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path, PurePosixPath
from shlex import quote
from time import monotonic, time
from typing import Any, Callable, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agent_utils.sandbox_session import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT, SandboxSession
from nemo_gym.agent_utils.task_timeout import AGENT_TIMEOUT_METADATA_KEY, resolve_agent_timeout
from nemo_gym.base_resources_server import NEMO_GYM_MCP_METADATA_KEY, BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config import SKILLS_REF_KEY_NAME, ModelServerRef, ResourcesServerRef, get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentObservationBundle,
    ObservationGap,
    SandboxObservation,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxSpec, create_provider, resolve_provider_config
from nemo_gym.server_utils import apply_rollout_prefix, get_response_json, raise_for_status
from nemo_gym.skills import stage_skills
from responses_api_agents.claude_code_agent.observability import extract_claude_code_observations
from responses_api_agents.claude_code_agent.parsing import _extract_instruction, _invocation_outcome, parse_stream_json
from responses_api_agents.claude_code_agent.runtime import install_sandbox_runtime
from responses_api_agents.claude_code_agent.sandbox import (
    ClaudeCodeSandboxSession,
    parse_artifact_observations,
)
from responses_api_agents.claude_code_agent.setup_claude_code import ensure_claude_code


LOG = logging.getLogger(__name__)
# Bootstrap with POSIX sh; the runtime installer, the harness runner and the supervisor require Bash.
_BASH_BOOTSTRAP = """if ! command -v bash >/dev/null 2>&1; then
    [ "$(id -u)" = 0 ] || { echo "Claude Code sandbox sessions require bash: preinstall it or use a root image." >&2; exit 1; }
    if command -v apk >/dev/null 2>&1; then
        apk add --no-cache bash || exit 1
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bash || exit 1
    else
        echo "Claude Code sandbox sessions require bash: preinstall it (automatic installation requires apt-get or apk)." >&2
        exit 1
    fi
fi
"""


class ClaudeCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: Optional[ResourcesServerRef] = None
    # When model_server is set, ANTHROPIC_BASE_URL is resolved from the Gym model
    # server's URL (requires the server to expose POST /v1/messages).
    # When None, anthropic_base_url is used directly.
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""  # pragma: allowlist secret
    anthropic_base_url: Optional[str] = None
    max_turns: Optional[int] = 30  # None -> unlimited turns
    timeout: int = 300
    system_prompt: Optional[str] = None
    allowed_tools: Optional[str] = None
    disallowed_tools: Optional[str] = None
    claude_code_version: Optional[str] = None
    thinking: Optional[str] = None
    max_thinking_tokens: Optional[int] = None
    # Runtime capability knobs. The default (bare=True, no mcp_config/settings) reproduces the original
    # isolated behavior: Claude Code skips hooks, LSP, plugin sync, attribution, auto-memory, background
    # prefetches, keychain reads, and CLAUDE.md auto-discovery (skills still resolve via /skill-name).
    bare: bool = True
    mcp_config: Optional[str] = None
    settings: Optional[str] = None

    # Used only when an episode does not supply a borrowed SandboxAccess.
    sandbox_provider: str | None = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_executable_path: str | None = None
    sandbox_local_executable_path: str | None = None
    sandbox_executable_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    sandbox_install_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)


class ClaudeCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ClaudeCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ClaudeCodeAgent(SimpleResponsesAPIAgent):
    config: ClaudeCodeAgentConfig
    sem: Semaphore = None
    _local_setup_task: asyncio.Task[None] | None = PrivateAttr(default=None)
    _static_mcp_config: Optional[dict[str, Any]] = PrivateAttr(default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    async def _ensure_local_runtime(self) -> None:
        if self._local_setup_task is None:
            self._local_setup_task = asyncio.create_task(
                asyncio.to_thread(ensure_claude_code, self.config.claude_code_version)
            )
            self._local_setup_task.add_done_callback(self._clear_failed_local_setup)
        task = self._local_setup_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._local_setup_task is task:
                self._local_setup_task = None
            raise

    def _clear_failed_local_setup(self, task: asyncio.Task[None]) -> None:
        if (task.cancelled() or task.exception() is not None) and self._local_setup_task is task:
            self._local_setup_task = None

    def _resolve_base_url(self) -> str:
        if self.config.model_server:
            cfg = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            return self.server_client._build_server_base_url(cfg)
        return self.config.anthropic_base_url or ""

    def _resolve_call_base_url(self, rollout_id: Optional[str]) -> str:
        """Return the CLI model-call URL with its rollout prefix.

        Apply the prefix only for a configured Gym model server.
        A real Anthropic endpoint has no prefix-stripping middleware.
        """
        base_url = self._resolve_base_url()
        if base_url and self.config.model_server:
            base_url = apply_rollout_prefix(
                base_url,
                rollout_id,
                token_capture=self._token_id_capture_enabled(),
            )
        return base_url

    def _build_settings(self) -> dict[str, Any]:
        """Settings written into the run's CLAUDE_CONFIG_DIR.

        The base settings disable telemetry/attribution. When ``config.settings`` points at a
        JSON file, its contents are layered on top: top-level keys override, and the ``env`` block
        is shallow-merged so the telemetry defaults are preserved unless explicitly overridden.
        """
        settings: dict[str, Any] = {
            "env": {
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        }
        if self.config.settings:
            user_settings = json.loads(Path(self.config.settings).expanduser().read_text())
            user_env = user_settings.get("env") or {}
            settings = {**settings, **user_settings, "env": {**settings["env"], **user_env}}
        return settings

    def _setup_config_dir(self, skills_path: Optional[str] = None) -> Path:
        """Create a per-run CLAUDE_CONFIG_DIR and stage settings (and optionally skills) into it.

        The directory lives for the duration of a single ``_run_claude_code`` call. When
        ``skills_path`` is provided, the directory of skills is copied into ``<dir>/skills/`` so
        Claude Code's native discovery can pick them up. Each request gets its own ephemeral copy,
        so concurrent requests with different skills do not contaminate one another. The caller is
        responsible for removing the directory on success; if setup fails partway (e.g. a bad
        ``skills_path``), this method cleans up the partially-created dir before re-raising so it
        does not leak (the caller never receives the path in that case).
        """
        claude_config_dir = Path.home() / ".claude_code_agent" / uuid4().hex
        claude_config_dir.mkdir(parents=True)
        try:
            (claude_config_dir / "settings.json").write_text(json.dumps(self._build_settings()))
            if skills_path:
                stage_skills(skills_path, claude_config_dir / "skills")
        except Exception:
            shutil.rmtree(claude_config_dir, ignore_errors=True)
            raise
        return claude_config_dir

    def _build_command(
        self,
        model: str,
        instruction: str,
        system_prompt: Optional[str] = None,
        mcp_config: Optional[str] = None,
        skills_active: bool = False,
    ) -> list[str]:
        """Construct the ``claude`` CLI argv from config.

        ``--bare`` is only passed when ``config.bare`` is True; it skips hooks, LSP, plugin sync,
        attribution, auto-memory, background prefetches, keychain reads, and CLAUDE.md auto-discovery
        (skills still resolve via /skill-name). Explicit capabilities like ``--mcp-config`` are passed
        regardless of ``--bare`` since they are not auto-discovered.

        When ``skills_active`` is True (skills were staged into CLAUDE_CONFIG_DIR for this request),
        ``--bare`` is forced off so Claude Code's native filesystem discovery picks the skills up.
        """
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        if self.config.bare and skills_active:
            LOG.warning(
                "skills are active for this request; ignoring bare=True so Claude Code can discover them. "
                "Note this re-enables ALL native auto-discovery, not just skills (hooks, plugins, MCP servers, "
                "memory, and CLAUDE.md), so the runtime broadens versus a bare baseline."
            )
        if self.config.bare and not skills_active:
            cmd.append("--bare")
        cmd += ["--model", model]
        effective_mcp_config = mcp_config if mcp_config is not None else self.config.mcp_config
        if effective_mcp_config:
            cmd += ["--mcp-config", effective_mcp_config]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        if self.config.allowed_tools:
            cmd += ["--allowedTools", self.config.allowed_tools]
        if self.config.disallowed_tools:
            cmd += ["--disallowedTools", self.config.disallowed_tools]
        if self.config.thinking:
            cmd += ["--thinking", self.config.thinking]
        if self.config.max_thinking_tokens is not None:
            cmd += ["--max-thinking-tokens", str(self.config.max_thinking_tokens)]
        if self.config.max_turns is not None:
            cmd += ["--max-turns", str(self.config.max_turns)]
        cmd += ["--", instruction]
        return cmd

    async def _run_claude_code(
        self,
        instruction: str,
        system_prompt: Optional[str] = None,
        mcp_config: Optional[str] = None,
        skills_path: Optional[str] = None,
        rollout_id: Optional[str] = None,
        observation_collector: Optional[Callable[[Path, dict[str, Any]], None]] = None,
    ) -> tuple[list[Any], str, dict[str, Any]]:
        """Run Claude Code and return parsed output, model name, and run metadata.

        When ``rollout_id`` is set and a model server is configured, the per-rollout capture prefix is
        applied to ANTHROPIC_BASE_URL so the CLI's streaming /v1/messages calls correlate to this rollout.
        """
        await self._ensure_local_runtime()
        base_url = self._resolve_call_base_url(rollout_id)
        # Keep full model name for local/custom endpoints; strip provider prefix for real Anthropic API.
        model = self.config.model if base_url else self.config.model.split("/")[-1]
        api_key = self.config.anthropic_api_key

        claude_config_dir = None
        run_metadata: dict[str, Any] = {"status": "unknown"}
        try:
            # Inside the try so a bad skills.path (raising in stage_skills) still cleans up the
            # partially-created config dir in the finally rather than leaking it per failing request.
            claude_config_dir = self._setup_config_dir(skills_path=skills_path)
            env = {
                **os.environ,
                "ANTHROPIC_API_KEY": api_key,  # pragma: allowlist secret
                "ANTHROPIC_MODEL": model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                "CLAUDE_CODE_SUBAGENT_MODEL": model,
                "IS_SANDBOX": "1",
                "CLAUDE_CONFIG_DIR": str(claude_config_dir),
            }
            if base_url:
                env["ANTHROPIC_BASE_URL"] = base_url
                env["ANTHROPIC_AUTH_TOKEN"] = api_key or "local"

            cmd = self._build_command(
                model,
                instruction,
                system_prompt=system_prompt,
                mcp_config=mcp_config,
                skills_active=bool(skills_path),
            )

            process_started_at = monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            communication = asyncio.create_task(proc.communicate())
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=self.config.timeout,
                )
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    with suppress(ProcessLookupError):
                        proc.kill()
                stdout, _ = await communication
                LOG.warning("claude-code timed out after %ds", self.config.timeout)
                # Keep the actual partial work so capture can corroborate the terminal call.
                output_items, run_metadata = parse_stream_json(
                    stdout.decode(errors="replace"), fold_thinking=not self._token_id_capture_enabled()
                )
                run_metadata.update(
                    status="incomplete",
                    error_type="timeout",
                    duration_ms=(monotonic() - process_started_at) * 1000,
                )
                return output_items, model, run_metadata
            except asyncio.CancelledError:
                if proc.returncode is None:
                    with suppress(ProcessLookupError):
                        proc.kill()
                await asyncio.gather(communication, return_exceptions=True)
                raise

            if proc.returncode not in (0, None):
                LOG.warning("claude-code exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            stdout_text = stdout.decode(errors="replace")
            LOG.debug("claude-code stdout (%d chars): %s", len(stdout), stdout_text[:2000])
            output_items, run_metadata = parse_stream_json(
                stdout_text, fold_thinking=not self._token_id_capture_enabled()
            )
            run_metadata.setdefault("duration_ms", (monotonic() - process_started_at) * 1000)
            status, error_type = _invocation_outcome(run_metadata, proc.returncode)
            run_metadata["status"] = status
            if error_type is not None:
                run_metadata["error_type"] = error_type
            return output_items, model, run_metadata
        finally:
            if claude_config_dir is not None:
                try:
                    if observation_collector is not None:
                        await asyncio.to_thread(observation_collector, claude_config_dir, run_metadata)
                except Exception:
                    LOG.exception("failed to collect Claude Code observations")
                finally:
                    shutil.rmtree(claude_config_dir, ignore_errors=True)

    def _resources_server_base_url(self) -> str:
        cfg = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.resources_server.name,
        )
        return self.server_client._build_server_base_url(cfg)

    def _load_static_mcp_config(self) -> dict[str, Any]:
        if not self.config.mcp_config:
            return {"mcpServers": {}}

        config_path = Path(self.config.mcp_config).expanduser()
        config = json.loads(config_path.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"Claude Code mcp_config must be a JSON object: {config_path}")
        mcp_servers = config.setdefault("mcpServers", {})
        if not isinstance(mcp_servers, dict):
            raise ValueError(f"Claude Code mcp_config has non-object mcpServers: {config_path}")
        return config

    def _get_static_mcp_config(self) -> dict[str, Any]:
        # The static mcp_config is immutable, so read it from disk at most once and reuse the cached
        # copy for every rollout instead of re-reading the file each time.
        if self._static_mcp_config is None:
            self._static_mcp_config = self._load_static_mcp_config()
        return self._static_mcp_config

    def _write_rollout_mcp_config(self, seed_response_json: dict[str, Any], output_dir: Path) -> Optional[str]:
        metadata = seed_response_json.get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, dict):
            return None

        server_name = metadata.get("server_name") or self.config.resources_server.name
        url_path = str(metadata.get("url_path") or "/mcp")
        url = f"{self._resources_server_base_url().rstrip('/')}/{url_path.lstrip('/')}"

        entry: dict[str, Any] = {
            "type": metadata.get("transport") or "http",
            "url": url,
        }
        headers = metadata.get("headers")
        if isinstance(headers, dict) and headers:
            entry["headers"] = {str(key): str(value) for key, value in headers.items()}
        else:
            LOG.warning(
                "MCP seed metadata for %r has no headers; the tool endpoint will be called without a "
                "session token and will reject the calls.",
                server_name,
            )

        # Start from a copy of the (cached) static config and add the per-rollout Gym entry. If a static
        # mcp_config server already uses this name, the per-rollout Gym entry takes precedence over it.
        config = copy.deepcopy(self._get_static_mcp_config())
        config.setdefault("mcpServers", {})[str(server_name)] = entry

        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "gym_mcp_config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
        return str(config_path)

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        mcp_config: Optional[str] = None,
        skills_path: Optional[str] = None,
        rollout_id: Optional[str] = None,
        observation_collector: Optional[Callable[[Path, dict[str, Any]], None]] = None,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, system_prompt = self._conversation_input(body)

        output_items, model_name, run_metadata = await self._run_claude_code(
            user_message,
            system_prompt=system_prompt,
            mcp_config=mcp_config,
            skills_path=skills_path,
            rollout_id=rollout_id,
            observation_collector=observation_collector,
        )

        return self._response_from_output(body, output_items, model_name, run_metadata)

    def _conversation_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str | None]:
        """Use the established local last-user/leading-system prompt contract in both modes."""
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        prompt, input_system = _extract_instruction(items)
        system = "\n\n".join(part for part in (self.config.system_prompt, input_system) if part) or None
        return prompt, system

    def _configured_skills_path(self) -> str | None:
        skills = getattr(self.server_client, "global_config_dict", {}).get("skills")
        return skills.get("path") if isinstance(skills, Mapping) else None

    def _response_from_output(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        output_items: list[Any],
        model_name: str,
        run_metadata: dict[str, Any],
        *,
        status: str | None = None,
    ) -> NeMoGymResponse:
        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("claude-code produced no assistant message; padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = run_metadata.get("input_tokens", 0)
        output_tokens = run_metadata.get("output_tokens", 0)

        # Under capture, name the response by the last served id so terminal attribution can match it directly.
        terminal_response_id = run_metadata.get("terminal_response_id")
        response_id = (
            terminal_response_id
            if self._token_id_capture_enabled() and isinstance(terminal_response_id, str) and terminal_response_id
            else f"resp_{uuid4().hex}"
        )
        return NeMoGymResponse(
            **({"status": status} if status is not None else {}),
            id=response_id,
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

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> ClaudeCodeSandboxSession:
        """Install/reuse the harness runtime before the first activation."""
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise HTTPException(
                422, "Claude sandbox execution uses native tools; required remote tools are unsupported"
            )
        version = self.config.claude_code_version
        if not version or not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise HTTPException(422, "Claude sandbox execution requires an exact claude_code_version")
        if self.config.sandbox_executable_path and self.config.sandbox_local_executable_path:
            raise HTTPException(422, "Choose either sandbox_executable_path or sandbox_local_executable_path")
        if (
            self.config.sandbox_executable_path or self.config.sandbox_local_executable_path
        ) and not self.config.sandbox_executable_sha256:
            raise HTTPException(422, "A supplied Claude binary requires sandbox_executable_sha256")
        access = body.sandbox_access
        owns_sandbox = access is None
        if owns_sandbox and not self.config.sandbox_provider:
            raise HTTPException(422, "Claude requires sandbox_access or a configured sandbox_provider")
        spec = SandboxSpec(**{"workdir": "/app", **self.config.sandbox_config}) if owns_sandbox else None
        workdir = access.workdir if access else spec.workdir
        path = PurePosixPath(workdir or "")
        if (
            not path.is_absolute()
            or ".." in path.parts
            or "\x00" in workdir
            or path in (PurePosixPath("/"), PurePosixPath("/tmp"))
            or str(path).startswith(("/tmp/nemo-gym-claude-sessions", "/tmp/nemo-gym-claude-runtime-"))
        ):
            raise HTTPException(
                422, "Claude sandbox workdir must be absolute and separate from runtime/session storage"
            )
        if self.config.timeout <= 0:
            raise HTTPException(422, "Claude sandbox execution timeout must be positive")
        provider_ref = access.connection.provider_config_ref if access else self.config.sandbox_provider
        provider = create_provider(resolve_provider_config(provider_ref, self.server_client.global_config_dict))
        if owns_sandbox:
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        directory = f"/tmp/nemo-gym-claude-sessions/{uuid4().hex}"
        runtime = f"/tmp/nemo-gym-claude-runtime-{version}"
        state = ClaudeCodeSandboxSession(
            request=body,
            runtime=runtime,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=directory,
                workdir=workdir,
                harness="Claude Code",
                owns_sandbox=owns_sandbox,
            ),
        )
        try:
            async with asyncio.timeout(self.config.sandbox_install_timeout_seconds):
                if owns_sandbox:
                    await sandbox.start(spec)
                    result = await sandbox.exec(f"mkdir -p -- {quote(workdir)}", cwd="/", timeout_s=30)
                    self._check_sandbox_setup(result)
                result = await sandbox.exec(
                    _BASH_BOOTSTRAP + f"test -d {quote(workdir)} && umask 077 && mkdir -p {quote(directory)}/config",
                    cwd="/",
                    timeout_s=self.config.sandbox_install_timeout_seconds,
                )
                self._check_sandbox_setup(result)
                await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
                installer = "install_claude_runtime.sh"
                await sandbox.upload(Path(__file__).with_name(installer), f"{directory}/{installer}")
                await install_sandbox_runtime(
                    sandbox,
                    version=version,
                    directory=directory,
                    runtime=runtime,
                    timeout=self.config.sandbox_install_timeout_seconds,
                    source=self.config.sandbox_executable_path,
                    local_source=self.config.sandbox_local_executable_path,
                    expected=self.config.sandbox_executable_sha256,
                )
            return state
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise

    @staticmethod
    def _check_sandbox_setup(result: SandboxExecResult) -> None:
        if result.return_code != 0 or result.error_type:
            details = "\n".join(part for part in (result.stderr, result.stdout) if part)
            raise RuntimeError(
                f"Claude sandbox setup failed (exit={result.return_code}, error={result.error_type}): {details[-16000:]}"
            )

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Confirm cleanup and return a bundle even when no activation ran."""
        assert isinstance(state, ClaudeCodeSandboxSession)
        await state.close(self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.session.cleanup_confirmed,
            agent_observations=state.observations
            or AgentObservationBundle(
                source="claude_code",
                gaps=[ObservationGap(code="agent_activation_interrupted")],
            ),
        )

    def _validate_sandbox_request(self, body: NeMoGymResponseCreateParamsNonStreaming) -> None:
        supported = {"input", "model"}
        for name, field in type(body).model_fields.items():
            value = getattr(body, name)
            if name in supported or (name in {"stream", "background"} and value is False):
                continue
            if name == "metadata" and value is not None and set(value) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"Claude sandbox execution does not support request field {name}")
        models = {None, "", self.config.model}
        if self.config.model_server:
            models.add(self.config.model_server.name)
        if body.model not in models or body.model_extra:
            raise HTTPException(422, "Claude sandbox model is selected by agent/model-server configuration")
        if not isinstance(body.input, str):
            roles = [getattr(item, "role", None) for item in body.input]
            if roles not in (["user"], ["system", "user"]):
                raise HTTPException(
                    422, "Claude sandbox execution accepts one user message with an optional leading system message"
                )
            for item in body.input:
                if not isinstance(item.content, str) and any(
                    (part.get("type") if isinstance(part, dict) else part.type) != "input_text"
                    for part in item.content
                ):
                    raise HTTPException(422, "Claude sandbox input requires text messages")

    async def _sandbox_response(
        self,
        state: ClaudeCodeSandboxSession,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        prompt: str,
        system: str | None,
    ) -> NeMoGymResponse:
        directory = state.session.session_dir
        base_url = self._resolve_call_base_url(state.request.episode_id.capture_key)
        model = self.config.model if base_url else self.config.model.split("/")[-1]
        env = {
            "CLAUDE_CONFIG_DIR": directory + "/config",
            "IS_SANDBOX": "1",
            "ANTHROPIC_API_KEY": "gym" if self.config.model_server else self.config.anthropic_api_key or "EMPTY",
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        if base_url:
            env.update(
                ANTHROPIC_BASE_URL=base_url,
                ANTHROPIC_AUTH_TOKEN="gym" if self.config.model_server else self.config.anthropic_api_key or "local",
            )
        artifacts = None
        failure = None
        try:
            # Close can fence the staging phase just as it fences the execution phase.
            async with self.sem:
                mcp_config = directory + "/mcp.json" if self.config.mcp_config else None
                skills_path = self._configured_skills_path()
                command = self._build_command(
                    model, prompt, system, mcp_config=mcp_config, skills_active=bool(skills_path)
                )
                command[0] = state.runtime + "/claude"
                artifacts = await state.execute(
                    {
                        "command": command,
                        "cwd": state.session.workdir,
                        "env": env,
                        "settings": self._build_settings(),
                        "mcp_config": self._get_static_mcp_config() if mcp_config else None,
                        "skills_path": skills_path,
                    },
                    timeout=resolve_agent_timeout(body.metadata, ceiling=self.config.timeout),
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            failure = error
            artifacts = state.session.artifacts
        items, run_metadata = ([], {})
        if artifacts is not None:
            items, run_metadata = await asyncio.to_thread(
                parse_stream_json,
                artifacts.events,
                fold_thinking=not self._token_id_capture_enabled(),
            )
        cleanup = state.session.cleanup
        if cleanup and cleanup["timed_out"]:
            status, reason = "incomplete", "timeout"
        elif isinstance(failure, asyncio.CancelledError):
            status, reason = "incomplete", "cancelled"
        elif failure is not None:
            status, reason = "failed", "sandbox_error"
        else:
            status, reason = _invocation_outcome(run_metadata, cleanup["return_code"] if cleanup else None)
        if artifacts is not None:
            state.observations = await asyncio.to_thread(
                parse_artifact_observations,
                artifacts,
                model_ref=self.config.model_server,
                status=status,
                error_type=reason,
                run_metadata=run_metadata,
            )
        else:
            state.observations = AgentObservationBundle(
                source="claude_code", gaps=[ObservationGap(code="agent_artifact_unavailable")]
            )
        if state.runtime_info is None:
            state.observations.gaps.append(ObservationGap(code="runtime_info_unavailable"))
        if cleanup:
            access = state.request.sandbox_access
            state.observations.records.append(
                SandboxObservation(
                    role="agent",
                    provider=access.connection.provider_config_ref if access else self.config.sandbox_provider,
                    outcome=(
                        "timeout"
                        if cleanup["timed_out"]
                        else "cancelled"
                        if reason == "cancelled"
                        else "failed"
                        if status == "failed"
                        else "completed"
                    ),
                    exit_code=cleanup["return_code"],
                    error_type=reason,
                )
            )
        if failure is not None:
            raise failure
        return self._response_from_output(body, items, model, run_metadata, status=status)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is not None:
            state = self._require_agent_session(session_id)
            assert isinstance(state, ClaudeCodeSandboxSession)
            if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
                raise HTTPException(409, "Claude activation does not match the seeded session and rollout")
            if state.session.closing:
                raise HTTPException(409, "Claude session is closing")
            self._validate_sandbox_request(body)
            prompt, system = self._conversation_input(body)
            if state.task is None:
                state.activation_request = body.model_copy(deep=True)
                state.task = asyncio.create_task(
                    self._sandbox_response(state, state.activation_request, prompt=prompt, system=system)
                )
            elif body != state.activation_request:
                raise HTTPException(409, "Claude sandbox sessions support one activation; retry the same request")
            return (await asyncio.shield(state.task)).model_copy(deep=True)
        return await self._create_response(body, rollout_id=request.path_params.get("rollout_id"))

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        mcp_config: Optional[str] = None,
        skills_path: Optional[str] = None,
        rollout_id: Optional[str] = None,
    ) -> AgentEpisode:
        observations: Optional[AgentObservationBundle] = None

        def collect(config_dir: Path, run_metadata: dict[str, Any]) -> None:
            nonlocal observations
            try:
                observations = extract_claude_code_observations(
                    config_dir,
                    model_ref=self.config.model_server,
                    root_status=run_metadata["status"],
                    root_duration_ms=run_metadata.get("duration_ms"),
                    root_error_type=run_metadata.get("error_type"),
                    compaction_attempts=run_metadata.get("compaction_attempts"),
                )
                if self.config.model_server is None:
                    observations.gaps.append(ObservationGap(code="model_call_ownership_unavailable"))
            except Exception:
                LOG.exception("failed to extract Claude Code observations")
                observations = AgentObservationBundle(
                    source="claude_code",
                    gaps=[ObservationGap(code="observation_parse_failed")],
                )

        response = await self._create_response(
            body,
            mcp_config=mcp_config,
            skills_path=skills_path,
            rollout_id=rollout_id,
            observation_collector=collect,
        )
        if observations is None:
            observations = AgentObservationBundle(
                source="claude_code",
                gaps=[ObservationGap(code="agent_transcript_unavailable")],
            )
        observations.gaps.append(ObservationGap(code="no_sandbox_runtime"))
        return AgentEpisode(response=response, observations=observations)

    async def run(self, request: Request, body: ClaudeCodeAgentRunRequest) -> ClaudeCodeAgentVerifyResponse:
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "Claude Code sandbox sessions must use EnvironmentServer /run")
        if self.config.resources_server is None:
            raise HTTPException(422, "Local Claude Code /run requires resources_server")
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
            # can stage the skills into its per-request CLAUDE_CONFIG_DIR. run() calls _create_response
            # in-process, so no metadata side-channel is needed (unlike the schema-forbidden HTTP path).
            skills_path = ((body.model_extra or {}).get(SKILLS_REF_KEY_NAME) or {}).get("path")
            rollout_id = self.rollout_id_from_run(body)

            with tempfile.TemporaryDirectory(prefix="nemo_gym_claude_mcp_") as mcp_config_dir:
                mcp_config = self._write_rollout_mcp_config(seed_resp_json, Path(mcp_config_dir))
                if rollout_id is not None:
                    episode = await self._create_episode(
                        body.responses_create_params,
                        mcp_config=mcp_config,
                        skills_path=skills_path,
                        rollout_id=rollout_id,
                    )
                    agent_resp, observations = episode.response, episode.observations
                else:
                    agent_resp = await self._create_response(
                        body.responses_create_params,
                        mcp_config=mcp_config,
                        skills_path=skills_path,
                    )
                    observations = None
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

            result = verify_json | {"turns_used": turns, "finished_naturally": naturally}
            if observations is not None:
                result["ng_agent_observations"] = observations.model_dump(mode="json")
            return ClaudeCodeAgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    ClaudeCodeAgent.run_webserver()
