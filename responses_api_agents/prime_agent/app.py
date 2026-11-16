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
import shlex
import shutil
import signal
import tempfile
from asyncio import Semaphore
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path, PurePosixPath
from time import time
from typing import Any, ClassVar, Literal, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, JsonValue, PrivateAttr

from nemo_gym.agents import (
    AGENT_TIMEOUT_METADATA_KEY,
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SandboxCommand,
    SandboxSession,
    SimpleResponsesAPIAgent,
    resolve_agent_timeout,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef, get_global_config_dict
from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
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
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
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
from responses_api_agents.prime_agent.sandbox import (
    CLI,
    PRIME_AGENT_VERSION,
    RUNTIME_ROOT,
    SESSION_ROOT,
    PrimeArtifacts,
    PrimeSession,
)
from responses_api_agents.prime_agent.setup_prime_agent import ensure_prime_agent


LOG = logging.getLogger(__name__)


def _descendant_pids(root_pid: int) -> list[int]:
    descendants: list[tuple[int, int]] = []
    pending = [(root_pid, 0)]
    while pending:
        parent_pid, depth = pending.pop()
        try:
            children = Path(f"/proc/{parent_pid}/task/{parent_pid}/children").read_text().split()
        except OSError:
            continue
        for child in children:
            child_pid = int(child)
            descendants.append((child_pid, depth + 1))
            pending.append((child_pid, depth + 1))
    return [pid for pid, _ in sorted(descendants, key=lambda item: item[1], reverse=True)]


def _process_groups_with_env(key: str, value: str, proc_root: Path = Path("/proc")) -> list[int]:
    marker = f"{key}={value}".encode()
    process_groups: list[int] = []
    try:
        entries = proc_root.iterdir()
    except OSError:
        return process_groups

    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            environ = (entry / "environ").read_bytes().split(b"\0")
            process_group = os.getpgid(int(entry.name))
        except (OSError, ProcessLookupError):
            continue
        if marker in environ and process_group not in process_groups:
            process_groups.append(process_group)
    return process_groups


def _kill_prime_processes(root_pid: int, agent_dir: Path) -> None:
    process_groups: list[int] = []
    for descendant_pid in _descendant_pids(root_pid):
        try:
            process_group = os.getpgid(descendant_pid)
        except ProcessLookupError:
            continue
        if process_group != root_pid and process_group not in process_groups:
            process_groups.append(process_group)

    for process_group in _process_groups_with_env("PRIME_AGENT_CODING_AGENT_DIR", str(agent_dir)):
        if process_group != root_pid and process_group not in process_groups:
            process_groups.append(process_group)

    for process_group in process_groups:
        with suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGKILL)
    with suppress(ProcessLookupError):
        os.killpg(root_pid, signal.SIGKILL)


def parse_prime_agent_events(stdout: str, *, timed_out: bool = False) -> tuple[list[Any], dict[str, int]]:
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    # Prime Agent retries a failed model call after printing its error message, so only a final error fails the run.
    # An error at the deadline is a retry still pending, so a timed-out run keeps its transcript.
    failed = False

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message_end":
            continue
        message = event.get("message") or {}
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, list):
            continue

        if role == "assistant":
            usage = message.get("usage") or {}
            input_tokens += int(usage.get("input") or 0) + int(usage.get("cacheRead") or 0)
            output_tokens += int(usage.get("output") or 0)
            cached_tokens += int(usage.get("cacheRead") or 0)
            failed = message.get("stopReason") in {"error", "aborted"}
            if failed:
                LOG.warning("Prime Agent stopped with an error: %s", message.get("errorMessage") or "unknown error")
                continue
            reasoning = [
                block["thinking"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "thinking"
                and isinstance(block.get("thinking"), str)
                and block["thinking"].strip()
            ]
            if reasoning:
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs-{len(output_items)}",
                        summary=[NeMoGymSummary(type="summary_text", text="\n".join(reasoning))],
                    )
                )
            texts = [
                block["text"] for block in content if isinstance(block, dict) and (block.get("text") or "").strip()
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
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                args = block.get("arguments")
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
            call_id = message.get("toolCallId", "")
            result_text = "".join(
                block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
            )
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=result_text,
                    status="incomplete" if message.get("isError") else "completed",
                )
            )

    return [] if failed and not timed_out else output_items, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
    }


def prime_stop(stdout: str) -> dict[str, Any]:
    """The last assistant message's stopReason and errorMessage, which say how Prime Agent ended its run."""
    last: dict[str, Any] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if isinstance(event, dict) and event.get("type") == "message_end" else None
        if isinstance(message, dict) and message.get("role") == "assistant":
            last = message
    return {"stopReason": last.get("stopReason"), "errorMessage": last.get("errorMessage")}


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_text(item) for item in content)
    if isinstance(content, dict):
        return str(content.get("text") or "")
    return str(getattr(content, "text", ""))


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    user_message = ""
    system_messages = []
    for item in body_input:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        text = _content_text(content)
        if role in {"system", "developer"} and text:
            system_messages.append(text)
        elif role == "user" and text:
            user_message = text

    return user_message, "\n\n".join(system_messages) or None


class PrimeAgentConfig(BaseResponsesAPIAgentConfig):
    # Set only for the host /run path; agent sessions get the benchmark from the Environment Server.
    resources_server: Optional[ResourcesServerRef] = None
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "prime-agent"
    model: str = "policy/model"
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/prime_agent/workspaces"
    kernel_venv: Optional[str] = "outputs/prime_agent/kernel-venv"
    thinking: Optional[str] = None
    system_prompt: Optional[str] = None
    timeout: int = 900
    extra_args: list[str] = Field(default_factory=list)
    models_config: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 262144
    max_output_tokens: int = 131072
    prime_agent_version: Optional[str] = None
    # Agent sessions: Prime Agent runs inside the task sandbox the Resources Server supplies (or an agent-owned one).
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    sandbox_provider: Optional[str] = None
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)
    network_access: Literal["inherit"] = Field(
        default="inherit",
        description=(
            "Egress of a sandbox Prime Agent creates. Only `inherit` (the provider's network): setup installs "
            "dependencies from the npm registry, the release host and PyPI."
        ),
    )

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class PrimeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class PrimeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class PrimeAgent(SimpleResponsesAPIAgent):
    config: PrimeAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Sandbox sessions run the CLI inside the task sandbox, which calls the model server from there.
    calls_model_from_sandbox: ClassVar[bool] = True
    _local_setup: Optional[asyncio.Task] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    async def _ensure_local_runtime(self) -> None:
        """Install the host CLI on the first host-path call; agent sessions install inside the sandbox instead."""
        if not self.config.command_parts or self.config.command_parts[0] != "prime-agent":
            return
        if self._local_setup is None or (self._local_setup.done() and self._local_setup.exception() is not None):
            self._local_setup = asyncio.create_task(
                asyncio.to_thread(ensure_prime_agent, self.config.prime_agent_version)
            )
        await asyncio.shield(self._local_setup)

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"prime_agent_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _kernel_venv(self) -> Optional[Path]:
        if self.config.kernel_venv is None:
            return None
        path = Path(self.config.kernel_venv).expanduser()
        return path if path.is_absolute() else Path.cwd() / path

    def _env(self, home: Path) -> dict[str, str]:
        agent_dir = home / ".prime" / "agent"
        env = {
            **os.environ,
            "HOME": str(home),
            "PRIME_AGENT_CODING_AGENT_DIR": str(agent_dir),
            "PI_SKIP_VERSION_CHECK": "1",
        }
        kernel_venv = self._kernel_venv()
        if kernel_venv is not None:
            env["PRIME_AGENT_KERNEL_VENV"] = str(kernel_venv)
        env.update({key: value for key, value in self.config.env.items() if value})
        return env

    def _effective_model(self) -> str:
        return f"nemo/{self.config.model}" if self.config.model_server else self.config.model

    def _build_models_config(self, rollout_id: Optional[str] = None) -> dict[str, Any]:
        config = copy.deepcopy(self.config.models_config)
        if self.config.model_server is None:
            return config
        providers = config.setdefault("providers", {})
        providers["nemo"] = {
            "baseUrl": self.resolve_model_base_url(self.config.model_server.name, rollout_id),
            "api": "openai-completions",
            "apiKey": "EMPTY",  # pragma: allowlist secret
            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": True},
            "models": [
                {
                    "id": self.config.model,
                    "reasoning": True,
                    "input": ["text"],
                    "contextWindow": self.config.context_window,
                    "maxTokens": self.config.max_output_tokens,
                }
            ],
        }
        return config

    def _build_command(
        self,
        instruction: str,
        system_prompt: Optional[str],
        daemon_socket: Optional[Path] = None,
    ) -> list[str]:
        effective_model = self._effective_model()
        provider, separator, model_id = effective_model.partition("/")
        cmd = [*self.config.command_parts, "--print", "--mode", "json", "--no-session"]
        if daemon_socket is not None:
            cmd += ["--daemon-socket", str(daemon_socket)]
        if separator:
            cmd += ["--provider", provider, "--model", model_id]
        else:
            cmd += ["--model", effective_model]
        if self.config.thinking:
            cmd += ["--thinking", self.config.thinking]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        cmd += self.config.extra_args
        # The terminator keeps a task that starts with "-" or "@" from being read as an option or a file argument.
        cmd += ["--", instruction]
        return cmd

    async def _run_prime_agent(
        self, instruction: str, system_prompt: Optional[str], rollout_id: Optional[str]
    ) -> tuple[list[Any], dict[str, int], str, bool]:
        work_dir = self._workspace_root()
        socket_dir = Path(tempfile.mkdtemp(prefix="pa-", dir="/tmp"))
        try:
            home = work_dir / ".prime-home"
            agent_dir = home / ".prime" / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            models_config = self._build_models_config(rollout_id)
            if models_config:
                (agent_dir / "models.json").write_text(json.dumps(models_config, indent=2))
            env = self._env(home)
            cmd = self._build_command(instruction, system_prompt, socket_dir / "daemon.sock")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
            communication = asyncio.create_task(proc.communicate())
            process_exit = asyncio.create_task(proc.wait())
            try:
                done, _ = await asyncio.wait(
                    {communication, process_exit},
                    timeout=self.config.timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                timed_out = not done
                if communication not in done:
                    _kill_prime_processes(proc.pid, agent_dir)
                stdout, stderr = await communication
                await process_exit
            except asyncio.CancelledError:
                _kill_prime_processes(proc.pid, agent_dir)
                await asyncio.gather(communication, process_exit, return_exceptions=True)
                raise

            if timed_out:
                LOG.warning("Prime Agent timed out after %ds", self.config.timeout)
                output_items, usage = parse_prime_agent_events(stdout.decode(errors="replace"), timed_out=True)
                return output_items, usage, self.config.model, True

            if proc.returncode not in (0, None):
                LOG.warning("Prime Agent exited %d: %s", proc.returncode, stderr.decode(errors="replace")[-1000:])
                return [], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}, self.config.model, False
            output_items, usage = parse_prime_agent_events(stdout.decode(errors="replace"))
            return output_items, usage, self.config.model, False
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            shutil.rmtree(socket_dir, ignore_errors=True)

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Read Prime Agent 0.7.0's terminal assistant error (Pi's provider messages); tested on that version."""
        if native_result.get("failure_kind"):
            return AgentOutcome(failure_kind=native_result["failure_kind"])
        message = str(native_result.get("errorMessage") or "")
        match = re.match(r"^(?:HTTP )?([45]\d{2})(?::|\s|$)", message)
        if match:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=int(match[1]))
        if message in {"Connection error.", "Connection error", "fetch failed", "Request timed out."}:
            return AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
        return AgentOutcome(failure_kind="unknown")

    def _rollout_sandbox_session(self, state: AgentSessionState) -> Optional[SandboxSession]:
        return state.session if isinstance(state, PrimeSession) else None

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> PrimeSession:
        """Connect to (or start) the task sandbox and install Prime Agent before the activation."""
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError(
                "Prime Agent sessions use Prime Agent's own tools; required tool grants are unsupported"
            )
        if self.config.model_server is None:
            raise AgentSeedRejectedError("Prime Agent sessions require a sandbox-reachable Gym model_server")
        if self.config.prime_agent_version not in (None, PRIME_AGENT_VERSION):
            raise AgentSeedRejectedError(f"Prime Agent sessions pin prime_agent_version {PRIME_AGENT_VERSION}")
        if self.config.command != "prime-agent" or self.config.extra_args or self.config.env:
            raise AgentSeedRejectedError("Prime Agent sessions do not support command, extra_args, or env overrides")
        access = body.sandbox_access
        owns_sandbox = access is None
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise AgentSeedRejectedError("Prime Agent requires sandbox_access or a configured sandbox_provider")
            provider_ref = self.config.sandbox_provider
            workdir = self.config.sandbox_config.get("workdir") or "/app"
        else:
            if not isinstance(access.connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("Prime Agent requires direct SandboxAccess")
            if access.user is not None:
                raise AgentSeedRejectedError("Prime Agent does not support a SandboxAccess user override")
            provider_ref = access.connection.provider_config_ref
            workdir = access.workdir
        path = PurePosixPath(workdir) if isinstance(workdir, str) else None
        if (
            path is None
            or not path.is_absolute()
            or ".." in path.parts
            or path in (PurePosixPath("/"), PurePosixPath("/tmp"))
            or str(path).startswith((SESSION_ROOT, RUNTIME_ROOT))
        ):
            raise AgentSeedRejectedError(
                "Prime Agent task workdir must be absolute and separate from its session storage"
            )

        global_config = get_global_config_dict()
        provider = create_provider(resolve_provider_config(provider_ref, global_config))
        if owns_sandbox:
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        state = PrimeSession(
            request=body,
            provider_name=provider.name,
            sandbox_id=sandbox.sandbox_id,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=f"{SESSION_ROOT}/{uuid4().hex}",
                workdir=str(path),
                harness="Prime Agent",
                owns_sandbox=owns_sandbox,
            ),
        )
        try:
            async with asyncio.timeout(self.config.sandbox_install_timeout_seconds):
                if owns_sandbox:
                    settings = dict(self.config.sandbox_config)
                    metadata = settings.pop("metadata", None) or {}
                    settings["metadata"] = {
                        **resolve_provider_metadata(provider_ref, global_config),
                        **metadata,
                        "agent_session_id": body.agent_session_id[:63],
                    }
                    settings["workdir"] = str(path)
                    settings.setdefault(
                        "ttl_s", int(self.config.sandbox_install_timeout_seconds + self.config.timeout + 600)
                    )
                    await sandbox.start(SandboxSpec(**settings))
                    state.sandbox_id = sandbox.sandbox_id
                    made = await sandbox.exec(f"mkdir -p -- {shlex.quote(str(path))}", cwd="/", timeout_s=30)
                    if made.return_code != 0 or made.error_type:
                        raise RuntimeError(f"Cannot create the Prime Agent task workdir {path}")
                await state.install(timeout=self.config.sandbox_install_timeout_seconds)
        except BaseException as error:
            try:
                await state.close(timeout=self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, PrimeSession):
            raise TypeError("Expected PrimeSession")
        await state.close(timeout=self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.session.cleanup_confirmed,
            agent_observations=state.observations
            or AgentObservationBundle(source="prime_agent", gaps=[ObservationGap(code="observation_capture_failed")]),
        )

    def _session_conversation(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, Optional[str]]:
        """The task and appended system prompt for a sandbox activation, rejecting controls it cannot apply."""
        if body.model is not None and body.model != self.config.model:
            raise HTTPException(422, "Prime Agent model is selected by the agent configuration")
        supported = {"input", "instructions", "model"}
        for name, field in type(body).model_fields.items():
            value = getattr(body, name)
            if name in supported or (name in ("stream", "background") and value is False):
                continue
            if name == "metadata" and value and set(value) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"Prime Agent does not support request field {name}")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"], ["developer", "user"]):
            raise HTTPException(
                422, "Prime Agent accepts one user message with an optional system or developer message"
            )
        # Prime Agent takes text only. Flattening would silently drop images or files from the task.
        for item in items:
            if not isinstance(item.content, str) and any(
                (part.get("type") if isinstance(part, dict) else getattr(part, "type", None)) != "input_text"
                for part in item.content
            ):
                raise HTTPException(422, "Prime Agent only supports text input")
        user_message, input_system = _extract_instruction(items)
        if not user_message.strip():
            raise HTTPException(422, "Prime Agent requires a nonempty task")
        system_parts = [part for part in [self.config.system_prompt, body.instructions, input_system] if part]
        system_prompt = "\n\n".join(system_parts) or None
        # The task and the system prompt are argv values, which Linux caps at 128 KiB each (MAX_ARG_STRLEN).
        if any(len(text.encode()) >= 128 * 1024 for text in (user_message, system_prompt or "")):
            raise HTTPException(422, "Prime Agent task and system prompt must each be under 128 KiB")
        return user_message, system_prompt

    async def _stage_session(
        self, state: PrimeSession, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> SandboxCommand:
        user_message, system_prompt = self._session_conversation(body)
        command = self._build_command(user_message, system_prompt, Path(state.session.session_dir) / "daemon.sock")
        command[: len(self.config.command_parts)] = [f"{state.runtime}/bin/node", f"{state.runtime}/{CLI}"]
        access = state.request.sandbox_access
        env = {
            key: value
            for key, value in (access.env if access else {}).items()
            if not key.startswith(("PRIME_AGENT_", "PI_", "OPENAI_"))
        }
        # Prime Agent's models, tools and state live in the session directory, so tool shells keep the task's HOME.
        env |= {
            "PRIME_AGENT_CODING_AGENT_DIR": state.agent_dir,
            "PRIME_AGENT_KERNEL_PYTHON": f"{state.runtime}/kernel/bin/python3",
            "PI_SKIP_VERSION_CHECK": "1",
        }
        # Only the Gym provider: the sandbox is model-controlled, so host providers' keys must not reach it.
        nemo = self._build_models_config(state.request.episode_id.capture_key)["providers"]["nemo"]
        models = {"providers": {"nemo": nemo}}
        return await state.stage(command=command, env=env, models=json.dumps(models, indent=2))

    def _session_observations(
        self, state: PrimeSession, *, status: str, error_type: Optional[str], output: list[Any]
    ) -> AgentObservationBundle:
        cleanup = state.session.cleanup
        conversation = []
        if state.activation_request is not None:
            user_message, _ = self._session_conversation(state.activation_request)
            conversation.append(NeMoGymEasyInputMessage(role="user", content=user_message))
        # Prime Agent 0.7.0 sends no per-call session header to the model server.
        gaps = ["model_call_ownership_unavailable", "harness_tool_timing_unavailable", "child_invocations_unavailable"]
        gaps += ["optional_tool_grant_skipped" for _ in self.effective_tool_accesses(state.request)]
        return AgentObservationBundle(
            source="prime_agent",
            records=[
                AgentInvocation(
                    invocation_id=state.request.episode_id.capture_key,
                    status=status,
                    error_type=error_type,
                    conversation=conversation + output,
                ),
                SandboxObservation(
                    role="agent",
                    provider=state.provider_name,
                    sandbox_id=state.sandbox_id,
                    outcome="timeout"
                    if cleanup and cleanup["timed_out"]
                    else "failed"
                    if status == "failed"
                    else "completed",
                    exit_code=cleanup["return_code"] if cleanup else None,
                    error_type=error_type,
                ),
            ],
            gaps=[ObservationGap(code=code) for code in sorted(set(gaps))],
        )

    async def _activate_session(
        self, state: PrimeSession, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        timeout = resolve_agent_timeout(
            body.metadata, ceiling=min(self.config.timeout, state.request.agent_timeout_sec or self.config.timeout)
        )
        try:
            async with self.sem:
                artifacts: PrimeArtifacts = await state.session.execute(
                    stage_activation=lambda: self._stage_session(state, body),
                    collect=state.collect,
                    timeout=timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            state.observations = self._session_observations(
                state, status="failed", error_type=type(error).__name__, output=[]
            )
            raise
        cleanup = state.session.cleanup
        if cleanup is None or not cleanup["cleanup_confirmed"]:
            raise RuntimeError("Prime Agent ended without confirmed supervisor cleanup")
        if cleanup["error"]:
            raise RuntimeError(f"Prime Agent supervision failed: {cleanup['error']}")
        return_code, timed_out = cleanup["return_code"], cleanup["timed_out"]
        if return_code in (125, 126, 127) and not timed_out:
            raise RuntimeError(f"Prime Agent did not start (runner exit {return_code}): {artifacts.stderr[-4096:]}")
        stop = prime_stop(artifacts.stdout)
        output, usage = parse_prime_agent_events(artifacts.stdout, timed_out=timed_out)
        status, incomplete, native = "completed", None, None
        if timed_out and not output and not state.session.expired:
            # The deadline passed before any model reply: a classified failure, retried rather than graded.
            status, native = "failed", {"failure_kind": "unknown"}
        elif timed_out:
            # A rollout stop is incomplete even before the first model reply, so report_rollout_stop marks it.
            status = "incomplete"
        elif stop["stopReason"] in ("error", "aborted"):
            # parse_prime_agent_events drops the transcript of a run that ended in an error.
            status, native = "failed", stop
        elif stop["stopReason"] == "length":
            status, incomplete = "incomplete", {"reason": "max_output_tokens"}
        elif return_code != 0:
            status, native = "failed", {"failure_kind": "unknown"}
        elif not any(getattr(item, "type", None) == "message" for item in output):
            status, native = "failed", {"failure_kind": "agent_invalid_output"}
        error_type = stop["stopReason"] if status != "completed" else None
        state.observations = self._session_observations(state, status=status, error_type=error_type, output=output)
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            object="response",
            model=self.config.model,
            output=output,
            # Each assistant message carries its model call's usage.
            usage=NeMoGymResponseUsage(
                input_tokens=usage["input_tokens"],
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=usage["cached_tokens"]),
                output_tokens=usage["output_tokens"],
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=usage["input_tokens"] + usage["output_tokens"],
            ),
            status=status,
            incomplete_details=incomplete,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            metadata={
                "harness_execution": "sandbox",
                "prime_agent_version": PRIME_AGENT_VERSION,
                "prime_agent_exit_code": str(return_code),
            },
        )
        if status == "failed":
            LOG.warning("Prime Agent run failed (%s); stderr tail: %s", stop, artifacts.stderr[-2048:])
            return self.report_outcome(response, native)
        return self.report_rollout_stop(response, state.session)

    async def _session_response(
        self, request: Request, session_id: str, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        state = self._require_agent_session(session_id)
        if not isinstance(state, PrimeSession):
            raise TypeError("Expected PrimeSession")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Prime Agent activation must use its seeded rollout route")
        if state.session.closing:
            raise HTTPException(409, "Prime Agent session is closing")
        self._session_conversation(body)
        if state.task is None:
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._activate_session(state, state.activation_request))
        elif body != state.activation_request:
            raise HTTPException(
                409, "Prime Agent sessions support one activation; retries must repeat the same request"
            )
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is not None:
            return await self._session_response(request, session_id, body)
        await self._ensure_local_runtime()
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [part for part in [self.config.system_prompt, body.instructions, input_system] if part]
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        rollout_id = request.path_params.get("rollout_id") if request is not None else None

        output_items, usage, model_name, timed_out = await self._run_prime_agent(
            user_message,
            system_prompt,
            rollout_id,
        )

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("Prime Agent produced no assistant message. Padding empty output")
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
            metadata={"prime_agent_timed_out": str(timed_out).lower()},
        )

    async def run(self, request: Request, body: PrimeAgentRunRequest) -> PrimeAgentVerifyResponse:
        if self.config.resources_server is None:
            raise HTTPException(
                422, "Prime Agent /run needs a resources_server; agent sessions use an Environment Server"
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

            return PrimeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    PrimeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = PrimeAgent.run_webserver()  # noqa: F401
