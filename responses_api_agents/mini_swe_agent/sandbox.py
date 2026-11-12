# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server side of mini-SWE agent sessions in borrowed or owned task sandboxes.

mini-SWE runs in the task sandbox with a private Python runtime that this server builds and uploads at seed with
``SandboxPythonRuntime``, so the sandbox needs no network and no Python of its own. The runtime is locked by
``runtime/requirements.txt``, independent of this server's environment, which keeps the SWE-Gym fork for ``/run``.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from shlex import quote

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agents import (
    OUTPUT_LOG_FILE,
    AgentSeedRejectedError,
    AgentSessionState,
    SandboxCommand,
    SandboxPythonRuntime,
    SandboxSession,
    probe_sandbox_platform,
)
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)

MINI_SWE_VERSION = "2.4.6"
SESSION_ROOT = "/tmp/nemo-gym-mini-swe-sessions"
# Where the Harbor environment server leaves the task's own MCP servers and skills directory at seed.
TASK_CONTEXT_FILE = "/tmp/.nemo-gym/task.json"
# Bounds the runner and MCP logs copied back to the agent server.
LOG_TAIL_BYTES = 1024 * 1024
# The MCP call records cross the provider and go into the response: a larger file is not read, and each call keeps at
# most this much of its output (the model saw at most mini-SWE's 10,000-character view of it).
MCP_CALLS_MAX_BYTES = 16 * 1024 * 1024
# The runtime's interpreter, linked into the session directory under a name without "python". The supervisor's command
# line carries the runner's argv, so a task's `pkill -9 -f python` would otherwise kill the supervisor itself, leaving no
# receipt; the runner and the MCP CLI it starts (with sys.executable) also run under this name.
INTERPRETER_LINK = "bin/mini-swe"
# LiteLLM's wheels in the runtime lock need glibc 2.28 or newer; an older task image can never run this runtime.
MIN_GLIBC_MINOR = 28
MCP_CALL_OUTPUT_MAX_CHARS = 64 * 1024
_RUNTIME_FILES = Path(__file__).with_name("runtime")


def _requirement_lines(path: Path) -> tuple[str, ...]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return tuple(line for line in lines if line and not line.startswith("#"))


MINI_SWE_RUNTIME = SandboxPythonRuntime(
    harness="mini-swe",
    requirements=_requirement_lines(_RUNTIME_FILES / "requirements.in"),
    import_check=(
        # LiteLLM would otherwise fetch its model cost map from GitHub at import; the check makes no network call.
        # mini-SWE makes its config directory at import; keep it in the check's HOME, not in the task's XDG_CONFIG_HOME.
        "import os; os.environ.update(LITELLM_LOCAL_MODEL_COST_MAP='true', MSWEA_SILENT_STARTUP='true', "
        "MSWEA_GLOBAL_CONFIG_DIR=os.path.join(os.environ['HOME'], 'mini-swe-agent')); "
        f"import minisweagent; assert minisweagent.__version__ == {MINI_SWE_VERSION!r}, minisweagent.__version__; "
        "from minisweagent.models.litellm_response_model import LitellmResponseModel; "
        "from mcp.client.streamable_http import streamable_http_client"
    ),
    overrides=_requirement_lines(_RUNTIME_FILES / "overrides.txt"),
    constraints=_requirement_lines(_RUNTIME_FILES / "requirements.txt"),
)


class HarnessProcessInfo(BaseModel):
    """Optional identity of the mini-SWE runner process, separate from cleanup evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int
    python: str | None = None


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read the runner's process diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("mini-SWE runtime metadata is missing or malformed")
        return None


@dataclass(frozen=True)
class MiniSWEArtifacts:
    """What one run left in the session directory, copied before close removes it, including interrupted runs.

    ``trajectory`` is mini-SWE's native transcript, saved after every step; ``None`` when the run saved none.
    ``output`` is the runner's account of how the run ended; ``None`` when the runner was killed before writing it.
    ``invalid_mcp_calls`` counts the lines of ``mcp_calls.jsonl`` that are not a call record; they are left out of
    ``mcp_calls``. ``mcp_calls_unread`` is set when the file was larger than ``MCP_CALLS_MAX_BYTES`` and not read.
    """

    trajectory: dict[str, JsonValue] | None
    output: dict[str, JsonValue] | None
    mcp_calls: list[dict[str, JsonValue]]
    log: str
    invalid_mcp_calls: int = 0
    mcp_calls_unread: bool = False


@dataclass
class MiniSWESession(AgentSessionState):
    """One mini-SWE episode; ``SandboxSession`` owns supervision, termination and release."""

    session: SandboxSession[MiniSWEArtifacts]
    provider_name: str
    sandbox_id: str | None
    task_context: dict[str, JsonValue] = field(default_factory=dict)
    mcp_servers: list[dict[str, JsonValue]] = field(default_factory=list)
    required_mcp_servers: list[str] = field(default_factory=list)
    observations: AgentObservationBundle | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None

    async def install(self, *, timeout: float) -> None:
        """Reuse or install the pinned runtime, then stage the runner and the MCP CLI in the session directory.

        The runtime directory is keyed by its inputs, so a runtime an earlier session left in this sandbox is reused
        when it passes the version and import check; ``install`` publishes a new one only after it passes the same
        check, and concurrent seeds never see a partial one.
        """
        sandbox, directory = self.session.sandbox, self.session.session_dir
        prepared = await sandbox.exec(f"umask 077 && mkdir -p {quote(directory)}", cwd="/", timeout_s=30)
        if prepared.return_code != 0 or prepared.error_type is not None:
            raise RuntimeError(
                prepared.stderr or prepared.stdout or "Failed to prepare the mini-SWE session directory"
            )
        if not await MINI_SWE_RUNTIME.is_installed(sandbox):
            platform = await probe_sandbox_platform(sandbox)
            if platform.libc == "gnu" and platform.glibc_minor is not None and platform.glibc_minor < MIN_GLIBC_MINOR:
                raise AgentSeedRejectedError(
                    f"mini-SWE's runtime needs glibc 2.{MIN_GLIBC_MINOR} or newer; the sandbox has glibc "
                    f"2.{platform.glibc_minor}"
                )
            await MINI_SWE_RUNTIME.install(sandbox, timeout_s=timeout)
        for name in ("sandbox_runner.py", "mcp_cli.py"):
            await sandbox.upload(Path(__file__).with_name(name), f"{directory}/{name}")

    async def read_task_context(self) -> dict[str, JsonValue]:
        """The task's own MCP servers and skills directory, from the file a Harbor environment server writes.

        A read that ran and found nothing returns ``{}``; a provider failure raises, because running the task without
        the tools it declares would score a silent zero. A file that is not a JSON object is the task's own error, so
        the seed is refused rather than retried.
        """
        path = quote(TASK_CONTEXT_FILE)
        # A file that exists but cannot be read fails the read instead of passing for an absent one.
        result = await self.session.sandbox.exec(f"if [ -e {path} ]; then cat {path}; fi", cwd="/", timeout_s=30)
        if result.error_type is not None or result.return_code != 0:
            raise RuntimeError(
                f"The sandbox could not read {TASK_CONTEXT_FILE}: {result.error_type or result.return_code}"
            )
        text = (result.stdout or "").strip()
        if not text:
            return {}
        try:
            context = json.loads(text)
        except json.JSONDecodeError as error:
            raise AgentSeedRejectedError(f"{TASK_CONTEXT_FILE} is not JSON") from error
        if not isinstance(context, dict):
            raise AgentSeedRejectedError(f"{TASK_CONTEXT_FILE} is not a JSON object")
        return context

    async def close(self, timeout: float) -> None:
        """Finish sandbox capture and release before cancelling the HTTP activation."""
        await self.session.close(timeout=timeout)
        if self.task is not None and not self.task.done() and not self.task.cancelling():
            self.task.cancel()
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except asyncio.CancelledError:
                if not self.task.cancelled():
                    raise
            except Exception:
                if not self.task.done():
                    raise
                # The agent base replays the activation error independently of close.

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Stage this activation's input and the MCP servers, then describe the runner process."""
        directory = self.session.session_dir
        if self.mcp_servers:
            await upload_text(
                self.session.sandbox, path=f"{directory}/mcp_servers.json", text=json.dumps(self.mcp_servers)
            )
        await upload_text(self.session.sandbox, path=f"{directory}/input.json", text=json.dumps(payload))
        interpreter = f"{directory}/{INTERPRETER_LINK}"
        linked = await self.session.sandbox.exec(
            f"mkdir -p {quote(str(PurePosixPath(interpreter).parent))}"
            f" && ln -sfn {quote(MINI_SWE_RUNTIME.python)} {quote(interpreter)}",
            cwd="/",
            timeout_s=30,
        )
        if linked.return_code != 0 or linked.error_type is not None:
            raise RuntimeError(f"Could not link the mini-SWE interpreter: {linked.error_type or linked.stderr}")
        return SandboxCommand(
            argv=[
                interpreter,
                # Ignore the task's Python settings without changing the environment its commands inherit.
                "-E",
                "-s",
                f"{directory}/sandbox_runner.py",
                f"{directory}/input.json",
                f"{directory}/output.json",
            ]
        )

    async def collect(self) -> MiniSWEArtifacts:
        """Copy the run's files after termination is confirmed; a file the run did not write is ``None`` or empty."""
        directory = self.session.session_dir
        trajectory = await self._read_json(f"{directory}/trajectory.json")
        output = await self._read_json(f"{directory}/output.json")
        if output is not None:
            self.runtime_info = parse_runtime_info(output.get("runtime"))
        mcp_calls, invalid_mcp_calls, mcp_calls_unread = await self._read_mcp_calls(f"{directory}/mcp_calls.jsonl")
        logs = []
        # Trimmed in the sandbox: a long run's output need not cross the provider.
        for name in (OUTPUT_LOG_FILE, "mcp.log"):
            tail = await self.session.sandbox.exec(
                f"tail -c {LOG_TAIL_BYTES} {quote(f'{directory}/{name}')} 2>/dev/null || true", cwd="/", timeout_s=30
            )
            if tail.stdout:
                logs.append(f"[{name}]\n{tail.stdout}")
        return MiniSWEArtifacts(
            trajectory=trajectory,
            output=output,
            mcp_calls=mcp_calls,
            log="\n".join(logs),
            invalid_mcp_calls=invalid_mcp_calls,
            mcp_calls_unread=mcp_calls_unread,
        )

    async def _read_mcp_calls(self, path: str) -> tuple[list[dict[str, JsonValue]], int, bool]:
        """The MCP call records, how many lines were not one, and whether the file was too large to read.

        The model's commands run as the user that appends to the file, so a line can be partial, malformed or written
        by the model itself. A run without MCP servers made no calls, and its file is not read. A call to a server the
        run was not granted is kept: the CLI records it as a failed call.
        """
        if not self.mcp_servers:
            return [], 0, False
        size = await self.session.sandbox.exec(f"wc -c < {quote(path)} 2>/dev/null || true", cwd="/", timeout_s=30)
        if (size.stdout or "").strip().isdigit() and int(size.stdout) > MCP_CALLS_MAX_BYTES:
            return [], 0, True
        calls: list[dict[str, JsonValue]] = []
        invalid = 0
        for line in (await self._read_optional(path) or "").split("\n"):
            if not line.strip():
                continue
            try:
                call = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if (
                isinstance(call, dict)
                and isinstance(call.get("server"), str)
                and isinstance(call.get("tool"), str)
                and isinstance(call.get("output"), str)
                and isinstance(call.get("tool_call_id"), str | None)
            ):
                if len(call["output"]) > MCP_CALL_OUTPUT_MAX_CHARS:
                    omitted = len(call["output"]) - MCP_CALL_OUTPUT_MAX_CHARS
                    call["output"] = (
                        f"{call['output'][:MCP_CALL_OUTPUT_MAX_CHARS]}\n[{omitted} characters omitted by NeMo Gym]"
                    )
                calls.append(call)
            else:
                invalid += 1
        return calls, invalid, False

    async def _read_optional(self, path: str) -> str | None:
        exists = await self.session.sandbox.exec(f"test -f {quote(path)}", cwd="/", timeout_s=30)
        if exists.error_type is not None or exists.return_code not in (0, 1):
            raise RuntimeError(f"Could not check {path} in the sandbox: {exists.error_type or exists.stderr}")
        if exists.return_code != 0:
            return None
        return await read_text(self.session.sandbox, path=path)

    async def _read_json(self, path: str) -> dict[str, JsonValue] | None:
        text = await self._read_optional(path)
        if text is None:
            return None
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise TypeError(f"mini-SWE sandbox payload at {path} is not an object")
        return payload
