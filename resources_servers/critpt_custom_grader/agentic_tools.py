# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent offline code execution for the opt-in agentic grading recipe."""

import asyncio
import hashlib
import json
import logging
import math
import shlex
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxConfig, SandboxSpec, create_provider
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    ResourcesSessionLostError,
)


LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_code_execution_params() -> type[BaseModel]:
    """Load Stirrup's canonical ``cmd`` schema only when agentic tools are enabled."""
    from stirrup.tools.code_backends.base import CodeExecutionParams

    return CodeExecutionParams


def _format_result(*, exit_code: int = 1, stdout: str = "", stderr: str = "", error_kind: str | None = None) -> str:
    # Import, rather than copy, the pinned framework's XML and 20,000-character truncation contract.
    from stirrup.tools.code_backends.base import CommandResult, format_result

    result = format_result(CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr, error_kind=error_kind))
    if not isinstance(result.content, str):
        raise RuntimeError("Unexpected Stirrup code execution result type")
    return result.content


class AgenticSandboxSpecConfig(SandboxConfig):
    """A prebuilt execution image; task files are never mounted or uploaded."""

    image: str = Field(min_length=1)
    workdir: str = Field(default="/sandbox", pattern=r"^/")
    user: str | int | None = None
    ttl_s: float = Field(default=21600, gt=0, allow_inf_nan=False)
    ready_timeout_s: float = Field(default=300, gt=0, allow_inf_nan=False)


class AgenticExecutionConfig(BaseModel):
    """Limits and provider selection for isolated per-episode code execution."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    sandbox_provider: str | dict[str, Any]
    sandbox_spec: AgenticSandboxSpecConfig
    command_timeout_s: float = Field(default=300, gt=0, le=3600, allow_inf_nan=False)
    provider_timeout_margin_s: float = Field(default=60, ge=10, le=600, allow_inf_nan=False)
    max_command_bytes: int = Field(default=65536, ge=1, le=1048576)
    max_output_bytes: int = Field(default=1048576, ge=20000, le=16777216)
    max_sessions: int = Field(default=16, ge=1, le=1024)
    create_timeout_s: float = Field(default=600, gt=0, allow_inf_nan=False)
    close_timeout_s: float = Field(default=60, gt=0, allow_inf_nan=False)


class AgenticExecutionError(RuntimeError):
    """A bounded infrastructure category, never a candidate correctness result."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class _CommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    exit_code: int
    stdout: str
    stderr: str
    error_kind: Literal["timeout", "output_limit"] | None = None


# This wrapper runs inside the execution sandbox, never in the resources-server process. Nonblocking reads cap
# normal stdout/stderr capture and avoid hanging on a background child's pipe. Process-group cleanup is best effort;
# the sandbox's resource limits and TTL bound descendants that detach themselves. This wrapper is not a separate
# security boundary against commands manipulating their own sandbox. Each command gets a fresh shell and keeps
# the episode's files. Provider timeouts remain separate failures.
_COMMAND_RUNNER = r"""
import json
import os
import selectors
import signal
import subprocess
import sys
import time

command, timeout, output_limit = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
process = subprocess.Popen(
    ["bash", "-c", command], stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
)

def stop_group():
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

buffers = {"stdout": bytearray(), "stderr": bytearray()}
error_kind = None
deadline = time.monotonic() + timeout
selector = selectors.DefaultSelector()
try:
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    while selector.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            error_kind = "timeout"
            break
        for key, _ in selector.select(min(remaining, 0.1)):
            chunk = os.read(key.fd, 65536)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            target = buffers[key.data]
            if len(target) + len(chunk) > output_limit:
                error_kind = "output_limit"
                break
            target.extend(chunk)
        if error_kind:
            break
    if error_kind is None:
        try:
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            error_kind = "timeout"
finally:
    stop_group()
    process.wait(timeout=5)
    selector.close()
    process.stdout.close()
    process.stderr.close()

if error_kind == "timeout":
    stdout, stderr = "", "Command exceeded its execution time limit."
elif error_kind == "output_limit":
    stdout, stderr = "", "Command exceeded its output byte limit."
else:
    stdout = buffers["stdout"].decode("utf-8", errors="replace")
    stderr = buffers["stderr"].decode("utf-8", errors="replace")
print(json.dumps({"exit_code": process.returncode, "stdout": stdout, "stderr": stderr, "error_kind": error_kind}))
"""


def _validate_offline(provider_config: dict[str, Any], spec: SandboxSpec) -> None:
    """Require explicit network isolation in the selected provider's supported configuration."""
    if len(provider_config) != 1:
        raise ValueError("Offline execution requires one resolved sandbox provider")
    name, options = next(iter(provider_config.items()))
    options = options or {}
    if not isinstance(options, dict):
        raise ValueError("Offline execution provider settings must be a mapping")
    create = options.get("create") or {}
    if not isinstance(create, dict):
        raise ValueError("Offline execution create settings must be a mapping")
    if name == "opensandbox":
        allowed = {"network_policy"}
        offline = (
            spec.provider_options.get("network_policy") == {"defaultAction": "deny", "egress": []}
            and create.get("renew_interval_s") is None
        )
    elif name == "docker":
        allowed = set()
        offline = create.get("network") == "none" and not create.get("extra_run_args")
    elif name == "e2b":
        allowed = {"template"}
        offline = create.get("allow_internet_access") is False
    else:
        allowed = set()
        offline = False
    if not offline or set(spec.provider_options) - allowed:
        raise ValueError("The selected provider configuration does not establish offline execution")
    if spec.files or spec.sidecars or spec.ports or spec.entrypoint:
        raise ValueError("Offline execution does not accept files, sidecars, ports or an entrypoint override")
    if not spec.image or not spec.image.strip():
        raise ValueError("Offline execution requires a prebuilt image")
    if (
        isinstance(spec.ttl_s, bool)
        or not isinstance(spec.ttl_s, (float, int))
        or not math.isfinite(spec.ttl_s)
        or spec.ttl_s <= 0
    ):
        raise ValueError("Offline execution requires a finite positive sandbox TTL")


@dataclass
class _ExecutionSession:
    episode_id: EpisodeId
    task_id: TaskId
    task_digest: str
    sandbox: AsyncSandbox
    expires_at: float
    usable: bool = False
    discard_requested: bool = False
    creation_task: asyncio.Task[None] | None = None
    stop_task: asyncio.Task[None] | None = None


class AgenticExecutionSessions:
    """Own isolated execution sandboxes and fence their seed, command and close operations."""

    def __init__(
        self,
        config: AgenticExecutionConfig,
        *,
        provider_config: dict[str, Any],
        sandbox_spec: SandboxSpec,
        close_retry_window_seconds: float,
    ) -> None:
        _validate_offline(provider_config, sandbox_spec)
        self.config = config
        self._spec = sandbox_spec
        self._provider = create_provider(provider_config)
        self._sessions: dict[str, _ExecutionSession] = {}
        self._fence = ResourcesSessionFence(close_retry_window_seconds)
        self._shutting_down = False
        self._provider_closed = False

    @staticmethod
    def _observe_background_result(task: asyncio.Task[None]) -> None:
        # A request can time out before its owned provider operation finishes. Retrieve any late exception.
        if not task.cancelled() and (error := task.exception()) is not None:
            LOGGER.warning("Execution sandbox background operation failed (%s)", type(error).__name__)

    async def _stop_started(self, state: _ExecutionSession) -> None:
        stop = state.stop_task
        if stop is None or (stop.done() and (stop.cancelled() or stop.exception() is not None)):
            stop = state.stop_task = asyncio.create_task(state.sandbox.stop())
            stop.add_done_callback(self._observe_background_result)
        try:
            # Keep the deletion operation owned after this request's budget expires; retries await the same task.
            await asyncio.wait_for(asyncio.shield(stop), self.config.close_timeout_s)
        except Exception as error:
            LOGGER.warning("Execution sandbox cleanup failed (%s)", type(error).__name__)
            raise AgenticExecutionError("execution_cleanup_failed") from None

    async def _start_owned(self, state: _ExecutionSession) -> None:
        try:
            await state.sandbox.start()
        finally:
            if state.discard_requested:
                try:
                    await self._stop_started(state)
                except Exception as error:
                    LOGGER.warning("Late execution sandbox cleanup deferred (%s)", type(error).__name__)

    async def _stop(self, state: _ExecutionSession) -> None:
        state.usable = False
        state.discard_requested = True
        if state.creation_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(state.creation_task), self.config.close_timeout_s)
            except TimeoutError:
                # Never close a handle-less AsyncSandbox while create can still return a live handle.
                if not state.creation_task.done():
                    raise AgenticExecutionError("execution_cleanup_failed") from None
            except Exception:
                # A failed create can still leave a facade handle available for cleanup.
                pass
        await self._stop_started(state)

    async def _abort(self, state: _ExecutionSession) -> None:
        # Keep the unusable record even after successful cleanup: retries may close it, but must not recreate it.
        state.usable = False
        try:
            await self._stop(state)
        except Exception as error:
            LOGGER.warning("Execution sandbox abort cleanup deferred (%s)", type(error).__name__)

    async def _reap_expired(self) -> None:
        # Each close takes its own fence. Do this before locking a new seed, so concurrent reapers cannot deadlock.
        expired = [
            ResourcesCloseSessionRequest(resources_session_id=key, episode_id=state.episode_id)
            for key, state in list(self._sessions.items())
            if time.monotonic() >= state.expires_at
        ]
        results = await asyncio.gather(*(self.close(request) for request in expired), return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                LOGGER.warning("Expired execution sandbox cleanup deferred (%s)", type(result).__name__)

    async def seed(self, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        """Create once per identity; task data contributes only an identity digest and is never delivered."""
        session_id = body.resources_session_id
        digest = hashlib.sha256(json.dumps(body.task_data, sort_keys=True, allow_nan=False).encode()).hexdigest()
        await self._reap_expired()
        async with self._fence.locked(session_id):
            if self._shutting_down:
                raise AgenticExecutionError("execution_shutdown")
            self._fence.raise_if_closed(session_id)
            state = self._sessions.get(session_id)
            if state is not None:
                if (state.episode_id, state.task_id, state.task_digest) != (body.episode_id, body.task_id, digest):
                    raise ResourcesSessionConflictError(
                        "Execution session is already bound to another episode or task"
                    )
                if state.usable and time.monotonic() >= state.expires_at:
                    await self._abort(state)
                if not state.usable:
                    raise ResourcesSessionConflictError("Execution session is closed or failed")
                return ResourcesSeedSessionResponse(resources_session_id=session_id)
            if len(self._sessions) >= self.config.max_sessions:
                raise AgenticExecutionError("execution_capacity_exceeded")
            spec = replace(
                self._spec,
                metadata=self._spec.metadata
                | {"phase": "agentic_execution", "episode": body.episode_id.capture_key[:63]},
            )
            state = _ExecutionSession(
                body.episode_id,
                body.task_id,
                digest,
                AsyncSandbox(self._provider, spec, owns_provider=False),
                time.monotonic() + float(spec.ttl_s),
            )
            self._sessions[session_id] = state
            state.creation_task = asyncio.create_task(self._start_owned(state))
            state.creation_task.add_done_callback(self._observe_background_result)
            try:
                await asyncio.wait_for(asyncio.shield(state.creation_task), self.config.create_timeout_s)
            except BaseException as error:
                await self._abort(state)
                if isinstance(error, asyncio.CancelledError):
                    raise
                LOGGER.warning("Execution sandbox creation failed (%s)", type(error).__name__)
                raise AgenticExecutionError("execution_create_failed") from None
            state.usable = True
        # Lending the execution sandbox would make the agent run its networked orchestration inside it.
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def execute(self, session_id: str, cmd: str) -> str:
        """Run one bounded command in a fresh shell, preserving only the episode's filesystem."""
        async with self._fence.locked(session_id):
            state = self._sessions.get(session_id)
            if state is None or not state.usable:
                raise ResourcesSessionLostError("Execution session is unavailable")
            if time.monotonic() >= state.expires_at:
                await self._abort(state)
                raise ResourcesSessionLostError("Execution session expired")
            try:
                valid = (
                    bool(cmd.strip())
                    and "\x00" not in cmd
                    and len(cmd.encode("utf-8")) <= self.config.max_command_bytes
                )
            except (AttributeError, UnicodeError):
                valid = False
            if not valid:
                return _format_result(error_kind="invalid_argument", stderr="Command is empty, invalid or too large.")
            command = " ".join(
                shlex.quote(arg)
                for arg in (
                    "python3",
                    "-I",
                    "-c",
                    _COMMAND_RUNNER,
                    cmd,
                    str(self.config.command_timeout_s),
                    str(self.config.max_output_bytes),
                )
            )
            deadline = self.config.command_timeout_s + self.config.provider_timeout_margin_s
            try:
                result = await asyncio.wait_for(state.sandbox.exec(command, timeout_s=deadline), deadline + 1)
                if result.error_type is not None or result.return_code != 0:
                    raise AgenticExecutionError("execution_provider_failed")
                # JSON escaping can expand each captured byte to six characters. Refuse malformed envelopes too.
                payload = result.stdout or ""
                if len(payload) > 12 * self.config.max_output_bytes + 4096:
                    raise AgenticExecutionError("execution_provider_failed")
                observation = _CommandResult.model_validate_json(payload)
                if (
                    max(len(observation.stdout.encode()), len(observation.stderr.encode()))
                    > 3 * self.config.max_output_bytes
                ):
                    raise AgenticExecutionError("execution_provider_failed")
            except BaseException as error:
                await self._abort(state)
                if isinstance(error, asyncio.CancelledError):
                    raise
                LOGGER.warning("Execution sandbox command failed (%s)", type(error).__name__)
                raise AgenticExecutionError("execution_provider_failed") from None
            return _format_result(**observation.model_dump())

    async def close(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Stop before forgetting state; a failed stop remains available for a later cleanup retry."""
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            closed_episode = self._fence.closed_episode(session_id)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed execution session")
            state = self._sessions.get(session_id)
            if state is not None:
                if state.episode_id != body.episode_id:
                    raise ResourcesSessionConflictError("episode_id does not match the seeded execution session")
                await self._stop(state)
                del self._sessions[session_id]
            self._fence.mark_closed(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def shutdown(self) -> None:
        """Reject new seeds, stop all remaining sessions and then close the shared provider."""
        self._shutting_down = True
        results = await asyncio.gather(
            *(
                self.close(ResourcesCloseSessionRequest(resources_session_id=key, episode_id=state.episode_id))
                for key, state in list(self._sessions.items())
            ),
            return_exceptions=True,
        )
        if any(isinstance(result, BaseException) for result in results):
            raise AgenticExecutionError("execution_cleanup_failed")
        if not self._provider_closed:
            try:
                await asyncio.wait_for(self._provider.aclose(), self.config.close_timeout_s)
            except Exception as error:
                LOGGER.warning("Execution provider cleanup failed (%s)", type(error).__name__)
                raise AgenticExecutionError("execution_cleanup_failed") from None
            self._provider_closed = True
