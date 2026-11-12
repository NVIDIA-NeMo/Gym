# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenHands runtime installation and activation artifacts over the shared sandbox lifecycle."""

import asyncio
import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    AgentSessionState,
    SandboxCommand,
    SandboxPythonRuntime,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NgFailureEvent
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)

OPENHANDS_VERSION = "1.59.0"
# openhands-tools pins func-timeout 4.3.5, which publishes only an sdist. The runtime installs wheels only, and
# builds a requirement given as a direct URL from source on the Agent Server host; uv checks the URL's sha256.
_FUNC_TIMEOUT_REQUIREMENT = (
    "func-timeout @ https://files.pythonhosted.org/packages/"
    "b3/0d/bf0567477f7281d9a3926c582bfef21bff7498fc0ffd3e9de21811896a0b/func_timeout-4.3.5.tar.gz"
    "#sha256=74cd3c428ec94f4edfba81f9b2f14904846d5ffccc27c92433b8b5939b5575dd"  # pragma: allowlist secret
)
RUNTIME = SandboxPythonRuntime(
    harness="openhands",
    requirements=(
        f"openhands-sdk=={OPENHANDS_VERSION}",
        f"openhands-tools=={OPENHANDS_VERSION}",
        _FUNC_TIMEOUT_REQUIREMENT,
    ),
    # The imports the runner needs, at the pinned version.
    import_check=(
        "import openhands.sdk, openhands.tools.file_editor, openhands.tools.task_tracker, openhands.tools.terminal, "
        f"openhands.tools.preset.default; assert openhands.sdk.__version__ == {OPENHANDS_VERSION!r}, "
        "openhands.sdk.__version__"
    ),
)
# The runtime directory's name ends with a key of its inputs; the seed refuses a workdir under any of them.
RUNTIME_PREFIX = RUNTIME.directory.rsplit("-", 1)[0]
# The runner runs under this name, outside python/, so 'pkill python3' or 'pkill -f python' in a task cannot stop it.
INTERPRETER = f"{RUNTIME.directory}/openhands-gym"
SESSION_ROOT = "/tmp/nemo-gym-openhands-sessions"
# The runner has no .py suffix in the sandbox, so 'pkill -f "\.py"' in a task cannot stop it.
RUNNER_NAME = "openhands_runner"
RUNNER_SOURCE = Path(__file__).with_name("sandbox_runner.py")
# Bounds what collect downloads from the runner's diagnostics.
_STDERR_TAIL_BYTES = 1024 * 1024


def check_exec(result: SandboxExecResult, what: str) -> None:
    """Raise with the command's output when a setup step in the sandbox fails."""
    if result.return_code != 0 or result.error_type:
        details = "\n".join(part for part in (result.stderr, result.stdout) if part)
        raise RuntimeError(
            f"OpenHands {what} failed (exit {result.return_code}, error_type={result.error_type}): {details[-16000:]}"
        )


async def prepare_runtime(sandbox: AsyncSandbox, *, timeout: float) -> str:
    """Reuse or install the pinned OpenHands runtime in the sandbox; return the interpreter the runner uses.

    The Agent Server builds the runtime with uv and uploads it (``SandboxPythonRuntime``), so the sandbox needs no
    network and the task image's Python is never used.
    """
    if not await RUNTIME.is_installed(sandbox):
        await RUNTIME.install(sandbox, timeout_s=timeout)
    link = shlex.quote(INTERPRETER)
    # Concurrent seeds in one sandbox may race to create the link; either one's link serves.
    command = f"test -L {link} || ln -s python/bin/python3 {link} || test -L {link}"
    check_exec(await sandbox.exec(command, cwd="/", timeout_s=30), "interpreter link")
    return INTERPRETER


class HarnessProcessInfo(BaseModel):
    """Optional runner identity, independent of the supervisor cleanup receipt."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read runner diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("OpenHands runtime metadata is missing or malformed")
        return None


@dataclass
class OpenHandsArtifacts:
    """Runner output copied after the supervisor confirmed that the harness stopped."""

    events: str
    result: str | None
    stderr: str


@dataclass
class OpenHandsSandboxSession(AgentSessionState):
    """OpenHands request and output state; SandboxSession owns execution and teardown."""

    session: SandboxSession[OpenHandsArtifacts]
    interpreter: str = ""
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    provider_wait_failure: NgFailureEvent | None = None

    def path(self, name: str) -> str:
        return f"{self.session.session_dir}/{name}"

    async def close(self, timeout: float) -> None:
        """Stop and release the harness before cancelling an unfinished activation."""
        await self.session.close(timeout=timeout)
        # Captured artifacts are converted by the activation after execute() returns; let it finish.
        if self.session.artifacts is None and self.task is not None and not self.task.done():
            if not self.task.cancelling():
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
                # The activation route replays its own error; close only needs it finished.

    async def execute(
        self, runner_input: dict[str, JsonValue], *, timeout: float, close_timeout: float
    ) -> OpenHandsArtifacts:
        """Run the activation through the shared lifecycle, retaining artifacts on failure."""
        artifacts = await self.session.execute(
            stage_activation=lambda: self.stage_activation(runner_input),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
        )
        if self.session.closing:
            raise asyncio.CancelledError("OpenHands session closed during the activation")
        return artifacts

    async def stage_activation(self, runner_input: dict[str, JsonValue]) -> SandboxCommand:
        """Upload the runner input and return the supervised runner command."""
        sandbox = self.session.sandbox
        await upload_text(sandbox, path=self.path("input.json"), text=json.dumps(runner_input))
        # Regular files keep a detached task service from holding the provider's output pipe open.
        params = {
            "cwd": self.session.workdir or "/",
            "command": [self.interpreter, "-I", "-B", self.path(RUNNER_NAME), self.path("input.json")],
            "env": {},
            "stdin": "/dev/null",
            "stdout": self.path("stdout.log"),
            "stderr": self.path("stderr.log"),
        }
        await upload_text(sandbox, path=self.path("params.sh"), text=shell_assignments(params))
        return SandboxCommand(argv=["bash", self.path(HARNESS_RUNNER_FILE), self.path("params.sh")])

    async def collect_artifacts(self) -> OpenHandsArtifacts:
        """Copy the event log, the run result and the end of stderr; a missing result is reported later."""
        if self.session.expired_before_launch:
            # A rollout stop arrived before launch, so OpenHands never ran.
            return OpenHandsArtifacts(events="", result=None, stderr="")
        sandbox = self.session.sandbox
        stderr = ""
        try:
            tail = await sandbox.exec(
                f"tail -c {_STDERR_TAIL_BYTES} {shlex.quote(self.path('stderr.log'))}", cwd="/", timeout_s=60
            )
            stderr = tail.stdout or ""
        except Exception:
            LOG.warning("OpenHands stderr is unavailable", exc_info=True)
        try:
            events = await read_text(sandbox, path=self.path("events.jsonl"))
        except Exception as error:
            cleanup = self.session.cleanup
            if cleanup and cleanup["timed_out"] and await self._events_absent():
                # The deadline or a rollout stop can end the runner after launch but before it records its first
                # event; the activation judges that run like any other without a model reply.
                LOG.warning("OpenHands recorded no events before its deadline or a rollout stop; keeping an empty run")
                return OpenHandsArtifacts(events="", result=None, stderr=stderr)
            logs = await self.session.read_output_log()
            raise RuntimeError(f"OpenHands runner recorded no events: {(stderr or logs)[-16000:]}") from error
        try:
            result = await read_text(sandbox, path=self.path("result.json"))
        except Exception:
            # The activation judges a missing result: graded after a deadline, otherwise a runner failure.
            LOG.warning("Could not read OpenHands result.json", exc_info=True)
            result = None
        try:
            runtime = json.loads(await read_text(sandbox, path=self.path("runtime.json")))
        except Exception:
            runtime = None
        self.runtime_info = parse_runtime_info(runtime)
        return OpenHandsArtifacts(events=events, result=result, stderr=stderr)

    async def _events_absent(self) -> bool:
        """Whether the session directory exists without events.jsonl; an unclear answer counts as present."""
        directory = shlex.quote(self.session.session_dir)
        events = shlex.quote(self.path("events.jsonl"))
        try:
            probe = await self.session.sandbox.exec(
                f"if [ -e {events} ]; then echo present; elif [ -d {directory} ]; then echo absent; fi",
                cwd="/",
                timeout_s=10,
            )
        except Exception:
            LOG.warning("Could not check for OpenHands events after the deadline", exc_info=True)
            return False
        return probe.return_code == 0 and not probe.error_type and (probe.stdout or "").strip() == "absent"
