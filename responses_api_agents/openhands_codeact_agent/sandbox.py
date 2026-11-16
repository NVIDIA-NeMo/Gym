# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenHands CodeAct session state: the staged runtime, one supervised activation, and its collected logs."""

import asyncio
import json
import logging
import shlex
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from nemo_gym.agents import HARNESS_RUNNER_FILE, AgentSessionState, SandboxCommand, SandboxSession, shell_assignments
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult, read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)

SESSION_ROOT = "/tmp/nemo-gym-openhands-codeact-sessions"
# The runner and the modules it imports, uploaded into each session directory.
RUNNER_FILES = ("run_openhands.py", "direct_model_client.py", "compaction_condenser.py")
SOURCE_DIR = Path(__file__).parent
# Bounds on the logs collected from the sandbox. Only the completion, model-request and compaction logs are read;
# OpenHands' own session store and event dump stay in the sandbox.
MAX_TRACE_FILES = 10_000
MAX_TRACE_BYTES = 1_073_741_824
MAX_TRACE_FILE_BYTES = 67_108_864
_STDERR_TAIL_BYTES = 1024 * 1024
_PACK_TRACE = """import sys, zipfile
from pathlib import Path
root, archive, max_files, max_bytes, max_file = Path(sys.argv[1]), sys.argv[2], *map(int, sys.argv[3:6])
paths = [p for p in [*(root / 'completions').glob('*.json'), root / 'model-requests.jsonl', root / 'compactions.jsonl']
         if p.is_file() and not p.is_symlink()]
sizes = [p.stat().st_size for p in paths]
if len(paths) > max_files or sum(sizes) > max_bytes or any(size > max_file for size in sizes):
    sys.exit('OpenHands trace exceeds collection limits')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in paths:
        bundle.write(path, path.relative_to(root))
"""


def check_exec(result: SandboxExecResult, what: str) -> None:
    """Raise with the command's output when a setup step in the sandbox fails."""
    if result.return_code != 0 or result.error_type:
        details = "\n".join(part for part in (result.stderr, result.stdout) if part)
        raise RuntimeError(
            f"OpenHands CodeAct {what} failed (exit {result.return_code}, error_type={result.error_type}): "
            f"{details[-16000:]}"
        )


def extract_trace(archive: Path, destination: Path) -> None:
    """Unpack the collected logs, refusing unsafe or oversized members (blocking; run it in a thread)."""
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if (
            len(members) > MAX_TRACE_FILES
            or sum(member.file_size for member in members) > MAX_TRACE_BYTES
            or any(member.file_size > MAX_TRACE_FILE_BYTES for member in members)
        ):
            raise ValueError("OpenHands trace exceeds collection limits")
        for member in members:
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe OpenHands trace entry: {member.filename}")
            if member.is_dir() or not path.parts:
                continue
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1_048_576)


@dataclass
class CodeActArtifacts:
    """What one activation left: the runner's result, its collected logs on this host, and its stderr tail."""

    result: str | None
    trace_dir: Path | None
    stderr: str

    def cleanup(self) -> None:
        if self.trace_dir is not None:
            shutil.rmtree(self.trace_dir.parent, ignore_errors=True)


@dataclass
class CodeActSandboxSession(AgentSessionState):
    """Request and output state of one agent session; SandboxSession owns execution and teardown."""

    session: SandboxSession[CodeActArtifacts]
    interpreter: str = ""
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    observations: AgentObservationBundle | None = None

    def path(self, name: str) -> str:
        return f"{self.session.session_dir}/{name}"

    async def close(self, timeout: float) -> None:
        """Stop and release the runner before cancelling an unfinished activation."""
        await self.session.close(timeout=timeout)
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

    async def upload_runner(self) -> None:
        for name in RUNNER_FILES:
            await self.session.sandbox.upload(SOURCE_DIR / name, self.path(name))

    async def execute(self, runner_input: dict[str, Any], *, timeout: float, close_timeout: float) -> CodeActArtifacts:
        """Run the activation through the shared lifecycle, retaining artifacts on failure."""
        artifacts = await self.session.execute(
            stage_activation=lambda: self.stage_activation(runner_input),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
        )
        if self.session.closing:
            raise asyncio.CancelledError("OpenHands CodeAct session closed during the activation")
        return artifacts

    async def stage_activation(self, runner_input: dict[str, Any]) -> SandboxCommand:
        """Upload the runner input and return the supervised runner command."""
        sandbox = self.session.sandbox
        await upload_text(sandbox, path=self.path("input.json"), text=json.dumps(runner_input))
        params = {
            "cwd": self.session.workdir or "/",
            # -E -s: no PYTHON* variables or user site; the runner's directory stays on sys.path for its modules.
            "command": [self.interpreter, "-E", "-s", self.path("run_openhands.py"), self.path("input.json")],
            "env": {},
            "stdin": "/dev/null",
            "stdout": self.path("stdout.log"),
            "stderr": self.path("stderr.log"),
        }
        await upload_text(sandbox, path=self.path("params.sh"), text=shell_assignments(params))
        return SandboxCommand(argv=["bash", self.path(HARNESS_RUNNER_FILE), self.path("params.sh")])

    async def collect_artifacts(self) -> CodeActArtifacts:
        """Copy the result, the logs the response is built from, and the end of stderr."""
        if self.session.expired_before_launch:
            return CodeActArtifacts(result=None, trace_dir=None, stderr="")
        sandbox = self.session.sandbox
        stderr = ""
        try:
            tail = await sandbox.exec(
                f"tail -c {_STDERR_TAIL_BYTES} {shlex.quote(self.path('stderr.log'))}", cwd="/", timeout_s=60
            )
            stderr = tail.stdout or ""
        except Exception:
            LOG.warning("OpenHands CodeAct stderr is unavailable", exc_info=True)
        try:
            result = await read_text(sandbox, path=self.path("result.json"))
        except Exception:
            LOG.warning("Could not read OpenHands CodeAct result.json", exc_info=True)
            result = None
        archive = self.path("trace.zip")
        packed = await sandbox.exec(
            shlex.join(
                [
                    self.interpreter,
                    "-I",
                    "-c",
                    _PACK_TRACE,
                    self.path("trace"),
                    archive,
                    str(MAX_TRACE_FILES),
                    str(MAX_TRACE_BYTES),
                    str(MAX_TRACE_FILE_BYTES),
                ]
            ),
            cwd="/",
            timeout_s=300,
        )
        check_exec(packed, "trace collection")
        local = Path(tempfile.mkdtemp(prefix="openhands-codeact-"))
        try:
            await sandbox.download(archive, local / "trace.zip")
            await asyncio.to_thread(extract_trace, local / "trace.zip", local / "trace")
        except BaseException:
            shutil.rmtree(local, ignore_errors=True)
            raise
        return CodeActArtifacts(result=result, trace_dir=local / "trace", stderr=stderr)
