# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenHands runtime installation and activation artifacts over the shared sandbox lifecycle."""

import asyncio
import hashlib
import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agents import HARNESS_RUNNER_FILE, AgentSessionState, SandboxCommand, SandboxSession, shell_assignments
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle
from responses_api_agents.miniswe_sandboxed_agent.bootstrap import cached_download


LOG = logging.getLogger(__name__)

OPENHANDS_VERSION = "1.59.0"
RUNTIME_ROOT = "/tmp/nemo-gym-openhands-runtime"
SESSION_ROOT = "/tmp/nemo-gym-openhands-sessions"
# The runner has no .py suffix in the sandbox, so 'pkill -f "\.py"' in a task cannot stop it.
RUNNER_NAME = "openhands_runner"
RUNNER_SOURCE = Path(__file__).with_name("sandbox_runner.py")
_INSTALLER = Path(__file__).with_name("install_openhands_runtime.sh")
_REQUIREMENTS = Path(__file__).with_name("sandbox_requirements.txt")
# Astral python-build-standalone 20260805, CPython 3.13.14 install_only builds, with their published digests.
# https://github.com/astral-sh/python-build-standalone/releases/tag/20260805
_PYTHON_RELEASE = "20260805"
_PYTHON_VERSION = "3.13.14"
_PYTHON_SHA256 = {
    "x86_64/gnu": "9f9ebf776b2d4913fe5a91508ab42e6ce17d266138f76a7782c05fa1c3cba1d1",  # pragma: allowlist secret
    "aarch64/gnu": "4777d7df2edb47b96e53abad5e1b9df1b2a1a9b2f7bdba12b5c0122163b3fed9",  # pragma: allowlist secret
}
# Bounds what collect downloads from the runner's diagnostics.
_STDERR_TAIL_BYTES = 1024 * 1024


def runtime_fingerprint() -> str:
    """Identify a runtime by everything that decides its contents."""
    digest = hashlib.sha256()
    digest.update(json.dumps([_PYTHON_RELEASE, _PYTHON_VERSION, _PYTHON_SHA256, OPENHANDS_VERSION]).encode())
    digest.update(_INSTALLER.read_bytes())
    digest.update(_REQUIREMENTS.read_bytes())
    return digest.hexdigest()


def check_exec(result: SandboxExecResult, what: str) -> None:
    """Raise with the command's output when a setup step in the sandbox fails."""
    if result.return_code != 0 or result.error_type:
        details = "\n".join(part for part in (result.stderr, result.stdout) if part)
        raise RuntimeError(
            f"OpenHands {what} failed (exit {result.return_code}, error_type={result.error_type}): {details[-16000:]}"
        )


async def prepare_runtime(sandbox: AsyncSandbox, *, timeout: float) -> str:
    """Reuse or install the pinned OpenHands runtime in the sandbox; return its interpreter.

    The interpreter is a standalone CPython that the Agent Server downloads, verifies and uploads, so the
    task image's Python is never used. The installer then installs the hash-locked OpenHands packages from
    PyPI, which the sandbox must reach. Cold setup needs Linux x86_64 or aarch64 with glibc, bash, tar,
    gzip, sha256sum, flock, tee, mv with -T support, and a writable /tmp.
    """
    fingerprint = runtime_fingerprint()
    scratch = f"/tmp/nemo-gym-openhands-setup-{uuid4().hex}"
    script = scratch + ".sh"
    await sandbox.upload(_INSTALLER, script)
    probe = await sandbox.exec(shlex.join(["bash", script, "probe"]), cwd="/", timeout_s=60)
    check_exec(probe, "sandbox probe")
    target = (probe.stdout or "").strip()
    if target not in _PYTHON_SHA256:
        raise RuntimeError(f"Unsupported OpenHands sandbox platform: {target!r}")
    runtime = f"{RUNTIME_ROOT}/{fingerprint}-{target.split('/')[0]}"
    interpreter = f"{runtime}/interp/bin/openhands-gym"
    cached = await sandbox.exec(shlex.join(["bash", script, "check", runtime, fingerprint]), cwd="/", timeout_s=120)
    if cached.return_code == 0 and not cached.error_type:
        await _remove_setup_files(sandbox, scratch, script)
        return interpreter

    arch, libc = target.split("/")
    expected = _PYTHON_SHA256[target]
    url = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{_PYTHON_RELEASE}/cpython-{_PYTHON_VERSION}%2B{_PYTHON_RELEASE}-{arch}-unknown-linux-{libc}-install_only.tar.gz"
    )
    archive = await cached_download(url, sha256=expected)
    with TemporaryDirectory(prefix="gym-openhands-runtime-") as directory:
        local = Path(directory) / "python.tar.gz"
        local.write_bytes(archive.read_bytes())
        # Check the private copy too, so a cache replacement cannot change the bytes before upload.
        if hashlib.sha256(local.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"OpenHands Python artifact checksum mismatch for {target}")
        await sandbox.upload(local, f"{scratch}/python.tar.gz")
    await sandbox.upload(_REQUIREMENTS, f"{scratch}/requirements.txt")
    requirements_sha256 = hashlib.sha256(_REQUIREMENTS.read_bytes()).hexdigest()
    command = shlex.join(["bash", script, "install", runtime, fingerprint, expected, requirements_sha256])
    check_exec(await sandbox.exec(command, cwd="/", timeout_s=timeout), "runtime installation")
    # A failed installation keeps its scratch directory and log for diagnosis.
    await _remove_setup_files(sandbox, scratch, script)
    return interpreter


async def _remove_setup_files(sandbox: AsyncSandbox, scratch: str, script: str) -> None:
    try:
        result = await sandbox.exec(f"rm -rf -- {shlex.quote(scratch)} {shlex.quote(script)}", cwd="/", timeout_s=60)
        if result.return_code != 0 or result.error_type:
            LOG.warning("Could not remove OpenHands setup files at %s", scratch)
    except Exception:
        # The runtime is ready; a cleanup transport failure must not fail the seed.
        LOG.warning("Could not remove OpenHands setup files at %s", scratch, exc_info=True)


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
            logs = await self.session.read_output_log()
            raise RuntimeError(f"OpenHands runner recorded no events: {(stderr or logs)[-16000:]}") from error
        try:
            result = await read_text(sandbox, path=self.path("result.json"))
        except Exception:
            result = None
        try:
            runtime = json.loads(await read_text(sandbox, path=self.path("runtime.json")))
        except Exception:
            runtime = None
        self.runtime_info = parse_runtime_info(runtime)
        return OpenHandsArtifacts(events=events, result=result, stderr=stderr)
