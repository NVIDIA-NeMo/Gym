# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cline runtime, input staging and artifacts over the shared task-sandbox lifecycle."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from time import perf_counter

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    AgentSessionState,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult, read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)
INSTALLER = "install_cline_runtime.sh"
STDERR_TAIL_BYTES = 16000

# SHA-512 (hex) of the npm tarballs @cline/cli-linux-{x64,arm64}, which carry Cline's self-contained binary.
# They are npm's own `dist.integrity` for each release; add a release here before pinning it in cline_version.
CLINE_RUNTIME_SHA512: dict[str, dict[str, str]] = {
    "3.0.55": {
        "x64": "bd5aaa113eb4b42e4a7c87dd1f02d3c9833846e9a62384b8c59bfeb1c68ec177"
        "f869aec3f4c4c0ac8f7f0745bc57eea586176050b45f3720cca66fe5fec106ba",
        "arm64": "f89fcd965bdaeff0ac49f93ab9dad263cba419ac5bbfd5af11b08f19662b99f3"
        "24d4067f4e1b5cdefe6b904944adbcc3c419d0436e2becad253d1a91264e4d5a",
    },
}

# Bootstrap with POSIX sh; the installer and the harness runner require Bash.
_BASH_BOOTSTRAP = """set -eu
set --
command -v bash >/dev/null 2>&1 || set -- "$@" bash
if [ "$#" -gt 0 ]; then
    [ "$(id -u)" = 0 ] || { echo "Cline sandbox sessions require $*: preinstall these tools or use a root image." >&2; exit 1; }
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y "$@"
    elif command -v yum >/dev/null 2>&1; then
        yum install -y "$@"
    else
        echo "Cline sandbox sessions require $*: preinstall these tools (automatic installation requires apt-get, dnf or yum)." >&2
        exit 1
    fi
fi
"""


class HarnessProcessInfo(BaseModel):
    """Optional runner identity, independent of the supervisor's cleanup receipt."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read runner diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("Cline runtime metadata is missing or malformed")
        return None


def check_sandbox_setup(result: SandboxExecResult, what: str) -> None:
    """Raise with both streams: background execution can leave only a generic message in stderr."""
    if result.return_code != 0 or result.error_type:
        details = "\n".join(part for part in (result.stderr, result.stdout) if part)
        raise RuntimeError(
            f"{what} failed (exit {result.return_code}, error_type={result.error_type}): {details[-16000:]}"
        )


@dataclass(frozen=True, kw_only=True)
class ClineArtifacts:
    """CLI output copied before the sandbox connection is released."""

    stdout: str
    stderr: str
    wall_time_s: float | None = None


@dataclass
class ClineSandboxSession(AgentSessionState):
    """Cline input/output state; SandboxSession owns process supervision, capture and release."""

    session: SandboxSession[ClineArtifacts]
    runtime: str
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    observations: AgentObservationBundle | None = None
    runtime_info: HarnessProcessInfo | None = None
    started_at: float | None = None

    @property
    def config_dir(self) -> str:
        """Cline's `--config` directory, which replaces `~/.cline`."""
        return f"{self.session.session_dir}/config"

    @property
    def data_dir(self) -> str:
        """Cline's `--data-dir`: provider settings, session database, logs."""
        return f"{self.session.session_dir}/data"

    @property
    def executable(self) -> str:
        return f"{self.runtime}/cline"

    async def prepare(self, *, timeout: float) -> None:
        """Make sure Bash exists and create the private session directories outside the task workdir."""
        workdir, directory = quote(self.session.workdir), quote(self.session.session_dir)
        result = await self.session.sandbox.exec(
            _BASH_BOOTSTRAP
            + f"test -d {workdir} || {{ echo 'Cline task workdir is missing' >&2; exit 1; }}\n"
            + f"umask 077 && mkdir -p -- {directory}/config {directory}/data/settings",
            cwd="/",
            timeout_s=timeout,
        )
        check_sandbox_setup(result, "Cline session preparation")

    async def install_runtime(self, *, version: str, timeout: float) -> None:
        """Install or reuse the pinned Cline binary, then stage the shared harness runner."""
        directory, sandbox = self.session.session_dir, self.session.sandbox
        digests = CLINE_RUNTIME_SHA512[version]
        await sandbox.upload(Path(__file__).with_name(INSTALLER), f"{directory}/{INSTALLER}")
        installed = await sandbox.exec(
            " ".join(
                quote(part)
                for part in (
                    "bash",
                    f"{directory}/{INSTALLER}",
                    self.runtime,
                    version,
                    digests["x64"],
                    digests["arm64"],
                )
            ),
            cwd="/",
            timeout_s=timeout,
        )
        check_sandbox_setup(installed, "Cline sandbox installation")
        await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")

    async def upload_json(self, path: str, payload: JsonValue) -> None:
        await upload_text(self.session.sandbox, path=path, text=json.dumps(payload))

    async def stage_activation(
        self,
        *,
        command: list[str],
        env: dict[str, str],
        providers: dict[str, JsonValue],
        global_settings: dict[str, JsonValue],
    ) -> SandboxCommand:
        """Write Cline's provider and global settings, then the runner input, without touching task files."""
        directory = self.session.session_dir
        # Written directly rather than through `cline auth`, which has no flag for request headers.
        await self.upload_json(f"{self.data_dir}/settings/providers.json", providers)
        await self.upload_json(f"{self.data_dir}/settings/global-settings.json", global_settings)
        params = {
            "cwd": self.session.workdir,
            "command": command,
            "env": env,
            # Do not inherit Cline state paths or provider credentials from the image.
            "unset_env": ["CLINE_*", "OPENAI_API_KEY", "OPENAI_BASE_URL"],
            # Cline 3.0.55 reads a prompt only from argv or a pipe, so stdin carries nothing.
            "stdin": "/dev/null",
            # File-backed stdout survives supervisor termination and cannot be held open by a detached tool.
            "stdout": f"{directory}/stdout.jsonl",
            "stderr": f"{directory}/stderr.log",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        self.started_at = perf_counter()
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def collect_artifacts(self) -> ClineArtifacts:
        """Copy the JSON event stream after confirmed termination, keeping partial output of a stopped run."""
        wall_time_s = perf_counter() - self.started_at if self.started_at is not None else None
        directory = self.session.session_dir
        try:
            stdout = await read_text(self.session.sandbox, path=f"{directory}/stdout.jsonl")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(f"Cline runner produced no readable output: {logs[-STDERR_TAIL_BYTES:]}") from error
        # Cline repeats SDK deprecation warnings on every model call; keep only the tail.
        result = await self.session.sandbox.exec(
            f"tail -c {STDERR_TAIL_BYTES} -- {quote(directory + '/stderr.log')}", cwd="/", timeout_s=30
        )
        stderr = (result.stdout or "") if result.return_code == 0 and not result.error_type else ""
        try:
            runtime = json.loads(await read_text(self.session.sandbox, path=f"{directory}/runtime.json"))
        except Exception:
            runtime = None
        self.runtime_info = parse_runtime_info(runtime)
        return ClineArtifacts(stdout=stdout, stderr=stderr, wall_time_s=wall_time_s)

    async def close(self, timeout: float) -> None:
        """Confirm termination and capture before cancelling the HTTP activation."""
        await self.session.close(timeout=timeout)
        # The activation parses captured artifacts after execute() returns; cancelling it then would discard them.
        if (
            self.session.artifacts is None
            and self.task is not None
            and not self.task.done()
            and not self.task.cancelling()
        ):
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
