# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cline runtime, input staging and artifacts over the shared task-sandbox lifecycle."""

import asyncio
import hashlib
import json
import logging
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    AgentSeedRejectedError,
    AgentSessionState,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.client import raise_for_status, request
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NgFailureEvent
from nemo_gym.sandbox import SandboxExecResult, read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)
INSTALLER = "install_cline_runtime.sh"
STDERR_TAIL_BYTES = 16000
_T = TypeVar("_T")

# SHA-512 (hex) of the npm tarballs @cline/cli-linux-{x64,arm64}, which carry Cline's self-contained binary.
# They are npm's own `dist.integrity` for each release; add a release here before pinning it in cline_version.
CLINE_RUNTIME_SHA512: dict[str, dict[str, str]] = {
    "3.0.55": {
        "x64": "bd5aaa113eb4b42e4a7c87dd1f02d3c9833846e9a62384b8c59bfeb1c68ec177"  # pragma: allowlist secret
        "f869aec3f4c4c0ac8f7f0745bc57eea586176050b45f3720cca66fe5fec106ba",  # pragma: allowlist secret
        "arm64": "f89fcd965bdaeff0ac49f93ab9dad263cba419ac5bbfd5af11b08f19662b99f3"  # pragma: allowlist secret
        "24d4067f4e1b5cdefe6b904944adbcc3c419d0436e2becad253d1a91264e4d5a",  # pragma: allowlist secret
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

_OFFLINE_PREREQUISITES = """set -eu
for tool in bash uname getconf mkdir dirname mktemp rm mv chmod touch cp sha512sum; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "Cline model_only requires $tool in the sandbox image; preinstall Bash and coreutils." >&2
        exit 1
    }
done
"""


async def _complete_file_io(operation: Callable[[], _T]) -> _T:
    """Join an active file operation before cancellation can close its files or remove its temporary directory."""
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        # Observe a worker failure without replacing the caller's cancellation.
        try:
            task.result()
        except Exception:
            pass
        raise


def _extract_cline_binary(archive_path: Path, destination: Path) -> str:
    """Extract only the verified archive's binary and hash the bytes that will be uploaded."""
    # Read only the regular binary member; package paths, links and dashboard assets are never extracted.
    binary_digest = hashlib.sha512()
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember("package/bin/cline")
        if not member.isfile():
            raise RuntimeError("Cline archive package/bin/cline must be a regular file")
        with archive.extractfile(member) as source, destination.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                binary_digest.update(chunk)
                output.write(chunk)
    return binary_digest.hexdigest()


async def download_cline_binary(*, version: str, architecture: str, destination: Path) -> str:
    """Fetch and verify the pinned npm archive on the Agent Server; never execute its binary there."""
    expected = CLINE_RUNTIME_SHA512[version][architecture]
    archive_path = destination.with_suffix(".tgz")
    response = await request(
        "GET",
        f"https://registry.npmjs.org/@cline/cli-linux-{architecture}/-/cli-linux-{architecture}-{version}.tgz",
        _max_num_tries=3,
    )
    digest = hashlib.sha512()
    try:
        await raise_for_status(response)
        with archive_path.open("wb") as output:

            def write_chunk(chunk: bytes) -> None:
                digest.update(chunk)
                output.write(chunk)

            async for chunk in response.content.iter_chunked(1024 * 1024):
                await _complete_file_io(lambda: write_chunk(chunk))
    finally:
        response.release()
    if digest.hexdigest() != expected:
        raise RuntimeError(f"Cline {version} {architecture} archive failed SHA-512 verification")
    return await _complete_file_io(lambda: _extract_cline_binary(archive_path, destination))


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
    provider_wait_failure: NgFailureEvent | None = None

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

    async def prepare(
        self, *, timeout: float, allow_package_install: bool = True, create_workdir: bool = False
    ) -> None:
        """Check prerequisites before creating session directories and, when owned, the task workdir."""
        workdir, directory = quote(self.session.workdir), quote(self.session.session_dir)
        result = await self.session.sandbox.exec(
            (_BASH_BOOTSTRAP if allow_package_install else _OFFLINE_PREREQUISITES)
            + (f"mkdir -p -- {workdir}\n" if create_workdir else "")
            + f"test -d {workdir} || {{ echo 'Cline task workdir is missing' >&2; exit 1; }}\n"
            + f"umask 077 && mkdir -p -- {directory}/config {directory}/data/settings",
            cwd="/",
            timeout_s=timeout,
        )
        if not allow_package_install and result.return_code != 0 and not result.error_type:
            raise AgentSeedRejectedError(
                f"Cline restricted sandbox preparation failed: {result.stderr or result.stdout}"
            )
        check_sandbox_setup(result, "Cline session preparation")

    async def install_runtime(self, *, version: str, timeout: float, download_on_host: bool = False) -> None:
        """Install or reuse the pinned Cline binary, then stage the shared harness runner."""
        directory, sandbox = self.session.session_dir, self.session.sandbox
        digests = CLINE_RUNTIME_SHA512[version]
        await sandbox.upload(Path(__file__).with_name(INSTALLER), f"{directory}/{INSTALLER}")
        extra = []
        if download_on_host:
            probe = await sandbox.exec(
                "set -eu\n"
                '[ "$(uname -s)" = Linux ] && getconf GNU_LIBC_VERSION >/dev/null 2>&1 || '
                "{ echo 'Cline requires glibc Linux' >&2; exit 1; }\n"
                f"if [ -f {quote(self.runtime + '/ready')} ] && "
                f'[ "$(CLINE_NO_AUTO_UPDATE=1 {quote(self.executable)} --version 2>/dev/null)" '
                f"= {quote(version)} ]; "
                "then echo ready; else uname -m; fi",
                cwd="/",
                timeout_s=timeout,
            )
            if probe.return_code != 0 and not probe.error_type:
                raise AgentSeedRejectedError(
                    f"Cline restricted runtime is unsupported: {probe.stderr or probe.stdout}"
                )
            check_sandbox_setup(probe, "Cline restricted runtime probe")
            platform = (probe.stdout or "").strip()
            if platform == "ready":
                await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
                return
            architecture = {"x86_64": "x64", "aarch64": "arm64"}.get(platform)
            if architecture is None:
                raise AgentSeedRejectedError(f"Cline supports Linux x86_64/aarch64 sandboxes only, got {platform!r}")
            with TemporaryDirectory(prefix="nemo-gym-cline-download-") as temporary:
                binary = Path(temporary) / "cline"
                digest = await download_cline_binary(version=version, architecture=architecture, destination=binary)
                uploaded = f"{directory}/cline.upload"
                await sandbox.upload(binary, uploaded)
            extra = [uploaded, digest]
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
                    *extra,
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
        if self.session.expired_before_launch:
            # A rollout stop arrived before launch, so Cline never ran and wrote no output.
            return ClineArtifacts(stdout="", stderr="", wall_time_s=wall_time_s)
        directory = self.session.session_dir
        try:
            stdout = await read_text(self.session.sandbox, path=f"{directory}/stdout.jsonl")
        except Exception as error:
            if self.session.expired and await self._stdout_absent():
                # A rollout stop can end the runner after launch but before it opens Cline's output.
                LOG.warning("Cline wrote no output before a rollout stop ended it; keeping an empty transcript")
                return ClineArtifacts(stdout="", stderr="", wall_time_s=wall_time_s)
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

    async def _stdout_absent(self) -> bool:
        """Whether the session directory exists without Cline's output; an unclear answer counts as present."""
        directory = quote(self.session.session_dir)
        stdout = quote(f"{self.session.session_dir}/stdout.jsonl")
        try:
            probe = await self.session.sandbox.exec(
                f"if [ -e {stdout} ]; then echo present; elif [ -d {directory} ]; then echo absent; fi",
                cwd="/",
                timeout_s=10,
            )
        except Exception:
            LOG.warning("Could not check for Cline output after a rollout stop", exc_info=True)
            return False
        return probe.return_code == 0 and not probe.error_type and (probe.stdout or "").strip() == "absent"

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
