# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox backends for the VCQA agent.

Three implementations of the same `Sandbox` protocol:

- `ApptainerSandbox` (production): boots an `apptainer instance` per rollout,
  bind-mounts the working tree at `/codebase:ro`, runs every tool call via
  `apptainer exec instance://... bash -lc <cmd>`. This is what the YAML
  config defaults to.

With `contain=True` (the default) both Apptainer backends start the container
with `--containall` and exec with `--cleanenv`: the host environment (API keys,
tokens) and the host's $HOME, /tmp and working directory are not visible to
the model's commands. Only `/codebase` (read-only) and the rollout scratch dir
are. With `network_none=True` the container also gets an empty network
namespace (`--net --network none`), so the model's commands cannot reach the
internet.
- `ApptainerDirectSandbox` (nested-container fallback): runs each tool call
  with direct `apptainer exec`, bind-mounting the working tree at
  `/codebase:ro` and a host scratch dir at `/tmp/scratch`. Use this when
  `apptainer exec instance://...` cannot enter the instance namespace.
- `LocalSandbox` (dev / macOS / no-apptainer hosts): runs commands directly
  with `cwd=<working_tree>`, no container at all. Cheaper to spin up and
  needs no daemon, but obviously gives the model access to the host
  filesystem outside `/codebase` if the path-containment helpers in
  `tools.py` ever miss a case. Use only against trusted artifact sources.

Both expose the same surface: `codebase_path`, `scratch_path`, `todos_path`,
`async start()`, `async exec(command, timeout_s) -> ExecResult`,
`async stop()`. Callers in `tools.py` and `app.py` only see the protocol,
not the concrete class.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Protocol


CODEBASE_PATH = "/codebase"
SCRATCH_PATH = "/tmp/scratch"
TODOS_PATH = f"{SCRATCH_PATH}/todos.md"


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


# Backwards-compat alias so existing tests / external callers keep working.
ApptainerExecResult = ExecResult


class Sandbox(Protocol):
    """Structural type used by `tools.py` and `app.py`."""

    codebase_path: str
    scratch_path: str
    todos_path: str

    async def start(self) -> None: ...

    async def exec(self, command: str, timeout_s: Optional[int] = None) -> ExecResult: ...

    async def stop(self) -> None: ...


########################################
# Apptainer backend
########################################


@dataclass
class ApptainerSandbox:
    """A running Apptainer instance scoped to a single rollout."""

    instance_name: str
    container_image: str
    working_tree: Path
    scratch_dir: Path
    exec_timeout_s: int = 30
    setup_done: bool = False
    started: bool = False
    cleaned_up: bool = False
    extra_setup_command: Optional[str] = None
    contain: bool = True
    network_none: bool = False
    _started_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    codebase_path: str = CODEBASE_PATH
    scratch_path: str = SCRATCH_PATH
    todos_path: str = TODOS_PATH

    async def start(self) -> None:
        async with self._started_lock:
            if self.started:
                return
            self.scratch_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "apptainer",
                "instance",
                "start",
                *(["--containall"] if self.contain else []),
                *(["--net", "--network", "none"] if self.network_none else []),
                "--writable-tmpfs",
                "--bind",
                f"{self.working_tree}:{self.codebase_path}:ro",
                self.container_image,
                self.instance_name,
            ]
            try:
                await _run_subprocess(cmd, timeout_s=300)
                self.started = True

                await self._exec_checked(
                    f"mkdir -p {shlex.quote(self.scratch_path)} && touch {shlex.quote(self.todos_path)}",
                    timeout_s=30,
                )

                if self.extra_setup_command and not self.setup_done:
                    await self._exec_checked(self.extra_setup_command, timeout_s=600)
                    self.setup_done = True
            except Exception:
                if self.started:
                    await self.stop()
                raise

    async def _exec_checked(self, command: str, timeout_s: int) -> ExecResult:
        result = await self.exec(command, timeout_s=timeout_s)
        if result.timed_out:
            raise RuntimeError(f"sandbox command timed out after {timeout_s}s: {command}")
        if result.exit_code != 0:
            raise RuntimeError(
                f"sandbox command failed (exit={result.exit_code}): {command}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result

    async def exec(self, command: str, timeout_s: Optional[int] = None) -> ExecResult:
        timeout_s = timeout_s if timeout_s is not None else self.exec_timeout_s
        cmd = [
            "apptainer",
            "exec",
            # An exec into an instance otherwise inherits the caller's environment.
            *(["--cleanenv"] if self.contain else []),
            "--pwd",
            self.codebase_path,
            f"instance://{self.instance_name}",
            "bash",
            "-lc",
            command,
        ]
        return await _exec_with_timeout(cmd, timeout_s=timeout_s, cwd=None)

    async def stop(self) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True
        if self.started:
            try:
                await _run_subprocess(
                    ["apptainer", "instance", "stop", self.instance_name],
                    timeout_s=60,
                )
            except Exception:
                pass
            finally:
                self.started = False
        try:
            shutil.rmtree(self.scratch_dir, ignore_errors=True)
        except Exception:
            pass


########################################
# Direct Apptainer exec backend
########################################


@dataclass
class ApptainerDirectSandbox:
    """Apptainer sandbox without persistent instances.

    Each command runs in a fresh `apptainer exec`, so filesystem mutations in
    the container layer do not persist across calls. The image must already
    contain the command-line tools exposed by `tools.py` (`bash`, `git`, `rg`,
    `fd`, `tree`, coreutils). A host scratch directory is bind-mounted so
    `write_todos` persists across calls.
    """

    container_image: str
    working_tree: Path
    scratch_dir: Path
    exec_timeout_s: int = 30
    started: bool = False
    cleaned_up: bool = False
    extra_setup_command: Optional[str] = None
    contain: bool = True
    network_none: bool = False
    _started_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    codebase_path: str = CODEBASE_PATH
    scratch_path: str = SCRATCH_PATH
    todos_path: str = TODOS_PATH

    def _exec_cmd(self, command: str) -> List[str]:
        return [
            "apptainer",
            "exec",
            *(["--containall"] if self.contain else []),
            *(["--net", "--network", "none"] if self.network_none else []),
            "--writable-tmpfs",
            "--bind",
            f"{self.working_tree}:{self.codebase_path}:ro",
            "--bind",
            f"{self.scratch_dir}:{self.scratch_path}:rw",
            "--pwd",
            self.codebase_path,
            self.container_image,
            "bash",
            "-lc",
            command,
        ]

    async def start(self) -> None:
        async with self._started_lock:
            if self.started:
                return
            if self.extra_setup_command:
                raise RuntimeError(
                    "sandbox_backend='apptainer_exec' requires a container image with tools preinstalled; "
                    "apptainer_setup_command is not persistent without an Apptainer instance"
                )
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
            await _run_subprocess(
                self._exec_cmd(f"mkdir -p {shlex.quote(self.scratch_path)} && touch {shlex.quote(self.todos_path)}"),
                timeout_s=60,
            )
            self.started = True

    async def exec(self, command: str, timeout_s: Optional[int] = None) -> ExecResult:
        timeout_s = timeout_s if timeout_s is not None else self.exec_timeout_s
        return await _exec_with_timeout(self._exec_cmd(command), timeout_s=timeout_s, cwd=None)

    async def stop(self) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True
        try:
            shutil.rmtree(self.scratch_dir, ignore_errors=True)
        except Exception:
            pass


########################################
# Local backend (no container)
########################################


@dataclass
class LocalSandbox:
    """No-container sandbox: runs commands with `cwd=working_tree` directly.

    Useful for development on hosts without apptainer (e.g. macOS) and for
    fast unit-test loops. Path-containment is enforced by `tools._resolve_*`.
    No bind-mount, no read-only protection; only point this at trusted
    artifact sources.
    """

    working_tree: Path
    scratch_dir: Path
    exec_timeout_s: int = 30
    started: bool = False
    cleaned_up: bool = False
    _started_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def codebase_path(self) -> str:
        return str(self.working_tree)

    @property
    def scratch_path(self) -> str:
        return str(self.scratch_dir)

    @property
    def todos_path(self) -> str:
        return str(self.scratch_dir / "todos.md")

    async def start(self) -> None:
        async with self._started_lock:
            if self.started:
                return
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
            (self.scratch_dir / "todos.md").touch(exist_ok=True)
            self.started = True

    async def exec(self, command: str, timeout_s: Optional[int] = None) -> ExecResult:
        timeout_s = timeout_s if timeout_s is not None else self.exec_timeout_s
        cmd = ["bash", "-c", command]
        return await _exec_with_timeout(
            cmd,
            timeout_s=timeout_s,
            cwd=str(self.working_tree) if self.working_tree.exists() else None,
        )

    async def stop(self) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True
        try:
            shutil.rmtree(self.scratch_dir, ignore_errors=True)
        except Exception:
            pass


########################################
# Shared subprocess helpers
########################################


async def _exec_with_timeout(cmd: List[str], *, timeout_s: int, cwd: Optional[str]) -> ExecResult:
    """Run a subprocess. Never raises on non-zero exit / timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        return ExecResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            timed_out=False,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        stdout_b, stderr_b = b"", b""
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=5)
        except asyncio.TimeoutError:
            pass
        return ExecResult(
            exit_code=-1,
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            timed_out=True,
        )


async def _run_subprocess(cmd: List[str], timeout_s: int) -> ExecResult:
    """Like `_exec_with_timeout` but raises on non-zero exit.

    Used for agent-internal operations (e.g. `apptainer instance start`)
    where a failure should abort the rollout, not be reflected back to the
    model as a tool-result error.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        raise RuntimeError(f"command timed out after {timeout_s}s: {' '.join(cmd)}")

    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed (exit={proc.returncode}): {' '.join(cmd)}\nstdout: {stdout}\nstderr: {stderr}"
        )
    return ExecResult(exit_code=0, stdout=stdout, stderr=stderr, timed_out=False)


def make_instance_name(prefix: str = "vcqa") -> str:
    """Apptainer instance names must be unique on the host."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# Re-exported in case callers want to override the apptainer search path.
def apptainer_on_path() -> bool:
    return shutil.which("apptainer") is not None or os.environ.get("VCQA_FAKE_APPTAINER") == "1"


########################################
# Sandbox-provider backend (OpenSandbox, ...)
########################################


class SandboxInfraError(RuntimeError):
    """The sandbox runtime failed (not the model's command): the rollout is aborted and masked."""


@dataclass
class ProviderSandbox:
    """A sandbox on a NeMo Gym sandbox provider (e.g. OpenSandbox), with the working tree materialized inside it.

    `start()` creates the sandbox from `spec`, uploads `uploads` (local path -> sandbox path), then runs
    `setup_commands` as root (tools, the non-root user, extracting or cloning the repo into `/codebase`). Tool commands
    then run as `run_as` in `/codebase`; a root-owned tree is read-only to that user, as the Apptainer `:ro` bind is.
    Provider failures raise `SandboxInfraError` instead of reaching the model.
    """

    provider: Any
    spec: Any
    uploads: List[tuple] = field(default_factory=list)
    setup_commands: List[str] = field(default_factory=list)
    run_as: Optional[str] = None
    exec_timeout_s: int = 30
    setup_timeout_s: int = 900
    upload_semaphore: Optional[asyncio.Semaphore] = None
    started: bool = False
    cleaned_up: bool = False
    _sandbox: Any = None

    codebase_path: str = CODEBASE_PATH
    scratch_path: str = SCRATCH_PATH
    todos_path: str = TODOS_PATH

    async def start(self) -> None:
        from nemo_gym.sandbox.api import AsyncSandbox

        async def setup(sandbox: AsyncSandbox) -> None:
            for local_path, remote_path in self.uploads:
                async with self.upload_semaphore if self.upload_semaphore is not None else nullcontext():
                    await sandbox.upload(local_path, remote_path)
            for command in self.setup_commands:
                result = await sandbox.exec(command, timeout_s=self.setup_timeout_s, user="root")
                if result.return_code != 0 or result.error_type is not None:
                    raise RuntimeError(
                        f"sandbox setup command failed (exit={result.return_code}, error={result.error_type}): "
                        f"{command[:300]}\nstdout: {(result.stdout or '')[-2000:]}\n"
                        f"stderr: {(result.stderr or '')[-2000:]}"
                    )

        sandbox = AsyncSandbox(self.provider, self.spec, owns_provider=False)
        self._sandbox = sandbox
        # Stops the sandbox if an upload or a setup command fails; a stop that fails there stays retryable by stop().
        await sandbox.start_with_setup(None, setup)
        self.started = True

    async def exec(self, command: str, timeout_s: Optional[int] = None) -> ExecResult:
        if self._sandbox is None or self.cleaned_up:
            raise SandboxInfraError("sandbox is not running")
        timeout_s = timeout_s if timeout_s is not None else self.exec_timeout_s
        try:
            # Tool commands are bash, as in the Apptainer backends; the provider runs another user's command with sh.
            result = await self._sandbox.exec(
                f"bash -lc {shlex.quote(command)}", cwd=self.codebase_path, timeout_s=timeout_s, user=self.run_as
            )
        except Exception as e:
            raise SandboxInfraError(f"sandbox exec failed: {type(e).__name__}: {e}") from e
        if result.error_type is not None and result.error_type != "timeout":
            # The runtime reported no exit code for the command (e.g. its stream ended): not the model's command.
            raise SandboxInfraError(f"sandbox exec failed ({result.error_type}): {(result.stderr or '')[-500:]}")
        return ExecResult(
            exit_code=result.return_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            timed_out=result.error_type == "timeout",
        )

    async def stop(self) -> None:
        """Stop the sandbox. A failed or cancelled stop raises and can be retried; a confirmed one is not repeated."""
        if self.cleaned_up:
            return
        if self._sandbox is not None:
            await self._sandbox.stop()
        self.cleaned_up = True
        self.started = False


def provider_setup_commands(
    *,
    dataset_kind: str,
    artifact_path: str,
    head_sha: Optional[str],
    tools_command: Optional[str],
    run_as: Optional[str],
    remove_artifact: bool,
    hidden_paths: Optional[List[str]] = None,
) -> List[str]:
    """Root commands that prepare a fresh provider sandbox for one VCQA task.

    The tree is materialized like `materialize.py` does on the host: a fileset tarball is extracted as-is (entering a
    single wrapping directory, if any), a githistory bundle is cloned and checked out at `head_sha`. With `run_as`, the
    tree is then made root-owned and read-only to that user, and `hidden_paths` become root-only last: an existing one
    that is not root's, or cannot be restricted, fails the setup.
    """
    q = shlex.quote
    commands: List[str] = []
    if tools_command:
        commands.append(tools_command)
    # A tree owned by root is read by the non-root tool user; git refuses that without safe.directory.
    commands.append("git config --system --add safe.directory '*'")
    scratch = f"mkdir -p {q(SCRATCH_PATH)} && touch {q(TODOS_PATH)}"
    if run_as:
        scratch += (
            f" && (id -u {q(run_as)} >/dev/null 2>&1 || useradd -m {q(run_as)})"
            f" && chown -R {q(run_as)} {q(SCRATCH_PATH)}"
        )
    commands.append(scratch)
    if dataset_kind == "fileset":
        commands.append(
            "set -e; rm -rf /codebase /tmp/vcqa-x; mkdir -p /tmp/vcqa-x; "
            f"tar -xzf {q(artifact_path)} -C /tmp/vcqa-x; "
            "n=$(ls -A /tmp/vcqa-x | wc -l); one=$(ls -A /tmp/vcqa-x | head -1); "
            'if [ "$n" = 1 ] && [ -d "/tmp/vcqa-x/$one" ] && [ ! -L "/tmp/vcqa-x/$one" ]; '
            'then mv "/tmp/vcqa-x/$one" /codebase && rmdir /tmp/vcqa-x; else mv /tmp/vcqa-x /codebase; fi'
        )
    elif dataset_kind == "githistory":
        if not head_sha:
            raise ValueError("githistory rows require verifier_metadata.head_sha")
        commands.append(
            f"set -e; rm -rf /codebase; git clone --quiet --no-checkout {q(artifact_path)} /codebase; "
            f"git -C /codebase checkout --quiet {q(head_sha)}"
        )
    else:
        raise ValueError(f"unknown dataset_kind: {dataset_kind!r}")
    if run_as:
        # Extraction keeps the archive's owners and modes (a member can be 0666 or owned by the tool user's UID):
        # make the tree root-owned and read-only to everyone else, without following its symlinks.
        commands.append(f"chown -hR 0:0 {CODEBASE_PATH} && chmod -R a+rX,go-w,a-s {CODEBASE_PATH}")
    if remove_artifact:
        commands.append(f"rm -f {q(artifact_path)}")
    if run_as and hidden_paths:
        # Mode 700 hides a path only from users other than its owner, so a hidden path must be root's: anything else
        # fails the setup rather than leave it readable.
        paths = " ".join(q(p) for p in hidden_paths)
        commands.append(
            f'for p in {paths}; do if [ -e "$p" ]; then '
            '[ "$(stat -Lc %u -- "$p")" = 0 ] && chmod 700 -- "$p" || exit 1; fi; done'
        )
    return commands
