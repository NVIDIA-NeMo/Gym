# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Apptainer-backed code execution provider for Stirrup.

Runs each tool call inside a persistent ``apptainer exec bash``
subprocess backed by a ``.sif`` image (e.g. a SWE-bench container with
the repo pre-checked out at ``/testbed``).  Filesystem changes persist
across calls because the same ``--writable-tmpfs`` overlay stays alive
for the lifetime of the subprocess.

Uses the same ``apptainer exec`` invocation that already works in the
``swe_agents`` wrapper (``run_openhands.py``), avoiding the more
restrictive ``apptainer instance start`` which requires extra kernel
capabilities not available in all Slurm container environments.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from stirrup.core.models import ImageContentBlock, Tool, ToolUseCountMetadata
from stirrup.tools.code_backends.base import (
    SHELL_TIMEOUT,
    CodeExecToolProvider,
    CodeExecutionParams,
    CommandResult,
    SavedFile,
    SaveOutputFilesResult,
    UploadedFile,
    UploadFilesResult,
)

logger = logging.getLogger(__name__)

IO_MOUNT_DEST = "/workspace_io"

_MARKER_PREFIX = "__STIRRUP_EXEC_DONE_"


class ApptainerCodeExecToolProvider(CodeExecToolProvider):
    """Execute Stirrup tool calls inside a long-running Apptainer shell.

    The provider lifecycle:

    1. ``__aenter__`` launches ``apptainer exec --writable-tmpfs ... bash``
       as a subprocess with piped stdin/stdout.  A host temp directory is
       bind-mounted at ``/workspace_io`` for file I/O.
    2. Each ``run_command`` / file operation writes a shell command to
       stdin and reads stdout until a unique end-marker appears.
       Stderr for each command is captured via a per-command file on the
       bind mount.
    3. ``__aexit__`` extracts ``git diff``, terminates the subprocess,
       and cleans up.
    """

    def __init__(
        self,
        sif_path: str,
        *,
        working_dir: str = "/testbed",
        allowed_commands: list[str] | None = None,
        memory_limit_mb: int | None = None,
        extra_mounts: list[str] | None = None,
        capture_git_diff: bool = True,
        env_passthrough: list[str] | None = None,
    ) -> None:
        super().__init__(allowed_commands=allowed_commands)
        self._sif_path = sif_path
        self._working_dir = working_dir.rstrip("/")
        self._memory_limit_mb = memory_limit_mb
        self._extra_mounts = extra_mounts or []
        self._capture_git_diff = capture_git_diff
        self._env_passthrough = env_passthrough or []

        self._process: asyncio.subprocess.Process | None = None
        self._temp_dir: Path | None = None
        self._patch: str | None = None
        self._stderr_log: Path | None = None
        self._has_timeout_cmd: bool = False

    def _serializable_kwargs(self) -> dict:
        """Return constructor kwargs that can be passed through Ray."""
        return {
            "sif_path": self._sif_path,
            "working_dir": self._working_dir,
            "allowed_commands": None,
            "memory_limit_mb": self._memory_limit_mb,
            "extra_mounts": self._extra_mounts,
            "capture_git_diff": self._capture_git_diff,
            "env_passthrough": self._env_passthrough,
        }

    @property
    def patch(self) -> str | None:
        """The git diff captured from the working dir when the session ended."""
        return self._patch

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        self._temp_dir = Path(tempfile.mkdtemp(prefix="apptainer_exec_env_"))

        mount_args = [
            f"--mount type=bind,src={self._temp_dir},dst={IO_MOUNT_DEST}",
        ]
        for extra in self._extra_mounts:
            mount_args.append(extra)
        mount_str = " ".join(mount_args)

        env_args = " ".join(
            f"--env {var}={shlex.quote(os.environ.get(var, ''))}"
            for var in self._env_passthrough
            if os.environ.get(var)
        )
        env_section = f"--env HOME=/root {env_args}" if env_args else "--env HOME=/root"

        exec_cmd = (
            f"apptainer exec "
            f"--writable-tmpfs --cleanenv --pid "
            f"--no-mount home,tmp,bind-paths "
            f"{env_section} "
            f"{mount_str} "
            f"{shlex.quote(self._sif_path)} "
            f"bash"
        )

        if self._memory_limit_mb and self._memory_limit_mb > 0:
            memory_kb = self._memory_limit_mb * 1024
            exec_cmd = f"ulimit -v {memory_kb} && {exec_cmd}"

        self._stderr_log = self._temp_dir / "apptainer_stderr.log"
        stderr_fh = open(self._stderr_log, "w")

        self._process = await asyncio.create_subprocess_shell(
            exec_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr_fh,
        )

        # Verify the shell is alive by running a trivial command
        rc, _, _ = await self._exec("echo ready", timeout=30)
        if rc != 0:
            appt_stderr = ""
            if self._stderr_log and self._stderr_log.exists():
                appt_stderr = self._stderr_log.read_text(errors="replace")[:2000]
            raise RuntimeError(
                f"Failed to start Apptainer shell for {self._sif_path}: {appt_stderr}"
            )

        # Mark all directories as safe for git (needed because --cleanenv
        # strips the environment, and the container user may differ from the
        # owner of /testbed).
        await self._exec("git config --global --add safe.directory '*'", timeout=10)

        # Check if GNU `timeout` is available for per-command time limits.
        # When present, each command is wrapped so that a hung command is
        # killed by the OS instead of blocking all subsequent commands on the
        # shared stdin/stdout stream.
        rc_to, _, _ = await self._exec("command -v timeout >/dev/null 2>&1", timeout=5)
        self._has_timeout_cmd = rc_to == 0
        if not self._has_timeout_cmd:
            logger.warning(
                "timeout command not found in container — "
                "in-shell time limits disabled; hung commands will block the stream"
            )

        logger.info("Started Apptainer shell (pid=%s) from %s", self._process.pid, self._sif_path)
        return self.get_code_exec_tool()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._process:
            if self._process.returncode is None:
                if self._capture_git_diff:
                    print("[apptainer] __aexit__: shell alive, extracting git diff", flush=True)

                    try:
                        rc_dbg, out_dbg, err_dbg = await self._exec(
                            "echo HOME=$HOME PATH=$PATH && which git && pwd && git status --short 2>&1 | head -20",
                            timeout=15,
                        )
                        print(
                            f"[apptainer] debug: rc={rc_dbg} "
                            f"stdout={out_dbg.decode('utf-8', errors='replace')[:500]} "
                            f"stderr={err_dbg.decode('utf-8', errors='replace')[:500]}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"[apptainer] debug failed: {exc}", flush=True)

                    try:
                        rc, diff_out, diff_err = await self._exec("git diff", timeout=60)
                        if rc == 0 and diff_out:
                            self._patch = diff_out.decode("utf-8", errors="replace").strip() or None
                        print(
                            f"[apptainer] git diff rc={rc}, "
                            f"output_len={len(diff_out)}, "
                            f"patch_len={len(self._patch) if self._patch else 0}, "
                            f"stderr={diff_err.decode('utf-8', errors='replace')[:300]}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"[apptainer] git diff failed: {exc}", flush=True)

                    if not self._patch:
                        try:
                            rc, diff_out, _ = await self._exec("git diff --cached", timeout=60)
                            if rc == 0 and diff_out:
                                self._patch = diff_out.decode("utf-8", errors="replace").strip() or None
                                print(
                                    f"[apptainer] git diff --cached: patch_len={len(self._patch) if self._patch else 0}",
                                    flush=True,
                                )
                        except Exception:
                            pass

                # Terminate the shell (always, regardless of capture_git_diff)
                try:
                    self._process.stdin.close()
                    await asyncio.wait_for(self._process.wait(), timeout=10)
                except (asyncio.TimeoutError, ProcessLookupError):
                    self._process.kill()
                    await self._process.wait()
                logger.info("Stopped Apptainer shell (pid=%s)", self._process.pid)
            else:
                print(
                    f"[apptainer] __aexit__: shell DEAD (rc={self._process.returncode})",
                    flush=True,
                )
                if self._stderr_log and self._stderr_log.exists():
                    stderr_tail = self._stderr_log.read_text(errors="replace")[-1000:]
                    print(f"[apptainer] stderr tail: {stderr_tail}", flush=True)
            self._process = None

        print(
            f"[apptainer] __aexit__: patch captured={self._patch is not None} "
            f"(len={len(self._patch) if self._patch else 0})",
            flush=True,
        )

        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _exec(
        self,
        cmd: str,
        *,
        timeout: int = SHELL_TIMEOUT,
    ) -> tuple[int, bytes, bytes]:
        """Run a command inside the persistent Apptainer shell.

        Sends the command to bash's stdin, using a unique marker to
        delimit the end of output and capture the exit code.  Stderr
        is redirected to a per-command file on the bind mount.
        """
        if self._process is None or self._process.returncode is not None:
            extra = ""
            if self._stderr_log and self._stderr_log.exists():
                extra = self._stderr_log.read_text(errors="replace")[-2000:]
            rc = self._process.returncode if self._process else "N/A"
            raise RuntimeError(
                f"Apptainer shell exited (rc={rc}). "
                f"Stderr tail: {extra or '(empty)'}"
            )

        marker = f"{_MARKER_PREFIX}{uuid.uuid4().hex[:12]}"
        stderr_name = f".stderr_{marker}"
        stderr_path = self._temp_dir / stderr_name if self._temp_dir else None

        # Wrap the command with GNU `timeout` so that a hung command is
        # killed by the OS.  Without this, a stuck command blocks the
        # shared bash stdin/stdout stream and ALL subsequent commands
        # (including the git-diff in __aexit__) time out as well.
        if self._has_timeout_cmd and timeout > 0:
            wrapped_cmd = (
                f"timeout -k 10 {timeout} bash -c {shlex.quote(cmd)}"
            )
            asyncio_timeout = timeout + 30
        else:
            wrapped_cmd = cmd
            asyncio_timeout = timeout

        script = (
            f"cd {self._working_dir} && "
            f"( {wrapped_cmd} ) 2>{IO_MOUNT_DEST}/{stderr_name}; "
            f"_rc=$?; echo ''; echo '{marker}:'$_rc\n"
        )

        self._process.stdin.write(script.encode())
        await self._process.stdin.drain()

        # Read stdout line by line until we see the marker
        stdout_lines: list[bytes] = []
        rc = 1  # default in case of EOF / timeout
        try:
            while True:
                line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=asyncio_timeout,
                )
                if not line:
                    # EOF — process died mid-command
                    extra = ""
                    if self._stderr_log and self._stderr_log.exists():
                        extra = self._stderr_log.read_text(errors="replace")[-2000:]
                    logger.error(
                        "Apptainer shell died during command: %s | stderr: %s",
                        cmd[:200], extra or "(empty)",
                    )
                    break

                line_str = line.decode("utf-8", errors="replace")
                if marker in line_str:
                    # Parse exit code from "MARKER:RC"
                    try:
                        rc = int(line_str.split(":")[-1].strip())
                    except (ValueError, IndexError):
                        rc = 1
                    break
                stdout_lines.append(line)
            else:
                rc = 1
        except asyncio.TimeoutError:
            rc = 1
            # The in-shell `timeout` should have killed the command.
            # If we still hit the asyncio safety-net, try to resync.
            await self._drain_after_timeout()
            stderr_content = f"Command timed out after {timeout} seconds".encode()
            return rc, b"".join(stdout_lines), stderr_content

        stdout_bytes = b"".join(stdout_lines)
        # Trim trailing empty line we injected before the marker
        if stdout_bytes.endswith(b"\n\n"):
            stdout_bytes = stdout_bytes[:-1]

        # Detect in-shell timeout (rc 124 = SIGTERM, 137 = SIGKILL)
        if rc in (124, 137) and self._has_timeout_cmd:
            file_stderr = b""
            if stderr_path and stderr_path.exists():
                try:
                    file_stderr = stderr_path.read_bytes()
                    stderr_path.unlink(missing_ok=True)
                except Exception:
                    pass
            timeout_msg = f"Command timed out after {timeout} seconds".encode()
            combined = (
                file_stderr + b"\n" + timeout_msg if file_stderr else timeout_msg
            )
            return 1, stdout_bytes, combined

        # Read per-command stderr from the bind mount file
        stderr_bytes = b""
        if stderr_path and stderr_path.exists():
            try:
                stderr_bytes = stderr_path.read_bytes()
                stderr_path.unlink(missing_ok=True)
            except Exception:
                pass

        return rc, stdout_bytes, stderr_bytes

    async def _drain_after_timeout(self) -> None:
        """Best-effort resync of the stdout stream after an asyncio timeout.

        The in-shell ``timeout`` wrapper should prevent this from ever
        being needed.  If the asyncio safety-net fires anyway, we send
        SIGINT to the process tree and try to read until a drain marker
        appears so that subsequent commands can still execute.
        """
        if self._process is None or self._process.returncode is not None:
            return

        import signal as signal_mod

        drain_marker = f"{_MARKER_PREFIX}DRAIN_{uuid.uuid4().hex[:12]}"
        try:
            try:
                self._process.send_signal(signal_mod.SIGINT)
            except (ProcessLookupError, OSError):
                return
            await asyncio.sleep(1)

            drain_script = f"echo '{drain_marker}:0'\n"
            self._process.stdin.write(drain_script.encode())
            await self._process.stdin.drain()

            for _ in range(1000):
                try:
                    line = await asyncio.wait_for(
                        self._process.stdout.readline(), timeout=5
                    )
                except asyncio.TimeoutError:
                    break
                if not line:
                    break
                if drain_marker in line.decode("utf-8", errors="replace"):
                    logger.info("Stream resynchronized after timeout drain")
                    return
            logger.warning(
                "Stream drain did not find marker — subsequent commands may fail"
            )
        except Exception as exc:
            logger.warning("Stream drain failed after timeout: %s", exc)

    def _resolve_path(self, path: str) -> str:
        """Resolve a relative or absolute path to a container-absolute path."""
        if path.startswith("/"):
            return path
        return f"{self._working_dir}/{path}"

    # ------------------------------------------------------------------
    # CodeExecToolProvider interface
    # ------------------------------------------------------------------

    async def run_command(
        self, cmd: str, *, timeout: int = SHELL_TIMEOUT
    ) -> CommandResult:
        if not self._check_allowed(cmd):
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command not allowed: '{cmd}' does not match any allowed patterns",
                error_kind="command_not_allowed",
                advice="Only commands matching the allowlist patterns are permitted.",
            )

        try:
            rc, stdout, stderr = await self._exec(cmd, timeout=timeout)
            return CommandResult(
                exit_code=rc,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            logger.warning("Command timed out after %ds: %s", timeout, cmd[:100])
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                error_kind="timeout",
            )
        except Exception as exc:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                error_kind="execution_error",
            )

    async def read_file_bytes(self, path: str) -> bytes:
        container_path = self._resolve_path(path)
        rc, stdout, stderr = await self._exec(
            f"cat {shlex.quote(container_path)}", timeout=30
        )
        if rc != 0:
            err = stderr.decode("utf-8", errors="replace")
            raise FileNotFoundError(f"Cannot read {path}: {err}")
        return stdout

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        if self._temp_dir is None:
            raise RuntimeError("Apptainer shell not started.")

        tmp_name = f".tmp_write_{uuid.uuid4().hex[:8]}"
        host_tmp = self._temp_dir / tmp_name
        host_tmp.write_bytes(content)

        container_path = self._resolve_path(path)
        try:
            rc, _, stderr = await self._exec(
                f"mkdir -p $(dirname {shlex.quote(container_path)}) && "
                f"cp {IO_MOUNT_DEST}/{tmp_name} {shlex.quote(container_path)}",
                timeout=30,
            )
            if rc != 0:
                err = stderr.decode("utf-8", errors="replace")
                raise OSError(f"Failed to write {path}: {err}")
        finally:
            host_tmp.unlink(missing_ok=True)

    async def file_exists(self, path: str) -> bool:
        try:
            container_path = self._resolve_path(path)
            rc, _, _ = await self._exec(
                f"test -f {shlex.quote(container_path)}", timeout=10
            )
            return rc == 0
        except RuntimeError:
            return False

    async def is_directory(self, path: str) -> bool:
        try:
            container_path = self._resolve_path(path)
            rc, _, _ = await self._exec(
                f"test -d {shlex.quote(container_path)}", timeout=10
            )
            return rc == 0
        except RuntimeError:
            return False

    async def list_files(self, path: str) -> list[str]:
        container_path = self._resolve_path(path)
        rc, stdout, _ = await self._exec(
            f"find {shlex.quote(container_path)} -type f 2>/dev/null",
            timeout=30,
        )
        if rc != 0:
            return []

        base = container_path.rstrip("/")
        files = []
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(base + "/"):
                files.append(line[len(base) + 1:])
            else:
                files.append(line)
        return files

    async def view_image(self, path: str) -> ImageContentBlock:
        data = await self.read_file_bytes(path)
        return ImageContentBlock(data=data)

    # ------------------------------------------------------------------
    # File transfer helpers
    # ------------------------------------------------------------------

    async def save_output_files(
        self,
        paths: list[str],
        output_dir: Path | str,
        dest_env: CodeExecToolProvider | None = None,
    ) -> SaveOutputFilesResult:
        if self._temp_dir is None:
            raise RuntimeError("Apptainer shell not started.")

        if dest_env is not None:
            return await super().save_output_files(paths, output_dir, dest_env)

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        result = SaveOutputFilesResult()

        for src_path in paths:
            try:
                container_path = self._resolve_path(src_path)
                staging_name = f"save_{uuid.uuid4().hex[:8]}_{Path(src_path).name}"

                rc, _, stderr = await self._exec(
                    f"cp {shlex.quote(container_path)} {shlex.quote(f'{IO_MOUNT_DEST}/{staging_name}')}",
                    timeout=30,
                )
                if rc != 0:
                    result.failed[src_path] = stderr.decode("utf-8", errors="replace")
                    continue

                host_staged = self._temp_dir / staging_name
                if not host_staged.exists():
                    result.failed[src_path] = "File not found after copy"
                    continue

                dest_path = output_dir_path / Path(src_path).name
                shutil.move(str(host_staged), str(dest_path))
                result.saved.append(
                    SavedFile(
                        source_path=src_path,
                        output_path=dest_path,
                        size=dest_path.stat().st_size,
                    )
                )
            except Exception as exc:
                result.failed[src_path] = str(exc)
                logger.exception("Failed to save file: %s", src_path)

        return result

    async def upload_files(
        self,
        *paths: Path | str,
        source_env: CodeExecToolProvider | None = None,
        dest_dir: str | None = None,
    ) -> UploadFilesResult:
        if self._temp_dir is None:
            raise RuntimeError("Apptainer shell not started.")

        if source_env is not None:
            return await super().upload_files(*paths, source_env=source_env, dest_dir=dest_dir)

        container_dest = f"{self._working_dir}/{dest_dir}" if dest_dir else self._working_dir
        result = UploadFilesResult()

        for source in paths:
            source = Path(source).resolve()
            if not source.exists():
                result.failed[str(source)] = "File or directory does not exist"
                continue

            try:
                if source.is_file():
                    staging_name = f"upload_{uuid.uuid4().hex[:8]}_{source.name}"
                    shutil.copy2(source, self._temp_dir / staging_name)

                    rc, _, stderr = await self._exec(
                        f"mkdir -p {shlex.quote(container_dest)} && "
                        f"cp {IO_MOUNT_DEST}/{staging_name} {shlex.quote(container_dest)}/{shlex.quote(source.name)}",
                        timeout=30,
                    )
                    (self._temp_dir / staging_name).unlink(missing_ok=True)

                    if rc != 0:
                        result.failed[str(source)] = stderr.decode("utf-8", errors="replace")
                        continue

                    result.uploaded.append(
                        UploadedFile(
                            source_path=source,
                            dest_path=f"{container_dest}/{source.name}",
                            size=source.stat().st_size,
                        )
                    )

                elif source.is_dir():
                    staging_name = f"upload_dir_{uuid.uuid4().hex[:8]}"
                    staging_dir = self._temp_dir / staging_name
                    shutil.copytree(source, staging_dir)

                    rc, _, stderr = await self._exec(
                        f"mkdir -p {shlex.quote(container_dest)} && "
                        f"cp -r {IO_MOUNT_DEST}/{staging_name}/. {shlex.quote(container_dest)}/",
                        timeout=60,
                    )
                    shutil.rmtree(staging_dir, ignore_errors=True)

                    if rc != 0:
                        result.failed[str(source)] = stderr.decode("utf-8", errors="replace")
                        continue

                    for file_path in source.rglob("*"):
                        if file_path.is_file():
                            rel = file_path.relative_to(source)
                            result.uploaded.append(
                                UploadedFile(
                                    source_path=file_path,
                                    dest_path=f"{container_dest}/{rel}",
                                    size=file_path.stat().st_size,
                                )
                            )
            except Exception as exc:
                result.failed[str(source)] = str(exc)
                logger.exception("Failed to upload: %s", source)

        return result
