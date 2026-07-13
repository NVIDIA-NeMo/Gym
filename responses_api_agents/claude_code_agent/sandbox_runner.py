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

"""Run Claude Code in a Gym sandbox and capture its workspace patch."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import sys
import tarfile
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any

from pydantic import BaseModel

from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxSpec


_GIT_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
LOG = logging.getLogger(__name__)


class SandboxRunResult(BaseModel):
    """Artifacts produced by one sandboxed Claude Code invocation."""

    stdout: str
    stderr: str
    return_code: int
    error_type: str | None = None
    workspace_patch: str
    base_revision: str
    elapsed_seconds: float


class ClaudeCodeSandboxRunner:
    """Execute a Claude Code command in an ephemeral sandbox.

    The sandbox image must contain a Git checkout at ``workspace`` and a
    ``claude`` executable. Runtime settings and skills are uploaded as a tar
    archive to a fresh ``CLAUDE_CONFIG_DIR``. The returned patch is computed
    against the revision that existed before the agent ran, so agent-created
    commits are included as well as staged, unstaged, and untracked files.
    """

    def __init__(
        self,
        *,
        provider: Mapping[str, Any],
        spec: SandboxSpec,
        workspace: str,
        timeout_s: int | float,
        user: str | int | None = "root",
        max_patch_bytes: int = 10 * 1024 * 1024,
        max_output_bytes: int = 50 * 1024 * 1024,
        cleanup_timeout_s: int | float = 30,
        require_clean_workspace: bool = True,
        forbidden_workspace_paths: tuple[str, ...] = (),
        sandbox_factory: Callable[..., AsyncSandbox] = AsyncSandbox,
    ) -> None:
        self._provider = provider
        self._spec = spec
        self._workspace = workspace
        self._timeout_s = timeout_s
        self._user = user
        self._max_patch_bytes = max_patch_bytes
        self._max_output_bytes = max_output_bytes
        self._cleanup_timeout_s = cleanup_timeout_s
        self._require_clean_workspace = require_clean_workspace
        self._forbidden_workspace_paths = tuple(
            self._validate_relative_path(path) for path in forbidden_workspace_paths
        )
        self._sandbox_factory = sandbox_factory

    async def run(
        self,
        *,
        command: list[str],
        env: dict[str, str],
        config_dir: Path,
    ) -> SandboxRunResult:
        """Run ``command`` and return process output plus a complete Git patch."""

        started_at = monotonic()
        sandbox = self._sandbox_factory(self._provider, self._spec)
        remote_config_dir = str(PurePosixPath("/tmp") / "nemo_gym_claude_config")
        remote_archive = f"{remote_config_dir}.tar.gz"
        remote_patch = "/tmp/nemo_gym_workspace.patch"
        remote_patch_status = "/tmp/nemo_gym_workspace_patch.status"

        try:
            async with asyncio.timeout(self._timeout_s):
                await sandbox.start()
                with tempfile.TemporaryDirectory(prefix="nemo_gym_claude_sandbox_") as temp_dir:
                    archive_path = Path(temp_dir) / "claude_config.tar.gz"
                    await asyncio.to_thread(self._archive_config, config_dir, archive_path)
                    await sandbox.upload(archive_path, remote_archive)

                setup_result = await sandbox.exec(
                    self._config_setup_command(remote_config_dir, remote_archive),
                    cwd="/tmp",
                    timeout_s=self._timeout_s,
                    user=self._user,
                )
                self._require_success(setup_result, "stage Claude Code configuration")
                await self._validate_workspace(sandbox)

                revision_result = await sandbox.exec(
                    "git rev-parse HEAD",
                    cwd=self._workspace,
                    timeout_s=self._timeout_s,
                    user=self._user,
                )
                self._require_success(revision_result, "resolve workspace base revision")
                base_revision = (revision_result.stdout or "").strip()
                if not _GIT_REVISION_RE.fullmatch(base_revision):
                    raise RuntimeError(f"Sandbox workspace returned an invalid Git revision: {base_revision!r}")

                command_env = {**env, "CLAUDE_CONFIG_DIR": remote_config_dir}
                run_result = await self._run_agent_command(sandbox, command, command_env)
                if run_result.error_type is not None:
                    detail = (run_result.stderr or run_result.stdout or "").strip()
                    raise RuntimeError(
                        "Sandboxed Claude Code did not complete safely "
                        f"(return_code={run_result.return_code}, error_type={run_result.error_type!r}): {detail}"
                    )

                intent_result = await sandbox.exec(
                    "git add -N -- .",
                    cwd=self._workspace,
                    timeout_s=self._timeout_s,
                    user=self._user,
                )
                self._require_success(intent_result, "mark untracked files for patch capture")

                patch_result = await sandbox.exec(
                    self._patch_capture_command(base_revision, remote_patch, remote_patch_status),
                    cwd=self._workspace,
                    timeout_s=self._timeout_s,
                    user=self._user,
                )
                self._require_success(patch_result, "capture workspace patch")
                diff_status_result = await sandbox.exec(
                    f"cat {shlex.quote(remote_patch_status)}",
                    cwd=self._workspace,
                    timeout_s=self._timeout_s,
                    user=self._user,
                )
                self._require_success(diff_status_result, "read workspace diff status")
                try:
                    diff_status = int((diff_status_result.stdout or "").strip())
                except ValueError as exc:
                    raise RuntimeError(
                        f"Sandbox returned an invalid git diff status: {diff_status_result.stdout!r}"
                    ) from exc
                size_result = await sandbox.exec(
                    f"wc -c < {shlex.quote(remote_patch)}",
                    cwd=self._workspace,
                    timeout_s=self._timeout_s,
                    user=self._user,
                )
                self._require_success(size_result, "measure workspace patch")
                try:
                    patch_size = int((size_result.stdout or "").strip())
                except ValueError as exc:
                    raise RuntimeError(
                        f"Sandbox returned an invalid workspace patch size: {size_result.stdout!r}"
                    ) from exc
                if patch_size > self._max_patch_bytes:
                    raise RuntimeError(
                        f"Sandbox workspace patch is {patch_size} bytes, "
                        f"exceeding the {self._max_patch_bytes}-byte limit"
                    )
                if diff_status != 0:
                    raise RuntimeError(f"git diff failed while capturing the workspace patch (status={diff_status})")

                with tempfile.TemporaryDirectory(prefix="nemo_gym_claude_patch_") as temp_dir:
                    local_patch = Path(temp_dir) / "workspace.patch"
                    await sandbox.download(remote_patch, local_patch)
                    workspace_patch = local_patch.read_text(encoding="utf-8", errors="replace")

                return SandboxRunResult(
                    stdout=run_result.stdout or "",
                    stderr=run_result.stderr or "",
                    return_code=run_result.return_code,
                    error_type=run_result.error_type,
                    workspace_patch=workspace_patch,
                    base_revision=base_revision,
                    elapsed_seconds=monotonic() - started_at,
                )
        finally:
            active_exception = sys.exception()
            try:
                await self._stop_sandbox(sandbox)
            except Exception as cleanup_error:
                if active_exception is None:
                    raise
                active_exception.add_note(f"Sandbox cleanup also failed: {cleanup_error!r}")
                LOG.exception("sandbox cleanup failed while handling another error")

    async def _stop_sandbox(self, sandbox: AsyncSandbox) -> None:
        stop_task = asyncio.create_task(sandbox.stop())
        try:
            await asyncio.wait_for(asyncio.shield(stop_task), timeout=self._cleanup_timeout_s)
        except TimeoutError:
            LOG.error(
                "sandbox cleanup exceeded %ss; waiting for provider termination",
                self._cleanup_timeout_s,
            )
            await stop_task

    async def _run_agent_command(
        self,
        sandbox: AsyncSandbox,
        command: list[str],
        env: dict[str, str],
    ) -> SandboxExecResult:
        stdout_path = "/tmp/nemo_gym_claude.stdout"
        stderr_path = "/tmp/nemo_gym_claude.stderr"
        status_path = "/tmp/nemo_gym_claude.status"
        stdout_pipe = "/tmp/nemo_gym_claude.stdout.pipe"
        stderr_pipe = "/tmp/nemo_gym_claude.stderr.pipe"
        limit = self._max_output_bytes
        wrapped_command = (
            f"rm -f {stdout_pipe} {stderr_pipe}; mkfifo {stdout_pipe} {stderr_pipe} || exit $?; "
            f"({{ head -c {limit}; cat >/dev/null; }} < {stdout_pipe} > {stdout_path}) & out_reader=$!; "
            f"({{ head -c {limit}; cat >/dev/null; }} < {stderr_pipe} > {stderr_path}) & err_reader=$!; "
            f"{{ {shlex.join(command)}; }} > {stdout_pipe} 2> {stderr_pipe}; agent_status=$?; "
            "wait $out_reader; wait $err_reader; "
            f"rm -f {stdout_pipe} {stderr_pipe}; printf '%s' $agent_status > {status_path}"
        )
        wrapper_result = await sandbox.exec(
            wrapped_command,
            cwd=self._workspace,
            env=env,
            timeout_s=self._timeout_s,
            user=self._user,
        )
        if wrapper_result.error_type is not None:
            return wrapper_result
        self._require_success(wrapper_result, "run sandboxed Claude Code wrapper")

        with tempfile.TemporaryDirectory(prefix="nemo_gym_claude_output_") as temp_dir:
            root = Path(temp_dir)
            local_stdout = root / "stdout"
            local_stderr = root / "stderr"
            local_status = root / "status"
            await sandbox.download(stdout_path, local_stdout)
            await sandbox.download(stderr_path, local_stderr)
            await sandbox.download(status_path, local_status)
            stdout = local_stdout.read_text(encoding="utf-8", errors="replace")
            stderr = local_stderr.read_text(encoding="utf-8", errors="replace")
            try:
                return_code = int(local_status.read_text().strip())
            except ValueError as exc:
                raise RuntimeError("Sandboxed Claude Code returned an invalid status") from exc

        return SandboxExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            error_type=None,
        )

    async def _validate_workspace(self, sandbox: AsyncSandbox) -> None:
        if self._require_clean_workspace:
            status_result = await sandbox.exec(
                "git status --porcelain=v1 --untracked-files=all",
                cwd=self._workspace,
                timeout_s=self._timeout_s,
                user=self._user,
            )
            self._require_success(status_result, "inspect sandbox workspace")
            dirty_paths = (status_result.stdout or "").strip()
            if dirty_paths:
                raise RuntimeError(f"Sandbox workspace must start clean; found:\n{dirty_paths}")

        if self._forbidden_workspace_paths:
            checks = " && ".join(
                f"test ! -e {shlex.quote(path)} && test ! -L {shlex.quote(path)}"
                for path in self._forbidden_workspace_paths
            )
            forbidden_result = await sandbox.exec(
                checks,
                cwd=self._workspace,
                timeout_s=self._timeout_s,
                user=self._user,
            )
            if forbidden_result.return_code != 0 or forbidden_result.error_type is not None:
                paths = ", ".join(self._forbidden_workspace_paths)
                raise RuntimeError(f"Sandbox workspace contains forbidden auto-discovery paths: {paths}")

    @staticmethod
    def _archive_config(config_dir: Path, archive_path: Path) -> None:
        if not config_dir.is_dir():
            raise ValueError(f"Claude Code config directory does not exist: {config_dir}")
        with tarfile.open(archive_path, mode="w:gz") as archive:
            for child in sorted(config_dir.iterdir()):
                archive.add(child, arcname=child.name, recursive=True)

    @staticmethod
    def _config_setup_command(remote_config_dir: str, remote_archive: str) -> str:
        config = shlex.quote(remote_config_dir)
        archive = shlex.quote(remote_archive)
        return f"rm -rf {config} && mkdir -p {config} && tar -xzf {archive} -C {config} && rm -f {archive}"

    def _patch_capture_command(self, base_revision: str, remote_patch: str, remote_status: str) -> str:
        revision = shlex.quote(base_revision)
        patch = shlex.quote(remote_patch)
        status = shlex.quote(remote_status)
        bounded_size = self._max_patch_bytes + 1
        return (
            "(git -c diff.external= -c core.attributesfile=/dev/null "
            f"diff --binary --no-ext-diff --no-textconv {revision} --; "
            f"printf '%s' $? > {status}) "
            f"| head -c {bounded_size} > {patch}"
        )

    @staticmethod
    def _validate_relative_path(path: str) -> str:
        parsed = PurePosixPath(path)
        if not path or parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"Forbidden workspace path must be a non-empty relative path: {path!r}")
        return str(parsed)

    @staticmethod
    def _require_success(result: Any, operation: str) -> None:
        if result.return_code == 0 and result.error_type is None:
            return
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Failed to {operation} in sandbox "
            f"(return_code={result.return_code}, error_type={result.error_type!r}): {detail}"
        )
