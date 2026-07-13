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

"""Apply an agent patch and run hidden checks in a fresh Gym sandbox."""

from __future__ import annotations

import asyncio
import logging
import shlex
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any

from pydantic import BaseModel

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec


LOG = logging.getLogger(__name__)


class PatchVerificationResult(BaseModel):
    """Result of applying and checking one agent workspace patch."""

    passed: bool
    status: str
    stdout: str = ""
    stderr: str = ""
    return_code: int
    error_type: str | None = None
    verifier_base_revision: str
    elapsed_seconds: float


class SandboxPatchVerifier:
    """Verify a patch in a fresh sandbox containing server-side hidden checks."""

    def __init__(
        self,
        *,
        provider: Mapping[str, Any],
        spec: SandboxSpec,
        workspace: str,
        check_command: str,
        timeout_s: int | float,
        hidden_files: Mapping[str, str] | None = None,
        check_cwd: str = "/tmp/nemo_gym_hidden_checks",
        user: str | int | None = "root",
        check_user: str | int | None = "nobody",
        max_patch_bytes: int = 10 * 1024 * 1024,
        max_log_chars: int = 20_000,
        cleanup_timeout_s: int | float = 30,
        sandbox_factory: Callable[..., AsyncSandbox] = AsyncSandbox,
    ) -> None:
        self._provider = provider
        self._spec = spec
        self._workspace = workspace
        self._check_command = check_command
        self._hidden_files = {
            self._validate_hidden_path(path): contents for path, contents in (hidden_files or {}).items()
        }
        self._check_cwd = self._validate_check_cwd(check_cwd, workspace)
        self._timeout_s = timeout_s
        self._user = user
        self._check_user = check_user
        self._max_patch_bytes = max_patch_bytes
        self._max_log_chars = max_log_chars
        self._cleanup_timeout_s = cleanup_timeout_s
        self._sandbox_factory = sandbox_factory

    async def verify(
        self,
        *,
        patch: str,
        expected_base_revision: str | None,
    ) -> PatchVerificationResult:
        """Apply ``patch`` and execute the configured hidden check command."""

        patch_size = len(patch.encode())
        if patch_size > self._max_patch_bytes:
            raise ValueError(
                f"Workspace patch is {patch_size} bytes, exceeding the {self._max_patch_bytes}-byte verifier limit"
            )
        if not patch.strip():
            raise ValueError("Workspace patch is empty")

        started_at = monotonic()
        sandbox = self._sandbox_factory(self._provider, self._spec)
        try:
            async with asyncio.timeout(self._timeout_s):
                await sandbox.start()
                revision = await self._require_clean_revision(sandbox)
                if expected_base_revision and revision != expected_base_revision:
                    raise ValueError(
                        "Agent and verifier fixtures use different base revisions: "
                        f"agent={expected_base_revision}, verifier={revision}"
                    )

                with tempfile.TemporaryDirectory(prefix="nemo_gym_agent_skill_verify_") as temp_dir:
                    local_patch = Path(temp_dir) / "submission.patch"
                    local_patch.write_text(patch)
                    remote_patch = "/tmp/nemo_gym_submission.patch"
                    await sandbox.upload(local_patch, remote_patch)

                quoted_patch = shlex.quote(remote_patch)
                check_apply = await sandbox.exec(
                    f"git apply --check --binary {quoted_patch}",
                    cwd=self._workspace,
                    timeout_s=self._timeout_s,
                    user=self._user,
                )
                self._require_success(check_apply, "validate submitted patch")
                apply_result = await sandbox.exec(
                    f"git apply --binary {quoted_patch}",
                    cwd=self._workspace,
                    timeout_s=self._timeout_s,
                    user=self._user,
                )
                self._require_success(apply_result, "apply submitted patch")
                check_dir_result = await sandbox.exec(
                    f"mkdir -p {shlex.quote(self._check_cwd)}",
                    cwd="/tmp",
                    timeout_s=self._timeout_s,
                    user=self._user,
                )
                self._require_success(check_dir_result, "create hidden check directory")
                await self._upload_hidden_files(sandbox)
                protect_result = await sandbox.exec(
                    f"chmod -R a-w {shlex.quote(self._check_cwd)}",
                    cwd="/tmp",
                    timeout_s=self._timeout_s,
                    user=self._user,
                )
                self._require_success(protect_result, "protect hidden check files")

                check_result, stdout, stderr = await self._run_hidden_check(sandbox)
                safely_completed = check_result.error_type is None
                passed = safely_completed and check_result.return_code == 0
                status = "pass" if passed else (check_result.error_type or "check_failed")
                return PatchVerificationResult(
                    passed=passed,
                    status=status,
                    stdout=stdout,
                    stderr=stderr,
                    return_code=check_result.return_code,
                    error_type=check_result.error_type,
                    verifier_base_revision=revision,
                    elapsed_seconds=monotonic() - started_at,
                )
        finally:
            active_exception = sys.exception()
            try:
                await self._stop_sandbox(sandbox)
            except Exception as cleanup_error:
                if active_exception is None:
                    raise
                active_exception.add_note(f"Verifier sandbox cleanup also failed: {cleanup_error!r}")
                LOG.exception("verifier sandbox cleanup failed while handling another error")

    async def _run_hidden_check(self, sandbox: AsyncSandbox) -> tuple[Any, str, str]:
        stdout_path = "/tmp/nemo_gym_hidden_check.stdout"
        stderr_path = "/tmp/nemo_gym_hidden_check.stderr"
        status_path = "/tmp/nemo_gym_hidden_check.status"
        stdout_pipe = "/tmp/nemo_gym_hidden_check.stdout.pipe"
        stderr_pipe = "/tmp/nemo_gym_hidden_check.stderr.pipe"
        wrapped_command = self._bounded_check_command(
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            status_path=status_path,
            stdout_pipe=stdout_pipe,
            stderr_pipe=stderr_pipe,
        )
        wrapper_result = await sandbox.exec(
            wrapped_command,
            cwd=self._check_cwd,
            env={"NEMO_GYM_WORKSPACE": self._workspace},
            timeout_s=self._timeout_s,
            user=self._check_user,
        )
        if wrapper_result.error_type is not None:
            return wrapper_result, "", (wrapper_result.stderr or "")[: self._max_log_chars]
        self._require_success(wrapper_result, "run hidden check wrapper")

        with tempfile.TemporaryDirectory(prefix="nemo_gym_hidden_check_output_") as temp_dir:
            root = Path(temp_dir)
            local_stdout = root / "stdout"
            local_stderr = root / "stderr"
            local_status = root / "status"
            await sandbox.download(stdout_path, local_stdout)
            await sandbox.download(stderr_path, local_stderr)
            await sandbox.download(status_path, local_status)
            stdout = local_stdout.read_text(encoding="utf-8", errors="replace")[: self._max_log_chars]
            stderr = local_stderr.read_text(encoding="utf-8", errors="replace")[: self._max_log_chars]
            try:
                return_code = int(local_status.read_text().strip())
            except ValueError as exc:
                raise RuntimeError("Hidden check returned an invalid status") from exc

        result = type(wrapper_result)(
            stdout="",
            stderr="",
            return_code=return_code,
            error_type=None,
        )
        return result, stdout, stderr

    def _bounded_check_command(
        self,
        *,
        stdout_path: str,
        stderr_path: str,
        status_path: str,
        stdout_pipe: str,
        stderr_pipe: str,
    ) -> str:
        limit = self._max_log_chars
        stdout = shlex.quote(stdout_path)
        stderr = shlex.quote(stderr_path)
        status = shlex.quote(status_path)
        out_pipe = shlex.quote(stdout_pipe)
        err_pipe = shlex.quote(stderr_pipe)
        return (
            f"rm -f {out_pipe} {err_pipe}; mkfifo {out_pipe} {err_pipe} || exit $?; "
            f"({{ head -c {limit}; cat >/dev/null; }} < {out_pipe} > {stdout}) & out_reader=$!; "
            f"({{ head -c {limit}; cat >/dev/null; }} < {err_pipe} > {stderr}) & err_reader=$!; "
            f"{{ {self._check_command}; }} > {out_pipe} 2> {err_pipe}; check_status=$?; "
            "wait $out_reader; wait $err_reader; "
            f"rm -f {out_pipe} {err_pipe}; printf '%s' $check_status > {status}"
        )

    async def _stop_sandbox(self, sandbox: AsyncSandbox) -> None:
        stop_task = asyncio.create_task(sandbox.stop())
        try:
            await asyncio.wait_for(asyncio.shield(stop_task), timeout=self._cleanup_timeout_s)
        except TimeoutError:
            LOG.error(
                "verifier sandbox cleanup exceeded %ss; waiting for provider termination",
                self._cleanup_timeout_s,
            )
            await stop_task

    async def _require_clean_revision(self, sandbox: AsyncSandbox) -> str:
        status_result = await sandbox.exec(
            "git status --porcelain=v1 --untracked-files=all",
            cwd=self._workspace,
            timeout_s=self._timeout_s,
            user=self._user,
        )
        self._require_success(status_result, "inspect verifier workspace")
        dirty_paths = (status_result.stdout or "").strip()
        if dirty_paths:
            raise RuntimeError(f"Verifier workspace must start clean; found:\n{dirty_paths}")

        revision_result = await sandbox.exec(
            "git rev-parse HEAD",
            cwd=self._workspace,
            timeout_s=self._timeout_s,
            user=self._user,
        )
        self._require_success(revision_result, "resolve verifier base revision")
        return (revision_result.stdout or "").strip()

    async def _upload_hidden_files(self, sandbox: AsyncSandbox) -> None:
        if not self._hidden_files:
            return
        with tempfile.TemporaryDirectory(prefix="nemo_gym_hidden_checks_") as temp_dir:
            root = Path(temp_dir)
            for index, (relative_path, contents) in enumerate(self._hidden_files.items()):
                local_path = root / f"hidden-{index}"
                local_path.write_text(contents)
                remote_path = str(PurePosixPath(self._check_cwd) / relative_path)
                await sandbox.upload(local_path, remote_path)

    @staticmethod
    def _validate_hidden_path(path: str) -> str:
        parsed = PurePosixPath(path)
        if not path or parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"Hidden check file must use a non-empty relative path: {path!r}")
        return str(parsed)

    @staticmethod
    def _validate_check_cwd(check_cwd: str, workspace: str) -> str:
        check_path = PurePosixPath(check_cwd)
        workspace_path = PurePosixPath(workspace)
        if not check_path.is_absolute():
            raise ValueError(f"Hidden check directory must be absolute: {check_cwd!r}")
        if check_path == workspace_path or workspace_path in check_path.parents:
            raise ValueError(f"Hidden check directory must be outside the agent workspace: {check_cwd!r}")
        return str(check_path)

    @staticmethod
    def _require_success(result: Any, operation: str) -> None:
        if result.return_code == 0 and result.error_type is None:
            return
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Failed to {operation} in verifier sandbox "
            f"(return_code={result.return_code}, error_type={result.error_type!r}): {detail}"
        )
