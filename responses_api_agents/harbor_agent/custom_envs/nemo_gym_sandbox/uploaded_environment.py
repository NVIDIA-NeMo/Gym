# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Harbor environment that uploads task files into an OpenSandbox sandbox."""

import shlex
import tempfile
from pathlib import Path
from typing import Optional
from uuid import uuid4

from opensandbox.exceptions.sandbox import SandboxApiException

from .environment import NemoGymSandboxEnvironment


class UploadedNemoGymSandboxEnvironment(NemoGymSandboxEnvironment):
    """NemoGymSandboxEnvironment with opt-in task payload materialization."""

    def __init__(
        self,
        *args,
        upload_environment_dir: bool = True,
        upload_target_dir: Optional[str] = None,
        sandbox_setup_command: Optional[str] = None,
        sandbox_setup_timeout_s: int = 900,
        verifier_recovery_checkpoint: bool = False,
        workdir: Optional[str] = None,
        **kwargs,
    ):
        self._upload_environment_dir = upload_environment_dir
        self._sandbox_setup_command = sandbox_setup_command
        self._sandbox_setup_timeout_s = sandbox_setup_timeout_s
        self._uploaded_workdir = workdir or "/app"
        self._upload_target_dir = upload_target_dir or self._uploaded_workdir
        self._verifier_recovery_checkpoint = verifier_recovery_checkpoint
        self._verifier_checkpoint: Optional[Path] = None
        self._verifier_tests_dir: Optional[Path] = None
        self._verifier_recovery_attempted = False
        self._recovering_verifier = False
        self._baseline_commit: Optional[str] = None
        # The base image may not contain the task workdir yet. Start at /, then
        # upload the task payload and use the requested workdir for commands.
        super().__init__(*args, workdir="/", **kwargs)

    async def start(self, force_build: bool) -> None:
        await super().start(force_build=force_build)
        target = self._uploaded_workdir
        upload_target = self._upload_target_dir

        # A distinct bootstrap directory lets setup scripts reproduce a
        # Dockerfile into an otherwise clean task workdir.
        result = await self._require_sandbox().exec(
            f"mkdir -p {shlex.quote(target)} {shlex.quote(upload_target)}",
            timeout_s=60,
        )
        if result.return_code != 0:
            detail = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"OpenSandbox workdir setup failed: {detail}")

        if self._upload_environment_dir:
            await self.upload_dir(self.environment_dir, upload_target)

        if self._sandbox_setup_command:
            result = await self.exec(
                self._sandbox_setup_command,
                cwd=target,
                timeout_sec=self._sandbox_setup_timeout_s,
            )
            if result.return_code != 0:
                detail = result.stderr or result.stdout or "<no output>"
                raise RuntimeError(f"OpenSandbox task setup failed: {detail}")

        if self._verifier_recovery_checkpoint and self._baseline_commit is None:
            result = await self._require_sandbox().exec(
                f"git -C {shlex.quote(target)} rev-parse HEAD",
                timeout_s=60,
            )
            if result.return_code != 0:
                detail = result.stderr or result.stdout or "<no output>"
                raise RuntimeError(f"OpenSandbox baseline lookup failed: {detail}")
            self._baseline_commit = result.stdout.strip()

    async def _checkpoint_for_verifier(self) -> None:
        """Retain the agent's Git changes across an OpenSandbox restart."""
        if self._baseline_commit is None:
            raise RuntimeError("OpenSandbox verifier checkpoint has no Git baseline")
        handle = tempfile.NamedTemporaryFile(
            prefix="harbor-opensandbox-state-",
            suffix=".tar.gz",
            delete=False,
        )
        handle.close()
        checkpoint = Path(handle.name)
        state_dir = f"/tmp/.harbor-state-{uuid4().hex}"
        remote = f"{state_dir}.tar.gz"
        sandbox = self._require_sandbox()
        result = await sandbox.exec(
            f"mkdir -p {shlex.quote(state_dir)} && "
            f"git -C {shlex.quote(self._uploaded_workdir)} diff --binary "
            f"{shlex.quote(self._baseline_commit)} > {shlex.quote(state_dir)}/agent.patch && "
            f"git -C {shlex.quote(self._uploaded_workdir)} "
            f"ls-files --others --exclude-standard -z | "
            f"tar --null -czf {shlex.quote(state_dir)}/untracked.tar.gz "
            f"-C {shlex.quote(self._uploaded_workdir)} --files-from=- && "
            f"tar -czf {shlex.quote(remote)} -C {shlex.quote(state_dir)} .",
            timeout_s=600,
        )
        if result.return_code != 0:
            checkpoint.unlink(missing_ok=True)
            detail = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"OpenSandbox verifier checkpoint failed: {detail}")
        try:
            await sandbox.download(remote, checkpoint)
        finally:
            try:
                await sandbox.exec(
                    f"rm -rf {shlex.quote(state_dir)} {shlex.quote(remote)}",
                    timeout_s=60,
                )
            except SandboxApiException as exc:
                self.logger.warning("OpenSandbox dropped after checkpoint download: %s", exc)
        if self._verifier_checkpoint is not None:
            self._verifier_checkpoint.unlink(missing_ok=True)
        self._verifier_checkpoint = checkpoint

    async def _recover_verifier_sandbox(self, *, restore_tests: bool) -> None:
        """Recreate a dropped sandbox and restore the completed agent state."""
        if self._verifier_recovery_attempted or self._verifier_checkpoint is None:
            raise RuntimeError("OpenSandbox verifier recovery is unavailable")
        self._verifier_recovery_attempted = True
        self._recovering_verifier = True
        old_sandbox, self._sandbox = self._sandbox, None
        if old_sandbox is not None:
            try:
                await old_sandbox.stop()
            except Exception as exc:
                self.logger.warning("Failed to close dropped OpenSandbox sandbox: %s", exc)

        try:
            self.logger.warning(
                "OpenSandbox dropped the sandbox during verifier handoff; "
                "recreating it from the pre-verifier app checkpoint."
            )
            await self.start(force_build=False)
            sandbox = self._require_sandbox()
            state_dir = f"/tmp/.harbor-state-restore-{uuid4().hex}"
            remote = f"{state_dir}.tar.gz"
            await sandbox.upload(self._verifier_checkpoint, remote)
            result = await sandbox.exec(
                f"mkdir -p {shlex.quote(state_dir)} && "
                f"tar -xzf {shlex.quote(remote)} -C {shlex.quote(state_dir)} && "
                f"git -C {shlex.quote(self._uploaded_workdir)} reset --hard "
                f"{shlex.quote(self._baseline_commit or 'HEAD')} && "
                f"git -C {shlex.quote(self._uploaded_workdir)} clean -fd && "
                f"git -C {shlex.quote(self._uploaded_workdir)} apply --allow-empty --binary "
                f"{shlex.quote(state_dir)}/agent.patch && "
                f"tar -xzf {shlex.quote(state_dir)}/untracked.tar.gz "
                f"-C {shlex.quote(self._uploaded_workdir)}; "
                f"status=$?; rm -rf {shlex.quote(state_dir)} {shlex.quote(remote)}; "
                f"exit $status",
                timeout_s=600,
            )
            if result.return_code != 0:
                detail = result.stderr or result.stdout or "<no output>"
                raise RuntimeError(f"OpenSandbox verifier restore failed: {detail}")
            if restore_tests and self._verifier_tests_dir is not None:
                await super().upload_dir(self._verifier_tests_dir, "/tests")
        finally:
            self._recovering_verifier = False

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        """Checkpoint before Harbor makes its post-agent verifier upload."""
        if not self._verifier_recovery_checkpoint or target_dir != "/tests" or self._recovering_verifier:
            return await super().upload_dir(source_dir, target_dir)

        self._verifier_tests_dir = Path(source_dir)
        try:
            await self._checkpoint_for_verifier()
        except Exception:
            if self._verifier_checkpoint is None:
                raise
            await self._recover_verifier_sandbox(restore_tests=False)
            return await super().upload_dir(source_dir, target_dir)
        try:
            return await super().upload_dir(source_dir, target_dir)
        except SandboxApiException:
            await self._recover_verifier_sandbox(restore_tests=False)
            return await super().upload_dir(source_dir, target_dir)

    async def exec(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout_sec: Optional[int] = None,
    ):
        """Default Harbor commands to the materialized task workdir."""
        try:
            result = await super().exec(
                command,
                cwd=cwd or self._uploaded_workdir,
                env=env,
                timeout_sec=timeout_sec,
            )
            if (
                self._verifier_recovery_checkpoint
                and not self._recovering_verifier
                and ("tmux wait done" in command or "tmux capture-pane" in command)
            ):
                try:
                    await self._checkpoint_for_verifier()
                except Exception as exc:
                    self.logger.warning("Could not refresh OpenSandbox agent checkpoint: %s", exc)
            return result
        except SandboxApiException:
            if self._verifier_checkpoint is None or self._recovering_verifier:
                raise
            await self._recover_verifier_sandbox(restore_tests=True)
            return await super().exec(
                command,
                cwd=cwd or self._uploaded_workdir,
                env=env,
                timeout_sec=timeout_sec,
            )

    async def stop(self, delete: bool):
        try:
            await super().stop(delete)
        finally:
            if self._verifier_checkpoint is not None:
                self._verifier_checkpoint.unlink(missing_ok=True)
