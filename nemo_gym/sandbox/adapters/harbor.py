# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor environment over a Gym sandbox that another component owns.

Harbor's agents and verifier act on a ``harbor.environments.base.BaseEnvironment``. This adapter lets them act on a
Gym ``AsyncSandbox`` instead, so a Resources Server can own the task sandbox while a Harbor agent borrows it and the
Harbor verifier later grades it. The adapter never creates or destroys the sandbox: ``start`` only prepares Harbor's
directory layout and ``stop`` is a no-op. The owner stops the sandbox; a borrower disconnects it.

Import this module only from servers that install ``harbor``.
"""

import logging
import shlex
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from harbor.environments.base import BaseEnvironment, ExecResult, transfer_tar_filter
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

from nemo_gym.sandbox import AsyncSandbox


HARBOR_ENVIRONMENT_TYPE = "nemo_gym_sandbox"
# Task-level agent settings from task.toml. The agent never reads the task, so prepared rows carry them in
# responses_create_params.metadata, and the Harbor harness agent reads them at activation.
HARBOR_AGENT_TIMEOUT_METADATA_KEY = "harbor_agent_timeout_sec"
HARBOR_AGENT_USER_METADATA_KEY = "harbor_agent_user"
# Harbor's agents and verifier bound their own phases with asyncio timeouts and often call exec without a timeout.
# Providers substitute a short default for None, so pass a bound longer than any Harbor phase instead.
DEFAULT_EXEC_TIMEOUT_SECONDS = 24 * 3600.0
_TRANSFER_DIR = PurePosixPath("/tmp")
_TRANSFER_TIMEOUT_SECONDS = 600.0


class HarborSandboxEnvironment(BaseEnvironment):
    """Expose a running Gym sandbox through Harbor's environment interface."""

    def __init__(
        self,
        sandbox: AsyncSandbox,
        *,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        exec_shell: str | None = "bash -c",
        default_exec_timeout_seconds: float = DEFAULT_EXEC_TIMEOUT_SECONDS,
        logger: logging.Logger | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._exec_shell = exec_shell
        self._default_exec_timeout_seconds = default_exec_timeout_seconds
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
        )

    @staticmethod
    def type() -> str:
        return HARBOR_ENVIRONMENT_TYPE

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # Logs are not bind-mounted, so Harbor downloads /logs to the host after each phase.
        return EnvironmentCapabilities(mounted=False)

    def _validate_definition(self) -> None:
        # The owner already started the sandbox from the task image; there is no definition to build here.
        return None

    async def start(self, force_build: bool) -> None:
        """Create Harbor's log directories and upload ``environment/`` for prebuilt-image tasks.

        Harbor environments do this while starting a container. The sandbox already runs, so only the layout remains.
        Call it once, from the sandbox owner, before handing the sandbox to an agent.
        """
        log_dirs = " ".join(
            shlex.quote(str(path))
            for path in (EnvironmentPaths.agent_dir, EnvironmentPaths.verifier_dir, EnvironmentPaths.artifacts_dir)
        )
        result = await self._sandbox.exec(
            f"mkdir -p {log_dirs} && chmod 777 {log_dirs}", cwd="/", user="root", timeout_s=60
        )
        if result.return_code != 0:
            raise RuntimeError(f"Cannot create Harbor log directories: {result.stderr or result.stdout}")
        workdir = self.task_env_config.workdir
        if workdir and workdir != "/":
            result = await self._sandbox.exec(f"mkdir -p {shlex.quote(workdir)}", cwd="/", user="root", timeout_s=60)
            if result.return_code != 0:
                raise RuntimeError(f"Cannot create task workdir {workdir}: {result.stderr or result.stdout}")
        await self._upload_environment_dir_after_start()

    async def stop(self, delete: bool) -> None:
        # The sandbox owner releases the sandbox, never a Harbor component.
        return None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self._exec_shell:
            command = f"{self._exec_shell} {shlex.quote(command)}"
        result = await self._sandbox.exec(
            command,
            cwd=cwd or self.task_env_config.workdir,
            env=self._merge_env(env),
            timeout_s=timeout_sec if timeout_sec is not None else self._default_exec_timeout_seconds,
            user=self._resolve_user(user),
        )
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.return_code)

    async def _root_exec(self, command: str, *, timeout_s: float = 60) -> None:
        result = await self._sandbox.exec(command, cwd="/", user="root", timeout_s=timeout_s)
        if result.return_code != 0:
            raise RuntimeError(f"Sandbox command failed ({result.return_code}): {command}: {result.stderr}")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        await self._root_exec(f"mkdir -p {shlex.quote(str(PurePosixPath(target_path).parent))}")
        await self._sandbox.upload(source, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source}")
        remote_tar = str(_TRANSFER_DIR / f".nemo-gym-harbor-upload-{uuid.uuid4().hex}.tar.gz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_tar = Path(tmp_dir) / "upload.tar.gz"
            with tarfile.open(local_tar, "w:gz") as tar:
                # Archive the directory's contents so they land directly in target_dir, as Harbor expects.
                tar.add(source, arcname=".")
            await self._sandbox.upload(local_tar, remote_tar)
        target, archive = shlex.quote(target_dir), shlex.quote(remote_tar)
        await self._root_exec(
            f"mkdir -p {target} && tar -xzf {archive} --no-same-owner -C {target}; status=$?; rm -f {archive}; "
            "exit $status",
            timeout_s=_TRANSFER_TIMEOUT_SECONDS,
        )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._sandbox.download(source_path, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        remote_tar = str(_TRANSFER_DIR / f".nemo-gym-harbor-download-{uuid.uuid4().hex}.tar.gz")
        source, archive = shlex.quote(source_dir), shlex.quote(remote_tar)
        try:
            await self._root_exec(f"tar -czf {archive} -C {source} .", timeout_s=_TRANSFER_TIMEOUT_SECONDS)
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_tar = Path(tmp_dir) / "download.tar.gz"
                await self._sandbox.download(remote_tar, local_tar)
                with tarfile.open(local_tar, "r:gz") as tar:
                    tar.extractall(target, filter=transfer_tar_filter)
        finally:
            await self._sandbox.exec(f"rm -f {archive}", cwd="/", user="root", timeout_s=60)
