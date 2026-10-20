# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit actor-only asset staging on Gym's Harbor/OpenSandbox backend."""

import os
import shlex
from copy import deepcopy
from pathlib import Path, PurePosixPath

from harbor.environments.base import ExecResult

from nemo_gym.global_config import GlobalConfigDictParserConfig
from nemo_gym.server_utils import get_global_aiohttp_client
from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox.environment import (
    NemoGymSandboxEnvironment,
    _cpu_pin_prefix,
)


def actor_directory(environment: Path, relative: str) -> Path:
    """Resolve declared actor assets, never grading assets outside environment/."""
    path = environment / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_dir():
        raise ValueError(f"Invalid actor directory: {relative}")
    if not path.resolve().is_relative_to(environment.resolve()) or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Actor directories may not contain escaping links")
    return path


class ArtifactEnvironment(NemoGymSandboxEnvironment):
    """Retain the native task bootstrap without routing on task names.

    Directories, packages and output directory are trusted execution-profile
    configuration, never model-provided parameters. Only actor assets are uploaded.
    """

    def __init__(
        self,
        *args,
        sandbox_provider: dict,
        actor_directories: dict[str, str],
        output_directory: str,
        bootstrap_packages: list[str] | None = None,
        **kwargs,
    ):
        provider = deepcopy(sandbox_provider)
        # Resolve inside the Ray worker; JobConfig receipts never serialize the key.
        provider["opensandbox"]["connection"]["api_key"] = os.environ["OPENSANDBOX_API_KEY"]
        self._actor_directories = dict(actor_directories)
        self._output_directory = output_directory
        self._bootstrap_packages = list(bootstrap_packages or [])
        for destination in [*actor_directories.values(), output_directory]:
            if not PurePosixPath(destination).is_absolute() or ".." in PurePosixPath(destination).parts:
                raise ValueError("Sandbox asset/output paths must be absolute without traversal")
        super().__init__(*args, sandbox_provider=provider, **kwargs)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | None = None,
    ) -> ExecResult:
        """Bridge Harbor0.24 user/scoped-env support without losing execution identity."""
        if self._exec_shell:
            command = f"{self._exec_shell} {shlex.quote(command)}"
        if self._cpu_pin_enabled:
            width = int(self.task_env_config.cpus or 0)
            if width > 0:
                command = f"{_cpu_pin_prefix(width)} {command}"
        result = await self._require_sandbox().exec(
            command,
            cwd=cwd,
            env=self._merge_env(env),
            user=user,
            timeout_s=timeout_sec if timeout_sec is not None else self._default_exec_timeout_s,
        )
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.return_code)

    async def start(self, force_build: bool) -> None:
        paths = [
            (actor_directory(self.environment_dir, source), destination)
            for source, destination in self._actor_directories.items()
        ]
        try:
            # A bare Ray worker has worker CLI flags, not Gym CLI overrides.
            # Initialize the shared SDK transport explicitly; reuse injected Gym
            # configuration when present, otherwise use library defaults only.
            get_global_aiohttp_client(
                GlobalConfigDictParserConfig(skip_load_from_cli=True, skip_load_from_dotenv=True)
            )
            await super().start(force_build)
            for source, destination in paths:
                await self.upload_dir(source, destination)
            bootstrap = (
                "python -m pip install --disable-pip-version-check "
                + shlex.join(self._bootstrap_packages)
                + " && python -m pip freeze > /logs/agent/partner-if-bootstrap-packages.txt && "
                if self._bootstrap_packages
                else ""
            )
            result = await self.exec(bootstrap + f"mkdir -p {shlex.quote(self._output_directory)}", timeout_sec=600)
            if result.return_code:
                raise RuntimeError(f"Actor bootstrap failed: {result.stderr}\n{(result.stdout or '')[-8000:]}")
        except BaseException:
            await self.stop(delete=True)
            raise
