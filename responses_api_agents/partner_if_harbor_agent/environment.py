# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit actor-only asset staging on Gym's Harbor/OpenSandbox backend."""

import os
import shlex
from copy import deepcopy
from pathlib import Path, PurePosixPath

from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox.environment import NemoGymSandboxEnvironment


# The Harbor worker reads the OpenSandbox key from its environment, so Harbor's job configs never hold it.
OPENSANDBOX_API_KEY = "OPENSANDBOX_API_KEY"


def actor_directory(environment: Path, relative: str) -> Path:
    """Resolve declared actor assets, never grading assets outside environment/."""
    path = environment / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_dir():
        raise ValueError(f"Invalid actor directory: {relative}")
    if not path.resolve().is_relative_to(environment.resolve()) or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Actor directories may not contain escaping links")
    return path


class ArtifactEnvironment(NemoGymSandboxEnvironment):
    """Upload the configured actor directories and create the output directory before the agent starts.

    Paths come from trusted config, never from the model.
    """

    def __init__(
        self,
        *args,
        sandbox_provider: dict,
        actor_directories: dict[str, str],
        output_directory: str,
        **kwargs,
    ):
        provider = deepcopy(sandbox_provider)
        provider["opensandbox"]["connection"]["api_key"] = os.environ[OPENSANDBOX_API_KEY]
        self._actor_directories = dict(actor_directories)
        self._output_directory = output_directory
        for destination in [*actor_directories.values(), output_directory]:
            if not PurePosixPath(destination).is_absolute() or ".." in PurePosixPath(destination).parts:
                raise ValueError("Sandbox asset/output paths must be absolute without traversal")
        super().__init__(*args, sandbox_provider=provider, **kwargs)

    async def start(self, force_build: bool) -> None:
        paths = [
            (actor_directory(self.environment_dir, source), destination)
            for source, destination in self._actor_directories.items()
        ]
        try:
            await super().start(force_build)
            for source, destination in paths:
                await self.upload_dir(source, destination)
            result = await self.exec(f"mkdir -p {shlex.quote(self._output_directory)}", timeout_sec=600)
            if result.return_code:
                raise RuntimeError(f"Cannot create output directory {self._output_directory}: {result.stderr}")
        except BaseException:
            await self.stop(delete=True)
            raise
