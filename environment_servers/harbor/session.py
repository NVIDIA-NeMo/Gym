# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-episode session and the in-sandbox workspace the agent and the verifier share.

HarborSession records what one episode owns: the task snapshot, the sandbox or Compose group,
the working directory, the entrypoint launch and the stop flags, with the phased stop methods
Harbor's collection order needs. Next to it live the workspace layout constants for the log
and artifact folders, the task context file the agent harness reads, and the host template
resolver for whole-value ${VAR} entries in task environments. The mixin holds the server
methods that prepare the working directory, write the task context and describe sandbox
access to the agent. The module sits below the verifier so that verifier code can be typed on
HarborSession without importing the server.
"""

import os
import re
import shlex
from dataclasses import dataclass

from fastapi import HTTPException

from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.transfer import exec_as_root, upload_dir
from nemo_gym.tasks.harbor import HarborTask


VERIFIER_LOGS_DIR = "/logs/verifier"
AGENT_LOGS_DIR = "/logs/agent"


@dataclass
class HarborSession:
    """One episode's sandbox, from seed until the final cleanup stops it."""

    task: HarborTask
    taskset: str
    identity: tuple[EpisodeId, TaskId]
    sandbox: AsyncSandbox
    workdir: str


_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_templates(env: dict[str, str]) -> dict[str, str]:
    """Resolve Harbor's whole-value host templates without changing the stored task."""
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = _ENV_TEMPLATE.fullmatch(value)
        if match is None:
            resolved[key] = value
        elif match.group(1) in os.environ:
            resolved[key] = os.environ[match.group(1)]
        elif match.group(2) is not None:
            resolved[key] = match.group(2)
        else:
            raise HTTPException(422, f"Environment variable {match.group(1)!r} is required by the task")
    return resolved


class HarborSessionMixin:
    """The per-episode session and the in-sandbox workspace the agent and the verifier share.

    HarborSession records what one episode owns: the task snapshot, the sandbox or Compose group,
    the working directory, the entrypoint launch and the stop flags, with the phased stop methods
    Harbor's collection order needs. Next to it live the workspace layout constants for the log
    and artifact folders, the task context file the agent harness reads, and the host template
    resolver for whole-value ${VAR} entries in task environments. The mixin holds the server
    methods that prepare the working directory, write the task context and describe sandbox
    access to the agent. The module sits below the verifier so that verifier code can be typed on
    HarborSession without importing the server.

    """

    async def _prepare_workdir(self, sandbox: AsyncSandbox, task: HarborTask, workdir: str | None) -> str:
        """Create the working directory, or resolve the image's own WORKDIR when the task sets none."""
        if workdir is None:
            result = await sandbox.exec("pwd", timeout_s=60)
            if result.return_code != 0 or not (result.stdout or "").strip():
                raise RuntimeError(f"Could not resolve the image working directory: {result.stderr or result.stdout}")
            workdir = (result.stdout or "").strip().splitlines()[-1]
        if workdir != "/":
            # Only a directory the task or image named below the root is created and, with a user
            # override, handed over; an image whose WORKDIR is `/` must keep root's ownership of it.
            commands = [f"mkdir -p {shlex.quote(workdir)}"]
            if task.user:
                commands.append(f"chown {shlex.quote(task.user)} {shlex.quote(workdir)}")
            result = await exec_as_root(sandbox, " && ".join(commands), timeout_s=60)
            if result.return_code != 0:
                raise RuntimeError(f"Could not prepare {workdir}: {result.stderr or result.stdout}")
        environment_dir = task.path / "environment"
        if (
            task.config.environment.docker_image
            and environment_dir.is_dir()
            and not any(
                (environment_dir / name).exists()
                for name in ("Dockerfile", "docker-compose.yaml", "docker-compose.yml")
            )
        ):
            await upload_dir(sandbox, environment_dir, workdir)
        return workdir

    async def _sandbox_access(self, session: HarborSession) -> SandboxAccess:
        return SandboxAccess(
            connection=DirectSandboxConnection(
                provider_config_ref=self.config.sandbox_provider,
                descriptor=await session.sandbox.serialize(),
            ),
            workdir=session.workdir,
        )
