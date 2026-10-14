# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-episode session and the in-sandbox workspace the agent and the verifier share.

HarborSession records what one episode owns: the task snapshot, the sandbox or Compose group,
the working directory, the entrypoint launch and the stop flags, with the phased stop methods
Harbor's collection order needs. Next to it live the workspace layout constants for the log
and artifact folders, the task context file the agent harness reads, and the host template
resolver for whole-value ${VAR} entries in task environments. The mixin holds the server
methods that prepare the working directory, write the task context, place the reference
solution for the oracle and describe sandbox access to the agent. The module sits below the
verifier so that verifier code can be typed on HarborSession without importing the server.
"""

import logging
import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field

from fastapi import HTTPException

from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.transfer import exec_as_root, upload_dir
from nemo_gym.tasks.harbor import HarborTask
from nemo_gym.tasks.harbor.cli import ORACLE_AGENT


LOGGER = logging.getLogger(__name__)


VERIFIER_LOGS_DIR = "/logs/verifier"
AGENT_LOGS_DIR = "/logs/agent"
# Where the oracle finds the task's reference solution: Harbor's oracle copies ``solution/`` here
# (0.22.0 ``agents/oracle.py``, ``EnvironmentPaths.solution_dir``) and runs ``solve.sh`` from it.
SOLUTION_DIR = "/solution"


@dataclass
class HarborSession:
    """One episode's sandbox, from seed until the final cleanup stops it."""

    task: HarborTask
    taskset: str
    identity: tuple[EpisodeId, TaskId]
    sandbox: AsyncSandbox
    workdir: str
    # Environment the harness runs its commands with, handed over in its sandbox access: the host-resolved
    # ``[solution].env`` when the harness is the oracle, nothing otherwise.
    agent_env: dict[str, str] = field(default_factory=dict)


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
    methods that prepare the working directory, write the task context, place the reference
    solution for the oracle and describe sandbox access to the agent. The module sits below the
    verifier so that verifier code can be typed on HarborSession without importing the server.

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

    def _agent_implementation(self) -> str | None:
        """The implementation behind the configured agent instance: the one key under its ``responses_api_agents``.

        ``None`` when the merged config does not show the instance in that shape (a test client, a block
        with several implementations); the lookup for token capture reads the same block positionally.
        """
        ref = self.config.agent_server
        instance = self.server_client.global_config_dict.get(ref.name)
        implementations = instance.get(ref.type) if isinstance(instance, Mapping) else None
        if not isinstance(implementations, Mapping) or len(implementations) != 1:
            return None
        return str(next(iter(implementations)))

    async def _seed_solution(self, session: HarborSession) -> None:
        """Place the task's reference solution for the oracle, as Harbor's oracle copies it into the environment.

        Only an agent instance implemented by ``harbor_oracle_agent`` gets it: the seeded sandbox receives the
        snapshot's ``solution/`` at ``/solution`` as root, opened to every user (the oracle runs ``solve.sh`` as
        the task user, or the image's default user, who could not read a ``0700`` script or enter a private
        folder the author's checkout kept), and the session carries the host-resolved ``[solution].env`` for the
        harness's commands (Harbor 0.22.0 ``agents/oracle.py`` uploads the folder and resolves ``solution.env``
        the same way). Any other harness never sees the solution. A task without ``solution/`` uploads nothing;
        the oracle then reports it unvalidated.
        """
        implementation = self._agent_implementation()
        if implementation is None:
            # Not an oracle run as far as this server can tell. Say so once per episode: an oracle wired through
            # a block this lookup cannot read would otherwise report every task as unvalidated.
            LOGGER.warning(
                "agent %r has no single responses_api_agents implementation in the run config, so no solution is "
                "placed for it (the oracle, if that is what it is, will report every task as unvalidated)",
                self.config.agent_server.name,
            )
            return
        if implementation != ORACLE_AGENT:
            return
        task = session.task
        session.agent_env = resolve_env_templates(task.config.solution.env)
        solution = task.path / "solution"
        if solution.is_dir():
            await upload_dir(session.sandbox, solution, SOLUTION_DIR, make_readable=True)

    async def _sandbox_access(self, session: HarborSession) -> SandboxAccess:
        return SandboxAccess(
            connection=DirectSandboxConnection(
                provider_config_ref=self.config.sandbox_provider,
                descriptor=await session.sandbox.serialize(),
            ),
            workdir=session.workdir,
            user=session.task.user,
            env=session.agent_env,
        )
