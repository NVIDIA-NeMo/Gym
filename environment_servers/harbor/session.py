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

import json
import logging
import os
import re
import shlex
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

from fastapi import HTTPException

from environment_servers.harbor.entrypoint import EntrypointLaunch, _cancel_entrypoint
from environment_servers.harbor.users import _is_root
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.data import ORACLE_AGENT, HarborMCPServer, HarborTask, HarborTurn, is_ignored, merged_mcp_servers
from nemo_gym.sandbox import (
    AsyncSandbox,
    AsyncSandboxCompose,
    DirectSandboxConnection,
    SandboxAccess,
    exec_as_root,
    upload_dir,
)


LOGGER = logging.getLogger(__name__)

VERIFIER_LOGS_DIR = "/logs/verifier"
AGENT_LOGS_DIR = "/logs/agent"
ARTIFACTS_DIR = "/logs/artifacts"
# Where the oracle finds the task's reference solution: Harbor's oracle copies ``solution/`` here
# (0.22.0 ``agents/oracle.py``, ``EnvironmentPaths.solution_dir``) and runs ``solve.sh`` from it.
SOLUTION_DIR = "/solution"

TASK_CONTEXT_DIR = "/tmp/.nemo-gym"
TASK_CONTEXT_FILE = f"{TASK_CONTEXT_DIR}/task.json"


@dataclass
class HarborSession:
    """One episode's sandbox, from seed until the final cleanup stops it."""

    task: HarborTask
    taskset: str
    identity: tuple[EpisodeId, TaskId]
    sandbox: AsyncSandbox
    workdir: str
    # The episode key the sandbox is owned under (see ``_sandboxes``).
    session_id: str
    provider_ref: str
    compose: AsyncSandboxCompose | None = None
    # The image entrypoint the seed started beside the keepalive, cancelled before the sandbox is stopped.
    entrypoint: EntrypointLaunch | None = None
    # Separate verification stops the agent's sandbox before test.sh runs; cleanup must not stop it twice.
    agent_sandbox_stopped: bool = False
    # In a Compose group the agent's own container is stopped before the sidecars' hooks run (see ``stop_main``).
    main_stopped: bool = False
    # Environment the harness runs its commands with, handed over in its sandbox access: the host-resolved
    # ``[solution].env`` when the harness is the oracle, nothing otherwise.
    agent_env: dict[str, str] = field(default_factory=dict)
    # The user the harness runs as, handed over in its sandbox access: the sandbox's default as the seed installed
    # it from the task's own ``HarborTask.user`` (``HarborUsersMixin._set_default_user``), then as a turn of a
    # multi-step task installed the step's ``[steps.agent].user`` (``HarborEnvironmentServer._apply_turn_user``).
    # Kept as installed: a uid is the name it resolved to, or the integer when it has no passwd entry
    # (OpenSandbox's native ``uid=`` exec path; the string would be taken for an account name). ``None`` is the
    # image's default.
    agent_user: str | int | None = None
    # That user as the task or the step wrote it (``HarborTask.user``, then the step's), which the next turn's own
    # is compared with to decide whether to switch (``_prepare_turn``); the installed form may spell it differently.
    requested_user: str | None = None
    # The image's uid-to-name table as the seed looked it up (``HarborUsersMixin._user_name``), reused when a
    # later turn installs a numeric user.
    user_names: dict[str, str] = field(default_factory=dict)
    # The sandbox's default exec user as the seed installed it (``None`` is the image's own): what a turn of a
    # multi-step task puts back around its verifier (``HarborSessionMixin._as_seed_default_user``), the way
    # Harbor's ``with_default_user`` around the agent phase is undone before the shared verifier runs.
    seed_default_user: str | int | None = None
    # ``uid:gid`` of the working directory as the seed left it, read before the first per-turn user switch and
    # restored when a later turn switches back to the image's default user (``_apply_turn_user``).
    workdir_owner: str | None = None
    # The harness is the Harbor oracle: the reference solution is placed at /solution for it, once at seed for a
    # single-turn task and before each turn of a multi-step task (``_seed_solution``, ``_place_turn_solution``).
    solution_for_oracle: bool = False

    def service(self, name: str | None) -> AsyncSandbox:
        """The sandbox of a Compose service; ``None`` or ``"main"`` is the agent's own."""
        if name in (None, "main"):
            return self.sandbox
        if self.compose is None or name not in self.compose.services:
            raise KeyError(f"No Compose service {name!r} in this task")
        return self.compose.services[name]

    async def stop_main(self) -> None:
        """Stop the agent's own container, leaving a Compose group's sidecars running.

        Harbor collects in phases: the main container's hooks and artifacts, then its stop, then the
        sidecars' hooks and artifacts, so nothing the agent left running can change a sidecar's
        state while it is read. Without a group this is the whole agent side.
        """
        if self.compose is None:
            await self.stop_agent_side()
            return
        if not self.main_stopped:
            await self.sandbox.stop()
            self.main_stopped = True

    async def stop_agent_side(self) -> None:
        """Stop the agent's sandbox, and its sidecars when the task is a Compose group."""
        if self.agent_sandbox_stopped:
            return
        await _cancel_entrypoint(self.entrypoint)
        if self.compose is not None:
            # A main container stopped by ``stop_main`` is skipped: a sandbox stop is idempotent.
            await self.compose.stop()
        else:
            await self.sandbox.stop()
        # Only a stop that succeeded releases the episode's final cleanup from stopping again.
        self.agent_sandbox_stopped = True


_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_templates(env: dict[str, str], secrets: Mapping[str, str] | None = None) -> dict[str, str]:
    """Resolve Harbor's whole-value host templates without changing the stored task.

    ``secrets`` (the server's operator secrets) are consulted before the process environment; they are
    passed only when resolving the verifier process's environment.
    """
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = _ENV_TEMPLATE.fullmatch(value)
        if match is None:
            resolved[key] = value
        elif secrets and match.group(1) in secrets:
            resolved[key] = secrets[match.group(1)]
        elif match.group(1) in os.environ:
            resolved[key] = os.environ[match.group(1)]
        elif match.group(2) is not None:
            resolved[key] = match.group(2)
        else:
            raise HTTPException(422, f"Environment variable {match.group(1)!r} is required by the task")
    return resolved


def _task_sandbox_env(task: HarborTask) -> dict[str, str]:
    """The agent sandbox's environment: the task's image layer as recorded, under the host-resolved ``[environment.env]``.

    The loader folds the Dockerfile's ``ENV`` (resolved as ``docker build`` resolves it, so a single-quoted
    ``'${X}'`` is the literal text) and the base image's ``Env`` into ``task.env`` under ``[environment.env]``.
    Only that task.toml layer is a host template surface: Harbor resolves ``task_env_config.env`` alone
    (0.22.0 environments/base.py 272-285, docker/docker.py 258) and the image layer is baked into the image.
    Resolving the merged ``task.env`` would expand a Dockerfile literal against the host, or refuse the task
    when the host has no variable of that name.
    """
    return {**task.env, **resolve_env_templates(task.config.environment.env)}


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
        await self._own_workdir(sandbox, workdir, task.user, create=True)
        environment_dir = task.path / "environment"
        if (
            task.config.environment.docker_image
            and environment_dir.is_dir()
            and not any(
                (environment_dir / name).exists()
                for name in ("Dockerfile", "docker-compose.yaml", "docker-compose.yml")
            )
        ):
            await upload_dir(sandbox, environment_dir, workdir, ignore=is_ignored)
        # Harbor mounts /logs into the agent's container; tasks may write to /logs/artifacts during the
        # episode. Best effort: an image that runs as a non-root user without the capability to switch
        # to root cannot create top-level folders, and the agent may still not need them.
        logs = await exec_as_root(
            sandbox,
            f"mkdir -p {AGENT_LOGS_DIR} {VERIFIER_LOGS_DIR} {ARTIFACTS_DIR} "
            f"&& chmod 777 {AGENT_LOGS_DIR} {VERIFIER_LOGS_DIR} {ARTIFACTS_DIR}",
            timeout_s=60,
        )
        if logs.return_code != 0:
            LOGGER.warning(
                f"{task.task_id}: could not create /logs folders in the agent sandbox: {logs.stderr or logs.stdout}"
            )
        return workdir

    @staticmethod
    async def _own_workdir(sandbox: AsyncSandbox, workdir: str, user: str | None, *, create: bool = False) -> None:
        """Hand the working directory to ``user`` (``chown``, as root), creating it first with ``create``.

        Only a directory the task or image named below the root is created and, with a user override, handed
        over; an image whose WORKDIR is `/` must keep root's ownership of it, and root itself needs no chown. A
        multi-step task runs this again for a turn whose ``[steps.agent].user`` differs from the task's, so the
        step's user owns the directory it works in, as a build's ``WORKDIR`` under that ``USER`` would.
        """
        if workdir == "/":
            return
        commands = [f"mkdir -p {shlex.quote(workdir)}"] if create else []
        if not _is_root(user):
            commands.append(f"chown {shlex.quote(str(user))} {shlex.quote(workdir)}")
        if not commands:
            return
        result = await exec_as_root(sandbox, " && ".join(commands), timeout_s=60)
        if result.return_code != 0:
            raise RuntimeError(f"Could not prepare {workdir}: {result.stderr or result.stdout}")

    @staticmethod
    async def _workdir_owner(session: HarborSession) -> str:
        """The ``uid:gid`` that owns the session's working directory now, read as root (``/`` is left alone)."""
        if session.workdir == "/":
            return "0:0"
        result = await exec_as_root(session.sandbox, f"stat -c %u:%g {shlex.quote(session.workdir)}", timeout_s=60)
        owner = (result.stdout or "").strip().splitlines()[-1:] if result.return_code == 0 else []
        if not owner:
            raise RuntimeError(f"Could not read the owner of {session.workdir}: {result.stderr or result.stdout}")
        return owner[0]

    @staticmethod
    async def _restore_workdir_owner(session: HarborSession) -> None:
        """Give the working directory back to the owner the seed left (:meth:`_workdir_owner`), as root."""
        if session.workdir == "/" or session.workdir_owner is None:
            return
        command = f"chown {shlex.quote(session.workdir_owner)} {shlex.quote(session.workdir)}"
        result = await exec_as_root(session.sandbox, command, timeout_s=60)
        if result.return_code != 0:
            raise RuntimeError(f"Could not restore the owner of {session.workdir}: {result.stderr or result.stdout}")

    @staticmethod
    @contextmanager
    def _as_seed_default_user(session: HarborSession) -> Iterator[None]:
        """The sandbox's default exec user put back to the seed's for the block, and the turn's own after.

        A multi-step task's verifier (its collect hooks and ``test.sh``) runs as its own user, never the step's
        ``[steps.agent].user``: the explicit user :func:`verifier._verifier_user` names, or, where that is the
        sandbox's own default (a root image with no ``[agent].user``), the default the seed installed, not the
        one the turn switched to. Harbor's ``with_default_user`` around the agent phase ends before its shared
        verifier runs under the verifier's own; this is that restore and re-apply.
        """
        sandbox = session.sandbox
        turn_user = sandbox.default_user
        if turn_user == session.seed_default_user:
            yield
            return
        sandbox.set_default_user(session.seed_default_user)
        try:
            yield
        finally:
            sandbox.set_default_user(turn_user)

    def _task_mcp_servers(self, task: HarborTask) -> list[HarborMCPServer]:
        """The MCP servers the harness gets for ``task``: its ``[[environment.mcp_servers]]`` with the job-level
        overlay's ``mcp_servers`` merged in by name, the overlay's winning, as Harbor merges the task's with the
        job's ``agents[].mcp_servers`` (:func:`~nemo_gym.data.tasks.harbor.compose_overlay.merged_mcp_servers`)."""
        return merged_mcp_servers(task.config.environment.mcp_servers, self.config.compose_overlay)

    async def _write_task_context(self, sandbox: AsyncSandbox, task: HarborTask) -> None:
        """Leave the task's in-sandbox tool declarations where a harness can read them.

        ``/tmp/.nemo-gym/task.json`` carries ``mcp_servers`` (Harbor's ``[[environment.mcp_servers]]`` with the
        job-level overlay's entries merged in by name, :meth:`_task_mcp_servers`) and ``skills_dir``. Harnesses
        that know how to talk to task MCP servers from inside the sandbox read it at session start; nothing in
        the episode protocol needs to change for that.
        """
        environment = task.config.environment
        servers = self._task_mcp_servers(task)
        if not servers and not environment.skills_dir:
            return
        context = {
            "mcp_servers": [server.model_dump(mode="json") for server in servers],
            "skills_dir": environment.skills_dir,
        }
        payload = shlex.quote(json.dumps(context))
        # Root first, falling back to the default user on sandboxes that cannot switch identities; /tmp
        # is writable either way, and the modes only need the agent's own user to read the file.
        result = await exec_as_root(
            sandbox,
            f"mkdir -p {TASK_CONTEXT_DIR} && printf %s {payload} > {TASK_CONTEXT_FILE} "
            f"&& (chmod 755 {TASK_CONTEXT_DIR} && chmod 644 {TASK_CONTEXT_FILE} || true)",
            timeout_s=60,
        )
        if result.return_code:
            raise RuntimeError(f"Could not write {TASK_CONTEXT_FILE}: {result.stderr or result.stdout}")

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
        snapshot's ``solution/`` at ``/solution`` as root, readable by the task user, and the session carries
        the host-resolved ``[solution].env`` for the harness's commands (Harbor 0.22.0 ``agents/oracle.py``
        uploads the folder and resolves ``solution.env`` the same way). Any other harness never sees the
        solution. A task without ``solution/`` uploads nothing; the oracle then reports it unvalidated. A
        multi-step task's solutions are per step: the seed only marks the session, and
        :meth:`_place_turn_solution` puts each turn's own folder there before the turn, as Harbor's oracle
        resolves the step's ``steps/<name>/solution/`` (falling back to the task's) on every step.
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
        session.solution_for_oracle = True
        if task.turns:
            return
        solution = task.path / "solution"
        if solution.is_dir():
            await upload_dir(session.sandbox, solution, SOLUTION_DIR, make_readable=True, ignore=is_ignored)

    async def _place_turn_solution(self, session: HarborSession, turn: HarborTurn) -> None:
        """Put the turn's reference solution at ``/solution`` for the oracle, replacing the previous turn's.

        ``/solution`` is emptied first (as root), so a step without any solution leaves the oracle nothing to
        run (it reports the turn unvalidated, where Harbor's oracle fails the step) instead of the previous
        step's script; then the step's own ``steps/<name>/solution/``, or the task's ``solution/`` when the step
        has none (:meth:`HarborTask.turn_solution_dir`), is uploaded as the seed uploads a single-turn task's.
        Nothing happens for any other harness.
        """
        if not session.solution_for_oracle:
            return
        reset = await exec_as_root(session.sandbox, f"rm -rf -- {SOLUTION_DIR}", timeout_s=60)
        if reset.return_code != 0:
            raise RuntimeError(f"Could not empty {SOLUTION_DIR}: {reset.stderr or reset.stdout}")
        solution = session.task.turn_solution_dir(turn)
        if solution.is_dir():
            await upload_dir(session.sandbox, solution, SOLUTION_DIR, make_readable=True, ignore=is_ignored)

    async def _sandbox_access(self, session: HarborSession) -> SandboxAccess:
        return SandboxAccess(
            connection=DirectSandboxConnection(
                provider_config_ref=session.provider_ref,
                descriptor=await session.sandbox.serialize(),
            ),
            workdir=session.workdir,
            user=session.agent_user,
            env=session.agent_env,
        )
