# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sandboxes a resources server lends to a harness, one per resources session."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Literal

import anyio
from fastapi import HTTPException

from nemo_gym.client.server_client import get_server_url
from nemo_gym.config.global_config import get_global_config_dict
from nemo_gym.core.episode_types import TaskId
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.config import SandboxConfig, resolve_provider_config
from nemo_gym.server_runtime.base_resources_server import ResourcesSeedRejectedError
from nemo_gym.utils.network_policy import restricted_network_policy


LOGGER = logging.getLogger(__name__)
# Provider label values stay within the common 63-character limit.
_LABEL_LIMIT = 63
# A provider stop that hangs must not hold a close, a failed seed or the shutdown forever.
STOP_TIMEOUT_S = 120.0
# Nor may a start that hangs (a provider's create without a time limit of its own) hold a seed and its session's lock:
# the default bound covers a remote provider's create that pulls a large image.
START_TIMEOUT_S = 1800.0
# The supervisor gives a server about a second to exit, so shutdown stops sandboxes concurrently, this many at a time.
SHUTDOWN_STOP_CONCURRENCY = 64

# Egress of a lent sandbox: ``inherit`` keeps the provider's network, ``model_only`` allows only the model server, and
# ``model_and_tools`` also the servers whose tools the harness calls.
LentSandboxNetworkAccess = Literal["inherit", "model_only", "model_and_tools"]


class LentSandboxes:
    """Start, lend and stop one sandbox per resources session, from the server's sandbox configuration.

    A server's episode seed calls :meth:`lend` and its episode close calls :meth:`release`, both under the server's
    :class:`ResourcesSessionLocks` lock. The server keeps any other session state, such as task files it serves or a
    tool workspace, and calls :meth:`release_all` at shutdown. A server that grades in the lent sandbox after the
    episode reads it with :meth:`get` before the close.

    The harness gets an ``operate`` descriptor, which asks the provider to let it use the sandbox but not destroy it
    (providers without leases ignore the scope). A stop goes on when the request that asked for it is cancelled, and
    gives up after ``STOP_TIMEOUT_S``. A session that is never closed keeps its sandbox until shutdown, or until the
    provider reclaims it after ``sandbox_config.ttl_s``, so set ``ttl_s`` above the episode deadline plus the time
    that grading and cleanup take. A sandbox from a failed seed whose stop also fails is kept apart from the lent
    ones, and the session's next lend, its release or the shutdown tries the stop again.

    A lend may give the sandbox its own image, more environment variables and a setup to run against it before it is
    lent, such as linking a task's commands. Its start runs to its end once begun, within ``start_timeout_s``, even
    while the request that asked for it is being cancelled: a create cut in flight leaves a sandbox whose id the
    provider never returned, which nothing here could stop. A lend cancelled meanwhile lends nothing; it stops the
    sandbox it started. Once :meth:`release_all` has begun, no lend starts a sandbox.

    With ``network_access="model_only"``, each sandbox is created with an egress rule that allows only the host of
    ``model_server``, the model server that a harness in the sandbox calls. ``model_and_tools`` also allows the hosts
    of ``tool_servers``, the servers whose tools the harness calls over MCP, such as this resources server. Every port
    of an allowed host is open, so other services on that host stay reachable. The harness that borrows the sandbox
    keeps this network; it cannot change it.
    """

    def __init__(
        self,
        sandbox_provider: str,
        sandbox_config: SandboxConfig,
        *,
        image: str | None,
        workdir: str,
        network_access: LentSandboxNetworkAccess = "inherit",
        model_server: str | None = None,
        tool_servers: Sequence[str] = (),
        start_timeout_s: float = START_TIMEOUT_S,
    ) -> None:
        if network_access != "inherit":
            if not model_server:
                raise ValueError(
                    f"network_access {network_access} needs model_server, the model server the sandbox calls"
                )
            if "network_policy" in sandbox_config.provider_options:
                raise ValueError("Set network_access or sandbox_config.provider_options.network_policy, not both")
        if network_access == "model_and_tools" and not tool_servers:
            raise ValueError(
                "network_access model_and_tools needs tool_servers, the servers whose tools the harness calls"
            )
        if network_access != "model_and_tools" and tool_servers:
            raise ValueError("tool_servers applies only to network_access model_and_tools")
        self._provider = sandbox_provider
        self._config = sandbox_config
        self._image = image
        self._workdir = workdir
        self._network_access = network_access
        self._model_server = model_server
        self._tool_servers = tuple(tool_servers)
        self._start_timeout_s = start_timeout_s
        self._sandboxes: dict[str, AsyncSandbox] = {}
        # Sandboxes of failed seeds whose stop failed too. They are never lent; a later stop of each is retried.
        self._failed_seed_sandboxes: dict[str, AsyncSandbox] = {}
        # The sessions whose sandbox is starting, each with an event set once its lend has lent or stopped it.
        self._starting: dict[str, asyncio.Event] = {}
        self._releasing_all = False

    def get(self, resources_session_id: str) -> AsyncSandbox | None:
        """Return the sandbox lent for ``resources_session_id``, or ``None``."""
        return self._sandboxes.get(resources_session_id)

    async def lend(
        self,
        resources_session_id: str,
        task_id: TaskId,
        *,
        files: Mapping[str, str | bytes] | None = None,
        image: str | None = None,
        env: Mapping[str, str] | None = None,
        setup: Callable[[AsyncSandbox], Awaitable[None]] | None = None,
    ) -> SandboxAccess:
        """Return the access a harness needs to the session's sandbox, and start the sandbox on the first call.

        ``files`` maps absolute paths in the sandbox to their contents; it can be empty. ``image`` replaces the
        configured image for this sandbox, ``env`` adds environment variables to the configured ones, and ``setup``
        runs against the started sandbox, its files delivered, before it is lent. A repeated call, as for a repeated
        seed, returns the same sandbox. If the new sandbox cannot start, receive its files, be set up or be described
        within ``start_timeout_s``, or the lend is cancelled meanwhile, it is stopped and the error is raised. If that
        stop fails too, the sandbox is kept for a later stop: the session's next lend stops it before it starts another
        one, and a failed stop there fails the lend. Once :meth:`release_all` has begun, a lend starts nothing and
        answers 503, which a caller may retry against another server.
        """
        sandbox = self._sandboxes.get(resources_session_id)
        if sandbox is not None:
            return await self._access(sandbox)
        self._refuse_after_release_all()
        await self._stop_failed_seed_sandbox(resources_session_id)
        named_configs = get_global_config_dict()
        spec = self._config.spec(
            self._provider,
            image=image or self._image,
            workdir=self._workdir,
            metadata={"task_id": task_id.task_id[:_LABEL_LIMIT]},
            files=files,
            named_configs=named_configs,
        )
        if env:
            spec = replace(spec, env={**spec.env, **env})
        provider_config = resolve_provider_config(self._provider, named_configs)
        if self._network_access != "inherit":
            spec = replace(
                spec,
                provider_options={**spec.provider_options, "network_policy": self._restricted_policy(provider_config)},
            )
        # Checked again after the awaits above, and from here to the start nothing awaits, so release_all either sees
        # this start or comes before it.
        self._refuse_after_release_all()
        sandbox = AsyncSandbox(provider_config)
        started = self._starting[resources_session_id] = asyncio.Event()
        try:
            # The start runs to its end once begun (the class docstring); start() and start_with_setup() stop the
            # sandbox themselves when delivering its files or its setup fails.
            with anyio.CancelScope(shield=True), anyio.fail_after(self._start_timeout_s):
                if setup is None:
                    await sandbox.start(spec)
                else:
                    await sandbox.start_with_setup(spec, setup)
                access = await self._access(sandbox)
            await anyio.lowlevel.checkpoint()  # a lend cancelled meanwhile lends nothing
            self._refuse_after_release_all()  # release_all began during the start: it would not stop this sandbox
        except BaseException as error:
            try:
                await _stop(sandbox)
            except Exception:
                LOGGER.exception(
                    "Failed to stop the sandbox of a failed seed for resources session %s; a later lend, release or "
                    "shutdown tries again",
                    resources_session_id,
                )
                self._failed_seed_sandboxes[resources_session_id] = sandbox
            else:
                if isinstance(error, anyio.get_cancelled_exc_class()) and sandbox.sandbox_id is not None:
                    LOGGER.info(
                        "Stopped sandbox %s of resources session %s: its seed was cancelled while it started",
                        sandbox.sandbox_id,
                        resources_session_id,
                    )
            raise
        finally:
            del self._starting[resources_session_id]
            started.set()
        self._sandboxes[resources_session_id] = sandbox
        return access

    def _refuse_after_release_all(self) -> None:
        if self._releasing_all:  # release_all would not find the sandbox
            raise HTTPException(status_code=503, detail="The server is shutting down and lends no more sandboxes")

    async def release(self, resources_session_id: str) -> None:
        """Stop the session's sandbox; safe to repeat. A failed stop raises and keeps it, so a close can retry.

        It also stops a sandbox that a failed seed of the session left running.
        """
        await self._stop_failed_seed_sandbox(resources_session_id)
        sandbox = self._sandboxes.get(resources_session_id)
        if sandbox is None:
            return
        await _stop(sandbox)
        self._sandboxes.pop(resources_session_id, None)

    async def release_all(self, *, timeout_s: float | None = None) -> dict[str, list[str]]:
        """Stop every sandbox, as at server shutdown: the lent ones, those failed seeds left, and those still starting
        once their lends stop them. From now on no lend starts a sandbox. A failed stop is logged and the rest go on.

        With ``timeout_s``, the time the server has to exit, the stops still running then are cut. Returns the
        sessions whose sandbox may outlive the process, by kind, and logs them for a reaper: ``lent`` (a stop that
        failed or was cut), ``failed_seeds`` (a failed seed's sandbox whose stop failed again) and ``starting`` (a start
        still in flight: its lend stops the sandbox once the provider returns it, if the process is still running).
        """
        self._releasing_all = True
        stopping = asyncio.Semaphore(SHUTDOWN_STOP_CONCURRENCY)

        async def release(resources_session_id: str) -> None:
            async with stopping:
                try:
                    await self.release(resources_session_id)
                except Exception:
                    LOGGER.exception("Failed to stop the sandbox lent for resources session %s", resources_session_id)

        async def release_all() -> None:
            # A lend still starting stops its own sandbox once its start returns (it sees release_all), so its wait
            # comes first: its sandbox is then stopped or kept for this stop.
            await asyncio.gather(*(started.wait() for started in list(self._starting.values())))
            sessions = dict.fromkeys([*self._sandboxes, *self._failed_seed_sandboxes])
            await asyncio.gather(*(release(resources_session_id) for resources_session_id in sessions))

        try:
            await asyncio.wait_for(release_all(), timeout_s)
        except TimeoutError:
            pass
        left = {
            "lent": sorted(self._sandboxes),
            "failed_seeds": sorted(self._failed_seed_sandboxes),
            "starting": sorted(self._starting),
        }
        if any(left.values()):
            LOGGER.error(
                "%d lent sandboxes may still run after the release of all of them; a reaper must stop them. "
                "Resources sessions with a lent sandbox: %s. Failed seeds: %s. Starting: %s",
                sum(len(sessions) for sessions in left.values()),
                left["lent"],
                left["failed_seeds"],
                left["starting"],
            )
        return left

    async def _stop_failed_seed_sandbox(self, resources_session_id: str) -> None:
        """Stop the sandbox a failed seed of the session left running, if any. A failed stop raises and keeps it."""
        sandbox = self._failed_seed_sandboxes.get(resources_session_id)
        if sandbox is None:
            return
        await _stop(sandbox)
        self._failed_seed_sandboxes.pop(resources_session_id, None)

    def _restricted_policy(self, provider_config: Mapping[str, object]) -> dict[str, object]:
        """The egress rule for ``network_access``, refused (422) where it cannot be enforced.

        It allows the model server's host, and for ``model_and_tools`` the tool servers' hosts too.
        """
        (provider_name,) = provider_config
        servers = [self._model_server, *self._tool_servers]
        try:
            return restricted_network_policy(provider_name, [get_server_url(server) for server in servers])
        except ValueError as error:
            # A provider without network policies or a server on a loopback address: no retry can fix it.
            raise ResourcesSeedRejectedError(f"Cannot lend a {self._network_access} sandbox: {error}") from error

    async def _access(self, sandbox: AsyncSandbox) -> SandboxAccess:
        return SandboxAccess(
            connection=DirectSandboxConnection(
                # The harness may operate the sandbox but not destroy it: this server stops it at close. Providers
                # without leases ignore the scope.
                provider_config_ref=self._provider,
                descriptor=await sandbox.serialize(scope="operate"),
            ),
            workdir=self._workdir,
        )


async def _stop(sandbox: AsyncSandbox) -> None:
    """Stop ``sandbox`` within ``STOP_TIMEOUT_S``, even while the request that owns it is being cancelled.

    A client that disconnects cancels its request handler and every later await in it, so an unshielded stop would
    never run, and the sandbox would outlive every reference to it.
    """
    with anyio.CancelScope(shield=True), anyio.fail_after(STOP_TIMEOUT_S):
        await sandbox.stop()
