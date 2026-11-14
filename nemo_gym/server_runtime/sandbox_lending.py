# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sandboxes a resources server lends to a harness, one per resources session."""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Literal

import anyio

from nemo_gym.config.global_config import get_global_config_dict
from nemo_gym.core.episode_types import TaskId
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.config import SandboxConfig, resolve_provider_config
from nemo_gym.server_runtime.sandbox_network import SandboxNetwork


LOGGER = logging.getLogger(__name__)
# Provider label values stay within the common 63-character limit.
_LABEL_LIMIT = 63
# A provider stop that hangs must not hold a close, a failed seed or the shutdown forever.
STOP_TIMEOUT_S = 120.0
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
    ) -> None:
        if network_access == "no_network":
            raise ValueError(
                "LentSandboxes cannot use no_network: the harness in a lent sandbox calls the model server"
            )
        self._network = SandboxNetwork(
            access=network_access,
            model_server=model_server if network_access != "inherit" else None,
            tool_servers=tuple(tool_servers),
        )
        self._network.check(sandbox_config)
        self._provider = sandbox_provider
        self._config = sandbox_config
        self._image = image
        self._workdir = workdir
        self._sandboxes: dict[str, AsyncSandbox] = {}
        # Sandboxes of failed seeds whose stop failed too. They are never lent; a later stop of each is retried.
        self._failed_seed_sandboxes: dict[str, AsyncSandbox] = {}

    def get(self, resources_session_id: str) -> AsyncSandbox | None:
        """Return the sandbox lent for ``resources_session_id``, or ``None``."""
        return self._sandboxes.get(resources_session_id)

    async def lend(
        self,
        resources_session_id: str,
        task_id: TaskId,
        *,
        files: Mapping[str, str | bytes] | None = None,
    ) -> SandboxAccess:
        """Return the access a harness needs to the session's sandbox, and start the sandbox on the first call.

        ``files`` maps absolute paths in the sandbox to their contents; it can be empty. A repeated call, as for a
        repeated seed, returns the same sandbox. If the new sandbox cannot start, receive its files or be described,
        it is stopped and the error is raised. If that stop fails too, the sandbox is kept for a later stop: the
        session's next lend stops it before it starts another one, and a failed stop there fails the lend.
        """
        sandbox = self._sandboxes.get(resources_session_id)
        if sandbox is not None:
            return await self._access(sandbox)
        await self._stop_failed_seed_sandbox(resources_session_id)
        named_configs = get_global_config_dict()
        provider_config = resolve_provider_config(self._provider, named_configs)
        spec = self._config.spec(
            self._provider,
            image=self._image,
            workdir=self._workdir,
            metadata={"task_id": task_id.task_id[:_LABEL_LIMIT]},
            files=files,
            network_policy=self._network.policy(provider_config),
            named_configs=named_configs,
        )
        sandbox = AsyncSandbox(provider_config)
        try:
            # start() stops the sandbox itself when delivering the files fails, but not under cancellation: its stop
            # is cancelled with the request. _stop() then finishes it, and does nothing after a stop that finished.
            await sandbox.start(spec)
            access = await self._access(sandbox)
        except BaseException:
            try:
                await _stop(sandbox)
            except Exception:
                LOGGER.exception(
                    "Failed to stop the sandbox of a failed seed for resources session %s; a later lend, release or "
                    "shutdown tries again",
                    resources_session_id,
                )
                self._failed_seed_sandboxes[resources_session_id] = sandbox
            raise
        self._sandboxes[resources_session_id] = sandbox
        return access

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

    async def release_all(self) -> None:
        """Stop every sandbox still lent, as at server shutdown. A failed stop is logged and the rest go on."""
        stopping = asyncio.Semaphore(SHUTDOWN_STOP_CONCURRENCY)

        async def release(resources_session_id: str) -> None:
            async with stopping:
                try:
                    await self.release(resources_session_id)
                except Exception:
                    LOGGER.exception("Failed to stop the sandbox lent for resources session %s", resources_session_id)

        sessions = dict.fromkeys([*self._sandboxes, *self._failed_seed_sandboxes])
        await asyncio.gather(*(release(resources_session_id) for resources_session_id in sessions))

    async def _stop_failed_seed_sandbox(self, resources_session_id: str) -> None:
        """Stop the sandbox a failed seed of the session left running, if any. A failed stop raises and keeps it."""
        sandbox = self._failed_seed_sandboxes.get(resources_session_id)
        if sandbox is None:
            return
        await _stop(sandbox)
        self._failed_seed_sandboxes.pop(resources_session_id, None)

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
