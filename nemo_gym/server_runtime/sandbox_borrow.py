# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a borrower uses a sandbox another server lends it: :func:`borrow`.

Every sandbox has one owner, the server that started it, and only the owner stops it (see "Sandbox owner" in the
glossary). A borrower, an agent whose seed carries a ``sandbox_access`` or a Resources Server grading in the sandbox,
runs commands and transfers files in it, and releases its client when it is done. Lent access is always operate scope.

No provider enforces scope yet, so code that connects to a sandbox some other way can still stop it. Enforcement
waits on provider leases; until then every borrower should connect through :func:`borrow` (a repository check that
keeps connections inside :func:`borrow` and ``OwnedSandboxes`` is planned, not in place).
"""

import logging
from collections.abc import Mapping
from typing import Any, NoReturn

import anyio

from nemo_gym.config.global_config import get_global_config_dict
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.api import STOP_TIMEOUT_S, AsyncSandbox, SandboxPty
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxPtySession
from nemo_gym.sandbox.providers.registry import create_provider


LOGGER = logging.getLogger(__name__)


class BorrowedSandboxError(RuntimeError):
    """A borrower tried what only the sandbox's owner does: start, stop, pause, resume or snapshot it."""


class _BorrowedPty(SandboxPty):
    """Terminal sessions of a borrowed sandbox, with the access's ``env`` under each session's own."""

    def __init__(self, sandbox: "BorrowedSandbox") -> None:
        super().__init__(sandbox)
        self._borrowed = sandbox

    async def create(
        self, command: str | None = None, *, env: dict[str, str] | None = None, **kwargs: Any
    ) -> SandboxPtySession:
        merged = {**self._borrowed._access_env, **(env or {})}
        return await super().create(command, env=merged or None, **kwargs)


class BorrowedSandbox(AsyncSandbox):
    """A lent sandbox: commands, terminals and file transfers work, the owner's lifecycle operations refuse.

    Commands and terminal sessions run as the access's ``user`` in its ``workdir``, with its ``env`` under each
    call's own ``env``. That includes a harness's own setup, such as installing its runtime: a command that must run
    as another user names it. :meth:`stop`, :meth:`start`, :meth:`pause`, :meth:`resume` and :meth:`snapshot` raise
    :class:`BorrowedSandboxError`. :meth:`disconnect` releases the client, shielded from cancellation and bounded as
    ``AsyncSandbox.disconnect`` is, and leaving an ``async with`` block disconnects instead of stopping. Sidecar
    commands run in another container and take none of the access's settings.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._access_env: dict[str, str] = {}
        self._lent: SandboxAccess | None = None
        self.pty = _BorrowedPty(self)

    def _refuse(self, operation: str) -> NoReturn:
        raise BorrowedSandboxError(
            f"this sandbox is borrowed: only the server that lent it may {operation} it; release the client with "
            "disconnect()"
        )

    async def stop(self, *, timeout_s: float | None = STOP_TIMEOUT_S) -> None:
        """Refuse: the owner stops the sandbox. Release this client with :meth:`disconnect`."""
        self._refuse("stop")

    async def start(self, *args: Any, **kwargs: Any) -> "AsyncSandbox":
        """Refuse: a borrower never creates a sandbox."""
        self._refuse("start")

    async def start_with_setup(self, *args: Any, **kwargs: Any) -> "AsyncSandbox":
        """Refuse: a borrower never creates a sandbox."""
        self._refuse("start")

    async def pause(self) -> None:
        """Refuse: pausing is the owner's, for example at a checkpoint."""
        self._refuse("pause")

    async def resume(self) -> None:
        """Refuse: resuming is the owner's."""
        self._refuse("resume")

    async def snapshot(self, *, name: str | None = None) -> str:
        """Refuse: the owner snapshots the sandbox at a checkpoint."""
        self._refuse("snapshot")

    async def serialize(self, *, scope: str | None = None) -> dict[str, Any]:
        """A descriptor with operate scope; :meth:`access` wraps it for a harness that borrows it elsewhere.

        Raises:
            ValueError: If ``scope`` asks for more than operate scope.
        """
        if scope not in (None, "operate"):
            raise ValueError(f"a borrower can pass on only operate scope, not {scope!r}")
        return await super().serialize(scope="operate")

    async def access(self) -> SandboxAccess:
        """The access this sandbox was lent with, for a harness that borrows it again, for example in a subprocess.

        That borrower calls :func:`borrow` with it, and with ``named_configs`` holding the provider block that
        ``access.connection.provider_config_ref`` names, since a subprocess has no global config.

        Raises:
            RuntimeError: For a sandbox that :func:`borrow` did not build.
        """
        if self._lent is None:
            raise RuntimeError("only a sandbox borrow() built has an access to pass on")
        return self._lent.model_copy(deep=True)

    async def exec(self, command: str, *, env: dict[str, str] | None = None, **kwargs: Any) -> SandboxExecResult:
        """Run ``command`` as :meth:`AsyncSandbox.exec` does, with the access's ``env`` under ``env``.

        The access's ``user`` is passed explicitly when the call names none, since a command that keeps background
        services runs through a provider call that does not read the sandbox's default user.
        """
        merged = {**self._access_env, **(env or {})}
        if kwargs.get("user") is None and self.default_user is not None:
            kwargs["user"] = self.default_user
        return await super().exec(command, env=merged or None, **kwargs)

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()


async def borrow(access: SandboxAccess, *, named_configs: Mapping[str, Any] | None = None) -> BorrowedSandbox:
    """Connect to the sandbox ``access`` lends, as its borrower.

    The provider is resolved from ``access.connection.provider_config_ref`` in ``named_configs``, the merged global
    config by default. The returned sandbox runs commands as ``access.user`` in ``access.workdir`` with
    ``access.env``, and its client is released with :meth:`BorrowedSandbox.disconnect`. A connect that fails or is
    cancelled closes the client it opened, shielded from the cancellation, and raises the connect's own error.

    Raises:
        ValueError: If the provider config cannot be resolved.
        RuntimeError: If the provider cannot reconnect to a sandbox.
    """
    configs = named_configs if named_configs is not None else get_global_config_dict()
    provider = create_provider(resolve_provider_config(access.connection.provider_config_ref, configs))
    try:
        sandbox = await BorrowedSandbox.connect(access.connection.descriptor, provider=provider, owns_provider=True)
    except BaseException:
        try:
            with anyio.CancelScope(shield=True), anyio.fail_after(STOP_TIMEOUT_S):
                await provider.aclose()
        except Exception as close_error:
            LOGGER.warning("the client of a failed borrow was not closed: %r", close_error)
        raise
    if not isinstance(sandbox, BorrowedSandbox):
        raise TypeError(f"connect() built a {type(sandbox).__name__}, not a BorrowedSandbox")
    sandbox.set_default_workdir(access.workdir)
    if access.user is not None:
        sandbox.set_default_user(access.user)
    sandbox._access_env = dict(access.env)
    sandbox._lent = access
    return sandbox
