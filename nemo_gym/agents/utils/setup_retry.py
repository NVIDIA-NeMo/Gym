# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retry a setup command that is safe to run twice when the sandbox provider fails to run it.

A sandbox that was just created or restored can refuse a connection for a few seconds even after it first answered,
so a seed's first commands can fail although the sandbox is fine. The provider does not retry a command itself: the
error can come after the command started, and running an arbitrary command twice is not safe. A caller whose command
is idempotent (a read, a ``mkdir -p``, an install that skips what is present) retries it here.
"""

import asyncio
import logging
from time import monotonic

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult


LOG = logging.getLogger(__name__)


async def exec_idempotent(
    sandbox: AsyncSandbox,
    command: str,
    *,
    what: str,
    attempts: int = 3,
    delay_s: float = 2.0,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: int | float | None = None,
) -> SandboxExecResult:
    """Run ``command``, retrying when ``sandbox.exec`` raises; a returned result, any exit code, ends the retries.

    ``cwd`` and ``env`` are passed to ``sandbox.exec``. ``timeout_s`` is the budget of all attempts and the waits
    between them together: each attempt gets what is left of it, and once it is spent the last error is raised. The
    wait doubles after each failed attempt (2 s, then 4 s). The last error is also raised once ``attempts`` run out.

    Use only for commands that are safe to run twice, and at the same time: the error can arrive after the command
    started, so a retry may begin while the first attempt is still running in the sandbox.
    """
    deadline = None if timeout_s is None else monotonic() + timeout_s
    for attempt in range(1, attempts + 1):
        remaining = None if deadline is None else deadline - monotonic()
        try:
            return await sandbox.exec(command, cwd=cwd, env=env, timeout_s=remaining)
        except Exception as error:
            wait = delay_s * 2 ** (attempt - 1)
            if attempt == attempts or (deadline is not None and monotonic() + wait >= deadline):
                raise
            LOG.warning("%s failed (attempt %d/%d): %r; retrying", what, attempt, attempts, error)
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover
