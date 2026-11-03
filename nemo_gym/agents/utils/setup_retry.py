# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retry a setup command that is safe to run twice when the sandbox provider fails to run it.

A sandbox that was just created or restored can refuse a connection for a few seconds even after it first answered
(a 502 from the gateway), so a seed's first commands can fail although the sandbox is fine. The provider does not
retry a command itself: the error can come after the command started, and running an arbitrary command twice is not
safe. A caller whose command is idempotent (a read, a ``mkdir -p``, an install that skips what is present) retries it
here.
"""

import asyncio
import logging

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
    **exec_kwargs: object,
) -> SandboxExecResult:
    """Run ``command``, retrying when ``sandbox.exec`` raises; a returned result, any exit code, ends the retries.

    The delay doubles after each failed attempt (2 s, then 4 s). The last error is raised once ``attempts`` run out.
    Use only for commands that are safe to run twice.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await sandbox.exec(command, **exec_kwargs)
        except Exception as error:
            if attempt == attempts:
                raise
            LOG.warning("%s failed (attempt %d/%d): %r; retrying", what, attempt, attempts, error)
            await asyncio.sleep(delay_s * 2 ** (attempt - 1))
    raise AssertionError("unreachable")  # pragma: no cover
