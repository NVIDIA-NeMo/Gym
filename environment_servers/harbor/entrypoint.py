# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the image ENTRYPOINT beside the provider's keepalive and gate the seed on healthchecks.

A Harbor image may carry an ENTRYPOINT that is not a keepalive. The server launches it in a
detached exec as the image's user, records its exit status in a runtime folder, polls it until
the healthcheck passes or the start timeout is reached, and after the rollout decides whether a
dead entrypoint masks the sample. Keepalive detection, the launcher script, the HEALTHCHECK to
Compose healthcheck translation, the launch bookkeeping dataclass and the exit and output
helpers are module level. The mixin holds the server methods that start, check, wait on and
collect the log of the entrypoint. The healthcheck wait predates the launcher, which is why the
module first appears at the milestone that added it.
"""

import asyncio

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.tasks.harbor import HarborTask


class HarborEntrypointMixin:
    """Run the image ENTRYPOINT beside the provider's keepalive and gate the seed on healthchecks.

    A Harbor image may carry an ENTRYPOINT that is not a keepalive. The server launches it in a
    detached exec as the image's user, records its exit status in a runtime folder, polls it until
    the healthcheck passes or the start timeout is reached, and after the rollout decides whether a
    dead entrypoint masks the sample. Keepalive detection, the launcher script, the HEALTHCHECK to
    Compose healthcheck translation, the launch bookkeeping dataclass and the exit and output
    helpers are module level. The mixin holds the server methods that start, check, wait on and
    collect the log of the entrypoint. The healthcheck wait predates the launcher, which is why the
    module first appears at the milestone that added it.

    """

    async def _wait_healthy(self, sandbox: AsyncSandbox, task: HarborTask) -> None:
        """Poll the task's ``[environment.healthcheck]`` until it passes or its retries run out."""
        check = task.config.environment.healthcheck
        if check is None:
            return
        loop = asyncio.get_running_loop()
        grace_until = loop.time() + check.start_period_sec
        failures = 0
        while True:
            # Decided before the attempt, as Harbor does: a check that starts inside the start period and
            # fails after it has ended is still a start-period attempt, not a counted failure.
            in_grace = loop.time() < grace_until
            result = await sandbox.exec(check.command, timeout_s=check.timeout_sec + 5)
            if result.return_code == 0:
                return
            if not in_grace:
                failures += 1
                if failures >= check.retries:
                    raise RuntimeError(f"Healthcheck failed {check.retries} times: {check.command!r}")
            await asyncio.sleep(check.start_interval_sec if in_grace else check.interval_sec)
