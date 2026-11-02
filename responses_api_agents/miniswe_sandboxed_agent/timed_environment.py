# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run mini-SWE with a local environment that times each bash command where it executes.

The harness runs this file inside the sandbox in place of ``python -m minisweagent.run.mini``. Running it as a
script puts its directory on the agent process's ``sys.path`` only, so mini-SWE can load the environment class
below by name while the commands it runs keep their environment unchanged.
"""

import time

from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted


class TimedLocalEnvironment(LocalEnvironment):
    """mini-SWE's local environment, adding when each command it ran started and ended.

    mini-SWE copies ``output["extra"]`` into the saved tool observation. A submission raises ``Submitted`` after
    its command exited 0 and saves no observation, so its timing and tool call id go on the exit message that
    mini-SWE saves instead. Any other exception propagates unchanged and leaves the timing unknown.
    """

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        started_at = time.time()
        started = time.monotonic()
        try:
            output = super().execute(action, cwd, timeout=timeout)
        except Submitted as submitted:
            extra = submitted.messages[0].setdefault("extra", {})
            extra.update(_timing(started_at, started), tool_call_id=action.get("tool_call_id"))
            raise
        output["extra"] = {**output.get("extra", {}), **_timing(started_at, started)}
        return output


def _timing(started_at: float, started: float) -> dict:
    return {"started_at": started_at, "completed_at": time.time(), "duration_ms": (time.monotonic() - started) * 1000}


if __name__ == "__main__":
    from minisweagent.run.mini import app

    app()
