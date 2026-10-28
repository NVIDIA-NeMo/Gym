# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native mini-SWE subprocess clocks; this file runs inside the task sandbox."""

import os
import subprocess
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from minisweagent.environments.local import LocalEnvironment, _run


# All tool threads in this interpreter share one wall clock. Never claim it is
# synchronized with the Gym host, another sandbox, or a restarted interpreter.
CLOCK_DOMAIN = f"miniswe:{uuid4().hex}:unix"


class TimedLocalEnvironment(LocalEnvironment):
    """Keep native subprocess semantics and attach spawn-through-reap clocks."""

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        env = os.environ | self.config.env
        started_at, tick = time(), perf_counter()
        try:
            result = _run(command, cwd, env, timeout or self.config.timeout)
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as error:
            raw = getattr(error, "output", None)
            output = {
                "output": raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or ""),
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {error}",
                "extra": {"exception_type": type(error).__name__, "exception": str(error)},
            }
            # Spawn errors do not establish a process execution interval.
            if not isinstance(error, subprocess.TimeoutExpired):
                self._check_finished(output)
                return output
        completed_at, duration_ms = time(), (perf_counter() - tick) * 1000
        timing = {
            "started_at": started_at if completed_at >= started_at else None,
            "completed_at": completed_at if completed_at >= started_at else None,
            "duration_ms": duration_ms,
            "clock_domain": CLOCK_DOMAIN,
            "timing_source": "miniswe.local._run.spawn_to_reap",
            "duration_source": "miniswe.local._run.perf_counter",
            "execution_id": action.get("tool_call_id"),
        }
        output.setdefault("extra", {})["execution_timing"] = timing
        self._check_finished(output)
        return output
