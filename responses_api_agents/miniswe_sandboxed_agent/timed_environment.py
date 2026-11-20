# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run mini-SWE with a local environment that records each bash command where it executes.

The harness runs this file inside the sandbox in place of ``python -m minisweagent.run.mini``. Running it as a
script puts its directory on the agent process's ``sys.path`` only, so mini-SWE can load the environment class
below by name, and the environment can import the staged ``tool_recording`` module, while the commands it runs keep
their environment unchanged.
"""

import socket
from uuid import uuid4

from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.exceptions import Submitted


class TimedLocalEnvironmentConfig(LocalEnvironmentConfig):
    tool_events_path: str = ""
    invocation_id: str = ""


class TimedLocalEnvironment(LocalEnvironment):
    """mini-SWE's local environment, recording each command it runs with the shared tool call recorder.

    With ``tool_events_path`` set, each command's start is appended before it runs and its finish after it returns,
    so a command that is still running when the process dies keeps its start. A submission raises ``Submitted``
    only after its command exited 0, so it is recorded as completed. Any other exception propagates unchanged and
    is recorded as failed.
    """

    def __init__(self, *, config_class: type = TimedLocalEnvironmentConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        self.recorder = None
        if self.config.tool_events_path:
            from tool_recording import ClockContext, ToolCallRecorder

            self.recorder = ToolCallRecorder(
                path=self.config.tool_events_path,
                producer_id=f"miniswe:{uuid4().hex}",
                invocation_id=self.config.invocation_id,
                agent_instance=None,
                clock=ClockContext(location="sandbox", owner=socket.gethostname().replace("/", "_") or "miniswe"),
                boundary="executor",
                timing_source="miniswe.environment",
            )

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        if self.recorder is None:
            return super().execute(action, cwd, timeout=timeout)
        call_id = action.get("tool_call_id")
        with self.recorder.call(native_call_id=call_id, tool_call_id=call_id, tool_name="bash") as handle:
            try:
                output = super().execute(action, cwd, timeout=timeout)
            except Submitted:
                handle.set_outcome("completed")
                raise
            handle.set_outcome(_status(output))
        return output


def _status(output: dict) -> str:
    if output.get("extra", {}).get("exception_type") == "TimeoutExpired":
        return "timeout"
    return "failed" if output.get("returncode") else "completed"


if __name__ == "__main__":
    from minisweagent.run.mini import app

    app()
