# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve the SWE-rebench server's sandbox calls from inside the sandbox they would have reached.

Adapted from ``resources_servers/scale_swe/harbor/task-template/tests/nemo_gym/local_sandbox.py``. The server's own
modules reach their sandbox only through ``await sandbox.exec(command, timeout_s=..., cwd=...)`` and
``await sandbox.upload(local_path, remote_path)`` (``swebench/anti_cheat.py``, ``swebench/patch_capture.py``,
``swe_rebench/verification.py``), and read ``stdout``, ``stderr``, ``return_code`` and ``error_type`` from the
result. In a converted task they already run inside that sandbox, so this class runs the command as a local process
and copies the file locally. Unlike the Scale-SWE shim, ``exec`` honours ``cwd``: the anti-cheat scrub runs its
script from the repository directory.

The command runs in its own process group: on a timeout the whole group is killed, so a test process left behind in
that group cannot keep the output pipes open (one that started its own session with setsid can, and then exec waits
until it exits), and the result carries ``error_type="timeout"`` as a provider's would.
Standard library only; the task image's Python (or the pinned standalone Python) runs it.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from typing import Any, Callable


class ExecResult:
    """The fields of ``nemo_gym.sandbox.SandboxExecResult`` the server's modules read."""

    def __init__(self, stdout: str, stderr: str, return_code: int | None, error_type: str | None = None) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.error_type = error_type


def _text(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", errors="replace")


class LocalSandbox:
    """A stand-in for ``AsyncSandbox`` whose ``exec`` runs ``bash -c`` here.

    ``preexec_fn`` runs in the command's process before ``exec``: the grader passes one that drops the capabilities
    the agent's code could use to reach the reward (``grade.py``); the seed and the capture hook pass none.
    """

    def __init__(self, *, preexec_fn: Callable[[], None] | None = None) -> None:
        self.preexec_fn = preexec_fn

    async def upload(self, local_path: Any, remote_path: str, **_: Any) -> None:
        shutil.copyfile(str(local_path), remote_path)

    async def exec(self, command: str, timeout_s: float | None = None, cwd: str | None = None, **_: Any) -> ExecResult:
        process = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            start_new_session=True,
            preexec_fn=self.preexec_fn,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return ExecResult(_text(stdout), _text(stderr), None, "timeout")
        return ExecResult(_text(stdout), _text(stderr), process.returncode)
