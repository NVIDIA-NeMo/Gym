# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shell tool's commands, run in the episode's task sandbox through ``AsyncSandbox``.

Each command runs in a fresh bash in the task directory, in a process group of its own. A deadline kills that group,
and so does agent close for a command still running. The command writes its output to a file in the session
directory, so a background process it leaves (a task service) cannot hold the provider's exec open. Only bash and
``head``, ``tail``, ``wc`` and ``sleep`` are needed in the task image.
"""

import math
from dataclasses import dataclass
from shlex import quote
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox, upload_text


SESSION_ROOT = "/tmp/nemo-gym-crewai-sessions"
RUNNER_FILE = "run_command.sh"
_STATUS_MARKER = "__nemo_gym_crewai_command_status__"
# Time the runner gets beyond the command's deadline to kill it and print its output.
_EXEC_GRACE_SECONDS = 30

# $1 command file, $2 deadline in whole seconds, $3 output file, $4 pid file, $5 output byte limit, $6 status marker.
# `set -m` gives the command, and the watchdog, a process group of its own. The watchdog marks a timeout before it
# kills the command's group; after the command exits, the watchdog's group is killed instead and the command's
# background processes keep running. Agent close kills the group named by a pid file that is still present.
RUNNER_SCRIPT = r"""set -m
bash "$1" </dev/null >"$3" 2>&1 &
command_pid=$!
echo "$command_pid" >"$4"
(sleep "$2"; : >"$4.timeout"; kill -KILL -- "-$command_pid") </dev/null >/dev/null 2>&1 &
watchdog_pid=$!
wait "$command_pid"
status=$?
kill -KILL -- "-$watchdog_pid" 2>/dev/null
rm -f "$4"
size=$(wc -c <"$3")
if [ "$size" -gt "$5" ]; then
  head -c "$(($5 / 2))" "$3"
  printf '\n[... %s bytes of output omitted ...]\n' "$((size - $5))"
  tail -c "$(($5 / 2))" "$3"
else
  cat "$3"
fi
timed_out=0
if [ -e "$4.timeout" ]; then timed_out=1; fi
printf '\n%s %s %s\n' "$6" "$status" "$timed_out"
"""


class ShellRunnerError(RuntimeError):
    """The sandbox did not run the command to completion: the provider failed, or the runner left no status."""


@dataclass(frozen=True)
class ShellResult:
    output: str
    exit_code: int
    timed_out: bool

    def for_model(self, timeout_seconds: float) -> str:
        """The text the model sees: the command's combined output, then how it ended."""
        if self.timed_out:
            ending = f"[command timed out after {timeout_seconds:g} s and was killed]"
        else:
            ending = f"[exit code: {self.exit_code}]"
        return f"{self.output}\n{ending}" if self.output else ending


class SandboxShell:
    """Runs shell-tool commands in a task sandbox the session borrows; it never starts or stops the sandbox."""

    def __init__(
        self,
        sandbox: AsyncSandbox,
        *,
        workdir: str,
        env: dict[str, str],
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> None:
        self.sandbox = sandbox
        self.workdir = workdir
        # The owner's environment for the harness's commands, over the sandbox's own.
        self.env = env
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        # Outside the task directory, so nothing of the agent's lands in what the verifier reads.
        self.session_dir = f"{SESSION_ROOT}/{uuid4().hex}"
        self._commands = 0

    async def setup(self) -> None:
        await upload_text(self.sandbox, path=f"{self.session_dir}/{RUNNER_FILE}", text=RUNNER_SCRIPT)

    async def run(self, command: str) -> ShellResult:
        self._commands += 1
        stem = f"{self.session_dir}/command-{self._commands}"
        # Send the command as a file: model text is never interpolated into the exec command line.
        await upload_text(self.sandbox, path=f"{stem}.sh", text=command)
        deadline = max(1, math.ceil(self.timeout_seconds))
        argv = [
            "bash",
            f"{self.session_dir}/{RUNNER_FILE}",
            f"{stem}.sh",
            str(deadline),
            f"{stem}.out",
            f"{stem}.pid",
            str(self.max_output_bytes),
            _STATUS_MARKER,
        ]
        result = await self.sandbox.exec(
            " ".join(quote(arg) for arg in argv),
            cwd=self.workdir,
            env=self.env or None,
            timeout_s=deadline + _EXEC_GRACE_SECONDS,
        )
        stdout = result.stdout or ""
        output, marker, status = stdout.rpartition(f"\n{_STATUS_MARKER} ")
        if result.error_type is not None or not marker:
            raise ShellRunnerError(
                f"sandbox exec ended without a command status (return code {result.return_code}, "
                f"error {result.error_type}): {(result.stderr or '').strip()[-500:]}"
            )
        exit_code, timed_out = status.split()
        return ShellResult(output=output, exit_code=int(exit_code), timed_out=timed_out == "1")

    async def release(self, *, timeout_seconds: float) -> None:
        """Kill commands still running and remove the session's files, leaving the sandbox to its owner."""
        directory = quote(self.session_dir)
        script = (
            f'for pid_file in {directory}/*.pid; do [ -e "$pid_file" ] || continue; '
            'kill -KILL -- "-$(cat "$pid_file")" 2>/dev/null; done; '
            f"rm -rf {directory}"
        )
        result = await self.sandbox.exec(script, timeout_s=timeout_seconds)
        if result.error_type is not None or result.return_code != 0:
            raise ShellRunnerError(
                f"could not release the shell session (return code {result.return_code}, "
                f"error {result.error_type}): {(result.stderr or '').strip()[-500:]}"
            )
