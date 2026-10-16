# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stop a black-box agent harness at a checkpoint: its launch and every process the launch started.

A harness such as OpenCode runs as one long shell command in a sandbox and starts processes Gym never sees: an
installer, the harness itself, and the commands the model runs, which the harness may start in sessions of their
own or which may detach (``nohup server &``). When a checkpoint asks the run to park, all of them must stop before
the run parks, or they keep changing the sandbox while it is snapshotted and next to the relaunched harness.

The launch runs with ``NEMO_GYM_HARNESS_LAUNCH=<id>`` in its environment. Every process it starts inherits that,
wherever it is re-parented, so a stop finds them by reading ``/proc/<pid>/environ``; a walk of parent links adds
children that cleared their environment, and the processes an earlier pass found are remembered (with their start
time, so a reused PID is never signalled), because a child orphaned by that pass is neither tagged nor anyone's
child any more. A stop leaves a file the launch checks before it starts anything, so a stop that arrives before
the launch is not missed. A process that outlives SIGKILL is stuck in an uninterruptible system call (a directory
walk of a slow mount, for one): it already has the kill pending and runs none of its own code again, so the run
parks and the stop reports it. The sandbox needs Linux ``/proc`` and bash; no procps, no privileges.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from shlex import quote
from typing import Optional
from uuid import uuid4

from anyio import CancelScope

from nemo_gym._checkpoint.agent import LegacyRun
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult


LOGGER = logging.getLogger(__name__)

LAUNCH_TAG = "NEMO_GYM_HARNESS_LAUNCH"
# SIGINT lets the commands a harness runs clean up; SIGKILL then stops whatever is left, ended launch or not.
_GRACEFUL_S = 10.0
_KILL_S = 30.0
# After the SIGKILL pass, what is still found is killed again this many times, this far apart.
_SURVIVOR_CHECKS = 3
_SURVIVOR_CHECK_INTERVAL_S = 1.0


class HarnessInterrupted(Exception):
    """The launch was stopped because a checkpoint asked the run to park; the message says how the stop went."""


@dataclass(frozen=True)
class ResumeDecision:
    """Whether a relaunch continues the harness's saved session, and why."""

    resume: bool
    reason: str

    def __str__(self) -> str:
        return f"{'continuing the saved session' if self.resume else 'starting over'}: {self.reason}"


def stdout_only(command: str) -> str:
    """``command`` with its stderr discarded, for output that is parsed.

    Harnesses report progress on stderr (OpenCode's ``export`` prints "Exporting session: ..."), and a sandbox may
    return stderr inside stdout, as OpenSandbox's background executions do.
    """
    return f"{{ {command}\n}} 2>/dev/null"


class HarnessLaunch:
    """One launch of a harness command that a checkpoint can stop completely.

    Files live in ``state_dir`` only once a stop is requested, and :meth:`cleanup_command` removes them.
    """

    def __init__(self, *, state_dir: str = "/tmp") -> None:
        self.id = uuid4().hex
        self._stop_file = f"{state_dir}/nemo-gym-harness-{self.id}.stop"
        self._found_file = f"{state_dir}/nemo-gym-harness-{self.id}.found"

    def command(self, command: str) -> str:
        """``command``, tagged, and exiting with 130 instead of starting once a stop was requested."""
        script = f"test ! -e {self._stop_file} || exit 130\n{command}"
        return f"env {LAUNCH_TAG}={self.id} bash -c {quote(script)}"

    def stop_command(self, signal_name: str) -> str:
        """Request the stop, then send ``signal_name`` to every process of the launch; prints them, one per line."""
        return self._bash(
            f"touch {self._stop_file}\n"
            f"{self._find_launch_processes()}\n"
            f'for p in $found; do t=$(starttime "$p") && echo "$p:$t" >> {self._found_file}; done\n'
            'for p in $found; do read -r n 2>/dev/null < "/proc/$p/comm" && echo "$p $n"; done\n'
            f'for p in $found; do kill -{signal_name} "$p" 2>/dev/null; done\n'
            "true"
        )

    def survivors_command(self) -> str:
        """Print the processes of the launch that are still alive (zombies are dead), one per line."""
        return self._bash(
            f"{self._find_launch_processes()}\n"
            'for p in $found; do read -r s 2>/dev/null < "/proc/$p/stat" || continue; n=${s#*(}; n=${n%) *}; '
            's=${s##*) }; [ "${s%% *}" = Z ] || echo "$p $n"; done\n'
            "true"
        )

    def cleanup_command(self) -> str:
        """Remove the stop and found files; only once the launch has ended, while the stop file still guards it."""
        return f"rm -f {self._stop_file} {self._found_file}"

    def _find_launch_processes(self) -> str:
        """Set ``$found``: tagged processes, those earlier passes recorded (if still the same process), and all their
        descendants. ``/proc/<pid>/stat`` holds the parent PID and start time after the parenthesised name."""
        return (
            'starttime() { local s; read -r s 2>/dev/null < "/proc/$1/stat" || return 1; s=${s##*) }; set -- $s; '
            'echo "${20}"; }\n'
            'found=""\n'
            'for d in /proc/[0-9]*; do p=${d#/proc/}; [ "$p" = "$$" ] && continue; '
            f'while IFS= read -r -d "" kv; do [ "$kv" = "{LAUNCH_TAG}={self.id}" ] && '
            '{ found="$found $p"; break; }; done 2>/dev/null < "$d/environ"; done\n'
            f"if [ -f {self._found_file} ]; then while IFS=: read -r p t; do "
            f'[ -n "$p" ] && [ "$(starttime "$p")" = "$t" ] && found="$found $p"; done < {self._found_file}; fi\n'
            'frontier=$found\nwhile [ -n "$frontier" ]; do next=""; for d in /proc/[0-9]*; do '
            'read -r s 2>/dev/null < "$d/stat" || continue; s=${s##*) }; set -- $s; c=${d#/proc/}; '
            'for f in $frontier; do [ "$2" = "$f" ] && case " $found $next " in *" $c "*) ;; *) next="$next $c";; '
            'esac; done; done; found="$found $next"; frontier=$next; done'
        )

    @staticmethod
    def _bash(script: str) -> str:
        return f"bash -c {quote(script)}"


async def run_until_park_requested(
    sandbox: AsyncSandbox, command: str, legacy_run: Optional[LegacyRun], *, timeout_s: Optional[float]
) -> SandboxExecResult:
    """Run the harness ``command`` in ``sandbox``; with a checkpointed run, stop it when a checkpoint asks to park.

    Returns the command's result when it ends first. When the park request comes first, SIGINT goes to every
    process of the launch, then, after up to ``_GRACEFUL_S`` for the launch to end, SIGKILL to whatever is left,
    which is killed again until nothing is found; then raises :class:`HarnessInterrupted`.
    """
    if legacy_run is None:
        return await sandbox.exec(command=command, timeout_s=timeout_s)
    launch = HarnessLaunch()
    exec_task = asyncio.ensure_future(sandbox.exec(command=launch.command(command), timeout_s=timeout_s))
    park_task = asyncio.ensure_future(legacy_run.wait_park_requested())
    try:
        await asyncio.wait({exec_task, park_task}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        # Cancelled from outside (the caller disconnected): the caller's cleanup stops the sandbox.
        exec_task.cancel()
        raise
    finally:
        park_task.cancel()
    if exec_task.done():
        return exec_task.result()

    started = time.monotonic()
    ended_by = None
    signalled: dict[str, list[str]] = {}
    for signal_name, grace_s in (("INT", _GRACEFUL_S), ("KILL", _KILL_S)):
        try:
            result = await sandbox.exec(command=launch.stop_command(signal_name), timeout_s=30)
            signalled[signal_name] = (result.stdout or "").split("\n")
        except Exception as error:
            LOGGER.warning("Could not send SIG%s to harness launch %s: %r", signal_name, launch.id, error)
        if not exec_task.done():
            await asyncio.wait({exec_task}, timeout=grace_s)
        if exec_task.done() and ended_by is None:
            ended_by = f"SIG{signal_name} ended it after {time.monotonic() - started:.1f} s"

    lingering = await _kill_until_gone(sandbox, launch)
    if exec_task.done():
        exec_task.exception()  # retrieved: an interrupted command's failure is expected
        try:
            await sandbox.exec(command=launch.cleanup_command(), timeout_s=30)
        except Exception as error:
            LOGGER.warning("Could not remove the stop files of harness launch %s: %r", launch.id, error)
    else:
        exec_task.cancel()
        with CancelScope(shield=True):
            await asyncio.gather(exec_task, return_exceptions=True)
    stopped = {name: [line for line in lines if line] for name, lines in signalled.items()}
    summary = (
        f"launch {launch.id}: "
        f"{ended_by or f'the launch did not end within {_GRACEFUL_S + _KILL_S:.0f} s'}; signalled {stopped}"
    )
    if lingering:
        summary += f"; killed, still finishing a system call: {lingering}"
        LOGGER.warning("Harness %s", summary)
    raise HarnessInterrupted(summary)


async def _kill_until_gone(sandbox: AsyncSandbox, launch: HarnessLaunch) -> list[str]:
    """SIGKILL whatever is still found of the launch until nothing is; returns what is left after the last try.

    A process found again either started during the kill pass, and gets its SIGKILL now, or is stuck in an
    uninterruptible system call with the kill pending, and exits when the call returns.
    """
    lingering: list[str] = []
    for attempt in range(_SURVIVOR_CHECKS):
        if attempt:
            await asyncio.sleep(_SURVIVOR_CHECK_INTERVAL_S)
        try:
            result = await sandbox.exec(command=launch.survivors_command(), timeout_s=30)
            lingering = [line for line in (result.stdout or "").split("\n") if line]
            if not lingering:
                return []
            await sandbox.exec(command=launch.stop_command("KILL"), timeout_s=30)
        except Exception as error:
            LOGGER.warning("Could not check harness launch %s for survivors: %r", launch.id, error)
            return lingering
    return lingering
