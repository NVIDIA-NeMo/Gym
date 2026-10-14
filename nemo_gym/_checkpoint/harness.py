# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pause or stop a black-box agent harness at a checkpoint: its launch and every process the launch started.

A harness such as OpenCode runs as one long shell command in a sandbox and starts processes Gym never sees: an
installer, the harness itself, and the commands the model runs, which the harness may start in sessions of their
own or which may detach (``nohup server &``). When a checkpoint asks the run to park, all of them must stop
changing the sandbox before the run parks, or they change it while it is snapshotted.

A run that can pause sends SIGSTOP to every process of the launch, parks while they are all stopped, and sends
SIGCONT once the checkpoint opens: the same harness process continues where it was, with no relaunch. Nothing
running in the sandbox changes while it is paused, so the snapshot holds that moment, and a restore after a crash
relaunches the harness on it as after a stop. A pause that cannot be confirmed (a process neither stopped nor dead
after the checks) falls back to the stop below. Time spent paused still counts toward the harness's own timers,
such as a per-command timeout.

A stop ends the launch for good instead: SIGINT, then SIGKILL, and the run relaunches the harness after the resume.

The launch runs with ``NEMO_GYM_HARNESS_LAUNCH=<id>`` in its environment. Every process it starts inherits that,
wherever it is re-parented, so a stop finds them by reading ``/proc/<pid>/environ``; a walk of parent links adds
children that cleared their environment, and the processes an earlier pass found are remembered (with their start
time, so a reused PID is never signalled), because a child orphaned by that pass is neither tagged nor anyone's
child any more. A stop leaves a file the launch checks before it starts anything, so a stop that arrives before
the launch is not missed. A process that outlives SIGKILL is stuck in an uninterruptible system call (a directory
walk of a slow mount, for one): it already has the kill pending and runs none of its own code again, so the run
parks and the stop reports it. The sandbox needs Linux ``/proc`` and bash; no procps, no privileges.

A launch can end just before the park request while its result is still on its way (OpenSandbox reports a
finished execution up to about 30 s late). The launch records its root process when it starts and replaces that
record with ``ended`` when its command returns, so the first stop pass can tell a launch that ended from one that
runs, or never started: an ended launch keeps its result instead of being interrupted and relaunched.

A launch whose caller is cancelled, such as a run whose client went away, is killed the same way, shielded and
time-limited, before the cancellation goes on: nothing else stops it once the sandbox outlives the run.
"""

import asyncio
import logging
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from shlex import quote
from typing import Awaitable, Callable, Literal, Optional
from uuid import uuid4

from anyio import CancelScope, move_on_after

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
# How long the result of a launch that had already ended may take to arrive.
_LATE_RESULT_S = 60.0
# How long the kill of a cancelled launch may hold up the cancellation.
_CANCEL_KILL_S = 30.0
# A pause checks this many times, this far apart, that every process of the launch has stopped.
_PAUSE_CHECKS = 50
_PAUSE_CHECK_INTERVAL_S = 0.1
# Tries to continue a paused launch before it is stopped instead.
_CONTINUE_ATTEMPTS = 3
# What the first stop pass prints, instead of signalling, for a launch whose command had already returned.
_ENDED = "NEMO_GYM_HARNESS_ENDED"


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
    """One launch of a harness command that a checkpoint can pause or stop completely.

    Files live in ``state_dir`` only once a pause or stop is requested, and :meth:`cleanup_command` removes them.
    """

    def __init__(self, *, state_dir: str = "/tmp") -> None:
        self.id = uuid4().hex
        self._stop_file = f"{state_dir}/nemo-gym-harness-{self.id}.stop"
        self._pause_file = f"{state_dir}/nemo-gym-harness-{self.id}.pause"
        self._found_file = f"{state_dir}/nemo-gym-harness-{self.id}.found"
        self._root_file = f"{state_dir}/nemo-gym-harness-{self.id}.root"

    def command(self, command: str) -> str:
        """``command``, tagged, exiting with 130 instead of starting once a stop was requested, and waiting
        instead of starting while a pause is.

        The root file holds the launch's ``pid:starttime`` while ``command`` runs and ``ended`` once it returned.
        The pause wait comes after the root file is written: a pause that finds no root file can count on the
        launch to wait.
        """
        script = (
            f"test ! -e {self._stop_file} || exit 130\n"
            f'read -r s < /proc/$$/stat; s=${{s##*) }}; set -- $s; echo "$$:${{20}}" > {self._root_file}\n'
            f"while test -e {self._pause_file}; do sleep 0.2; done\n"
            f"{command}\n"
            f"rc=$?; echo ended > {self._root_file}; exit $rc"
        )
        return f"env {LAUNCH_TAG}={self.id} bash -c {quote(script)}"

    def stop_command(self, signal_name: str, *, unless_ended: bool = False) -> str:
        """Request the stop, then send ``signal_name`` to every process of the launch; prints them, one per line.

        With ``unless_ended``, a launch whose command had already returned is not signalled: the command prints
        ``NEMO_GYM_HARNESS_ENDED`` instead.
        """
        ended_check = (
            f'[ "$(cat {self._root_file} 2>/dev/null)" = ended ] && {{ echo {_ENDED}; exit 0; }}\n'
            if unless_ended
            else ""
        )
        return self._bash(
            f"touch {self._stop_file}\n"
            f"{ended_check}"
            f"{self._find_launch_processes()}\n"
            f'for p in $found; do t=$(starttime "$p") && echo "$p:$t" >> {self._found_file}; done\n'
            'for p in $found; do read -r n 2>/dev/null < "/proc/$p/comm" && echo "$p $n"; done\n'
            f'for p in $found; do kill -{signal_name} "$p" 2>/dev/null; done\n'
            "true"
        )

    def pause_command(self) -> str:
        """Request a pause, then SIGSTOP every process of the launch until each is stopped or dead.

        Prints ``PAUSED <n>`` once all are (``n`` stopped, not counting those already dead), ``NOT_STARTED`` for a
        launch that will wait instead of starting, ``NEMO_GYM_HARNESS_ENDED`` for one whose command had already
        returned (nothing is signalled), or ``PENDING`` and what did not stop within the checks. A stopped process
        forks nothing, so the checks settle.
        """
        record = f'for p in $found; do t=$(starttime "$p") && echo "$p:$t" >> {self._found_file}; done'
        return self._bash(
            f"touch {self._pause_file}\n"
            f"r=$(cat {self._root_file} 2>/dev/null)\n"
            f'[ "$r" = ended ] && {{ echo {_ENDED}; exit 0; }}\n'
            '[ -z "$r" ] && { echo NOT_STARTED; exit 0; }\n'
            "n=0\n"
            "while :; do\n"
            f"{self._find_launch_processes()}\n"
            f"{record}\n"
            '  pending=""; count=0\n'
            '  for p in $(echo $found | tr " " "\\n" | sort -nu); do '
            'read -r s 2>/dev/null < "/proc/$p/stat" || continue; s=${s##*) }; st=${s%% *}; '
            'case "$st" in T|t) count=$((count+1));; Z|X) ;; '
            '*) pending="$pending $p:$st"; kill -STOP "$p" 2>/dev/null;; esac; done\n'
            '  [ -z "$pending" ] && { echo "PAUSED $count"; exit 0; }\n'
            f'  n=$((n+1)); [ "$n" -gt {_PAUSE_CHECKS} ] && {{ echo "PENDING$pending"; exit 0; }}\n'
            f"  sleep {_PAUSE_CHECK_INTERVAL_S}\n"
            "done"
        )

    def continue_command(self) -> str:
        """Withdraw the pause and SIGCONT every process of the launch; prints ``CONTINUED <n>``."""
        return self._bash(
            f"rm -f {self._pause_file}\n"
            f"{self._find_launch_processes()}\n"
            'n=0; for p in $(echo $found | tr " " "\\n" | sort -nu); do '
            'kill -CONT "$p" 2>/dev/null && n=$((n+1)); done\n'
            'echo "CONTINUED $n"'
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
        """Remove the launch's files; only once the launch has ended, while the stop file still guards it."""
        return f"rm -f {self._stop_file} {self._pause_file} {self._found_file} {self._root_file}"

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
    sandbox: AsyncSandbox,
    command: str,
    legacy_run: Optional[LegacyRun],
    *,
    timeout_s: Optional[float],
    park: Optional[Callable[[], Awaitable[None]]] = None,
) -> SandboxExecResult:
    """Run the harness ``command`` in ``sandbox``; with a checkpointed run, pause or stop it when a checkpoint asks
    to park.

    Returns the command's result when it ends first, or when it had ended by the time the park request came and
    its result was still on its way. With ``park``, a park request pauses the launch, then awaits ``park()``, which
    records the run's bookmark and returns once the checkpoint opens, then continues the launch and keeps waiting
    for it. Without ``park``, or when the pause cannot be confirmed, SIGINT goes to every process of the launch,
    then, after up to ``_GRACEFUL_S`` for the launch to end, SIGKILL to whatever is left, which is killed again
    until nothing is found; then raises :class:`HarnessInterrupted`. Cancelled, or when ``park()`` fails, it kills
    the launch before it re-raises.
    """
    if legacy_run is None:
        return await sandbox.exec(command=command, timeout_s=timeout_s)
    launch = HarnessLaunch()
    exec_task = asyncio.ensure_future(sandbox.exec(command=launch.command(command), timeout_s=timeout_s))
    try:
        while True:
            park_task = asyncio.ensure_future(legacy_run.wait_park_requested())
            try:
                await asyncio.wait({exec_task, park_task}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                park_task.cancel()
            if exec_task.done():
                result = exec_task.result()
                await _remove_files(sandbox, launch)
                return result
            if park is None or await _pause(sandbox, launch) != "paused":
                return await _stop(sandbox, launch, exec_task)
            try:
                await park()
            except Exception:
                await _kill(sandbox, launch)
                exec_task.cancel()
                raise
            if not await _continue(sandbox, launch):
                return await _stop(sandbox, launch, exec_task)
    except asyncio.CancelledError:
        # Nothing else stops the launch: the sandbox outlives a checkpointed run, which never stops it.
        await _kill(sandbox, launch)
        exec_task.cancel()
        raise


async def _kill(sandbox: AsyncSandbox, launch: HarnessLaunch) -> None:
    """SIGKILL the launch, shielded and time-limited; SIGKILL also ends a paused process."""
    with move_on_after(_CANCEL_KILL_S, shield=True):
        try:
            await sandbox.exec(command=launch.stop_command("KILL"), timeout_s=30)
            await _kill_until_gone(sandbox, launch)
        except Exception as error:
            LOGGER.warning("Could not kill harness launch %s: %r", launch.id, error)


async def _pause(sandbox: AsyncSandbox, launch: HarnessLaunch) -> Literal["paused", "ended", "failed"]:
    """Pause the launch. ``ended``: its command had returned, and the stop keeps its late result. ``failed``: the
    pause was not confirmed and is withdrawn, so the stop can end the launch."""
    started = time.monotonic()
    try:
        result = await sandbox.exec(command=launch.pause_command(), timeout_s=30)
        lines = [line for line in (result.stdout or "").split("\n") if line]
    except Exception as error:
        lines = [f"error: {error!r}"]
    if _ENDED in lines:
        return "ended"
    if lines and (lines[-1].startswith("PAUSED ") or lines[-1] == "NOT_STARTED"):
        LOGGER.info(
            "Harness launch %s paused for a checkpoint in %.2f s: %s", launch.id, time.monotonic() - started, lines[-1]
        )
        return "paused"
    LOGGER.warning("Could not pause harness launch %s (%s); stopping it instead", launch.id, "; ".join(lines))
    await _continue(sandbox, launch)
    return "failed"


async def _continue(sandbox: AsyncSandbox, launch: HarnessLaunch) -> bool:
    """Continue a paused launch, shielded: a launch left paused would hold its run until the command's timeout."""
    with CancelScope(shield=True):
        for attempt in range(_CONTINUE_ATTEMPTS):
            if attempt:
                await asyncio.sleep(_SURVIVOR_CHECK_INTERVAL_S)
            try:
                result = await sandbox.exec(command=launch.continue_command(), timeout_s=30)
            except Exception as error:
                LOGGER.warning("Could not continue harness launch %s: %r", launch.id, error)
                continue
            LOGGER.info(
                "Harness launch %s continued after the checkpoint: %s", launch.id, (result.stdout or "").strip()
            )
            return True
    return False


async def _stop(
    sandbox: AsyncSandbox, launch: HarnessLaunch, exec_task: "asyncio.Future[SandboxExecResult]"
) -> SandboxExecResult:
    """Stop the launch inside the sandbox's checkpoint polling scope: the stopped launch's end, and the late
    result of one that had ended, are seen within the scope's short poll interval instead of the command's
    backed-off one (up to 30 s on OpenSandbox)."""
    with _checkpoint_polling(sandbox):
        return await _stop_launch(sandbox, launch, exec_task)


def _checkpoint_polling(sandbox: AsyncSandbox) -> AbstractContextManager[None]:
    polling = getattr(sandbox, "checkpoint_exec_polling", None)
    return polling() if callable(polling) else nullcontext()


async def _stop_launch(
    sandbox: AsyncSandbox, launch: HarnessLaunch, exec_task: "asyncio.Future[SandboxExecResult]"
) -> SandboxExecResult:
    started = time.monotonic()
    ended_by = None
    signalled: dict[str, list[str]] = {}
    for signal_name, grace_s in (("INT", _GRACEFUL_S), ("KILL", _KILL_S)):
        try:
            result = await sandbox.exec(
                command=launch.stop_command(signal_name, unless_ended=signal_name == "INT"), timeout_s=30
            )
            lines = (result.stdout or "").split("\n")
        except Exception as error:
            LOGGER.warning("Could not send SIG%s to harness launch %s: %r", signal_name, launch.id, error)
            lines = []
        if _ENDED in lines:
            # The command had returned before the park request; its result is still on its way.
            await asyncio.wait({exec_task}, timeout=_LATE_RESULT_S)
            if exec_task.done():
                LOGGER.info("Harness launch %s had ended before the park request; keeping its result", launch.id)
                result = exec_task.result()
                await _remove_files(sandbox, launch)
                return result
            LOGGER.warning(
                "Harness launch %s had ended but its result did not arrive in %.0f s; stopping it as interrupted",
                launch.id,
                _LATE_RESULT_S,
            )
            continue
        signalled[signal_name] = lines
        if not exec_task.done():
            await asyncio.wait({exec_task}, timeout=grace_s)
        if exec_task.done() and ended_by is None:
            ended_by = f"SIG{signal_name} ended it after {time.monotonic() - started:.1f} s"

    lingering = await _kill_until_gone(sandbox, launch)
    if exec_task.done():
        exec_task.exception()  # retrieved: an interrupted command's failure is expected
        await _remove_files(sandbox, launch)
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


async def _remove_files(sandbox: AsyncSandbox, launch: HarnessLaunch) -> None:
    try:
        await sandbox.exec(command=launch.cleanup_command(), timeout_s=30)
    except Exception as error:
        LOGGER.warning("Could not remove the files of harness launch %s: %r", launch.id, error)


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
