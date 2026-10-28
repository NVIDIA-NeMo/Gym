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
import logging
import posixpath
import re
import shlex
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from environment_servers.harbor.overlay import OVERLAY_OUTPUT_TAIL, _output_tail, overlay_workdir_command
from environment_servers.harbor.users import _exec_as_root_user, _is_root
from nemo_gym.data import DEFAULT_SHELL, HarborHealthcheck, HarborTask
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, download_path, exec_as_root


LOGGER = logging.getLogger(__name__)

# Where the launcher keeps the image entrypoint's markers and log inside the sandbox, the way the Compose adapter
# keeps a service's; world-writable so an entrypoint that runs as the image's user can write them.
ENTRYPOINT_DIR = "/tmp/.nemo-gym-entrypoint"
ENTRYPOINT_LOG = f"{ENTRYPOINT_DIR}/log"
# The entrypoint's arguments: what Harbor's Compose files give it in place of CMD (``command: sh -c "sleep infinity"``).
ENTRYPOINT_ARGS = ("sh", "-c", "sleep infinity")
# How long the launcher waits for its started marker before the provider is judged unable to run it.
ENTRYPOINT_START_TIMEOUT_S = 60.0
ENTRYPOINT_POLL_INTERVAL_S = 0.5
# The entrypoint's output, collected best effort into the verification's logs folder, and the exit status
# it had recorded by the time the agent's sandbox was verified.
ENTRYPOINT_LOG_NAME = "entrypoint-stdout.txt"
VERIFIER_ENTRYPOINT_LOG_NAME = "verifier-entrypoint-stdout.txt"
ENTRYPOINT_EXIT_NAME = "entrypoint-exit.txt"
# The image entrypoint exited non-zero after the seed: during the rollout (when the exec holding it returned),
# or in the separate verifier's fresh sandbox. Under Harbor the container would have died and the trial errored.
ENTRYPOINT_FAILED_KIND = "harbor:entrypoint_failed"


class HarborEntrypointError(RuntimeError):
    """The image's ``ENTRYPOINT`` exited non-zero at start; the task cannot be set up."""


class HarborEntrypointUnavailable(RuntimeError):
    """The sandbox provider could not run the entrypoint launcher at all (not the entrypoint failing): retry."""


@dataclass
class EntrypointLaunch:
    """The image entrypoint running in a sandbox: its argv and the exec that holds it open."""

    argv: list[str]
    process: asyncio.Task
    # The entrypoint returned 0 without running its keepalive argument (warned once, then left alone).
    returned: bool = False
    # The provider ended the holding exec while no exit status was recorded (logged once; see _check_entrypoint).
    noticed: bool = False


_SHELLS = frozenset({"sh", "bash", "dash"})
_KEEPALIVE_PROGRAMS = frozenset({"sleep", "tail"})
# A shell script that is one sleep or tail command (``sleep infinity``, ``exec tail -f /dev/null``) or loops
# over a sleep. A compound script that starts with one (``sleep 5; service redis-server start``) does work and runs.
_KEEPALIVE_SCRIPT = re.compile(r"^\s*(?:exec\s+)?(?:sleep|tail)\s+[^;&|\n]*$")
_KEEPALIVE_LOOP = re.compile(r"^\s*while\s+(?:true|:)\s*;\s*do\s+sleep\s+\S+\s*;?\s*done\s*;?\s*$")


def is_keepalive_entrypoint(argv: list[str] | tuple[str, ...]) -> bool:
    """Whether ``argv`` only keeps a container alive, which the sandbox's own keepalive already does."""
    if not argv:
        return False
    if posixpath.basename(argv[0]) in _KEEPALIVE_PROGRAMS:
        return True
    if len(argv) == 3 and posixpath.basename(argv[0]) in _SHELLS and argv[1] == "-c":
        script = argv[2]
        return bool(_KEEPALIVE_SCRIPT.match(script) or _KEEPALIVE_LOOP.match(script))
    return False


def entrypoint_to_run(task: HarborTask, *, compose_managed: bool = False) -> list[str] | None:
    """The image ``ENTRYPOINT`` the seed starts beside the keepalive, or ``None`` when nothing is to run.

    Nothing runs in a sandbox the Compose adapter started (``compose_managed``: the agent's ``main`` of a
    Compose task, whose entrypoint the adapter ran), for an image whose recorded configuration has no
    entrypoint or none is recorded (a prebuilt ``docker_image``: today's behaviour), a Dockerfile that
    cleared it with ``ENTRYPOINT []``, and an entrypoint that is itself a keepalive. The decision is the
    sandbox's, not the task's: a separate verifier is always a plain sandbox of its own, under a Compose
    task too, so its image's entrypoint runs there (Harbor ``up --wait``s the verifier image on its own).
    """
    if compose_managed or not task.entrypoint:
        return None
    argv = list(task.entrypoint)
    return None if is_keepalive_entrypoint(argv) else argv


def entrypoint_script(argv: list[str], runtime: str = ENTRYPOINT_DIR) -> str:
    """The ``run.sh`` that marks the start, runs ``argv`` with Harbor's arguments and records the exit status."""
    command = shlex.join([*argv, *ENTRYPOINT_ARGS])
    return f"touch {runtime}/started\n{command}\nstatus=$?\nprintf '%s' \"$status\" > {runtime}/exit\n"


def image_healthcheck_settings(
    record: Mapping[str, Any] | None, shell: Sequence[str] | None = None
) -> HarborHealthcheck | None:
    """The recorded Dockerfile ``HEALTHCHECK`` as the settings the seed polls, or ``None`` when it has none.

    Docker records ``Test`` as ``["CMD", ...]`` or ``["CMD-SHELL", script]`` (``["NONE"]`` or empty
    disables it) and the durations in nanoseconds, absent when the Dockerfile left Docker's defaults.
    The daemon runs a ``CMD-SHELL`` script through the image's configured ``Shell`` (``shell``, the
    recorded ``HarborTask.image_shell``; ``/bin/sh -c`` when the image sets none: moby's
    ``daemon/health.go``), so the command carries that shell's whole argument list and the provider's
    own exec shell only starts it: ``SHELL ["/bin/bash", "-o", "pipefail", "-c"]`` fails ``false | true``
    here as it does under Docker. A ``CMD`` vector runs as written, with no shell of the image's.
    """
    if not record:
        return None
    test = record.get("Test") or []
    if not test or test[0] == "NONE":
        return None
    if test[0] == "CMD-SHELL" and len(test) == 2:
        command = shlex.join([*(shell or DEFAULT_SHELL), str(test[1])])
    elif test[0] == "CMD" and len(test) > 1:
        command = shlex.join(str(part) for part in test[1:])
    else:
        return None

    def seconds(key: str, default: float) -> float:
        value = record.get(key)
        return value / 1_000_000_000 if value else default

    return HarborHealthcheck(
        command=command,
        interval_sec=seconds("Interval", 30.0),
        timeout_sec=seconds("Timeout", 30.0),
        start_period_sec=seconds("StartPeriod", 0.0),
        start_interval_sec=seconds("StartInterval", 5.0),
        retries=int(record.get("Retries") or 3),
    )


async def _cancel_entrypoint(launch: EntrypointLaunch | None) -> None:
    """End the exec that holds the entrypoint open; the sandbox is stopped right after."""
    if launch is None:
        return
    launch.process.cancel()
    await asyncio.gather(launch.process, return_exceptions=True)


def _launch_outcome(launch: EntrypointLaunch) -> str | None:
    """How the exec holding the entrypoint ended, or ``None`` while it is still held open."""
    process = launch.process
    if not process.done():
        return None
    if process.cancelled():
        return "cancelled"
    error = process.exception()
    if error is not None:
        return f"{type(error).__name__}: {error}"
    result = process.result()
    return result.error_type or f"exit {result.return_code}"


def _launch_returned(launch: EntrypointLaunch) -> bool:
    """Whether the holding exec returned on its own: ``run.sh`` ran to its end, so the entrypoint had exited.

    A provider that capped the exec (docker's ``exec.default_timeout_s``, OpenSandbox's request timeout
    without ``background_exec``) reports ``error_type`` instead, and the entrypoint may well be running still.
    """
    process = launch.process
    if not process.done() or process.cancelled() or process.exception() is not None:
        return False
    return process.result().error_type is None


class HealthcheckFailed(RuntimeError):
    """A healthcheck that ran and never passed: ``retries`` consecutive failures after its start period, Harbor's
    ``HealthcheckError``. A sandbox that could not run the command raises its own error instead, never this one, so
    a caller that scores the check as the task's own failure catches this type alone."""


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

    async def _start_entrypoint(
        self,
        sandbox: AsyncSandbox,
        task: HarborTask,
        user: str | None,
        *,
        prepared_workdir: str | None = None,
        compose_managed: bool = False,
    ) -> EntrypointLaunch | None:
        """Start the image's ``ENTRYPOINT`` beside the keepalive, as Harbor's Compose start leaves it running.

        Harbor starts the image with ``command: sh -c "sleep infinity"``, which replaces CMD only, so the
        entrypoint is PID 1 with that as its arguments, as the image's user, in its working directory.
        Gym's providers start their own keepalive instead, so the same process is started here as a
        long-lived exec through a ``run.sh`` that records a started marker, the exit status and the
        output under :data:`ENTRYPOINT_DIR`, the way the Compose adapter starts a service. ``user`` is
        the image's user by name (:meth:`_image_user_name`); ``compose_managed`` says the adapter started this
        sandbox's service, entrypoint included, so nothing is started here (see :func:`entrypoint_to_run`).
        For a Dockerfile the loader resolved (not a
        built image, which holds its ``WorkingDir``) the entrypoint's working directory is created as
        ``docker build`` would when it is not ``prepared_workdir``, the one the seed made. The call
        returns once the marker exists and the first exit probe has passed (:meth:`_check_entrypoint`).
        A launcher that ends before the marker exists is classified as an overlay step is: the provider
        could not run it (:class:`HarborEntrypointUnavailable`, retried), or its own shell failed (a
        ``su`` or ``chdir`` for the image's user or directory: :class:`HarborEntrypointError`, terminal).
        """
        argv = entrypoint_to_run(task, compose_managed=compose_managed)
        if argv is None:
            return None
        made = await exec_as_root(sandbox, f"mkdir -p {ENTRYPOINT_DIR} && chmod 777 {ENTRYPOINT_DIR}", timeout_s=60)
        if made.return_code:
            raise HarborEntrypointUnavailable(f"could not create {ENTRYPOINT_DIR}: {made.stderr or made.stdout}")
        # The recorded User and WorkingDir, named explicitly: `[agent].user` and `[environment].workdir` shape
        # the sandbox's defaults for the commands run beside the entrypoint, not the entrypoint itself.
        cwd = task.entrypoint_workdir or "/"
        if task.built_image is None and task.entrypoint_workdir and task.entrypoint_workdir != prepared_workdir:
            # The Dockerfile's final WORKDIR, which `docker build` creates in the image; only a RUN line after it
            # made the overlay create it, and the seed makes the task's workdir, which may be another.
            workdir = await _exec_as_root_user(
                sandbox, overlay_workdir_command(cwd, user), configured_user=task.sandbox_user, spec_user=task.user
            )
            if workdir.error_type is not None:
                raise HarborEntrypointUnavailable(f"WORKDIR {cwd} could not be created: {workdir.error_type}")
            if workdir.return_code:
                raise HarborEntrypointError(
                    f"WORKDIR {cwd} for ENTRYPOINT {shlex.join(argv)} could not be created; "
                    f"output tail: {_output_tail(workdir)}"
                )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.sh"
            path.write_text(entrypoint_script(argv))
            await sandbox.upload(path, f"{ENTRYPOINT_DIR}/run.sh")
        command = f"sh {ENTRYPOINT_DIR}/run.sh >{ENTRYPOINT_LOG} 2>&1 </dev/null"
        if _is_root(user):
            run = _exec_as_root_user(
                sandbox, command, configured_user=task.sandbox_user, spec_user=task.user, cwd=cwd, timeout_s=None
            )
        else:
            run = sandbox.exec(command, cwd=cwd, timeout_s=None, user=user)
        # The exec is held open for as long as the provider allows: some providers reap background children as
        # soon as their launching shell exits. Docker caps it at `exec.default_timeout_s` and OpenSandbox without
        # `operations.background_exec` at `connection.request_timeout_s`; past that the process is untracked
        # (noticed and logged by `_check_entrypoint`). It is cancelled before the sandbox is stopped.
        launch = EntrypointLaunch(argv=argv, process=asyncio.create_task(run))
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + ENTRYPOINT_START_TIMEOUT_S
            while True:
                finished = launch.process.done()
                started = await sandbox.exec(f"test -f {ENTRYPOINT_DIR}/started", timeout_s=60)
                if started.return_code == 0:
                    break
                if finished:
                    error = launch.process.exception()
                    if error is not None:
                        raise HarborEntrypointUnavailable(f"the entrypoint launcher could not run: {error}") from error
                    outcome = launch.process.result()
                    if outcome.error_type is None and outcome.return_code != 0:
                        # `touch started` is run.sh's first line, in a world-writable folder: the launcher's own
                        # shell (`su`, `chdir`, `sh`) never reached the script. The image's fault, as a RUN
                        # line's non-zero exit is; another sandbox would repeat it.
                        raise HarborEntrypointError(
                            f"the launcher for ENTRYPOINT {shlex.join(argv)} (as {user or 'root'} in {cwd}) exited "
                            f"{outcome.return_code} before starting it; output tail: {_output_tail(outcome)}"
                        )
                    raise HarborEntrypointUnavailable(
                        f"the entrypoint launcher exited before starting: "
                        f"{outcome.error_type or outcome.stderr or outcome.stdout or 'no output'}"
                    )
                if loop.time() >= deadline:
                    raise HarborEntrypointUnavailable(
                        f"the entrypoint launcher did not start within {ENTRYPOINT_START_TIMEOUT_S:g}s"
                    )
                await asyncio.sleep(ENTRYPOINT_POLL_INTERVAL_S)
            await self._check_entrypoint(sandbox, task, launch)
        except BaseException:
            await _cancel_entrypoint(launch)
            raise
        return launch

    async def _entrypoint_exit_status(self, sandbox: AsyncSandbox) -> str | None:
        """What ``run.sh`` recorded as the entrypoint's exit status, or ``None`` while it has not exited."""
        probe = await sandbox.exec(
            f"if test -f {ENTRYPOINT_DIR}/exit; then cat {ENTRYPOINT_DIR}/exit; fi", timeout_s=60
        )
        if probe.return_code:
            raise HarborEntrypointUnavailable(
                f"cannot read the entrypoint's exit status: {probe.stderr or probe.stdout}"
            )
        return (probe.stdout or "").strip() or None

    async def _entrypoint_output_tail(self, sandbox: AsyncSandbox) -> str:
        tail = await sandbox.exec(f"tail -c {OVERLAY_OUTPUT_TAIL} {ENTRYPOINT_LOG}", timeout_s=60)
        return (tail.stdout or "").strip() or "(no output)"

    @staticmethod
    def _notice_launcher_end(task: HarborTask, launch: EntrypointLaunch) -> None:
        """Log once when the provider ended the exec holding the entrypoint while it had not exited.

        Docker's exec returns ``timeout`` at ``exec.default_timeout_s`` with the process still running,
        OpenSandbox's foreground exec at ``connection.request_timeout_s``: from then on the entrypoint
        is untracked (its exit file is still probed), and the cancel at stop has nothing to end.
        """
        if launch.noticed:
            return
        outcome = _launch_outcome(launch)
        if outcome is None:
            return
        launch.noticed = True
        LOGGER.warning(
            "%s: the exec holding ENTRYPOINT %s ended (%s) before the entrypoint exited: the provider capped it, "
            "so the process is no longer tracked (OpenSandbox deployments need operations.background_exec for "
            "long-lived launchers)",
            task.task_id,
            shlex.join(launch.argv),
            outcome,
        )

    async def _check_entrypoint(self, sandbox: AsyncSandbox, task: HarborTask, launch: EntrypointLaunch) -> None:
        """Fail the seed when the entrypoint has exited non-zero; warn once when it returned 0.

        Under Harbor an entrypoint that returns ends the container and ``up --wait`` fails the trial
        either way. A non-zero status is the task's fault (:class:`HarborEntrypointError`, with the
        output tail). A zero status means it returned without running its keepalive argument: the
        sandbox's own keepalive is still up, so the episode goes on, the one deliberate leniency.
        """
        if launch.returned:
            return
        status = await self._entrypoint_exit_status(sandbox)
        if status is None:
            self._notice_launcher_end(task, launch)
            return
        output = await self._entrypoint_output_tail(sandbox)
        if status != "0":
            raise HarborEntrypointError(f"ENTRYPOINT {shlex.join(launch.argv)} exited {status}; output tail: {output}")
        launch.returned = True
        LOGGER.warning(
            "%s: ENTRYPOINT %s exited 0 without running its keepalive argument (under Harbor the container would have "
            "exited); continuing with the sandbox's own keepalive. Output tail: %s",
            task.task_id,
            shlex.join(launch.argv),
            output,
        )

    async def _entrypoint_died_during_rollout(
        self, sandbox: AsyncSandbox, task: HarborTask, launch: EntrypointLaunch, logs_dir: Path
    ) -> str | None:
        """Read the entrypoint's exit status back after the rollout; the reason to mask the sample when it died.

        Under Harbor the entrypoint is PID 1: when it exits the container dies and the trial errors
        instead of scoring. Here the keepalive outlives it, so the status is read from the runtime folder
        and kept as :data:`ENTRYPOINT_EXIT_NAME` beside the output. That folder is world-writable (a
        non-root entrypoint writes the markers) and the agent had the sandbox, so the file alone does not
        void a rollout: the sample is masked only when the exec holding the entrypoint returned on its own
        (``run.sh`` finished, so the entrypoint had exited) and the status is non-zero. A status the exec
        does not corroborate is logged as an error and the sample is scored. A zero status gets the seed's
        leniency. Nothing here fails the verification: a probe that cannot run is logged.
        """
        if launch.returned:
            return None
        try:
            status = await self._entrypoint_exit_status(sandbox)
        except Exception:
            LOGGER.warning("Could not read the entrypoint's exit status for %s", task.task_id, exc_info=True)
            return None
        if status is None:
            self._notice_launcher_end(task, launch)
            return None
        try:
            (logs_dir / ENTRYPOINT_EXIT_NAME).write_text(status + "\n")
        except OSError:
            LOGGER.warning("Could not record the entrypoint's exit status for %s", task.task_id, exc_info=True)
        argv = shlex.join(launch.argv)
        if status == "0":
            launch.returned = True
            LOGGER.warning(
                "%s: ENTRYPOINT %s exited 0 during the rollout without running its keepalive argument (under Harbor "
                "the container would have exited); the sample is scored",
                task.task_id,
                argv,
            )
            return None
        try:
            output = await self._entrypoint_output_tail(sandbox)
        except Exception:
            output = "(unavailable)"
        if _launch_returned(launch):
            LOGGER.error(
                "Image ENTRYPOINT %s exited %s during the rollout for %s; the sample is masked (under Harbor the "
                "container would have died). Output tail: %s",
                argv,
                status,
                task.task_id,
                output,
            )
            return f"Image ENTRYPOINT {argv} exited {status} during the rollout; output tail: {output}"
        LOGGER.error(
            "%s: ENTRYPOINT %s has exit status %s recorded, but the exec holding it did not return (%s): scored, "
            "not masked; the status is kept as %s. Output tail: %s",
            task.task_id,
            argv,
            status,
            _launch_outcome(launch) or "still held",
            ENTRYPOINT_EXIT_NAME,
            output,
        )
        return None

    async def _wait_healthy(
        self,
        sandbox: AsyncSandbox,
        task: HarborTask,
        entrypoint: EntrypointLaunch | None = None,
        *,
        user: str | None = None,
        workdir: str | None = None,
        task_check: bool = True,
        compose_managed: bool = False,
    ) -> None:
        """Poll the image's ``HEALTHCHECK``, then the task's ``[environment.healthcheck]``, until each passes.

        That is Harbor's order: ``up --wait`` gates on the image HEALTHCHECK, then ``run_healthcheck()``
        runs the task's, before the agent user is set. Both run as the container's user (``user``, the
        image's ``User`` by name; ``None`` is root): the daemon runs the image check in the image
        ``WorkingDir``, Harbor's exec runs the task check in the task workdir (``[environment].workdir`` is
        the ``cwd`` of its execs, harbor 0.22.0 ``environments/docker/docker.py`` 1104, never the
        container's directory). ``workdir`` is that directory, the one the seed prepared
        (``_prepare_workdir``: ``task.workdir``, else the image's own), named explicitly for the task check
        as every other exec of the seed names its ``cwd``: a single sandbox is created in that directory,
        so its default is the same, while the ``main`` of a Compose group runs its command in the
        Dockerfile's ``WORKDIR`` (``synthesized_group``) and would run the check there too. ``None`` leaves
        the check to the sandbox's default directory. ``compose_managed`` leaves
        the image check out: the adapter already waited for the recorded healthcheck of the service it
        started (the agent's ``main`` of a Compose task), while a separate verifier's sandbox is a plain
        one under a Compose task too and gets its image check here. ``task_check=False`` leaves the
        task's check out (the separate verifier's sandbox, where Harbor never runs it). An entrypoint
        started by the seed is probed before every attempt, so one that dies fails fast with its output
        instead of waiting out the retries. A check that never passes is a retryable seed failure.
        """
        checks: list[tuple[str, HarborHealthcheck, str | None]] = []
        image_check = None if compose_managed else image_healthcheck_settings(task.image_healthcheck, task.image_shell)
        if image_check is not None:
            checks.append((" (the image HEALTHCHECK)", image_check, task.entrypoint_workdir or "/"))
        if task_check and task.config.environment.healthcheck is not None:
            checks.append(("", task.config.environment.healthcheck, workdir))
        for label, check, cwd in checks:

            async def attempt(check: HarborHealthcheck = check, cwd: str | None = cwd) -> SandboxExecResult:
                if _is_root(user):
                    return await _exec_as_root_user(
                        sandbox,
                        check.command,
                        configured_user=task.sandbox_user,
                        spec_user=task.user,
                        cwd=cwd,
                        timeout_s=check.timeout_sec + 5,
                    )
                return await sandbox.exec(check.command, cwd=cwd, timeout_s=check.timeout_sec + 5, user=user)

            await self._poll_healthcheck(sandbox, task, check, attempt, label=label, entrypoint=entrypoint)

    async def _poll_healthcheck(
        self,
        sandbox: AsyncSandbox,
        task: HarborTask,
        check: HarborHealthcheck,
        attempt: Callable[[], Awaitable[SandboxExecResult]],
        *,
        label: str = "",
        entrypoint: EntrypointLaunch | None = None,
    ) -> None:
        """One healthcheck polled to a pass with Harbor's ``run_healthcheck`` semantics (Docker's): attempts
        every ``start_interval_sec`` inside the start period, where a failure counts for nothing, then every
        ``interval_sec``, where ``retries`` consecutive failures fail the check. ``attempt`` runs the command once
        (the seed's image and task checks run it as the image's user, a step's check as the turn's agent user).
        With ``entrypoint``, the seed's launch is probed before every attempt so a dead entrypoint fails fast.
        Raises :class:`HealthcheckFailed` naming the command (and ``label``) when the check never passes; what
        ``attempt`` raises (a stopped sandbox, a provider error) propagates as it is."""
        loop = asyncio.get_running_loop()
        grace_until = loop.time() + check.start_period_sec
        failures = 0
        while True:
            # Decided before the attempt, as Harbor does: a check that starts inside the start period and
            # fails after it has ended is still a start-period attempt, not a counted failure.
            in_grace = loop.time() < grace_until
            if entrypoint is not None:
                await self._check_entrypoint(sandbox, task, entrypoint)
            result = await attempt()
            if result.return_code == 0:
                return
            if not in_grace:
                failures += 1
                if failures >= check.retries:
                    ended = ""
                    if entrypoint is not None and entrypoint.noticed:
                        ended = f" after the exec holding the entrypoint ended ({_launch_outcome(entrypoint)})"
                    raise HealthcheckFailed(
                        f"Healthcheck failed {check.retries} times: {check.command!r}{label}{ended}"
                    )
            await asyncio.sleep(check.start_interval_sec if in_grace else check.interval_sec)

    async def _collect_entrypoint_log(self, sandbox: AsyncSandbox, task: HarborTask, target: Path) -> None:
        """Keep the entrypoint's output with the verification's logs; best effort, never a masked sample."""
        try:
            kind = await download_path(sandbox, ENTRYPOINT_LOG, target)
        except Exception:
            LOGGER.warning("Could not collect the entrypoint log for %s", task.task_id, exc_info=True)
            return
        if kind is None:
            LOGGER.warning("%s: no entrypoint log at %s to collect", task.task_id, ENTRYPOINT_LOG)
