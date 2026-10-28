# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply a Dockerfile overlay at seed, the way docker build would have applied it.

Overlay mode is a pullable FROM plus RUN, WORKDIR, ENV and USER lines that the loader recorded
without a build. At seed the server replays them in the pulled base image: each RUN through the
image's shell as the user docker build would use, each WORKDIR created with the right owner,
and every failure classified as terminal or retryable from the exec result. The module holds
the shell command builders, the output tail and result check, the two overlay errors and the
mixin method that drives the replay.
"""

import asyncio
import shlex
from collections.abc import Mapping

from environment_servers.harbor.users import (
    _exec_as_root_user,
    _exec_user,
    _is_numeric_user,
    _is_plain_root,
    _is_root,
    _user_group,
)
from nemo_gym.data import HarborTask, OverlayRun
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult


# How much of a failed overlay RUN line's output the seed error carries.
OVERLAY_OUTPUT_TAIL = 1500


class HarborOverlayError(RuntimeError):
    """A ``RUN`` line of the task's overlay-mode Dockerfile failed; the task cannot be set up."""


class HarborOverlayUnavailable(RuntimeError):
    """The sandbox provider could not run an overlay step at all (not the step failing): retry elsewhere."""


def overlay_shell_command(step: OverlayRun, sandbox_env: Mapping[str, str] | None = None) -> str:
    """The shell line that runs one Dockerfile ``RUN`` with its ``ENV`` and ``WORKDIR`` in effect.

    The step's environment is already resolved to literals by the loader, so every value is
    exported quoted (nothing expands again here). ``sandbox_env`` is what the sandbox was
    created with (the Dockerfile's final ``ENV`` plus ``task.toml``); a key of it the step does
    not have yet is unset first, so the line sees the environment of its point in the Dockerfile,
    as a build step would, not the finished image's. The working directory exists by the time
    the line runs (see :func:`overlay_workdir_command`), so the line only enters it. The
    command's own exit status is the line's.

    The line itself runs as the build would run it: a shell-form line through the image's
    ``Shell`` when that is not Docker's default (``bash -o pipefail -c`` makes ``false | true``
    fail, as it did in the build), an exec-form line as its argument vector with no shell at all.
    The default shell is invoked explicitly too (``exec /bin/sh -c '<cmd>'``): the provider's exec picks
    its own shell (bash where the image has it), and a line written for the image's ``/bin/sh`` (dash on
    Debian, where ``[[`` is not a builtin) must behave as it did in the build.
    """
    parts = [f"unset {shlex.quote(key)}" for key in sorted(set(sandbox_env or ()) - set(step.env))]
    parts += [f"export {key}={shlex.quote(value)}" for key, value in step.env.items()]
    if step.workdir:
        parts.append(f"cd {shlex.quote(step.workdir)} || exit 1")
    if step.shell is None:
        parts.append(f"exec {step.command}")
    else:
        parts.append(f"exec {shlex.join(step.shell)} {shlex.quote(step.command)}")
    return "; ".join(parts)


def overlay_workdir_command(workdir: str, user: str | None) -> str:
    """Create a ``WORKDIR`` as root the way BuildKit does: missing directories are made and owned by the user.

    An existing directory (one the image ships) is left as it is. ``root:1234`` is not plain root: the
    directory is handed to that user and group, as a build would.
    """
    quoted = shlex.quote(workdir)
    if _is_plain_root(user):
        return f"mkdir -p {quoted}"
    return f"test -d {quoted} || (mkdir -p {quoted} && chown {shlex.quote(user)} {quoted})"


def _output_tail(result: SandboxExecResult, limit: int = OVERLAY_OUTPUT_TAIL) -> str:
    text = "\n".join(part.rstrip() for part in (result.stdout, result.stderr) if part and part.strip())
    return text[-limit:] if text else "(no output)"


def _check_overlay_result(result: SandboxExecResult, label: str, what: str | None, budget: float) -> None:
    """Classify one overlay exec: the task's fault (:class:`HarborOverlayError`) or the provider's."""
    subject = f"{label}: {what}" if what else label
    if result.error_type == "timeout":
        raise HarborOverlayError(f"{subject} exceeded [environment].build_timeout_sec={budget}")
    if result.error_type is not None:
        raise HarborOverlayUnavailable(f"{subject} could not run in the sandbox: {result.error_type}")
    if result.return_code != 0:
        raise HarborOverlayError(f"{subject} exited {result.return_code}; output tail: {_output_tail(result)}")


# The separate verifier's sandbox could not take the task's Dockerfile overlay after the rollout ran (a shared-mode
# task the server verifies separately, see `separate_verifier_for_artifacts`).
OVERLAY_FAILED_KIND = "harbor:overlay_failed"


class HarborOverlayMixin:
    """Apply a Dockerfile overlay at seed, the way docker build would have applied it.

    Overlay mode is a pullable FROM plus RUN, WORKDIR, ENV and USER lines that the loader recorded
    without a build. At seed the server replays them in the pulled base image: each RUN through the
    image's shell as the user docker build would use, each WORKDIR created with the right owner,
    and every failure classified as terminal or retryable from the exec result. The module holds
    the shell command builders, the output tail and result check, the two overlay errors and the
    mixin method that drives the replay.

    """

    async def _apply_overlay(self, sandbox: AsyncSandbox, task: HarborTask) -> dict[str, str]:
        """Run the Dockerfile ``RUN`` lines the loader recorded, in order, before anything else uses the sandbox.

        Each line runs with the ``ENV``, ``WORKDIR`` and ``USER`` the loader resolved for it, the
        way ``docker build`` would: every ``WORKDIR`` instruction creates its directory as root, in
        Dockerfile order (owned by the user and group in effect at it when that is not plain root, left
        alone when the image ships it), a numeric ``USER`` is mapped to its name in the image (a uid
        without a passwd entry is passed as the number: Docker's ``--user`` and OpenSandbox's ``uid`` take
        it, the ``su``-backed providers refuse it with their own message), then the line runs. A ``USER``
        that names a group (``root:1234``, ``app:1234``, ``1000:1000``) keeps that group on its ``WORKDIR``
        and runs its lines as the pair, which only Docker's ``--user`` applies: the ``su``-backed providers
        (OpenSandbox, enroot, apptainer) would reject ``root:1234`` as an account name or drop the group, so
        :meth:`HarborEnvironmentServer._sandbox_spec` refuses such a line's user on them before the sandbox
        exists. Root's own group (``0:0``, ``root:root``) is plain root everywhere. The whole overlay
        shares ``[environment].build_timeout_sec``, as a build would. A line that exits non-zero or
        runs out of budget raises :class:`HarborOverlayError` with the command and the tail of its
        output: the task's Dockerfile does not apply, so retrying would not help. A provider that
        cannot run a step at all raises :class:`HarborOverlayUnavailable` instead, so the seed is retried.
        """
        user_names: dict[str, str] = {}
        if not task.overlay:
            return user_names
        budget = task.config.environment.build_timeout_sec
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget
        total = len(task.overlay)
        prepared_workdirs: set[str] = {"/"}  # the root always exists; every other WORKDIR is made once

        async def as_root(command: str, label: str, what: str) -> SandboxExecResult:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise HarborOverlayError(f"{label} not run: [environment].build_timeout_sec={budget} is used up")
            result = await _exec_as_root_user(
                sandbox, command, configured_user=task.sandbox_user, spec_user=task.user, timeout_s=remaining
            )
            _check_overlay_result(result, label, what, budget)
            return result

        async def effective_user(user: str | None, label: str) -> str | None:
            """A numeric ``USER`` by the name the image gives it; a uid without a passwd entry stays numeric."""
            if not _is_numeric_user(user):
                return user
            uid, _, group = str(user).partition(":")
            if uid not in user_names:
                lookup = await as_root(f"getent passwd {shlex.quote(uid)} | cut -d: -f1", label, f"USER {uid} lookup")
                name = (lookup.stdout or "").strip().splitlines()
                user_names[uid] = name[-1] if name and name[-1] else uid
            return user_names[uid] + (f":{group}" if group else "")

        async def make_workdir(path: str, user: str | None, label: str) -> None:
            if path in prepared_workdirs:
                return
            await as_root(overlay_workdir_command(path, await effective_user(user, label)), label, f"WORKDIR {path}")
            prepared_workdirs.add(path)

        workdirs = list(task.overlay_workdirs)
        for index, step in enumerate(task.overlay, 1):
            label = f"RUN line {index}/{total} {step.command!r}"
            # The WORKDIR instructions before this line, in their order, each owned by the user of its point.
            while workdirs and workdirs[0].before < index:
                instruction = workdirs.pop(0)
                await make_workdir(instruction.path, instruction.user, label)
            user = await effective_user(step.user, label)
            await make_workdir(step.workdir, user, label)
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise HarborOverlayError(f"{label} not run: [environment].build_timeout_sec={budget} is used up")
            command = overlay_shell_command(step, task.env)
            if _is_root(user) and not _user_group(user):
                # Plain root (root's own group included): the one exec path every provider runs as root.
                result = await _exec_as_root_user(
                    sandbox, command, configured_user=task.sandbox_user, spec_user=task.user, timeout_s=remaining
                )
            else:
                # A named user, a bare uid, or a user:group pair as written, `root:1234` included: Docker's --user
                # applies the pair, and `_sandbox_spec` refused it on every other provider before the sandbox existed.
                result = await sandbox.exec(command, cwd="/", timeout_s=remaining, user=_exec_user(user))
            _check_overlay_result(result, label, None, budget)
        # WORKDIR instructions after the last RUN line exist in the built image too.
        for instruction in workdirs:
            await make_workdir(instruction.path, instruction.user, f"after RUN line {total}/{total}")
        return user_names
