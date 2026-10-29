# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who runs what inside a Harbor sandbox.

An image has a USER, the task's [agent] table may name another, the SandboxSpec carries the
default the sandbox was created with, and some providers switch user by name only. The
predicates and helpers here answer those questions in one place: whether a user is root,
whether it is numeric, which user an exec should name, how an image USER is resolved to a
name before it is installed as the default, and how a command is run as root on every
provider. The mixin holds the server methods that install the default user and read the
image's user names from the running sandbox. The module imports nothing else from the
environment server, so every other module may import it.
"""

import logging
import shlex
from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException

from nemo_gym.data import HarborTask
from nemo_gym.sandbox import AsyncSandbox


LOGGER = logging.getLogger(__name__)


def _is_root(user: str | int | None) -> bool:
    """Whether ``user`` runs with uid 0, whatever group it names."""
    return user is None or str(user).split(":")[0] in ("root", "0")


def _is_plain_root(user: str | None) -> bool:
    """Whether ``user`` is root with no group of its own (``root:1234`` is not: the group must be kept)."""
    return user is None or str(user) in ("root", "0")


async def _exec_as_root_user(
    sandbox: AsyncSandbox,
    command: str,
    *,
    configured_user: str | None,
    spec_user: str | int | None = None,
    cwd: str | None = "/",
    timeout_s: float = 60,
):
    """Run ``command`` as root (``cwd=None`` is the sandbox's default directory).

    ``configured_user`` is the OCI ``User`` of the image the sandbox started from
    (``HarborTask.sandbox_user``); ``spec_user`` is the default the sandbox was created with
    (``SandboxSpec.user``, the Dockerfile's final ``USER`` or ``[agent].user``), which the sandbox
    facade applies to every exec that names no user. Only when both are root does the plain exec
    path, which every provider supports, already run as root; otherwise ``user="root"`` is asked for.
    """
    user = None if _is_root(configured_user) and _is_root(spec_user) else "root"
    return await sandbox.exec(command, cwd=cwd, timeout_s=timeout_s, user=user)


def _is_numeric_user(user: str | None) -> bool:
    return user is not None and str(user).split(":")[0].isdigit() and not _is_root(user)


def _exec_user(user: str) -> str | int:
    """The ``user`` of a ``RUN`` line's exec: a bare uid as the number (a uid the image has no name for), the rest as written.

    A numeric string reaches the ``su``-backed providers as an account name, which no uid is;
    the number takes their uid path. A ``name:group`` or ``uid:gid`` pair stays a string: only
    Docker's ``--user`` takes the pair, and the other providers fail the line with their own message.
    """
    return int(user) if user.isdigit() else user


# Providers whose exec takes a ``name:group`` or ``uid:gid`` pair, as Docker's ``--user`` does (Harbor's Docker
# environment passes ``-u <user>`` to ``docker exec`` as written). Every other provider switches user by name only:
# OpenSandbox, enroot, Apptainer and ECS ``su`` to it, E2B names a username, Daytona makes it the sandbox's creation
# user, and none of them can apply a group.
GROUP_CAPABLE_PROVIDERS = frozenset({"docker"})


class HarborUserError(HTTPException):
    """A task's user this deployment's sandbox provider cannot run as: a ``name:group`` on a name-only provider.

    A 422: the task and the deployment disagree, which no retry mends. The seed stops what it claimed and
    keeps the status; separate verification lets it through as the verification's own terminal failure.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(422, detail)


def _user_group(user: str | int | None) -> str:
    """The group ``user`` names (``name:group``, ``uid:gid``) and only Docker's ``--user`` applies; empty when it
    names none, root's own group on root (``0:0``, ``root:root``) included, which is plain root everywhere."""
    if user is None:
        return ""
    name, _, group = str(user).partition(":")
    if name in ("root", "0") and group in ("root", "0"):
        return ""
    return group


def _check_user_group(task: HarborTask, user: str | int | None, provider: str | None, where: str = "") -> None:
    """Refuse a ``name:group`` default on a provider that cannot apply the group.

    Dropping the group would run the task with another gid than it asked for and lose the files readable
    only by that group, so the mismatch fails the episode instead of logging a warning. ``provider`` is the
    registry name of the block that runs the task; a provider with no name (``None``) is not known to take
    the pair either. ``where`` names the part of the task that carries the pair when it is not the task's own
    user (``" step 'build'"`` for a ``[steps.verifier].user``), so a multi-step task's author need not search
    every step for it; it follows the task id, as the loader labels its per-step refusals.
    """
    group = _user_group(user)
    if not group or provider in GROUP_CAPABLE_PROVIDERS:
        return
    raise HarborUserError(
        f"{task.task_id}{where}: user {user!r} names group {group!r}; the {provider!r} sandbox provider switches user "
        "by name only and cannot apply a group (Docker's --user takes the pair)"
    )


def _normalised_user(task: HarborTask, user: str | None) -> str | None:
    """A default exec user (a Dockerfile's final ``USER`` or ``[agent].user``) as the sandbox facade may install
    it. Root is ``None`` (the plain exec path every provider supports) unless the image itself starts as another
    user, where it has to be asked for by name; ``0`` is spelled ``root``, as providers that switch by name take
    it; a ``name:group`` keeps its group (``root:1234`` included, ``0:0`` is plain root), which Docker's ``--user``
    applies and :func:`_check_user_group` refuses on the other providers. A numeric uid passes through for
    :meth:`HarborEnvironmentServer._set_default_user` to resolve to a name."""
    if user is None:
        return None
    name, _, group = str(user).partition(":")
    if name in ("root", "0"):
        if _user_group(user):
            return f"root:{group}"
        return None if _is_root(task.sandbox_user) else "root"
    return str(user)


def _creation_user(task: HarborTask, user: str | None) -> str | None:
    """The default a sandbox in the task image is created with: :func:`_normalised_user`, except a numeric uid
    (with or without a group), which no provider should get raw (OpenSandbox would ``su`` to it); it is resolved
    by name once the sandbox runs, in :meth:`HarborEnvironmentServer._set_default_user`."""
    name = _normalised_user(task, user)
    return None if name is not None and name.partition(":")[0].isdigit() else name


class HarborUsersMixin:
    """Who runs what inside a Harbor sandbox.

    An image has a USER, the task's [agent] table may name another, the SandboxSpec carries the
    default the sandbox was created with, and some providers switch user by name only. The
    predicates and helpers here answer those questions in one place: whether a user is root,
    whether it is numeric, which user an exec should name, how an image USER is resolved to a
    name before it is installed as the default, and how a command is run as root on every
    provider. The mixin holds the server methods that install the default user and read the
    image's user names from the running sandbox. The module imports nothing else from the
    environment server, so every other module may import it.

    """

    async def _set_default_user(
        self,
        sandbox: AsyncSandbox,
        task: HarborTask,
        names: dict[str, str],
        *,
        role: str = "agent",
        user: str | None = None,
    ) -> None:
        """Make the task's user (the image's own ``USER`` for a separate verifier, see :func:`_sandbox_spec`), or
        ``user`` when given (a step's ``[steps.agent].user`` for one turn of a multi-step task, see
        :meth:`HarborEnvironmentServer._apply_turn_user`), the sandbox's default for every exec, once it exists
        in the sandbox, normalised as :func:`_normalised_user` says.

        A digit-only ``USER`` is resolved to its name with ``getent`` (``names`` is what the overlay
        already looked up), because OpenSandbox and PTY sessions switch user by name. When the uid
        names no user it is installed as the integer uid, with a warning: OpenSandbox's plain exec then
        takes its native ``uid=`` path and Docker's ``--user`` is unchanged, while OpenSandbox's PTY path
        (which resolves an int with ``id -nu``) and Daytona still cannot run as a uid with no passwd entry
        and fail with their own error naming it. A ``uid:gid`` keeps its group after the name lookup
        (``agent:1000``, or ``1000:1000`` as written when the uid has no name): only Docker takes the pair,
        and :func:`_check_user_group` has refused it on every other provider.
        """
        if user is None:
            user = task.entrypoint_user if role == "verifier" else task.user
        user = _normalised_user(task, user)
        if user is None:
            # Plain root on an image that starts as root: the provider's own default already is root. A turn
            # switching back to root from another user has to say so, since the facade's default was changed.
            if sandbox.default_user is not None:
                sandbox.set_default_user(None)
            return
        # `_sandbox_spec` refused the pair before the sandbox existed; the one place the default is installed
        # checks too, for a sandbox created another way.
        self._refuse_unsupported_group(task, user)
        name, _, group = user.partition(":")
        if name.isdigit():
            resolved = await self._user_name(sandbox, task, name, names)
            if group:
                user = f"{resolved or name}:{group}"
            else:
                user = resolved if resolved is not None else int(name)
        sandbox.set_default_user(user)

    def _refuse_unsupported_group(
        self,
        task: HarborTask,
        user: str | int | None,
        global_config_dict: Mapping[str, Any] | None = None,
        where: str = "",
    ) -> None:
        """:func:`_check_user_group` for ``user`` on the provider that runs ``task``, resolved only when there is a
        group to check (:meth:`HarborEnvironmentServer._task_provider_name`); ``where`` labels the step that
        carries the pair, as there."""
        if _user_group(user):
            _check_user_group(task, user, self._task_provider_name(task, global_config_dict), where)

    async def _user_name(self, sandbox: AsyncSandbox, task: HarborTask, uid: str, names: dict[str, str]) -> str | None:
        """The name of ``uid`` in the image, looked up once with ``getent``; ``None`` (warned once) when it has none."""
        if uid not in names:
            # Asked for as root explicitly whenever the sandbox's current default is another user.
            lookup = await _exec_as_root_user(
                sandbox,
                f"getent passwd {shlex.quote(uid)} | cut -d: -f1",
                configured_user=task.sandbox_user,
                spec_user=sandbox.default_user,
            )
            lines = (lookup.stdout or "").strip().splitlines()
            names[uid] = lines[-1] if lookup.return_code == 0 and lines and lines[-1] else ""
            if not names[uid]:
                LOGGER.warning(
                    f"{task.task_id}: USER {uid} names no user in {task.image}; providers that switch user by name "
                    "cannot run commands as it"
                )
        return names[uid] or None

    async def _image_user_name(self, sandbox: AsyncSandbox, task: HarborTask, names: dict[str, str]) -> str | None:
        """The user the image starts as (its recorded ``User``; ``None`` is root), by name when it was numeric.

        It is what the entrypoint and the image ``HEALTHCHECK`` run as. Docker runs a numeric ``User``
        without a passwd entry; a provider that switches user by name cannot, so the number stays
        (with :meth:`_user_name`'s warning) and the launcher's own failure then names the cause.
        """
        user = task.entrypoint_user
        if not _is_numeric_user(user):
            return user
        uid, _, group = str(user).partition(":")
        name = await self._user_name(sandbox, task, uid, names)
        return user if name is None else name + (f":{group}" if group else "")
