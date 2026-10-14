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

from nemo_gym.sandbox import AsyncSandbox


def _is_root(user: str | None) -> bool:
    """Whether ``user`` runs with uid 0, whatever group it names."""
    return user is None or str(user).split(":")[0] in ("root", "0")


def _is_plain_root(user: str | None) -> bool:
    """Whether ``user`` is root with no group of its own (``root:1234`` is not: the group must be kept)."""
    return user is None or str(user) in ("root", "0")


async def _exec_as_root_user(
    sandbox: AsyncSandbox, command: str, *, configured_user: str | None, cwd: str = "/", timeout_s: float = 60
):
    """Run ``command`` as root.

    ``configured_user`` is the user the sandbox runs commands as by default: the OCI ``User``
    of the image the sandbox started from (``HarborTask.sandbox_user``), which is not the
    Dockerfile's final ``USER`` or ``[agent].user``. Only a non-root default needs the
    ``user="root"`` override; root images keep the plain exec path every provider supports.
    """
    user = None if _is_root(configured_user) else "root"
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
