# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``agent_user``: the account the in-sandbox OpenCode process, and so every tool call the model makes, runs as.

* ``None``, ``"root"`` and ``0`` all mean "run as the image default": no identity check, unchanged behavior.
* A non-root identity keeps the model away from privileged locations such as ``/tests`` and ``/logs/verifier``; the
  image default stays root so the verifier can run as root.
* :func:`check_agent_user` is the fail-closed gate: running as root when a non-root identity was requested must never
  happen silently, so a violation raises before anything model-controlled runs.

The TB2.1 resources server applies the same rules (``resources_servers/terminal_bench_2_1/agent_user.py``); keep the
two in step.
"""

from typing import Any, Protocol


AgentUser = str | int | None

_AGENT_USER_ERROR = "agent_user must be an account name, a uid, or null"

# POSIX-sh-safe reset of $HOME to the effective uid's passwd home, falling back to the inherited HOME when `getent` has
# no entry for the uid. The provider runs an int identity through execd's uid switch WITHOUT `su`, so such an exec
# inherits the image default's $HOME (/root); OpenCode's install goes to $HOME/.opencode/bin and its session store to
# $HOME/.local/share/opencode, so every command that touches them resets HOME first and they agree on the location.
_AGENT_HOME_RESET = 'ng_home="$(getent passwd "$(id -u)" | cut -d: -f6)" && export HOME="${ng_home:-$HOME}" && '


class _ExecResultLike(Protocol):
    return_code: int
    stdout: str | None
    stderr: str | None


class SupportsUserExec(Protocol):
    async def exec(self, command: str, *, user: AgentUser = None, **kwargs: Any) -> _ExecResultLike: ...


def normalize_agent_user(value: Any) -> Any:
    """Normalize an ``agent_user`` value (a pydantic ``mode="before"`` validator body).

    A ``str`` is an account NAME for ``su``; an ``int`` is a uid. Digit-only strings become ints (``"1000"`` ->
    ``1000``, so the provider uses the execd uid path instead of ``su 1000``). ``isdecimal`` rather than ``isdigit``,
    so ``"²"`` stays a string instead of making ``int()`` raise. Booleans are rejected because pydantic's lax mode
    would otherwise coerce ``true`` to uid 1. Empty and option-like names (``""``, ``"-m"``) are rejected because
    ``su`` would parse them as options. Any other type is returned unchanged for pydantic to reject.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(_AGENT_USER_ERROR)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.isdecimal():
            return int(value)
        if value == "" or value.startswith("-"):
            raise ValueError(_AGENT_USER_ERROR)
        return value
    return value


def is_root_agent_user(agent_user: AgentUser) -> bool:
    """``None``, ``"root"`` and ``0`` all mean "run as the image default (root on the supported images)"."""
    return agent_user is None or agent_user == "root" or agent_user == 0


def agent_home_prefix(agent_user: AgentUser) -> str:
    """Command prefix pinning ``$HOME`` to the agent's own home under a non-root identity; ``""`` for the default."""
    return "" if is_root_agent_user(agent_user) else _AGENT_HOME_RESET


async def check_agent_user(executor: SupportsUserExec, agent_user: str | int) -> None:
    """Fail-closed identity check, run before anything model-controlled.

    ``id -u`` as ``"root"`` must print ``0`` (the image default must be root so the verifier can run as root), and
    ``id -u && id -g`` as ``agent_user`` (the same ``user`` argument the agent will use) must succeed with a non-root
    uid and gid; for an int identity the uid must match. A violation raises ``RuntimeError`` naming the identity, the
    condition and the observed return code / stdout / stderr.
    """

    def fail(condition: str, result: _ExecResultLike) -> None:
        raise RuntimeError(
            f"agent_user={agent_user!r} identity check failed: {condition}. "
            f"Observed return_code={result.return_code} stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    root_result = await executor.exec("id -u", user="root")
    if root_result.return_code != 0 or (root_result.stdout or "").strip() != "0":
        fail("the image default user must be root (uid 0), otherwise the verifier cannot run as root", root_result)

    agent_result = await executor.exec("id -u && id -g", user=agent_user)
    lines = (agent_result.stdout or "").strip().splitlines()
    if agent_result.return_code != 0 or len(lines) != 2:
        fail(f"could not run a command as {agent_user!r} (does the account exist in the image?)", agent_result)
    uid, gid = lines[0].strip(), lines[1].strip()
    if uid == "0":
        fail(f"commands run as {agent_user!r} still resolve to uid 0", agent_result)
    if gid == "0":
        fail(f"commands run as {agent_user!r} resolve to gid 0", agent_result)
    if isinstance(agent_user, int) and uid != str(agent_user):
        fail(f"commands run as uid {agent_user} resolved to uid {uid}", agent_result)
