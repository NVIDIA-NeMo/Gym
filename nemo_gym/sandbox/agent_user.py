# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ``agent_user`` identity helpers for sandboxed agents and their resources servers.

The containment contract behind ``agent_user``:

* The agent identity is the uid (or account name) under which every model-controlled command runs inside the
  sandbox: the Terminus 2 tmux session, the in-sandbox OpenCode process and every tool call it makes, the golden
  ``solve.sh`` in the resources server's verification path, and so on. A non-root identity keeps the model from
  reading or altering privileged locations such as ``/tests`` and ``/logs/verifier``.
* The image default stays root. The verifier (test upload, ``test.sh``, reward collection) runs as the image
  default, which is why a non-root ``agent_user`` requires a root-default image on which that account exists.
* ``None``, ``"root"`` and ``0`` all mean "run as the image default": no identity check, unchanged behavior.
* :func:`check_agent_user` is the fail-closed gate. Running as root when a non-root identity was requested must
  never happen silently, so any violation raises with a fleet-diagnosable message before anything
  model-controlled runs.

The same rules are enforced in the agent harness (lane config plus the per-row echo from ``/seed_session``) and in
the resources server (row schema), so both import from here rather than keeping hand-synchronized copies.
"""

from typing import Any, Protocol


AgentUser = str | int | None

_AGENT_USER_ERROR = "agent_user must be an account name, a uid, or null"


class _ExecResultLike(Protocol):
    return_code: int
    stdout: str | None
    stderr: str | None


class SupportsUserExec(Protocol):
    """Anything with ``async exec(command, *, user=...)``: ``AsyncSandbox`` or the Harbor environment adapter."""

    async def exec(self, command: str, *, user: AgentUser = None, **kwargs: Any) -> _ExecResultLike: ...


def normalize_agent_user(value: Any) -> Any:
    """Normalize an ``agent_user`` value (a pydantic ``mode="before"`` validator body).

    A ``str`` is an account NAME for ``su``; an ``int`` is a uid. Digit-only strings become ints (``"1000"`` ->
    ``1000``, so the provider uses the execd uid path instead of ``su 1000``; GNU ``id`` would accept ``"1000"`` as
    a name and hide the misconfiguration). ``isdecimal`` rather than ``isdigit``, so ``"²"`` stays a string instead
    of making ``int()`` raise. Booleans are rejected because pydantic's lax mode would otherwise coerce ``true`` to
    uid 1. Empty and option-like names (``""``, ``"-m"``) are rejected because ``su`` would parse them as options
    (``shlex.quote`` leaves ``"-m"`` unquoted). Any other type is returned unchanged for pydantic to reject.
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


async def check_agent_user(executor: SupportsUserExec, agent_user: str | int) -> None:
    """Fail-closed, outcome-based identity check to run before anything model-controlled.

    ``executor`` is anything with ``async exec(command, *, user=...)`` returning an object with ``.return_code``,
    ``.stdout`` and ``.stderr`` (``stdout``/``stderr`` may be ``None`` and are treated as ``""``). Two commands are
    issued: ``id -u`` as ``"root"`` must print ``0`` (the image default must be root so the verifier can run as
    root), and ``id -u && id -g`` as ``agent_user`` (the SAME ``user`` kwarg the agent will use, so this exercises
    the provider's su/uid path) must succeed with a non-root uid and gid, and for an int identity the uid must
    match. Any violation raises ``RuntimeError`` naming the identity, the condition and the observed
    return code / stdout / stderr.
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
