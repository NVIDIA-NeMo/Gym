# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``agent_user``: the account every model-controlled command runs as inside the task sandbox.

* ``None``, ``"root"`` and ``0`` all mean "run as the image default": no identity switch, unchanged behavior.
* A non-root identity keeps the model away from privileged locations such as ``/tests`` and ``/logs/verifier``; the
  image default stays root so the verifier can run as root.

The OpenCode sandboxed agent applies the same rules (``responses_api_agents/opencode_sandboxed_agent/agent_user.py``);
keep the two in step.
"""

from typing import Any


AgentUser = str | int | None

_AGENT_USER_ERROR = "agent_user must be an account name, a uid, or null"


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
