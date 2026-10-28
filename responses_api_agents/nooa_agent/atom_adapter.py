# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke Atom (``nooa_atom.gym:AtomHarness``) on the task text.

The harness runs one Atom session per rollout with Gym's model client as the only model, root
turn method ``handle_orchestrate`` by default (children run ``handle_work``), and returns the
session's ``TaskResult``. Pair it with ``agent_class: nooa_atom.gym:AtomHarness`` and pass
``turn_method`` or ``sessions_dir`` through ``init_kwargs``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.task_agent import _latest_user_text


if TYPE_CHECKING:
    from nooa.interactive import TaskResult
    from nooa_atom.gym import AtomHarness


async def invoke_atom(agent: AtomHarness, request: NeMoGymResponseCreateParamsNonStreaming) -> TaskResult:
    """Hand the latest user message to Atom and return its structured result.

    Atom's session reports a failed turn as ``TurnFailedError`` wrapping the error the agent's
    strategy raised. The wrapper is re-raised as that original error so the runner's termination
    policy sees the same CodeAct stops and model rejections it sees from any other NOOA agent.
    """
    try:
        return await agent.solve(_latest_user_text(request))
    except Exception as error:
        cause = error.__cause__
        if type(error).__name__ == "TurnFailedError" and isinstance(cause, Exception):
            raise cause
        raise
