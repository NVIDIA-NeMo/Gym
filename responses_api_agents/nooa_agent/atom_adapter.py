# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke Atom (``nooa_atom.gym:AtomHarness``) on the task text.

The harness runs one Atom session per rollout with Gym's model client as the only model, root
turn method ``handle_orchestrate`` by default (children run ``handle_work``), and returns the
session's ``TaskResult``. Pair it with ``agent_class: nooa_atom.gym:AtomHarness`` and pass
``turn_method`` or ``sessions_dir`` through ``init_kwargs``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.task_agent import _latest_user_text


logger = logging.getLogger(__name__)

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


SUBMIT_DIR = Path("/root/outputs")


async def invoke_atom_with_submission(
    agent: AtomHarness, request: NeMoGymResponseCreateParamsNonStreaming
) -> TaskResult:
    """Run Atom, then submit the files it saved in ``SUBMIT_DIR`` through the resources server's ``finish``.

    Atom has no ``finish`` tool, so a benchmark that grades submitted files (i.e GDPval) would see none. The prompt asks the
    agent to save its deliverables in ``SUBMIT_DIR``; this adapter, running in the task sandbox, sends them with
    ``finish(reason, paths)`` once Atom's turn ends. Use it only with a benchmark whose resources server serves
    ``finish`` (the request's tool list carries it); an empty folder submits nothing, as an agent that never finished.
    """
    if not callable(getattr(agent, "finish", None)):
        raise RuntimeError("invoke_atom_with_submission needs a benchmark whose resources server serves `finish`")
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = await invoke_atom(agent, request)
    except Exception as error:
        # A failed turn (e.g. a malformed Done) can leave finished files behind; submit them before the error is
        # graded, or the judge sees none.
        await _submit(agent, f"Atom's turn ended with an error: {error}")
        raise
    await _submit(agent, result.report or result.solution_description)
    return result


async def _submit(agent: AtomHarness, reason: str) -> None:
    paths = sorted(str(path) for path in SUBMIT_DIR.rglob("*") if path.is_file())
    if not paths:
        logger.warning("Atom saved no files in %s; nothing submitted", SUBMIT_DIR)
        return
    output = await agent.finish(reason=reason, paths=paths)
    logger.info("finish(%d files) -> %s", len(paths), output)
