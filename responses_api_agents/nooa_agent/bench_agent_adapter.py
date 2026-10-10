# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke the separately pinned upstream BenchAgent without its Harbor error wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.task_agent import _latest_user_text


if TYPE_CHECKING:
    from nooa_bench.bench_agent import BenchAgent, TaskResult


async def invoke_bench_agent(agent: BenchAgent, request: NeMoGymResponseCreateParamsNonStreaming) -> TaskResult:
    """Keep the Resources-provided cwd, native structured result and honest failures.

    Upstream's Harbor entrypoint catches every exception into a success=false
    dictionary. Calling its same decorated solve method directly lets Gym keep
    policy-budget/transport/cancellation outcomes distinct. Defaults for the
    summarizer, delegation and CodeActV2 strategy remain upstream-owned.
    """
    try:
        cwd = Path.cwd()
        if Path(agent.shell.cwd).resolve() != cwd:
            await agent.shell.close()
            agent._install_python_tools(str(cwd))
        return await agent._solve_task(_latest_user_text(request))
    finally:
        # Drain summaries and close the shell before Gym projects the final trace.
        await agent.aclose()
