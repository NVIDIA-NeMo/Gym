# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke upstream BenchAgent without its Harbor error wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from nooa.tools.shell_lifecycle import preserve_background_services

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.task_agent import _latest_user_text


LOGGER = logging.getLogger(__name__)


if TYPE_CHECKING:
    from nooa_bench.bench_agent import BenchAgent, TaskResult


def _retain_cleanup_failure(failure: BaseException | None, cleanup_error: Exception) -> BaseException:
    if failure is None:
        failure = cleanup_error
    else:
        LOGGER.warning("BenchAgent cleanup failed during invocation failure", exc_info=True)
    # Preserve the task exception's identity and cause while telling the runner
    # that this invocation cannot be graded as a normal policy termination.
    failure.nooa_cleanup_error = cleanup_error
    return failure


async def invoke_bench_agent(agent: BenchAgent, request: NeMoGymResponseCreateParamsNonStreaming) -> TaskResult:
    """Keep the Resources-provided cwd, structured result and honest failures.

    Upstream's Harbor entrypoint catches every exception into a success=false
    dictionary. Calling its same decorated solve method directly lets Gym keep
    policy-budget/transport/cancellation outcomes distinct. Defaults for the
    summarizer, delegation and CodeActV2 strategy remain upstream-owned.
    """
    failure: BaseException | None = None
    try:
        async with preserve_background_services() as shells:
            shells.adopt(agent.shell.session)
            try:
                cwd = Path.cwd()
                if Path(agent.shell.cwd).resolve() != cwd:
                    await agent.shell.close()
                    agent._install_python_tools(str(cwd))
                result = await agent._solve_task(_latest_user_text(request))
            except BaseException as error:
                # Defer propagation so the upstream scope does not suppress its
                # own cleanup failure while unwinding the task exception.
                failure = error
            # Drain summaries and close the shell before Gym projects the final trace.
            try:
                await agent.aclose()
            except Exception as error:
                failure = _retain_cleanup_failure(failure, error)
    except Exception as error:
        failure = _retain_cleanup_failure(failure, error)
    # Cancellation delivered during either cleanup await bypasses the Exception
    # handlers above; a previously caught task cancellation also remains intact.
    if failure is not None:
        raise failure
    return result
