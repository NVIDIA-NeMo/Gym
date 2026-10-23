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
from responses_api_agents.nooa_agent.termination import cleanup_errors


LOGGER = logging.getLogger(__name__)


if TYPE_CHECKING:
    from nooa_bench.bench_agent import BenchAgent, TaskResult


def _retain_cleanup_failure(failure: BaseException | None, cleanup_error: BaseException) -> BaseException:
    retained = cleanup_errors(failure) if failure is not None else ()
    if failure is None or not isinstance(cleanup_error, Exception):
        # Cancellation during cleanup must propagate with the earlier task/setup
        # failure still inspectable, including any cleanup errors already retained.
        if failure is not None and failure is not cleanup_error:
            cleanup_error.__cause__ = failure
        failure = cleanup_error
    # Preserve the task exception's identity and cause while telling the runner
    # that this invocation cannot be graded as a normal policy termination.
    failure.nooa_cleanup_errors = (*retained, cleanup_error)
    LOGGER.warning("BenchAgent cleanup failed", exc_info=True)
    return failure


async def invoke_bench_agent(agent: BenchAgent, request: NeMoGymResponseCreateParamsNonStreaming) -> TaskResult:
    """Keep the Resources-provided cwd, structured result and honest failures.

    Upstream's Harbor entrypoint catches every exception into a success=false
    dictionary. Calling its same decorated solve method directly lets Gym keep
    policy-budget/transport/cancellation outcomes distinct. Defaults for the
    summarizer, delegation and CodeActV2 strategy remain upstream-owned.
    """
    failure: BaseException | None = None
    scope = preserve_background_services()
    entered = False
    try:
        try:
            shells = await scope.__aenter__()
            entered = True
            shells.adopt(agent.shell.session)
            cwd = Path.cwd()
            if Path(agent.shell.cwd).resolve() != cwd:
                await agent.shell.close()
                agent._install_python_tools(str(cwd))
            task = _latest_user_text(request)
            result = await agent._solve_task(task)
        except BaseException as error:
            failure = error
        # An already-created agent owns background work even when scope entry or
        # adoption fails. Always attempt its close before closing the shell scope.
        try:
            await agent.aclose()
        except BaseException as error:
            failure = _retain_cleanup_failure(failure, error)
    finally:
        if entered:
            # Defer propagation so upstream cannot suppress a scope cleanup error
            # while unwinding the original task failure or cleanup cancellation.
            try:
                await scope.__aexit__(None, None, None)
            except BaseException as error:
                failure = _retain_cleanup_failure(failure, error)
    if failure is not None:
        raise failure
    return result
