# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic RLMBenchAgent wired to a Gym-hosted seeded sandbox.

RLMBenchAgent's host-side ShellTools is replaced by SandboxShellTools so every
shell command, file read, and file edit lands inside the swebench resources
server's seeded docker sandbox. The swebench verifier extracts the model patch
from that same sandbox via uncommitted ``git diff``, so edits made through
these tools are exactly what verification scores.

This adapter imports ``nooa_bench`` (RLMBenchAgent) and ``nooa_cli``
(RepoTools). Those packages land with the NOOA bench runtime (dev/tui); until
a containing NOOA release is published and pinned in requirements.txt, run
with them available on PYTHONPATH.
"""

from __future__ import annotations

from typing import Any

from nooa_bench.bench_agent import RLMBenchAgent
from nooa_cli.tools.repo_tools import RepoTools

from responses_api_agents.nooa_agent.sandbox_shell import SandboxShellTools


class GymBenchAgent(RLMBenchAgent):
    """RLMBenchAgent operating inside a Gym-hosted seeded sandbox."""

    def __init__(
        self,
        llm: Any | None = None,
        *,
        sandbox: Any = None,
        working_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        if sandbox is None:
            sandbox = getattr(type(self), "_gym_sandbox", None)
        if sandbox is None:
            raise ValueError("GymBenchAgent requires the seeded Gym sandbox")
        self._sandbox = sandbox
        super().__init__(llm=llm, working_dir=working_dir, **kwargs)

    async def solve(self, problem_statement: str, instance_id: str | None = None) -> Any:
        """Public Gym entrypoint: forward the problem to the strategy-decorated solver.

        The NOOA Gym config validator only accepts public, async entrypoints, so the
        strategy-decorated ``_solve_task`` is wrapped here. ``instance_id`` is kept in
        the description so the agent can cite the task it is solving.
        """
        description = problem_statement
        if instance_id:
            description = f"{problem_statement}\n\n(SWE-bench instance_id: {instance_id})"
        return await self._solve_task(description)

    def _install_python_tools(self, cwd: str) -> None:
        """Install shell/repo tools backed by the seeded sandbox."""
        init_command = getattr(self, "_worker_init_command", None)
        self.shell = SandboxShellTools(
            self._sandbox,
            cwd=cwd,
            init_command=init_command,
        )
        self.repo = RepoTools(root=cwd, session=self.shell.session)
