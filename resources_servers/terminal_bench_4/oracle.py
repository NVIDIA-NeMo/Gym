# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference solution execution using the same agent identity and grading lifecycle."""

import asyncio
import json
from pathlib import Path
from shlex import quote

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.sandbox import AsyncSandbox
from resources_servers.terminal_bench_4.transfers import stage_trusted_directory
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, HarnessOutcome


class OracleHarness:
    """Stage as root, execute as agent.user (or the unchanged image default)."""

    def __init__(
        self,
        *,
        sandbox: AsyncSandbox,
        context: HarnessContext,
        solution_dir: Path,
        directory: Path,
        response: NeMoGymResponse,
    ) -> None:
        self.sandbox = sandbox
        self.context = context
        self.solution_dir = solution_dir
        self.directory = directory
        self.response = response

    async def setup(self) -> None:
        if not (self.solution_dir / "solve.sh").is_file():
            raise FileNotFoundError("Oracle mode requires solution/solve.sh")
        self.directory.mkdir(parents=True, exist_ok=True)
        await stage_trusted_directory(self.sandbox, self.solution_dir, "/solution")
        identity = await self.sandbox.exec(
            "command -v setsid && id && pwd && test -r /solution/solve.sh",
            user=self.context.user,
            cwd=self.context.workdir,
            timeout_s=30,
        )
        (self.directory / "identity.json").write_text(
            json.dumps(
                {
                    "requested_user": self.context.user,
                    "workdir": self.context.workdir,
                    "stdout": identity.stdout,
                    "stderr": identity.stderr,
                    "return_code": identity.return_code,
                },
                indent=2,
            )
        )
        if identity.return_code:
            raise RuntimeError(f"Oracle agent-identity setup failed: {identity.stderr}")

    async def execute(self, budget: float) -> tuple[NeMoGymResponse, HarnessOutcome, dict[str, object]]:
        command = "setsid --wait bash -c " + quote(
            f"echo $$ >> /tmp/{self.context.session_id}.pids; exec bash /solution/solve.sh"
        )
        try:
            async with asyncio.timeout(budget):
                result = await self.sandbox.exec(
                    command,
                    user=self.context.user,
                    cwd=self.context.workdir,
                    timeout_s=budget,
                )
            (self.directory / "stdout.txt").write_text(result.stdout or "")
            (self.directory / "stderr.txt").write_text(result.stderr or "")
            outcome = HarnessOutcome(
                reason="completed" if result.return_code == 0 else "nonzero_exit",
                exit_code=result.return_code,
                detail=f"solution/solve.sh exit code {result.return_code}",
            )
            return self.response, outcome, {"oracle_exit_code": result.return_code}
        except TimeoutError:
            return self.response, HarnessOutcome(reason="timeout", detail="Reference solution timed out"), {}
