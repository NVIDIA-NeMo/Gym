# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource-owned staging for trusted reference solutions, never execution."""

import asyncio
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox
from resources_servers.terminal_bench_4.archive_workers import ArchiveWorkers
from resources_servers.terminal_bench_4.transfers import stage_trusted_directory


STAGING_TIMEOUT_SEC = 360


async def stage_solution(
    sandbox: AsyncSandbox, solution_dir: Path, *, archive_workers: ArchiveWorkers | None = None
) -> None:
    """Upload the pinned solution as root before handing the sandbox to an agent."""
    if not (solution_dir / "solve.sh").is_file():
        raise FileNotFoundError("Oracle mode requires solution/solve.sh")
    async with asyncio.timeout(STAGING_TIMEOUT_SEC):
        await stage_trusted_directory(sandbox, solution_dir, "/solution", archive_workers=archive_workers)
