# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Generate a standalone Slurm job for post-evaluation exports."""

import re
import shlex
from pathlib import Path


def render_export_job(
    *,
    bench_dir: Path,
    partition: str,
    account: str,
    walltime: str,
    export_command: str,
) -> str:
    """Render a CPU-only Slurm export job without submitting it."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", partition):
        raise ValueError("Invalid Slurm export partition")

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", account):
        raise ValueError("Invalid Slurm account")

    if not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", walltime):
        raise ValueError("Invalid Slurm export walltime")

    if not export_command.strip():
        raise ValueError("Export command cannot be empty")

    return "\n".join(
        [
            "#!/usr/bin/env bash",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --account={account}",
            f"#SBATCH --time={walltime}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=1",
            "set -euo pipefail",
            f"cd {shlex.quote(str(bench_dir))}",
            export_command,
            "",
        ]
    )
