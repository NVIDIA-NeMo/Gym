# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare AutomationBench data from the installed package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from subprocess import run


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "automationbench_benchmark.jsonl"

REPO_ROOT = BENCHMARK_DIR.parents[1]
VERIFIERS_AGENT_DIR = REPO_ROOT / "responses_api_agents" / "verifiers_agent"
CREATE_DATASET = VERIFIERS_AGENT_DIR / "scripts" / "create_dataset.py"

VF_ENV_ID = "AutomationBench"
VF_ENV_ARGS = {
    "domains": "all",
    "toolset": "api",
    "max_steps": 50,
    "num_examples": -1,
}

AGENT_NAME = "automationbench_benchmark_agent"


def _interpreter() -> str:
    """Python with verifiers importable."""
    venv_python = VERIFIERS_AGENT_DIR / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def prepare(force: bool = False) -> Path:
    """Materialize the 600-task public benchmark. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FPATH.exists() and not force:
        print(f"{OUTPUT_FPATH} already exists; pass --force to regenerate.")
        return OUTPUT_FPATH

    run(
        [
            _interpreter(),
            str(CREATE_DATASET),
            "--env-id",
            VF_ENV_ID,
            "--env-args",
            json.dumps(VF_ENV_ARGS),
            "--agent-name",
            AGENT_NAME,
            "--output",
            str(OUTPUT_FPATH),
        ],
        check=True,
        cwd=str(VERIFIERS_AGENT_DIR),
    )

    with open(OUTPUT_FPATH) as f:
        num_tasks = sum(1 for _ in f)
    print(f"Prepared {num_tasks} AutomationBench tasks -> {OUTPUT_FPATH}")

    return OUTPUT_FPATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the dataset even if it already exists.",
    )
    args = parser.parse_args()
    prepare(force=args.force)


if __name__ == "__main__":
    main()
