# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export native task references without copying solutions or grader fixtures."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from responses_api_agents.claweval_agent.worker import source_root


def prepare_split(
    split: str,
    *,
    output: str | Path | None = None,
    task_ids: list[str] | None = None,
    agent_name: str | None = None,
) -> Path:
    value = os.environ.get("CLAW_EVAL_ROOT")
    if not value:
        raise ValueError("Set CLAW_EVAL_ROOT to the local Claw-Eval checkout")
    root = source_root(value)
    output = Path(output or Path(__file__).parent / "data" / f"{split}.jsonl").resolve()
    payload = {
        "operation": "prepare",
        "claweval_root": str(root),
        "split": split,
        "output": str(output),
        "task_ids": task_ids or [],
        "agent_name": agent_name or f"claweval_{split}_agent",
    }
    worker = Path(__file__).resolve().parents[2] / "responses_api_agents/claweval_agent/worker.py"
    subprocess.run(
        [os.environ.get("CLAW_EVAL_PYTHON", sys.executable), str(worker)],
        input=json.dumps(payload),
        text=True,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("general", "multimodal", "multi_turn"), required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--agent-name")
    args = parser.parse_args()
    prepare_split(args.split, output=args.output, task_ids=args.task_id, agent_name=args.agent_name)
