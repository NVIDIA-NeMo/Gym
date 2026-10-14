# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare CaP-Bench (CaP-X) task rows.

CaP-Bench evaluates 7 Robosuite manipulation tasks at several tiers with 100 trials per cell. This writes the text-only
tiers: S1 ``privileged``, S2 ``full``, S3 ``reduced_api``, S4 ``reduced_api_exampleless``, and M1 ``multiturn``. As in
the CaP-X paper, Re-Stack is scored only in the multi-turn tier, because it requires checking the initial condition
before acting. Each row names a CaP-X config and a trial seed; the task prompt with its API documentation comes from
the simulator at reset. Sampling follows CaP-X's harness defaults.

    python benchmarks/capx/prepare.py                    # full set: 3,100 rows
    python benchmarks/capx/prepare.py --trials 10 --tiers privileged multiturn --output subset.jsonl
"""

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = "You are a helpful assistant that generates Python code to directly solve the task."

# Robosuite task -> CaP-X config directory and file stem.
TASKS = {
    "cube_lifting": ("cube_lifting", "franka_robosuite_cube_lifting"),
    "cube_stack": ("cube_stack", "franka_robosuite_cube_stack"),
    "cube_restack": ("cube_restack", "franka_robosuite_cube_restack"),
    "nut_assembly": ("nut_assembly", "franka_robosuite_nut_assembly"),
    "spill_wipe": ("spill_wipe", "franka_robosuite_spill_wipe"),
    "two_arm_lift": ("two_arm_lift", "franka_robosuite_two_arm_lift"),
    "two_arm_handover": ("two_arm_handover", "two_arm_handover"),
}
# The two-arm tasks name their reduced-API configs "_reduced" rather than "_reduced_api".
_TWO_ARM = {"two_arm_lift", "two_arm_handover"}
# Tasks CaP-Bench scores only in multi-turn tiers.
_MULTITURN_ONLY = {"cube_restack"}

# Text-only CaP-Bench tiers. Visual-feedback and visual-differencing tiers need image observations or a separate VLM.
TIERS = {
    "privileged": lambda task: "_privileged",
    "full": lambda task: "",
    "reduced_api": lambda task: "_reduced" if task in _TWO_ARM else "_reduced_api",
    "reduced_api_exampleless": lambda task: "_reduced_exampleless" if task in _TWO_ARM else "_reduced_api_exampleless",
    "multiturn": lambda task: "_multiturn",
}

# CaP-X harness defaults (capx/envs/launch.py): temperature 1.0, max_tokens 20480.
SAMPLING = {"temperature": 1.0, "max_output_tokens": 20480}

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FPATH = DATA_DIR / "capbench_benchmark.jsonl"
EXAMPLE_FPATH = DATA_DIR / "example.jsonl"
AGENT_REF = {"type": "responses_api_agents", "name": "capx_gymnasium_agent"}


def build_rows(tasks: list[str], tiers: list[str], trials: int) -> list[dict]:
    rows = []
    for tier in tiers:
        for task in tasks:
            if task in _MULTITURN_ONLY and tier != "multiturn":
                continue
            config_dir, stem = TASKS[task]
            config = f"env_configs/{config_dir}/{stem}{TIERS[tier](task)}.yaml"
            for seed in range(trials):
                rows.append(
                    {
                        "responses_create_params": {
                            "input": [{"role": "system", "content": SYSTEM_PROMPT}],
                            **SAMPLING,
                        },
                        "agent_ref": AGENT_REF,
                        "capx_config": config,
                        "seed": seed,
                        "task": task,
                        "tier": tier,
                    }
                )
    return rows


def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def prepare(trials: int = 100) -> Path:
    """Write the full CaP-Bench text-tier set and a small privileged-tier example set."""
    _write(build_rows(sorted(TASKS), list(TIERS), trials), OUTPUT_FPATH)
    _write(build_rows(["cube_lifting", "cube_stack"], ["privileged"], 3), EXAMPLE_FPATH)
    return OUTPUT_FPATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS))
    parser.add_argument("--tiers", nargs="+", choices=list(TIERS), default=list(TIERS))
    parser.add_argument("--trials", type=int, default=100, help="Trials (seeds 0..N-1) per task and tier.")
    parser.add_argument("--output", type=Path, default=None, help="Write a custom subset here instead.")
    args = parser.parse_args()
    if args.output is None:
        print(f"Wrote {prepare(args.trials)}")
        return
    rows = build_rows(args.tasks, args.tiers, args.trials)
    _write(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
