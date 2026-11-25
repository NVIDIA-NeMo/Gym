# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Filter a source dataset by mean reward from repeated Gym rollouts.

Example (run from Gym):
    python benchmarks/nemotron_3.5_super/reward_profiling.py \
        --input-jsonl data/swe_8n-11.jsonl \
        --rollouts-jsonl results/eval/rollouts.jsonl \
        --output-dir results/reward_filtered \
        --low-threshold 0.125 --high-threshold 0.75

Outputs filtered_inputs.jsonl (original rows, in original order),
reward_profile.jsonl (one decision per task), and summary.json. Bounds are
exclusive, as in RL dynamic sampling; arbitrary finite numeric rewards work.

Use the exact original input file used for collection, before repeat expansion.
Gym's _ng_task_index identifies tasks; rollout order and dataset 'idx' do not.
Completeness is checked against the adjacent *_materialized_inputs.jsonl when
available, otherwise --num-repeats is required. --allow-partial-rollouts opts
into means over available measured results, which can bias task selection.
Masked or evaluation_completed=false results never contribute rewards.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm


TASK_INDEX = "_ng_task_index"
ROLLOUT_INDEX = "_ng_rollout_index"


@dataclass
class Task:
    """Source input and observed/expected repeat results for one Gym task."""

    source: str
    expected: set[int] = field(default_factory=set)
    seen: set[int] = field(default_factory=set)
    rewards: list[float] = field(default_factory=list)
    excluded: int = 0
    agents: set[str] = field(default_factory=set)


def _read_jsonl(path: Path) -> Iterator[tuple[int, str, dict[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, line, row


def _index(row: dict[str, object], key: str, location: str) -> int:
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{location}: {key} must be a nonnegative integer")
    return value


def _load_inputs(path: Path) -> tuple[list[tuple[int, str]], dict[int, Task]]:
    rows: list[tuple[int, str]] = []
    tasks: dict[int, Task] = {}
    unstamped: dict[str, int] = {}
    for line_number, line, row in tqdm(_read_jsonl(path), desc="Loading inputs", unit="rows"):
        location = f"{path}:{line_number}"
        if ROLLOUT_INDEX in row:
            raise ValueError(f"{location}: use the original input dataset, not repeated materialized inputs")
        # Match Gym's file-based collector: explicit task IDs take precedence;
        # otherwise identical source lines share an ID in first-occurrence order.
        task_index = (
            _index(row, TASK_INDEX, location) if TASK_INDEX in row else unstamped.setdefault(line, len(unstamped))
        )
        if task_index in tasks and json.loads(tasks[task_index].source) != row:
            raise ValueError(f"{location}: conflicting inputs share {TASK_INDEX}={task_index}")
        tasks.setdefault(task_index, Task(source=line))
        rows.append((task_index, line))
    if not rows:
        raise ValueError(f"{path}: input dataset is empty")
    return rows, tasks


def _record_agent(task: Task, row: dict[str, object], location: str) -> None:
    agent = row.get("agent_ref")
    if isinstance(agent, dict) and isinstance(name := agent.get("name"), str):
        task.agents.add(name)
        if len(task.agents) > 1:
            raise ValueError(f"{location}: multiple agents evaluated one task; filter each agent's run separately")


def _rollout_task(row: dict[str, object], tasks: dict[int, Task], location: str) -> tuple[Task, int]:
    task_index = _index(row, TASK_INDEX, location)
    rollout_index = _index(row, ROLLOUT_INDEX, location)
    if task_index not in tasks:
        raise ValueError(f"{location}: task {task_index} is absent from the input dataset")
    task = tasks[task_index]
    _record_agent(task, row, location)
    return task, rollout_index


def _mean_reward(task: Task) -> float | None:
    return math.fsum(reward / len(task.rewards) for reward in task.rewards) if task.rewards else None


def filter_inputs(
    input_jsonl: Path,
    rollouts_jsonl: Path,
    output_dir: Path,
    *,
    low_threshold: float = 0.2,
    high_threshold: float = 0.8,
    num_repeats: int | None = None,
    allow_partial_rollouts: bool = False,
) -> dict[str, object]:
    """Write a filtered dataset and audit reports, returning the summary counts.

    A task with no measured rewards is excluded, never treated as reward zero.
    Duplicate rollout identities, invalid rewards, and unknown tasks are errors.
    The rollout file is streamed; full trajectories are not retained in memory.
    """
    if not math.isfinite(low_threshold) or not math.isfinite(high_threshold) or low_threshold >= high_threshold:
        raise ValueError("thresholds must be finite and low_threshold < high_threshold")
    if num_repeats is not None and num_repeats < 1:
        raise ValueError("num_repeats must be positive")
    materialized = rollouts_jsonl.with_name(f"{rollouts_jsonl.stem}_materialized_inputs.jsonl")
    outputs = [output_dir / name for name in ("filtered_inputs.jsonl", "reward_profile.jsonl", "summary.json")]
    sources = {path.resolve() for path in (input_jsonl, rollouts_jsonl, materialized)}
    if any(path.resolve() in sources for path in outputs):
        raise ValueError("output files must not overwrite inputs, rollouts, or materialized inputs")

    print(f"[1/5] Loading source inputs from {input_jsonl}", file=sys.stderr, flush=True)
    inputs, tasks = _load_inputs(input_jsonl)
    print(f"Loaded {len(inputs):,} input rows covering {len(tasks):,} tasks", file=sys.stderr, flush=True)
    print("[2/5] Determining expected rollouts", file=sys.stderr, flush=True)
    expected_known = materialized.is_file() or num_repeats is not None
    if materialized.is_file():
        print(f"Using materialized inputs: {materialized}", file=sys.stderr, flush=True)
        for line_number, _, row in tqdm(_read_jsonl(materialized), desc="Reading expected rollouts", unit="rollouts"):
            location = f"{materialized}:{line_number}"
            task, rollout_index = _rollout_task(row, tasks, location)
            if rollout_index in task.expected:
                raise ValueError(f"{location}: duplicate materialized rollout identity")
            task.expected.add(rollout_index)
        if num_repeats is not None and any(len(task.expected) != num_repeats for task in tasks.values()):
            raise ValueError("--num-repeats disagrees with the materialized inputs")
    elif num_repeats is not None:
        print(f"Expecting {num_repeats} repeats per task", file=sys.stderr, flush=True)
        for task in tasks.values():
            task.expected = set(range(num_repeats))
    elif not allow_partial_rollouts:
        raise ValueError(
            f"No materialized inputs at {materialized}; supply --num-repeats to verify completeness "
            "or explicitly use --allow-partial-rollouts"
        )

    expected_total = sum(len(task.expected) for task in tasks.values()) if expected_known else None
    print(f"[3/5] Reading rewards from {rollouts_jsonl}", file=sys.stderr, flush=True)
    print(
        "kept_tasks_provisional counts tasks within the reward bounds so far; remaining repeats can change it.",
        file=sys.stderr,
        flush=True,
    )
    provisional_kept: set[int] = set()
    excluded_rollouts = 0
    with tqdm(_read_jsonl(rollouts_jsonl), desc="Reading rollouts", total=expected_total, unit="rollouts") as progress:
        for line_number, _, row in progress:
            location = f"{rollouts_jsonl}:{line_number}"
            task, rollout_index = _rollout_task(row, tasks, location)
            if rollout_index in task.seen:
                raise ValueError(f"{location}: duplicate rollout identity")
            if expected_known and rollout_index not in task.expected:
                raise ValueError(f"{location}: rollout identity is absent from the expected collection")
            task.seen.add(rollout_index)
            if row.get("mask_sample") or row.get("evaluation_completed") is False:
                task.excluded += 1
                excluded_rollouts += 1
            else:
                reward = row.get("reward")
                if not isinstance(reward, (int, float)) or not math.isfinite(reward):
                    raise ValueError(
                        f"{location}: reward must be a finite number for an unmasked, completed evaluation"
                    )
                task.rewards.append(float(reward))
                task_index = _index(row, TASK_INDEX, location)
                mean = _mean_reward(task)
                if mean is not None and low_threshold < mean < high_threshold:
                    provisional_kept.add(task_index)
                else:
                    provisional_kept.discard(task_index)
            progress.set_postfix(
                kept_tasks_provisional=len(provisional_kept), excluded_rollouts=excluded_rollouts, refresh=False
            )

    print("[4/5] Validating completeness and filtering tasks", file=sys.stderr, flush=True)
    missing_tasks = [index for index, task in tasks.items() if not task.seen or task.expected - task.seen]
    if missing_tasks and not allow_partial_rollouts:
        raise ValueError(
            f"Incomplete rollouts for {len(missing_tasks)} tasks (first: {missing_tasks[:10]}). "
            "Use --allow-partial-rollouts to filter using available measured rewards."
        )

    profiles: list[dict[str, object]] = []
    kept: set[int] = set()
    decisions: Counter[str] = Counter()
    progress = tqdm(tasks.items(), desc="Filtering tasks", unit="tasks")
    for task_index, task in progress:
        mean = _mean_reward(task)
        if mean is None:
            decision = "no_measured_rewards"
        elif mean <= low_threshold:
            decision = "rejected_low"
        elif mean >= high_threshold:
            decision = "rejected_high"
        else:
            decision = "kept"
        decisions[decision] += 1
        if decision == "kept":
            kept.add(task_index)
        progress.set_postfix(
            kept_tasks=len(kept),
            rejected_low=decisions["rejected_low"],
            rejected_high=decisions["rejected_high"],
            no_measured_rewards=decisions["no_measured_rewards"],
            refresh=False,
        )
        profiles.append(
            {
                TASK_INDEX: task_index,
                "agent": next(iter(task.agents), None),
                "expected_rollouts": len(task.expected) if expected_known else None,
                "observed_rollouts": len(task.seen),
                "measured_rollouts": len(task.rewards),
                "excluded_rollouts": task.excluded,
                "missing_rollouts": len(task.expected - task.seen) if expected_known else None,
                "mean_reward": mean,
                "decision": decision,
            }
        )

    summary: dict[str, object] = {
        "input_jsonl": str(input_jsonl.resolve()),
        "rollouts_jsonl": str(rollouts_jsonl.resolve()),
        "materialized_inputs_jsonl": str(materialized.resolve()) if materialized.is_file() else None,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "num_repeats": num_repeats,
        "allow_partial_rollouts": allow_partial_rollouts,
        "input_rows": len(inputs),
        "input_tasks": len(tasks),
        "kept_rows": sum(task_index in kept for task_index, _ in inputs),
        "kept_tasks": decisions["kept"],
        "rejected_low_tasks": decisions["rejected_low"],
        "rejected_high_tasks": decisions["rejected_high"],
        "no_measured_reward_tasks": decisions["no_measured_rewards"],
        "tasks_with_missing_rollouts": len(missing_tasks) if expected_known else None,
        "excluded_rollouts": sum(task.excluded for task in tasks.values()),
    }
    # Validate everything before creating/truncating any output file.
    print(f"[5/5] Writing filtered inputs, reward profiles, and summary to {output_dir}", file=sys.stderr, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("w", encoding="utf-8") as handle:
        for task_index, line in tqdm(inputs, desc="Writing filtered inputs", unit="rows"):
            if task_index in kept:
                handle.write(line if line.endswith("\n") else line + "\n")
    with outputs[1].open("w", encoding="utf-8") as handle:
        for profile in tqdm(profiles, desc="Writing reward profiles", unit="tasks"):
            handle.write(json.dumps(profile, allow_nan=False) + "\n")
    outputs[2].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"Done: kept {summary['kept_rows']:,}/{len(inputs):,} input rows ({len(kept):,}/{len(tasks):,} tasks)",
        file=sys.stderr,
        flush=True,
    )
    return summary


def main() -> None:
    """Filter completed reward-profiling results from sbatch_external_vllm.sh."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Original, unexpanded collection input")
    parser.add_argument("--rollouts-jsonl", type=Path, required=True, help="Gym rollout results, in any order")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for filtered inputs and reports")
    parser.add_argument(
        "--low-threshold", type=float, default=0.2, help="Reject mean reward <= this value (default: 0.2)"
    )
    parser.add_argument(
        "--high-threshold", type=float, default=0.8, help="Reject mean reward >= this value (default: 0.8)"
    )
    parser.add_argument(
        "--num-repeats", type=int, help="Expected repeats per task if the materialized companion is absent"
    )
    parser.add_argument(
        "--allow-partial-rollouts", action="store_true", help="Allow missing results; omit unscored tasks"
    )
    args = parser.parse_args()
    try:
        summary = filter_inputs(**vars(args))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    print(f"Filtered dataset: {args.output_dir / 'filtered_inputs.jsonl'}")


if __name__ == "__main__":
    main()
