# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned task inventory and direct interpretation of upstream Harbor artifacts."""

import asyncio
import hashlib
import json
import math
import os
import signal
import subprocess
from collections import Counter
from pathlib import Path


REVISION = "610c4ecc69ac56fc62e8cfbd3b28dddd88f22863"
FAMILIES = {"repair": 18, "assembly": 18, "sequencing": 28, "repurpose": 36}


def inventory(root: Path) -> dict[str, dict]:
    root = root.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected Agentic-VBench {REVISION}, found {revision}")
    changed = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "tasks"], text=True)
    if changed.strip():
        raise ValueError("Benchmark tasks differ from the pinned checkout")
    tasks = {}
    for path in sorted((root / "tasks").glob("*/*/task.toml")):
        family = path.parent.parent.name.removeprefix("agentic_vbench_")
        prompt = (path.parent / "steps/solve/instruction.md").read_text()
        task_id = path.parent.name
        if task_id in tasks or family not in FAMILIES:
            raise ValueError(f"Unexpected or duplicate task: {path}")
        tasks[task_id] = {
            "task_id": task_id,
            "family": family,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "benchmark_revision": REVISION,
        }
    counts = Counter(t["family"] for t in tasks.values())
    if counts != FAMILIES:
        raise ValueError(f"Expected complete 100-task inventory, got {dict(counts)}")
    return tasks


def dataset_rows(tasks: dict[str, dict], selector: str = "all") -> list[dict]:
    selected = set(tasks) if selector == "all" else set()
    if not selector.strip():
        raise ValueError("Task selection must not be empty")
    if selector != "all":
        for item in selector.split():
            family = item.removeprefix("agentic_vbench_")
            matches = (
                {item}
                if item in tasks
                else {name for name, task in tasks.items() if family in FAMILIES and task["family"] == family}
            )
            if not matches:
                raise ValueError(f"Unknown task/family: {item}")
            if selected & matches:
                raise ValueError(f"Task selection overlaps: {item}")
            selected.update(matches)
    return [
        {
            "responses_create_params": {"input": [{"role": "user", "content": task["prompt"]}]},
            "verifier_metadata": {k: v for k, v in task.items() if k != "prompt"},
        }
        for task in tasks.values()
        if task["task_id"] in selected
    ]


def read_result(output: Path, task: dict) -> dict:
    task_results = sorted(output.glob("jobs/trial/*/result.json"))
    if len(task_results) != 1:
        raise RuntimeError(f"Expected one Harbor trial result, found {len(task_results)}")
    trial = json.loads(task_results[0].read_text())
    task_name = trial.get("task_name", "").removeprefix("agentic-vbench/")
    if task_name != task["task_id"]:
        raise ValueError(f"Harbor returned a mismatched task: {task_name}")
    trial_dir = task_results[0].parent
    reward_files = list(trial_dir.glob("steps/solve/verifier/reward.json"))
    if len(reward_files) != 1:
        raise RuntimeError(f"Missing native verifier reward: {trial_dir}")
    reward = json.loads(reward_files[0].read_text())["reward"]
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
        or not 0 <= reward <= 1
    ):
        raise ValueError(f"Invalid verifier reward: {reward}")
    trajectories = list(trial_dir.glob("steps/solve/agent/trajectory.json"))
    if len(trajectories) != 1:
        raise RuntimeError(f"Expected one model trajectory: {trial_dir}")
    trajectory = json.loads(trajectories[0].read_text())
    if not any(step.get("source") == "agent" for step in trajectory.get("steps", [])):
        raise RuntimeError("No model trajectory: cannot count this episode as a model outcome")
    return {
        "reward": reward,
        "status": "OK",
        "trajectory": trajectory,
        "artifacts": str(output),
    }


async def run_process(command: list[str], output: Path, runtime: Path) -> int:
    """Give each episode its own Podman state, even within the same Slurm job."""
    runtime.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, SLURM_TMPDIR=str(runtime))
    env.pop("AGENTIC_VBENCH_PODMAN_GRAPH_ROOT", None)
    with (output / "runner.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            return await process.wait()
        except asyncio.CancelledError:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), timeout=30)
                except TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
            raise
