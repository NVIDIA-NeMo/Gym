# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare fresh SWE training or raw evaluation from a ready config and frozen inputs.

The template supplies the model, sampling settings, harnesses, tasksets and runtime.
This command changes only the requested budget, input and physical output paths.
It does not prepare images, inspect checkpoints, submit jobs or select checkpoints.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from omegaconf import OmegaConf
from scripts.swe_training_continuation import _move_outputs


def prepare_experiment(
    template: Path,
    inputs: Path,
    output: Path,
    *,
    mode: str,
    policy: str,
    harness: str | None = None,
    steps: int = 100,
    tasks_per_step: int = 16,
    generations: int = 4,
    checkpoint_period: int = 25,
    trials: int = 3,
    batch_size: int = 16,
    weights: Path | None = None,
    step: int = 0,
) -> dict[str, object]:
    """Keep prepared assets in place and write a fresh config with an explicit population.

    Training inputs contain one materialized row per distinct task, including
    fixed harness assignments. Evaluation inputs also contain one row per task;
    this function expands each into ``trials`` identical rows. The evaluation
    template must already route the selected harness and benchmark correctly.
    """
    template, inputs, output = template.resolve(), inputs.resolve(), output.resolve()
    if mode not in {"train", "evaluate"}:
        raise ValueError("Choose train or evaluate")
    values = (steps, tasks_per_step, generations, checkpoint_period) if mode == "train" else (trials, batch_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("Budgets must be positive integers")
    if step < 0 or (weights is None) != (step == 0):
        raise ValueError("Use step zero without weights, or a positive step with explicit published weights")
    if mode == "train" and weights is not None:
        raise ValueError("Use same-run continuation for training checkpoints")
    if mode == "evaluate" and not harness:
        raise ValueError("Evaluation requires a harness label matching its prepared template")
    input_bytes = inputs.read_bytes()
    rows = [json.loads(line) for line in input_bytes.decode().splitlines() if line.strip()]
    identities = [row.get("task_id", {}).get("task_id") or row.get("instance_id") for row in rows]
    if not rows or any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("Inputs need materialized task_id.task_id or old-route instance_id")
    if len(set(identities)) != len(rows):
        raise ValueError("Supply one row per distinct task; evaluation trials are expanded here")
    if mode == "evaluate":
        tasksets = {row.get("task_id", {}).get("taskset") for row in rows}
        if len(tasksets) != 1 or any(
            not isinstance(name, str) or not name.endswith("_" + harness) for name in tasksets
        ):
            raise ValueError("Evaluation inputs must name one taskset for the selected harness")
    if mode == "train" and (tasks_per_step > len(rows) or len(rows) % tasks_per_step):
        raise ValueError("Training task count must be divisible by the distinct-task batch size")
    config = OmegaConf.to_container(OmegaConf.load(template), resolve=False)
    if config["policy"].get("pretrained_checkpoint"):
        raise ValueError("Use an original-model template; checkpoint evaluation takes explicit --weights")
    original = Path(config["logger"]["log_dir"]).parent
    _move_outputs(config["env"], original=original, output=output)
    gym = config["env"]["nemo_gym"]
    # Older prepared templates can omit this field and default inside read-only source.
    for name, agent in gym.items():
        if isinstance(agent, dict):
            settings = agent.get("responses_api_agents", {}).get("pi_sandboxed_agent")
            if isinstance(settings, dict):
                settings["artifacts_dir"] = str(output / "harness-artifacts" / name)
    config["logger"]["log_dir"] = str(output / "logs")
    wandb = config["logger"].setdefault("wandb", {})
    wandb["name"] = output.name
    for key in ("id", "resume"):
        wandb.pop(key, None)
    config["checkpointing"]["checkpoint_dir"] = str(output / "checkpoints")
    config["checkpointing"].pop("pretrained_checkpoint", None)
    config["data"]["shuffle"] = False
    receipt = {
        "policy": policy,
        "template": str(template),
        "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "frozen_input": str(inputs),
        "frozen_input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "mode": mode,
        "tasks": len(rows),
        "output": str(output),
        "qualification": "Preparation only; no native execution or checkpoint restore established.",
    }
    if mode == "train":
        for key in (
            "is_trajectory_collection",
            "trajectory_collection_raw_results",
            "trajectory_collection_batch_size",
        ):
            gym.pop(key, None)
        config["data"]["train"]["data_path"] = str(inputs)
        config["data"]["shuffle"] = False
        config["grpo"].update(
            max_num_steps=steps,
            max_num_epochs=math.ceil(steps * tasks_per_step / len(rows)),
            num_prompts_per_step=tasks_per_step,
            num_generations_per_prompt=generations,
        )
        config["policy"]["train_global_batch_size"] = tasks_per_step * generations
        config["checkpointing"].update(enabled=True, save_period=checkpoint_period, keep_top_k=4)
        receipt.update(
            steps=steps,
            distinct_tasks_per_step=tasks_per_step,
            generations=generations,
            expected_episodes=steps * tasks_per_step * generations,
        )
    else:
        config["data"]["validation"] = dict(config["data"].get("validation") or config["data"]["train"])
        for dataset in ("train", "validation"):
            config["data"][dataset]["data_path"] = str(output / "evaluation.jsonl")
        config["checkpointing"]["enabled"] = False
        config["grpo"]["max_val_samples"] = None
        gym.update(
            is_trajectory_collection=True,
            trajectory_collection_raw_results=True,
            trajectory_collection_batch_size=batch_size,
        )
        if weights is not None:
            config["policy"]["pretrained_checkpoint"] = {"path": str(weights.resolve()), "format": "megatron_bridge"}
        receipt.update(
            harness=harness,
            step=step,
            weights=str(weights.resolve()) if weights else None,
            trials=trials,
            expected_episodes=len(rows) * trials,
            expected_optimizer_updates=0,
            input=str(output / "evaluation.jsonl"),
        )
    output.mkdir(parents=True, exist_ok=False)
    if mode == "evaluate":
        (output / "evaluation.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows for _ in range(trials))
        )
    OmegaConf.save(OmegaConf.create(config), output / "config.yaml")
    receipt["config_sha256"] = hashlib.sha256((output / "config.yaml").read_bytes()).hexdigest()
    (output / ("EVALUATION.json" if mode == "evaluate" else "EXPERIMENT.json")).write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def main(argv: list[str] | None = None) -> None:
    """Prepare a fresh run; use the existing training launcher to execute it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    for name in ("template", "inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--policy", required=True, help="Policy label recorded in the result manifest")
    parser.add_argument("--harness", help="Evaluation harness already wired in the template")
    for name, default in (
        ("steps", 100),
        ("tasks-per-step", 16),
        ("generations", 4),
        ("checkpoint-period", 25),
        ("trials", 3),
        ("batch-size", 16),
        ("step", 0),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument(
        "--weights", type=Path, help="Explicit published step_N/policy/weights; never latest or tmp_step"
    )
    print(json.dumps(prepare_experiment(**vars(parser.parse_args(argv)))))


if __name__ == "__main__":
    main()
