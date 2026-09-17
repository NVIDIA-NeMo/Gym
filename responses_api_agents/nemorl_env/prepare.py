# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from itertools import islice
from pathlib import Path

import numpy as np
from datasets import load_dataset


ROOT = Path(__file__).parent
PROMPTS = {
    "loss": "Improve recipe.yaml or NeMo-RL/ source for the one-GPU GRPO math task; new core implementations are accepted, focusing on implementing a loss function or improving optimization. Wire new code into the training path and test it. After pristine setup, your patched code has a hard 60-minute budget for authored dependency builds, model initialization, 8 prompts x 8 generations training, and checkpoint export. A fresh unpatched evaluator then scores the weights on held-out math plus AIME25.",
    "generation": "Improve recipe.yaml or NeMo-RL/ source for the one-GPU GRPO math task; new core implementations are accepted, focusing on training-time generation settings. After pristine setup, your patched code has a hard 60-minute budget for authored dependency builds, model initialization, 8 prompts x 8 generations training, and checkpoint export. A fresh unpatched evaluator then scores the weights on held-out math plus AIME25.",
    "schedule": "Improve recipe.yaml or NeMo-RL/ source for the one-GPU GRPO math task; new core implementations are accepted, focusing on the optimizer or learning-rate schedule. After pristine setup, your patched code has a hard 60-minute budget for authored dependency builds, model initialization, 8 prompts x 8 generations training, and checkpoint export. A fresh unpatched evaluator then scores the weights on held-out math plus AIME25.",
    "capacity": "Improve recipe.yaml or NeMo-RL/ source for the one-GPU GRPO math task; new core implementations are accepted, focusing on trainable capacity and memory use. After pristine setup, your patched code has a hard 60-minute budget for authored dependency builds, model initialization, 8 prompts x 8 generations training, and checkpoint export. A fresh unpatched evaluator then scores the weights on held-out math plus AIME25.",
    "context": "Improve recipe.yaml or NeMo-RL/ source for the one-GPU GRPO math task; new core implementations are accepted, focusing on training sequence length or packing. After pristine setup, your patched code has a hard 60-minute budget for authored dependency builds, model initialization, 8 prompts x 8 generations training, and checkpoint export. A fresh unpatched evaluator then scores the weights on held-out math plus AIME25.",
    "stability": "Improve recipe.yaml or NeMo-RL/ source for the one-GPU GRPO math task; new core implementations are accepted, focusing on numerical stability or regularization. After pristine setup, your patched code has a hard 60-minute budget for authored dependency builds, model initialization, 8 prompts x 8 generations training, and checkpoint export. A fresh unpatched evaluator then scores the weights on held-out math plus AIME25.",
    "throughput": "Improve recipe.yaml or NeMo-RL/ source for the one-GPU GRPO math task; new core implementations are accepted, focusing on throughput so more useful updates fit in the hard 60-minute execution budget after pristine setup. This budget includes authored dependency builds, model initialization, training, and checkpoint export. The immutable generation batch is 8 prompts x 8 generations.",
    "recipe": "Make your best justified implementation in recipe.yaml or NeMo-RL/ source, including a new loss, optimizer, or data-processing method, for the fixed one-GPU NeMo-RL GRPO math task. After pristine setup, your patched code has a hard 60-minute budget for authored dependency builds, model initialization, 8 prompts x 8 generations training, and checkpoint export. A fresh unpatched evaluator then scores the weights on held-out math plus AIME25.",
}


def prepare() -> None:
    dataset = load_dataset(
        "nvidia/OpenMathInstruct-2",
        revision="469216e3f46f4dacf476b382e192485ea51a143e",
        split="train",
        streaming=True,
    )
    rows = [{"input": row["problem"], "output": str(row["expected_answer"])} for row in islice(dataset, 512)]
    if len(rows) != 512:
        raise ValueError("Expected at least 512 OpenMathInstruct-2 rows")
    held_out = np.random.default_rng(42).permutation(512)[:32].tolist()
    authors = [
        {
            "responses_create_params": {
                "input": [],
                "metadata": {
                    "instance_id": f"nemorl-env-aime25-{name}",
                    "dataset_name": "nemorl-env",
                    "problem_statement": prompt,
                    "instance_dict": "{}",
                    "image": "nemo-gpu-researcher:dev",
                },
            },
            "agent_ref": {"type": "responses_api_agents", "name": "nemorl_env"},
        }
        for name, prompt in PROMPTS.items()
    ]
    for path, records in (
        (ROOT / "task_environment/train_math.jsonl", [row for i, row in enumerate(rows) if i not in held_out]),
        (ROOT / "data/math_eval.jsonl", [rows[i] for i in held_out]),
        (ROOT / "data/train.jsonl", authors),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        print(f"Wrote {len(records)} rows to {path}")


if __name__ == "__main__":
    prepare()
