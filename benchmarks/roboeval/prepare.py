# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare RoboEval benchmark rows from a CodeBotler checkout.

RoboEval has 16 service-robot tasks with 5 prompt paraphrases each. Each row holds CodeBotler's default chat prompt
(system message with the robot API plus few-shot examples, from
``code_generation/openai_chat_completion_prefix.py``) followed by one task paraphrase. Gym prompt configs only express
a system and a user message, so the few-shot conversation is written into ``responses_create_params.input`` directly.

The benchmark's license forbids using its contents as ML training data; the prepared rows are for evaluation only and
are gitignored.

    python benchmarks/roboeval/prepare.py   # fetches CodeBotler at the pinned commit unless ROBOEVAL_ROOT is set
"""

import importlib.util
import json
import sys
from pathlib import Path

from resources_servers.roboeval.setup_codebotler import ensure_codebotler


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "roboeval_benchmark.jsonl"
EXAMPLE_FPATH = DATA_DIR / "example.jsonl"
NUM_TASKS = 16
NUM_PARAPHRASES = 5

# RoboEval's generation defaults (roboeval.py). max_tokens is raised from 512 so reasoning models can finish.
SAMPLING = {"temperature": 0.2, "top_p": 0.95, "max_output_tokens": 16384}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(roboeval_root: Path | None = None) -> Path:
    root = ensure_codebotler(roboeval_root)
    sys.path.insert(0, str(root))

    prefix = _load_module("roboeval_chat_prefix", root / "code_generation" / "openai_chat_completion_prefix.py")
    rows = []
    for task_path in sorted((root / "benchmark" / "tasks").glob("*.py")):
        task = _load_module(f"roboeval_task_{task_path.stem}", task_path)
        for prompt_index, prompt in enumerate(task.prompts):
            rows.append(
                {
                    "responses_create_params": {
                        "input": [*prefix.messages, {"role": "user", "content": prompt}],
                        **SAMPLING,
                    },
                    "task": task.name,
                    "prompt_index": prompt_index,
                }
            )
    if len(rows) != NUM_TASKS * NUM_PARAPHRASES:
        raise RuntimeError(f"expected {NUM_TASKS * NUM_PARAPHRASES} RoboEval prompts, got {len(rows)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    with EXAMPLE_FPATH.open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows if row["prompt_index"] == 0)
    print(f"Wrote {len(rows)} rows to {OUTPUT_FPATH} and {NUM_TASKS} to {EXAMPLE_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
