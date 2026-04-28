# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare RULER v2 benchmark data.

Runs the 12 sub-task generators ported from
https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ruler2/
and concatenates their outputs into a single Gym benchmark JSONL with a
per-row `task`, `eval_type`, and `match_type` so the `ruler2` resources
server can route verification correctly per row.

The three generator scripts are vendored under
``benchmarks/ruler2/sources/`` (see that directory's README for upstream
provenance). They install third-party deps (``wonderwords``, ``nltk``,
``inflect``, ``transformers``, ``editdistance``) into a private venv on
first run.

Configurable knobs
------------------
- ``RULER2_MODEL`` / ``--model``: tokenizer to use for length estimation
  (default: ``nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16``).
- ``RULER2_MAX_SEQ_LENGTH`` / ``--max-seq-length``: total input-token
  budget per sample (default: 16384).
- ``RULER2_DATASET_SIZE`` / ``--dataset-size``: per-task sample count
  (default: 100).
- ``RULER2_TASKS`` / ``--tasks``: comma-separated subset of the 12 task
  names. Defaults to all 12.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


BENCHMARK_DIR = Path(__file__).parent
SOURCES_DIR = BENCHMARK_DIR / "sources"
DATA_DIR = BENCHMARK_DIR / "data"
VENV_DIR = BENCHMARK_DIR / ".venv"
TMP_DIR = BENCHMARK_DIR / "_ruler2_tmp"
OUTPUT_FPATH = DATA_DIR / "ruler2_benchmark.jsonl"

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_MAX_SEQ_LENGTH = 16384
DEFAULT_DATASET_SIZE = 100

# Per-task evaluator routing — mirrors prepare_task_for_ns in Skills.
TASK_TO_ROUTING = {
    "mk_niah_basic": ("ruler2", "all"),
    "mk_niah_easy": ("ruler2", "all"),
    "mk_niah_medium": ("multichoice", "all"),  # match_type unused for multichoice
    "mk_niah_hard": ("multichoice", "all"),
    "mv_niah_basic": ("ruler2", "all"),
    "mv_niah_easy": ("ruler2", "all"),
    "mv_niah_medium": ("ruler2", "2steps"),
    "mv_niah_hard": ("ruler2", "all"),
    "qa_basic": ("ruler2", "part"),
    "qa_easy": ("ruler2", "part"),
    "qa_medium": ("ruler2", "part"),
    "qa_hard": ("ruler2", "part"),
}

# Per-task generator-script + CLI arguments (ported verbatim from
# nemo_skills.dataset.ruler2.prepare).
TASK_TO_ARGS: dict[str, tuple[str, list[str]]] = {
    "mk_niah_basic": (
        "prepare_niah",
        [
            "--num_needle_k",
            "1",
            "--num_needle_v",
            "1",
            "--num_needle_q",
            "1",
            "--type_haystack",
            "needle",
            "--type_needle_k",
            "words",
            "--type_needle_v",
            "numbers",
            "--num_digits_v",
            "10",
        ],
    ),
    "mk_niah_easy": (
        "prepare_mmlu",
        [
            "--dataset",
            "mmlu",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--num_order",
            "0",
            "--task_type",
            "retrieve",
            "--algo_type",
            "single",
        ],
    ),
    "mk_niah_medium": (
        "prepare_mmlu",
        [
            "--dataset",
            "mmlu",
            "--fewshot",
            "5",
            "--prompt_type",
            "instruct",
            "--num_order",
            "0",
            "--task_type",
            "solve",
            "--algo_type",
            "2steps",
        ],
    ),
    "mk_niah_hard": (
        "prepare_mmlu",
        [
            "--dataset",
            "mmlu",
            "--fewshot",
            "5",
            "--prompt_type",
            "instruct",
            "--num_order",
            "0",
            "--task_type",
            "solve",
            "--algo_type",
            "single",
        ],
    ),
    "mv_niah_basic": (
        "prepare_niah",
        [
            "--num_needle_k",
            "1",
            "--num_needle_v",
            "4",
            "--num_needle_q",
            "1",
            "--type_haystack",
            "needle",
            "--type_needle_k",
            "words",
            "--type_needle_v",
            "numbers",
            "--num_digits_v",
            "10",
        ],
    ),
    "mv_niah_easy": (
        "prepare_mmlu",
        [
            "--dataset",
            "mmlu",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--num_order",
            "4",
            "--task_type",
            "niah",
            "--algo_type",
            "single",
        ],
    ),
    "mv_niah_medium": (
        "prepare_mmlu",
        [
            "--dataset",
            "mmlu",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--num_order",
            "4",
            "--task_type",
            "retrieve",
            "--algo_type",
            "2steps",
        ],
    ),
    "mv_niah_hard": (
        "prepare_mmlu",
        [
            "--dataset",
            "mmlu",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--num_order",
            "4",
            "--task_type",
            "retrieve",
            "--algo_type",
            "single",
        ],
    ),
    "qa_basic": (
        "prepare_qa",
        [
            "--dataset",
            "hotpotqa",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--task_type",
            "retrieve",
            "--query_type",
            "doc",
        ],
    ),
    "qa_easy": (
        "prepare_qa",
        [
            "--dataset",
            "hotpotqa",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--task_type",
            "retrieve",
            "--query_type",
            "question",
        ],
    ),
    "qa_medium": (
        "prepare_qa",
        [
            "--dataset",
            "hotpotqa",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--task_type",
            "solve",
            "--algo_type",
            "2steps",
        ],
    ),
    "qa_hard": (
        "prepare_qa",
        [
            "--dataset",
            "hotpotqa",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--task_type",
            "solve",
            "--algo_type",
            "single",
        ],
    ),
}

ALL_TASKS = tuple(TASK_TO_ARGS.keys())


def _ensure_venv() -> Path:
    """Create a private venv with prepare-script dependencies if absent."""
    venv_python = VENV_DIR / "bin" / "python"
    if venv_python.exists():
        return venv_python

    print(f"Creating venv at {VENV_DIR} for ruler2 prepare scripts...")
    subprocess.run(
        ["uv", "venv", "--python", "3.12", "--allow-existing", "--seed", str(VENV_DIR)],
        check=True,
        cwd=BENCHMARK_DIR,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python),
            "wonderwords",
            "nltk",
            "inflect",
            "numpy",
            "transformers",
            "tiktoken",
            "tenacity",
            "datasets",
            "tqdm",
            "editdistance",
        ],
        check=True,
        cwd=BENCHMARK_DIR,
    )
    return venv_python


def _run_one_task(
    task: str,
    venv_python: Path,
    tokenizer_path: str,
    max_seq_length: int,
    dataset_size: int,
) -> Path:
    script, extra = TASK_TO_ARGS[task]
    out_dir = TMP_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(venv_python),
        "-m",
        f"sources.{script}",
        "--output_folder",
        str(out_dir),
        "--tokenizer_type",
        "hf",
        "--tokenizer_path",
        tokenizer_path,
        "--max_seq_length",
        str(max_seq_length),
        "--num_samples",
        str(dataset_size),
        "--random_seed",
        "42",
        *extra,
    ]
    print(f"[{task}] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=BENCHMARK_DIR)
    return out_dir / "test.jsonl"


def _convert_row_to_gym(row: dict, task: str) -> dict:
    eval_type, match_type = TASK_TO_ROUTING[task]
    expected = row["expected_answer"]
    # For ruler2 routes the expected_answer is always a list (the upstream
    # generators wrap single-answer tasks in a 1-element list). For
    # multichoice we want a single uppercase letter.
    if eval_type == "multichoice":
        if isinstance(expected, list):
            expected = expected[0] if expected else ""
        expected = str(expected).strip().upper()
    elif not isinstance(expected, list):
        expected = [str(expected)]

    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": row["question"]}],
        },
        # Plain `question` field for the `prompt_config: prompts/default.yaml`
        # placeholder — needed because Gym's benchmark mode uses prompt_config
        # to bake input messages from raw rows at rollout time.
        "question": row["question"],
        "expected_answer": expected,
        "eval_type": eval_type,
        "match_type": match_type,
        "task": task,
        "length": row.get("length"),
        "index": row.get("index"),
    }


def prepare(
    tasks: Iterable[str] = ALL_TASKS,
    model: str = DEFAULT_MODEL,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    dataset_size: int = DEFAULT_DATASET_SIZE,
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    venv_python = _ensure_venv()

    # Run the 12 generators in parallel — they're independent.
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_task = {
            executor.submit(_run_one_task, task, venv_python, model, max_seq_length, dataset_size): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            future.result()  # propagate exceptions

    # Concatenate per-task JSONLs into the single benchmark JSONL.
    n_total = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for task in tasks:
            test_jsonl = TMP_DIR / task / "test.jsonl"
            if not test_jsonl.exists():
                raise FileNotFoundError(f"Generator for {task} did not produce {test_jsonl}")
            with test_jsonl.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    gym_row = _convert_row_to_gym(row, task)
                    fout.write(json.dumps(gym_row) + "\n")
                    n_total += 1

    print(f"Wrote {n_total} samples across {len(list(tasks))} tasks to {OUTPUT_FPATH}")

    # Best-effort cleanup of the TMP dir to avoid bloating the workspace.
    if os.environ.get("RULER2_KEEP_TMP", "0") != "1":
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    return OUTPUT_FPATH


def _parse_tasks_arg(value: str | None) -> tuple[str, ...]:
    if not value:
        return ALL_TASKS
    items = tuple(s.strip() for s in value.split(",") if s.strip())
    for item in items:
        if item not in TASK_TO_ARGS:
            raise ValueError(f"Unknown task {item!r}. Valid options: {list(ALL_TASKS)}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("RULER2_MODEL", DEFAULT_MODEL),
        help="Tokenizer path / HF model id (default: %(default)s).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=int(os.environ.get("RULER2_MAX_SEQ_LENGTH", DEFAULT_MAX_SEQ_LENGTH)),
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=int(os.environ.get("RULER2_DATASET_SIZE", DEFAULT_DATASET_SIZE)),
    )
    parser.add_argument(
        "--tasks",
        default=os.environ.get("RULER2_TASKS"),
        help="Comma-separated list of tasks. Defaults to all 12.",
    )
    args = parser.parse_args()

    prepare(
        tasks=_parse_tasks_arg(args.tasks),
        model=args.model,
        max_seq_length=args.max_seq_length,
        dataset_size=args.dataset_size,
    )


if __name__ == "__main__":
    main()
