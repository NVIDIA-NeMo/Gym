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
"""Prepare the IOI'24 benchmark for CCC-backed evaluation.

Mirrors NeMo Skills' ``nemo_skills/dataset/ioi/prepare.py`` (downloads
``open-r1/ioi`` for problem statements + subtask scoring weights and
``open-r1/ioi-test-cases`` for the test inputs/outputs) and emits CCC-shaped
artifacts:

- ``benchmarks/ioi/data/ioi24_benchmark.jsonl``: one row per
  ``(problem, subtask)`` with top-level fields CCC expects —
  ``competition_id``, ``problem_id`` (= lowercase ioi_id; matches both
  metadata lookup AND the grader filename ``graders/{problem_id}.cpp``
  expected by IOI's ``compile.sh``), ``subtask``, ``subtask_score``,
  ``name``, ``question``.

- ``benchmarks/ioi/data/ioi24_metadata.json``: single-line JSONL with CCC's
  competition wrapper ``{"competition_id": "ioi24", "metadata": {"nile":
  {...}, ...}}``. Each problem value is a subtask-keyed dict with per-subtask
  ``tests`` / ``subtask_score`` / ``score_precision`` / ``run`` / ``compile``
  / ``grader_files``. CCC's ``_normalize_problem_metadata`` converts this
  legacy shape into the normalized ``subtasks`` / ``all_tests`` structure at
  load time.
"""

import json
from collections import defaultdict
from pathlib import Path

import requests
from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
BENCHMARK_FPATH = DATA_DIR / "ioi24_benchmark.jsonl"
METADATA_FPATH = DATA_DIR / "ioi24_metadata.json"

HF_PROBLEMS_REPO = "open-r1/ioi"
HF_PROBLEMS_SPLIT = "test"
HF_TESTS_REPO = "open-r1/ioi-test-cases"
HF_TESTS_CONFIG = "2024"

RUN_URL = "https://raw.githubusercontent.com/huggingface/ioi/refs/heads/main/run_tests/custom_setup/run"
COMPILE_URL = "https://raw.githubusercontent.com/huggingface/ioi/refs/heads/main/run_tests/custom_setup/compile"

COMPETITION_ID = "ioi24"


def prepare() -> Path:
    """Download IOI'24 and emit the benchmark + metadata files.

    Returns the benchmark JSONL path (framework's expected primary artifact).
    The metadata JSONL is written alongside and referenced by config.yaml's
    ``test_file``.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading IOI problems from {HF_PROBLEMS_REPO}...")
    ds = load_dataset(HF_PROBLEMS_REPO, split=HF_PROBLEMS_SPLIT)

    # IOI's problem name (capitalized, e.g. "Nile") is the row's `name`; its
    # short id (lowercase, e.g. "nile") is `id`. CCC uses one `problem_id`
    # string for both metadata lookup AND the grader filename, and IOI's
    # compile.sh derives that filename from the lowercase .h file — so we use
    # the lowercase id throughout.
    benchmark_rows = []
    for row in ds:
        # Score-0 entries on the test split are sample subtasks with no
        # evaluation weight — skip them.
        if row["score"] == 0:
            continue
        benchmark_rows.append(
            {
                "competition_id": COMPETITION_ID,
                "problem_id": row["id"],
                "subtask": row["subtask"],
                "subtask_score": row["score"],
                "name": row["name"],
                "question": row["problem"],
            }
        )

    with open(BENCHMARK_FPATH, "w") as f:
        f.write("\n".join(json.dumps(r) for r in benchmark_rows))
    print(f"Wrote {len(benchmark_rows)} rows to {BENCHMARK_FPATH}")

    print("Downloading run/compile scripts from github.com/huggingface/ioi ...")
    run_code = requests.get(RUN_URL).text
    compile_code = requests.get(COMPILE_URL).text

    print(f"Loading IOI test cases from {HF_TESTS_REPO} config={HF_TESTS_CONFIG}...")
    tests_ds = load_dataset(HF_TESTS_REPO, name=HF_TESTS_CONFIG, split="train")

    # Build {problem_name (capitalized) -> {test_name -> {input, output}}}.
    test_cases: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for tc in tests_ds:
        test_cases[tc["problem_name"]][tc["test_name"]] = {
            "input": tc["test_input"],
            "output": tc["test_output"],
        }

    # Build per-problem metadata keyed by lowercase problem_id.
    metadata: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in ds:
        subtask = row["subtask"]
        if subtask == "00-samples":
            continue
        problem_id = row["id"]
        problem_name = row["name"]
        tests = {test_name: test_cases[problem_name][test_name] for test_name in row["test_names"]}
        metadata[problem_id][subtask] = {
            "tests": tests,
            "subtask_score": row["score"],
            "score_precision": row["score_precision"],
            "run": run_code,
            "compile": compile_code,
            "grader_files": row["grader_files"],
        }

    wrapped = {"competition_id": COMPETITION_ID, "metadata": dict(metadata)}
    with open(METADATA_FPATH, "w") as f:
        json.dump(wrapped, f)
    print(
        f"Wrote CCC-wrapped metadata for {len(metadata)} problems to "
        f"{METADATA_FPATH} (keys: {sorted(metadata.keys())})"
    )

    return BENCHMARK_FPATH


if __name__ == "__main__":
    prepare()
