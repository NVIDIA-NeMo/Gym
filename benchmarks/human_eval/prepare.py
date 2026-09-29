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

"""Prepare HumanEval+ benchmark data for NeMo Gym.

Mirrors NeMo-Skills' `nemo_skills/dataset/human-eval/prepare.py`:
  - Source: `evalplus.data.get_human_eval_plus()` (HumanEvalPlus, default version).
  - Per-row transform: replace 4-space indentation with `\\t` in `prompt`
    (Skills comment: "models like tabs more than spaces").
  - Drop `base_input` / `plus_input` (re-fetched at eval time by the
    EvalPlus library — keeping them here just bloats the JSONL).

Per-row schema written:
  - question        : prompt with 4-space → \\t replacement (the field
                      the prompt template's `{question}` placeholder binds to)
  - entry_point     : str (informational; not consumed by the verifier)
  - canonical_solution : str (informational)
  - verifier_metadata.task_id : str (e.g. "HumanEval/0") — the ONLY field
                                the `evalplus` resource server reads
                                beyond the model output. The server looks
                                up base + plus tests via
                                `evalplus.data.get_human_eval_plus()` at
                                runtime using this id.
"""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "human_eval_benchmark.jsonl"


def prepare() -> Path:
    """Download HumanEval+ and write a Gym-format JSONL."""
    from evalplus.data import get_human_eval_plus

    print("Downloading HumanEval+ via evalplus.data.get_human_eval_plus()...")
    problems = get_human_eval_plus()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for problem in problems.values():
        # Skills' transform: problem["question"] = problem["prompt"].replace("    ", "\t")
        question = problem["prompt"].replace("    ", "\t")
        row = {
            "question": question,
            "entry_point": problem["entry_point"],
            "canonical_solution": problem["canonical_solution"],
            "verifier_metadata": {"task_id": problem["task_id"]},
        }
        rows.append(json.dumps(row) + "\n")

    with open(OUTPUT_FPATH, "w") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    if len(rows) != 164:
        # HumanEval has exactly 164 tasks; a different count means the
        # upstream dataset shifted and parity to Skills is no longer
        # guaranteed.
        raise RuntimeError(f"Expected 164 HumanEval+ problems, got {len(rows)}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
