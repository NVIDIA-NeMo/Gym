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
"""Prepare PHYSICS data for NeMo Gym.

Downloads ``desimfj/PHYSICS::test`` (the same source NeMo Skills uses) and
emits Gym JSONL with one row per problem. The ``expected_answer``
transformation matches Skills' ``nemo_skills/dataset/physics/prepare.py``
byte-for-byte; ``domain`` is the upstream field that Skills calls
``subset_for_metrics``, renamed so the physics_judge server can pick it up
via ``compute_subset_metrics(subset_key="domain")``.
"""

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "physics_benchmark.jsonl"

# Upstream HF identifier — must match the same source NeMo Skills uses
# (nemo_skills/dataset/physics/prepare.py:63).
_HF_REPO = "desimfj/PHYSICS"
_HF_SPLIT = "test"


def _strip_boxed(s: str) -> str:
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[7:-1]
    return s


def _process_answer(answer) -> str:
    r"""Flatten a list-of-lists of answers, strip any outer ``\boxed{...}``,
    and re-wrap the comma-joined contents in a single ``\boxed{...}``.

    Matches Skills' ``nemo_skills/dataset/physics/prepare.py::process_answer``.
    """
    all_answers = [_strip_boxed(item) for sublist in answer for item in sublist]
    return f"\\boxed{{{', '.join(all_answers)}}}"


def _format_entry(entry: dict) -> dict:
    return {
        "question": entry["question"],
        "expected_answer": _process_answer(entry["answer"]),
        "domain": entry["domain"],
        "difficulty": entry["difficulty"],
        "answer_type": entry["answer_type"],
        "language": entry["language"],
        "solution": entry["solution"],
    }


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {_HF_REPO}::{_HF_SPLIT} from HuggingFace ...")
    dataset = load_dataset(_HF_REPO, split=_HF_SPLIT)
    # Skills' default `physics` benchmark uses only the English subset; the
    # `zh` and `en_zh` splits Skills writes alongside aren't exercised here.
    en_data = [entry for entry in dataset if entry["language"] == "en"]

    count = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as out:
        for entry in en_data:
            out.write(json.dumps(_format_entry(entry), ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
