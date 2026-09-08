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

"""Prepare SECQUE benchmark data for NeMo Gym.

Downloads the SECQUE (SEC Question Understanding Evaluation) dataset from
HuggingFace and converts it to Gym JSONL format compatible with the
``equivalence_llm_judge`` resource server (open-book Q&A with context).

Each row carries the SEC filing context in the ``context`` field so the
agent receives the same supporting text the human annotators saw, and the
LLM judge scores the candidate answer against ``expected_answer``.

Reference:
    HuggingFace: https://huggingface.co/datasets/nogabenyoash/SecQue
"""

import json
import uuid
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "secque_benchmark.jsonl"

HF_DATASET = "nogabenyoash/SecQue"
HF_SPLIT = "train"  # SECQUE only ships a single 565-row split on HuggingFace


def _to_gym_row(item: dict) -> dict:
    """Convert one HuggingFace SECQUE row into a Gym benchmark row.

    The UUID is derived deterministically from the dataset's ``QID`` so that
    reruns of ``prepare.py`` produce byte-identical rows and downstream
    per-question aggregation stays stable.
    """
    qid = item["QID"]
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"secque/{qid}")),
        "question": item["Question"],
        "expected_answer": item["ground_truth_answer"],
        "context": item.get("context_markdown_with_headers", ""),
        "qid": qid,
        "question_type": item.get("question_type", "unknown"),
    }


def prepare() -> Path:
    """Download SECQUE and write the Gym benchmark JSONL.

    Returns the output file path.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {HF_DATASET}:{HF_SPLIT} ...")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)

    tmp_fpath = OUTPUT_FPATH.with_suffix(".jsonl.tmp")
    count = 0
    with tmp_fpath.open("w", encoding="utf-8") as out:
        for entry in tqdm(ds, desc="Writing SECQUE JSONL"):
            row = _to_gym_row(entry)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp_fpath.replace(OUTPUT_FPATH)

    print(f"Wrote {count} examples to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
