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

"""Prepare FinanceBench benchmark data for NeMo Gym.

Downloads the FinanceBench dataset from HuggingFace (Patronus AI) and
converts it to Gym JSONL format compatible with the
``equivalence_llm_judge`` resource server (open-book financial Q&A).

The 150-question benchmark covers publicly traded companies; each row's
``context`` field holds the concatenated full-page evidence text so the
agent receives the same supporting filings the human annotators worked
from, and the LLM judge scores the candidate answer against
``expected_answer``.

References:
    Paper:      https://arxiv.org/abs/2311.11944
    HuggingFace: https://huggingface.co/datasets/PatronusAI/financebench
    GitHub:     https://github.com/patronus-ai/financebench
"""

import json
import uuid
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "financebench_benchmark.jsonl"

HF_DATASET = "PatronusAI/financebench"
HF_SPLIT = "train"  # FinanceBench ships a single 150-row split on HuggingFace


def _join_evidence(evidence_list: list) -> str:
    """Combine FinanceBench evidence entries into a single context string.

    Each evidence entry is a dict with ``evidence_text_full_page`` (full
    surrounding page) and/or ``evidence_text`` (the human-annotated
    excerpt).  Prefer full-page text so the model sees the same supporting
    material the annotators worked from; fall back to excerpt-only when a
    full page isn't available.
    """
    if not evidence_list:
        return ""

    parts = []
    for i, ev in enumerate(evidence_list, 1):
        if isinstance(ev, dict):
            text = ev.get("evidence_text_full_page") or ev.get("evidence_text") or ""
        elif isinstance(ev, str):
            text = ev
        else:
            text = ""
        if text:
            parts.append(f"[Evidence {i}]\n{text}")
    return "\n\n".join(parts)


def _to_gym_row(item: dict) -> dict:
    """Convert one HuggingFace FinanceBench row into a Gym benchmark row.

    The UUID is derived deterministically from the upstream
    ``financebench_id`` so reruns of ``prepare.py`` produce byte-identical
    rows and downstream per-question aggregation stays stable.
    """
    fb_id = item["financebench_id"]
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"financebench/{fb_id}")),
        "question": item["question"],
        "expected_answer": item["answer"],
        "context": _join_evidence(item.get("evidence", [])),
        "financebench_id": fb_id,
        "company": item.get("company", ""),
        "doc_name": item.get("doc_name", ""),
        "question_type": item.get("question_type") or "unknown",
        "question_reasoning": item.get("question_reasoning") or "unknown",
    }


def prepare() -> Path:
    """Download FinanceBench and write the Gym benchmark JSONL.

    Returns the output file path.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {HF_DATASET}:{HF_SPLIT} ...")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)

    tmp_fpath = OUTPUT_FPATH.with_suffix(".jsonl.tmp")
    count = 0
    with tmp_fpath.open("w", encoding="utf-8") as out:
        for entry in tqdm(ds, desc="Writing FinanceBench JSONL"):
            row = _to_gym_row(entry)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp_fpath.replace(OUTPUT_FPATH)

    print(f"Wrote {count} examples to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
