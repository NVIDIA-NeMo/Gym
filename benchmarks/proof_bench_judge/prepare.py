# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare ProofBench evaluation data for NeMo Gym.

Port of ``nemo_skills/dataset/proof-bench-judge/prepare.py``
(``prepare_verification_data``). Downloads ``wenjiema02/ProofBench`` train
split, applies the Qwen3-token filter (proofs <= 10k tokens) and the seed-42
shuffle so ordering matches Skills byte-for-byte. Emits raw data (no prompt
baked in); the prompt is applied at rollout time via ``prompt_config`` in
``benchmarks/proof_bench_judge/config.yaml``.

Each output row carries ``expected_judgement`` — the gold "Judgement: Yes/No"
label derived from the upstream ``expert_rating`` threshold (>=6 → Yes) —
which the ``math_proof_judgement`` resource server reads during ``verify``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "proof_bench_judge_benchmark.jsonl"

# Must match nemo_skills/dataset/proof-bench-judge/prepare.py exactly.
HF_REPO = "wenjiema02/ProofBench"
HF_SPLIT = "train"
TOKENIZER_REPO = "Qwen/Qwen3-0.6B"
MAX_QWEN_TOKENS = 10_000
SHUFFLE_SEED = 42

JUDGEMENT_YES = "Judgement: Yes"
JUDGEMENT_NO = "Judgement: No"


def _load_hf_rows():
    """Load and normalise ProofBench rows — mirror Skills' ``load_hf_data``."""
    from datasets import load_dataset

    ds = load_dataset(HF_REPO)[HF_SPLIT]
    rows = []
    for x in ds:
        rows.append(
            {
                "problem_id": x["problem_id"],
                "problem": x["problem"],
                "proof": x["model_solution"],
                "rubric": x["marking_scheme"],
                "ground_truth_proof": x["reference_solution"],
                "expected_judgement": JUDGEMENT_YES if x["expert_rating"] >= 6 else JUDGEMENT_NO,
                "metadata": {
                    "model_id": x["generator"],
                    "ground_truth_proof": x["reference_solution"],
                    "expert_rating": x["expert_rating"],
                },
            }
        )
    return rows


def prepare() -> Path:
    from transformers import AutoTokenizer

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ProofBench from HuggingFace ({HF_REPO} / {HF_SPLIT})...")
    rows = _load_hf_rows()

    # Skills strips whitespace then shuffles with seed=42 before filtering.
    for r in rows:
        r["problem"] = r["problem"].strip()
        r["proof"] = r["proof"].strip()
    random.seed(SHUFFLE_SEED)
    random.shuffle(rows)

    print(f"Loading Qwen3 tokenizer ({TOKENIZER_REPO}) for <={MAX_QWEN_TOKENS}-token filter...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    filtered = [r for r in rows if len(tokenizer.encode(r["proof"])) <= MAX_QWEN_TOKENS]
    dropped = len(rows) - len(filtered)
    print(f"Filtered out {dropped} proofs due to length.")

    with OUTPUT_FPATH.open("w") as f:
        for r in filtered:
            f.write(json.dumps(r) + "\n")

    n_yes = sum(1 for r in filtered if r["expected_judgement"] == JUDGEMENT_YES)
    n_no = sum(1 for r in filtered if r["expected_judgement"] == JUDGEMENT_NO)
    print(f"Wrote {len(filtered)} problems to {OUTPUT_FPATH}")
    print(f"- Correct Proofs (Judgement: Yes): {n_yes}")
    print(f"- Incorrect Proofs (Judgement: No): {n_no}")

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
