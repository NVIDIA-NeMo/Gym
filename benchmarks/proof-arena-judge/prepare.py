# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Proof-Arena-Judge benchmark data for NeMo Gym."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import datasets
import requests


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "proof-arena-judge_benchmark.jsonl"

JUDGEMENT_YES = "Judgement: Yes"
JUDGEMENT_NO = "Judgement: No"
SUBSET_PROOFS = "proofs"
MAX_QWEN_TOKENS = 10_000
TOKENIZER_REPO = "Qwen/Qwen3-0.6B"
# Pinned HuggingFace dataset revisions aligned with the Skills-side workaround.
IMO_OUTPUTS_REVISION = "d995fc906b58"  # pragma: allowlist secret
USAMO_OUTPUTS_REVISION = "0fafbf629a32"  # pragma: allowlist secret
IMC_OUTPUTS_REVISION = "d4f93c209272"  # pragma: allowlist secret


def _grading_scheme_to_rubric(grading_scheme, desc_key: str = "grading_scheme_desc") -> str:
    if desc_key != "grading_scheme_desc":
        assert "grading_scheme_desc" not in grading_scheme[0]
    return "\n".join(f"- {scheme[desc_key]}" for scheme in grading_scheme)


def _load_openai_imo_proofs() -> list[dict]:
    imo_data = datasets.load_dataset("MathArena/imo_2025")["train"]
    rows = []
    for i in range(1, 6):
        url = f"https://raw.githubusercontent.com/aw31/openai-imo-2025-proofs/main/problem_{i}.txt"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        assert int(imo_data[i - 1]["problem_idx"]) == i
        rows.append(
            {
                "problem": imo_data[i - 1]["problem"],
                "proof": response.text,
                "rubric": _grading_scheme_to_rubric(imo_data[i - 1]["grading_scheme"], desc_key="desc"),
                "expected_judgement": JUDGEMENT_YES,
                "subset_for_metrics": SUBSET_PROOFS,
                "metadata": {
                    "source": "openai_imo",
                    "problem_idx": i,
                    "model_id": "openai_imo",
                },
            }
        )
    return rows


def _load_gemini_imo_proofs() -> list[dict]:
    imo_data = datasets.load_dataset("MathArena/imo_2025")["train"]
    rows = []
    for i in range(1, 6):
        fpath = Path(__file__).parent / "gemini_imo_2025" / f"{i}.txt"
        with fpath.open("r", encoding="utf-8") as fin:
            text = fin.read()
        assert int(imo_data[i - 1]["problem_idx"]) == i
        rows.append(
            {
                "problem": imo_data[i - 1]["problem"],
                "proof": text,
                "rubric": _grading_scheme_to_rubric(imo_data[i - 1]["grading_scheme"], desc_key="desc"),
                "expected_judgement": JUDGEMENT_YES,
                "subset_for_metrics": SUBSET_PROOFS,
                "metadata": {
                    "source": "gemini_imo",
                    "problem_idx": i,
                    "model_id": "gemini_imo",
                },
            }
        )
    return rows


def _process_imo_usamo_rows(raw_rows, source: str) -> list[dict]:
    rows = []
    for item in raw_rows:
        points1, points2 = item["points_judge_1"], item["points_judge_2"]
        if points2 is None:
            points2 = points1
        if points1 < 6 and points2 < 6:
            judgement = JUDGEMENT_NO
        elif points1 == 7 and points2 == 7:
            judgement = JUDGEMENT_YES
        else:
            continue
        rows.append(
            {
                "problem": item["problem"],
                "proof": item["answer"].split("</think>")[-1].strip(),
                "expected_judgement": judgement,
                "rubric": _grading_scheme_to_rubric(item["grading_details_judge_1"]),
                "subset_for_metrics": SUBSET_PROOFS,
                "metadata": {
                    "source": source,
                    "problem_idx": item["problem_idx"],
                    "grading_details_judge_1": item["grading_details_judge_1"],
                    "grading_details_judge_2": item["grading_details_judge_2"],
                    "model_id": item["model_name"],
                },
            }
        )
    return rows


def _process_imc_rows(raw_rows) -> list[dict]:
    rows = []
    for item in raw_rows:
        points1 = item["points_judge_1"]
        if points1 < 8:
            judgement = JUDGEMENT_NO
        elif points1 == 10:
            judgement = JUDGEMENT_YES
        else:
            continue
        rows.append(
            {
                "problem": item["problem"],
                "proof": item["answer"].split("</think>")[-1].strip(),
                "expected_judgement": judgement,
                "rubric": _grading_scheme_to_rubric(item["grading_details_judge_1"]),
                "subset_for_metrics": SUBSET_PROOFS,
                "metadata": {
                    "source": "imc",
                    "problem_idx": item["problem_idx"],
                    "grading_details_judge_1": item["grading_details_judge_1"],
                    "model_id": item["model_name"],
                },
            }
        )
    return rows


def prepare(output_fpath: Path = OUTPUT_FPATH) -> Path:
    from transformers import AutoTokenizer

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    imo_rows = datasets.load_dataset("MathArena/imo_2025_outputs", revision=IMO_OUTPUTS_REVISION)["train"]
    usamo_rows = datasets.load_dataset("MathArena/usamo_2025_outputs", revision=USAMO_OUTPUTS_REVISION)["train"]
    imc_rows = datasets.load_dataset("MathArena/imc_2025_outputs", revision=IMC_OUTPUTS_REVISION)["train"]

    rows = []
    rows.extend(_load_openai_imo_proofs())
    rows.extend(_load_gemini_imo_proofs())
    rows.extend(_process_imo_usamo_rows(usamo_rows, "usamo"))
    rows.extend(_process_imo_usamo_rows(imo_rows, "imo"))
    rows.extend(_process_imc_rows(imc_rows))

    for row in rows:
        row["problem"] = row["problem"].strip()
        row["proof"] = row["proof"].strip()

    random.seed(42)
    random.shuffle(rows)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    filtered_rows = [row for row in rows if len(tokenizer.encode(row["proof"])) <= MAX_QWEN_TOKENS]

    with output_fpath.open("w", encoding="utf-8") as fout:
        for row in filtered_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Filtered out {len(rows) - len(filtered_rows)} proofs due to length.")
    print(f"Wrote {len(filtered_rows)} problems to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=OUTPUT_FPATH)
    args = parser.parse_args()
    prepare(output_fpath=args.output_path)
