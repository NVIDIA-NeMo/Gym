# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the BabyVision TSV (VLMEvalKit format, 388 rows) into Gym benchmark JSONL.

Reads $BABYVISION_TSV, else $LMUData/BabyVision.tsv. Each row is one user message: the
image as a data URL, then Python_call_gen's opencode prompt (prompts/input_prompt.txt),
"---", the question and the boxed-answer suffix, the exact text that harness gives OpenCode.
The gold answer stays out of the prompt (expected_answer).
"""

import base64
import csv
import hashlib
import json
import os
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "babyvision_opencode_benchmark.jsonl"
NUM_ROWS = 388
# Two copies of the same 388 rows: EFB's LMUData and the lmu_cache copy Python_call_gen read.
KNOWN_MD5 = {"b3a93182cce0e658e4f81601275e1f04", "4216333e6620668edb267b032250f635"}

INSTRUCTION = """You are a meticulous visual-reasoning agent and you are provided with a question and a set of images.

Look at the image first. If the answer is clear, answer directly without code. Use code
when the answer needs precision your eyes cannot give: exact counts, coordinates, pixel
comparisons, or tracing a path. Before each tool call, say in one sentence what you are
checking. You have python3 with pillow (PIL), numpy, scipy and opencv (cv2) via the bash
tool and you have access to tools including bash, read, glob, grep, write, edit, todowrite. Please provide your final response inside \\boxed{}.
"""
SUFFIX = "\nPlease answer the question and put the final answer within \\boxed{}."


def tsv_path() -> Path:
    if os.environ.get("BABYVISION_TSV"):
        return Path(os.environ["BABYVISION_TSV"])
    if os.environ.get("LMUData"):
        return Path(os.environ["LMUData"]) / "BabyVision.tsv"
    raise FileNotFoundError("set BABYVISION_TSV or LMUData (the directory holding BabyVision.tsv)")


def convert(row: dict) -> dict:
    image = base64.b64decode(row["image"])
    mime = "image/png" if image[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    question = row["question"].strip()
    return {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "type": "message",
                    "content": [
                        {"type": "input_image", "image_url": f"data:{mime};base64,{row['image']}", "detail": "auto"},
                        {"type": "input_text", "text": f"{INSTRUCTION}\n\n---\n\n{question}{SUFFIX}"},
                    ],
                }
            ]
        },
        "expected_answer": row["answer"],
        "question": question,
        "source_id": row["index"],
        "category": row["category"],
        "l2_category": row["l2_category"],
        "ans_type": row["ans_type"],
    }


def prepare() -> Path:
    path = tsv_path()
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 not in KNOWN_MD5:
        raise ValueError(f"{path} has md5 {md5}, expected one of {sorted(KNOWN_MD5)}")
    csv.field_size_limit(10**9)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != NUM_ROWS:
        raise ValueError(f"{path} has {len(rows)} rows, expected {NUM_ROWS}")
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(convert(row)) + "\n")
    print(f"Wrote {len(rows)} rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
