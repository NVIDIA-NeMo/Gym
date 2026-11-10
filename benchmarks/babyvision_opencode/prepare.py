# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert BabyVision (388 rows) into Gym benchmark JSONL.

Downloads the BabyVision release (UnipatAI/BabyVision on Hugging Face, MIT license) at a pinned
revision and checks its parquet file's SHA-256. $BABYVISION_PARQUET reads a local copy of that
file instead, for a host without internet access.
Each row is one user message: the image as a PNG data URL, then a visual-reasoning instruction
(INSTRUCTION), "---", the question and the boxed-answer suffix.
With question_only (benchmarks/babyvision_opencode_question_only) the message is only the
question and the suffix, for models trained on OpenCode's own system prompt.
The gold answer stays out of the prompt (expected_answer).
"""

import base64
import hashlib
import io
import json
import os
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "babyvision_opencode_benchmark.jsonl"
NUM_ROWS = 388

REPO_ID = "UnipatAI/BabyVision"
REVISION = "849cbeea2b1d00ad334e9e1e2b49a1960c9ff3f3"  # pragma: allowlist secret
PARQUET_FILENAME = "data/train-00000-of-00001.parquet"
PARQUET_SHA256 = "7f6ce0d1daff90165b635f60c900fa71f92db1f6d5b38c10b0116d1f026ff3b1"  # pragma: allowlist secret
PARQUET_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{PARQUET_FILENAME}"
CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

INSTRUCTION = """You are a meticulous visual-reasoning agent and you are provided with a question and a set of images.

Look at the image first. If the answer is clear, answer directly without code. Use code
when the answer needs precision your eyes cannot give: exact counts, coordinates, pixel
comparisons, or tracing a path. Before each tool call, say in one sentence what you are
checking. You have python3 with pillow (PIL), numpy, scipy and opencv (cv2) via the bash
tool and you have access to tools including bash, read, glob, grep, write, edit, todowrite. Please provide your final response inside \\boxed{}.
"""
SUFFIX = "\nPlease answer the question and put the final answer within \\boxed{}."


def parquet_path() -> Path:
    if os.environ.get("BABYVISION_PARQUET"):
        return Path(os.environ["BABYVISION_PARQUET"])
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, PARQUET_FILENAME, repo_type="dataset", revision=REVISION))


def load_records(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 != PARQUET_SHA256:
        raise ValueError(f"{path} has SHA-256 {sha256}, expected {PARQUET_SHA256} ({PARQUET_URL})")
    records = pq.read_table(path).to_pylist()
    if len(records) != NUM_ROWS:
        raise ValueError(f"{path} has {len(records)} rows, expected {NUM_ROWS}")
    return records


def png_base64(image: bytes) -> str:
    """The image as a base64 RGB PNG. The release stores PNGs, JPEGs and WebPs; one lossless format
    keeps every decoded pixel (the release's alpha channels are fully opaque)."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.open(io.BytesIO(image)).convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def convert(record: dict, question_only: bool = False) -> dict:
    """One release record as a benchmark row. A multiple-choice question lists its options as
    (A), (B), … after "Choices:", and its answer is the option's letter."""
    question = record["question"]
    if record["ansType"] == "choice":
        options = "\n".join(f"({CHOICE_LETTERS[i]}) {option}" for i, option in enumerate(record["options"]))
        question = f"{question}\nChoices:\n{options}"
        answer = CHOICE_LETTERS[record["choiceAns"]]
    else:
        answer = record["blankAns"]
    question = question.strip()
    text = f"{question}{SUFFIX}" if question_only else f"{INSTRUCTION}\n\n---\n\n{question}{SUFFIX}"
    image_url = f"data:image/png;base64,{png_base64(record['image']['bytes'])}"
    return {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "type": "message",
                    "content": [
                        {"type": "input_image", "image_url": image_url, "detail": "auto"},
                        {"type": "input_text", "text": text},
                    ],
                }
            ]
        },
        "expected_answer": answer,
        "question": question,
        # BabyVision's taskId is the stable id that `gym eval plan` and task selection need (row order is not).
        "task_id": str(record["taskId"]),
        "source_id": str(record["taskId"]),
        "category": record["type"],
        "l2_category": record["subtype"],
        "ans_type": record["ansType"],
    }


def prepare(output_fpath: Path = OUTPUT_FPATH, question_only: bool = False) -> Path:
    records = load_records(parquet_path())
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w") as f:
        for record in records:
            f.write(json.dumps(convert(record, question_only)) + "\n")
    print(f"Wrote {len(records)} rows to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    prepare()
