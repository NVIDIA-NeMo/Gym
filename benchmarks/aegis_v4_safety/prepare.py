# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the built-in Aegis v4 safety smoke benchmark."""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
SOURCE_PATH = BENCHMARK_DIR / "data" / "source.jsonl"
OUTPUT_PATH = BENCHMARK_DIR / "data" / "aegis_v4_safety_benchmark.jsonl"


def prepare(source: Path = SOURCE_PATH, output: Path = OUTPUT_PATH) -> Path:
    """Validate the project-authored prompts and write the benchmark input."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open(encoding="utf-8") as source_stream,
        output.open("w", encoding="utf-8") as output_stream,
    ):
        for line_number, line in enumerate(source_stream, start=1):
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("sample_id"), str)
                or not isinstance(row.get("prompt"), str)
                or not row["prompt"].strip()
            ):
                raise ValueError(f"invalid source row {line_number}")
            output_stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "dataset_name": "aegis_v4_safety",
                        "prompt": row["prompt"],
                        "responses_create_params": {"max_output_tokens": 131072},
                    }
                )
                + "\n"
            )
    return output


if __name__ == "__main__":
    prepare()
