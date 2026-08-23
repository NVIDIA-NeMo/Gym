# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare source rows for the terminal_bench_2_1 benchmark."""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
SOURCE_PATH = BENCHMARK_DIR / "data" / "source.jsonl"
OUTPUT_PATH = BENCHMARK_DIR / "data" / "example.jsonl"


def prepare(source: Path = SOURCE_PATH, output: Path = OUTPUT_PATH) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open(encoding="utf-8") as source_stream,
        output.open("w", encoding="utf-8") as output_stream,
    ):
        for line_number, line in enumerate(source_stream, start=1):
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("question"), str)
                or not isinstance(row.get("expected_answer"), str)
            ):
                raise ValueError(f"invalid source row {line_number}")
            output_stream.write(
                json.dumps({"question": row["question"], "expected_answer": row["expected_answer"]}) + "\n"
            )
    return output


if __name__ == "__main__":
    prepare()
