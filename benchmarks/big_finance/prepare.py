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
"""Download and convert the pinned 50-question BigFinanceBench public subset."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


ENV_DIR = Path(__file__).parent
DATA_DIR = ENV_DIR / "data"
SPEC_FPATH = ENV_DIR / "upstream_spec.json"
# Gym invokes benchmark prepare scripts from the repository root.
OUTPUT_FPATH = Path("benchmarks/big_finance/data/big_finance_public_50.jsonl")
EXPECTED_COUNT = 50
_UPSTREAM_SHA = "d794a65fe583edc6852b44c817b0a2aef33ca831"  # pragma: allowlist secret


def load_spec() -> dict:
    spec = json.loads(SPEC_FPATH.read_text(encoding="utf-8"))
    sha = spec.get("upstream_commit_id", "")
    if len(sha) != 40 or f"/{sha}/" not in spec.get("dataset_url", ""):
        raise ValueError("upstream_spec.json must pin the dataset URL to its 40-character commit")
    if sha != _UPSTREAM_SHA:
        raise ValueError(
            f"{SPEC_FPATH} describes upstream commit {sha}, but prepare.py is pinned "
            f"to {_UPSTREAM_SHA}; update the snapshot and pin together"
        )
    return spec


def build_tools(spec: dict | None = None) -> list[dict]:
    return json.loads(json.dumps((spec or load_spec())["tools"]))


def convert_rows(rows: list[dict], *, expected_count: int | None = None) -> list[dict]:
    spec = load_spec()
    tools = build_tools(spec)
    converted: list[dict] = []
    for row in rows:
        required = (
            "id",
            "query",
            "reference_answer",
            "rubric",
            "evaluation_only",
            "do_not_train",
            "benchmark_canary",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"dataset row missing required keys: {missing}")
        if row["evaluation_only"] is not True or row["do_not_train"] is not True:
            raise ValueError(f"{row['id']}: BigFinanceBench rows must remain evaluation_only and do_not_train")
        if not isinstance(row["benchmark_canary"], str) or not row["benchmark_canary"].strip():
            raise ValueError(f"{row['id']}: benchmark_canary must be a non-empty string")
        rubric = row["rubric"]
        if not isinstance(rubric, list) or not rubric:
            raise ValueError(f"{row['id']}: rubric must be a non-empty list")
        converted.append(
            {
                "id": row["id"],
                "query": row["query"],
                "reference_answer": row["reference_answer"],
                "rubric": rubric,
                "evaluation_only": row["evaluation_only"],
                "do_not_train": row["do_not_train"],
                "benchmark_canary": row["benchmark_canary"],
                "sources": row.get("sources", []),
                "license": spec["dataset_license"],
                "provenance": {
                    "repository": spec["repository"],
                    "upstream_commit_id": spec["upstream_commit_id"],
                    "dataset_url": spec["dataset_url"],
                    "attribution": (
                        "Big Finance benchmark, public release subset (n = 50), "
                        "Rogo Technologies (2026), licensed under CC BY 4.0."
                    ),
                },
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": spec["system_prompt"], "type": "message"},
                        {"role": "user", "content": row["query"], "type": "message"},
                    ],
                    "tools": tools,
                    "parallel_tool_calls": True,
                },
            }
        )
    if expected_count is not None and len(converted) != expected_count:
        raise ValueError(f"expected {expected_count} public questions, found {len(converted)}")
    return converted


def convert_file(input_file: Path, output_file: Path, *, expected_count: int | None = None) -> int:
    rows = [json.loads(line) for line in input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    converted = convert_rows(rows, expected_count=expected_count)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as stream:
        for row in converted:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(converted)


def _download() -> Path:
    spec = load_spec()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source = DATA_DIR / "upstream_public_50.jsonl"
    urllib.request.urlretrieve(spec["dataset_url"], source)
    return source


def prepare() -> Path:
    source = _download()
    count = convert_file(source, OUTPUT_FPATH, expected_count=EXPECTED_COUNT)
    print(f"Wrote {count} BigFinanceBench samples to {OUTPUT_FPATH}")
    return OUTPUT_FPATH.absolute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.input:
        output = args.output or args.input.with_name(f"{args.input.stem}.gym.jsonl")
        count = convert_file(args.input, output)
        print(f"Wrote {count} samples to {output}")
        return 0
    prepare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
