#!/usr/bin/env python3
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
"""Count tokens in stored LMArena rollout fields."""

import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken


FIELDS = ("baseline_answer", "policy_answer", "policy_reasoning")


def read_jsonl(path: Path) -> list[dict]:
    with path.expanduser().open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_rollout_tokens(rollouts: Path, tokenizer: str) -> dict:
    """Return token counts and rollout-level diagnostics."""
    records = read_jsonl(rollouts)
    if not records:
        raise ValueError(f"No records found in {rollouts}")

    encoding = tiktoken.encoding_for_model(tokenizer)
    counts = {field: [] for field in FIELDS}
    empty = {field: 0 for field in FIELDS}

    for index, record in enumerate(records):
        if "policy_answer" not in record:
            raise ValueError(f"Record {index} is missing policy_answer")
        for field in FIELDS:
            if field not in record:
                continue
            text = record[field] or ""
            empty[field] += not text
            # Empty responses remain zero so aggregate lengths include failures.
            counts[field].append(len(encoding.encode(text, disallowed_special=())))

    failed_judgments = 0
    for record in records:
        incomplete_reason = ((record.get("response") or {}).get("incomplete_details") or {}).get("reason")
        if record.get("self_comparison") or (
            record.get("category") == "lmarena_v3" and incomplete_reason == "max_output_tokens"
        ):
            continue
        games = record.get("games") or []
        failed_judgments += len(games) != 2 or any((game or {}).get("verdict") is None for game in games)
    # Detect reasoning tags leaked into the final answer, including unclosed tags.
    policy_answers = (record["policy_answer"] or "" for record in records)
    contains_think_block = sum("<think>" in answer or "<thinking>" in answer for answer in policy_answers)

    return {
        "records": len(records),
        "failed_judgments": failed_judgments,
        "contains_think_block": contains_think_block,
        "empty": empty,
        "counts": counts,
    }


def report_rollout_tokens(rollouts: Path, tokenizer: str, max_number_tokens: int | None = None) -> None:
    summary = count_rollout_tokens(rollouts, tokenizer)
    print(f"path: {rollouts.expanduser()}")
    print(f"tokenizer: {tokenizer}")
    print(f"records: {summary['records']}")
    print(f"failed_judgments: {summary['failed_judgments']}")

    for field in FIELDS:
        values = np.asarray(summary["counts"][field])
        print(f"\n{field}:")
        if not len(values):
            print("  not available")
            continue
        print(f"  empty: {summary['empty'][field]}")
        if field == "policy_answer":
            print(f"  contains_think_block: {summary['contains_think_block']}")
        print(f"  mean: {values.mean():.2f}")
        if max_number_tokens is not None:
            capped_mean = np.minimum(values, max_number_tokens).mean()
            print(f"  mean_capped_at_{max_number_tokens}: {capped_mean:.2f}")
        print(f"  median: {np.median(values):.2f}")
        print(f"  p05: {np.percentile(values, 5):.2f}")
        print(f"  p95: {np.percentile(values, 95):.2f}")
        print(f"  min: {values.min()}")
        print(f"  max: {values.max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--tokenizer", default="gpt-4o")
    parser.add_argument("--max_number_tokens", type=int)
    args = parser.parse_args()
    if args.max_number_tokens is not None and args.max_number_tokens <= 0:
        parser.error("--max_number_tokens must be positive")
    report_rollout_tokens(args.rollouts, args.tokenizer, args.max_number_tokens)


if __name__ == "__main__":
    main()
