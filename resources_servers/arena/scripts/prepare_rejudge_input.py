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
"""Prepare saved LMArena rollouts for rejudging."""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def response_from_text(text: str) -> dict:
    return {
        "id": "saved-response",
        "created_at": 0.0,
        "model": "saved-rollout",
        "object": "response",
        "output": [
            {
                "id": "saved-message",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def answer_text(row: dict) -> str:
    if row.get("policy_answer") is not None:
        return row["policy_answer"] or ""
    assistant_messages = [message for message in row.get("messages", []) if message.get("role") == "assistant"]
    content = assistant_messages[-1]["content"] if assistant_messages else ""
    while isinstance(content, dict):
        content = content.get("answer", "")
    return content if isinstance(content, str) else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--benchmark-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--answer-field", help="Use this text field as the saved model response")
    parser.add_argument(
        "--allow-unmatched-prompts", action="store_true", help="Skip rollouts absent from the benchmark input"
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")

    benchmark = {row["question_id"]: row for row in read_jsonl(args.benchmark_input)}
    prepared = []
    unmatched = []
    for rollout in read_jsonl(args.rollouts):
        question_id = rollout.get("question_id") or rollout.get("uid")
        if question_id not in benchmark:
            unmatched.append(question_id)
            continue
        row = benchmark[question_id].copy()
        if args.answer_field:
            answer = rollout.get(args.answer_field)
            if answer is None:
                raise ValueError(f"Missing {args.answer_field!r} for question {question_id}")
            response = response_from_text(answer)
        else:
            response = rollout.get("response") or response_from_text(answer_text(rollout))
        row["response"] = response
        prepared.append(row)

    if unmatched and not args.allow_unmatched_prompts:
        raise ValueError(
            f"{len(unmatched)} rollouts are absent from the benchmark input; use --allow-unmatched-prompts to skip them"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared)
    print(f"wrote {len(prepared)} rows: {args.output}")
    if unmatched:
        print(f"skipped {len(unmatched)} unmatched rollouts")


if __name__ == "__main__":
    main()
