#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert the original CVDP agentic dataset directly to NeMo-Gym JSONL.
#
# Expected input row shape:
#   {
#     "id": "...",
#     "system_message": "...",
#     "prompt": "...",
#     "context": {"path": "content", ...},
#     "patch": {"target/path.sv": "...", ...},
#     "harness": {"docker-compose.yml": "...", ...},
#     "categories": ["cid003", "medium"]
#   }
#
# Output row shape matches resources_servers/cvdp/scripts/convert_to_gym.py:
#   {
#     "responses_create_params": {"input": [{"role": "system", ...}, {"role": "user", ...}]},
#     "verifier_metadata": {...}
#   }

import argparse
import json
from pathlib import Path
from typing import Any


CODE_COMPREHENSION_CATEGORIES = {6, 8, 9, 10}
DIFFICULTY_LABELS = {"easy", "medium", "hard"}


def _category_num(categories: list[str]) -> int | None:
    for category in categories:
        if category.startswith("cid") and category[3:].isdigit():
            return int(category[3:])
    return None


def _difficulty(categories: list[str], row: dict[str, Any]) -> str:
    if isinstance(row.get("difficulty"), str):
        return row["difficulty"]
    for category in categories:
        if category in DIFFICULTY_LABELS:
            return category
    return ""


def _target_files(row: dict[str, Any]) -> list[str]:
    patch = row.get("patch") or {}
    if not isinstance(patch, dict):
        return []
    return list(patch.keys())


def _harness_files(row: dict[str, Any]) -> dict[str, str | None]:
    harness = row.get("harness") or {}
    if not isinstance(harness, dict):
        return {}
    if isinstance(harness.get("files"), dict):
        return harness["files"]
    return harness


def _context_files(row: dict[str, Any], target_files: list[str]) -> dict[str, str]:
    context = row.get("context") or {}
    if not isinstance(context, dict):
        return {}
    targets = set(target_files)
    return {path: content for path, content in context.items() if path not in targets and content}


def _subjective_reference(row: dict[str, Any]) -> str | None:
    output = row.get("output") or {}
    if isinstance(output, dict) and isinstance(output.get("response"), str):
        return output["response"]
    for key in ("reference", "answer", "expected_answer"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _user_prompt(row: dict[str, Any], target_files: list[str], include_output_hints: bool) -> str:
    # NOTE: context files are seeded into the agent's workspace on disk (see
    # verifier_metadata["context_files"]); the agent reads them with ls/cat. We do
    # NOT inline their contents into the prompt -- that would defeat the agentic
    # exploration the original CVDP benchmark intends and can blow past the model's
    # context window (e.g. tasks that ship a multi-hundred-KB lookup-table file).
    parts: list[str] = []

    prompt = row.get("prompt") or ""
    if prompt:
        parts.append(f"\nProvide me one answer for this request: {prompt}")

    if include_output_hints:
        if len(target_files) == 1:
            parts.append(
                "\nPlease provide your response as plain text without any JSON formatting. "
                f"Your response will be saved directly to: {target_files[0]}."
            )
        elif target_files:
            parts.append(f"\nName the files as: {target_files}.")

    return "\n".join(parts)


def _convert_row(row: dict[str, Any], include_output_hints: bool) -> dict[str, Any] | None:
    task_id = row.get("id")
    if not task_id:
        return None

    categories = row.get("categories") or []
    if not isinstance(categories, list):
        categories = []

    target_files = _target_files(row)
    cat_num = _category_num(categories)
    is_comprehension = cat_num in CODE_COMPREHENSION_CATEGORIES
    if not target_files and not is_comprehension:
        return None

    verifier_metadata: dict[str, Any] = {
        "task_id": task_id,
        "categories": categories,
        "difficulty": _difficulty(categories, row),
        "target_files": target_files,
        "harness_files": _harness_files(row),
    }

    context_files = _context_files(row, target_files)
    if context_files:
        verifier_metadata["context_files"] = context_files

    if is_comprehension:
        reference = _subjective_reference(row)
        if not reference:
            print(f"WARNING: no subjective reference for comprehension task {task_id}, skipping")
            return None
        verifier_metadata["subjective_reference"] = reference

    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": row.get("system_message") or ""},
                {"role": "user", "content": _user_prompt(row, target_files, include_output_hints)},
            ],
        },
        "verifier_metadata": verifier_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert original CVDP agentic JSONL to NeMo-Gym JSONL")
    parser.add_argument("--dataset", required=True, help="Original CVDP agentic dataset JSONL")
    parser.add_argument("--output", required=True, help="Output NeMo-Gym JSONL")
    parser.add_argument(
        "--no-output-hints",
        action="store_true",
        help="Do not append target-file formatting hints to the user message.",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with open(args.dataset, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            gym_row = _convert_row(row, include_output_hints=not args.no_output_hints)
            if gym_row is None:
                skipped += 1
                continue
            fout.write(json.dumps(gym_row) + "\n")
            written += 1

    print(f"Wrote {written} entries to {args.output}")
    if skipped:
        print(f"Skipped {skipped} entries (missing id, target files, or subjective reference)")


if __name__ == "__main__":
    main()
