# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#!/usr/bin/env python3
"""
Preprocesses MultiChallenge dataset to the format required by NeMo-Gym.

Supports two input modes:
1. JSONL mode (default): Reads from pre-compiled .jsonl files
   Input: data/{split}.jsonl  ->  Output: data/{split}_preprocessed.jsonl

2. JSON directory mode: Reads from individual JSON files
   Input: data/{split}/*.json  ->  Output: data/{split}.jsonl

Each output line contains the task data formatted for the simple_agent.
"""

import argparse
import json
from pathlib import Path
from typing import Any


# Hardcoded path for raw multichallenge data
DEFAULT_RAW_DATA_DIR = Path("/lustre/fsw/portfolios/llmservice/users/mfathi/data/multichallenge")


def build_input_messages(task: dict) -> list[dict]:
    """
    Build the input messages for the policy model from the task data.

    Excludes 'thinking' role messages and the final user message (which the model should respond to).
    """
    messages = task.get("messages", [])
    system_prompt = task.get("system", None)

    input_msgs = []

    # Add system message if present
    if system_prompt:
        input_msgs.append({"role": "system", "content": system_prompt})

    # Add all messages (the agent will handle the conversation flow)
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Skip thinking messages - these shouldn't be sent to the policy model
        if role == "thinking":
            continue

        input_msgs.append({"role": role, "content": content})

    return input_msgs


def build_context_string(task: dict) -> str:
    """Build a readable context string from messages for the judge."""
    messages = task.get("messages", [])
    system_prompt = task.get("system", None)

    context_parts = []

    if system_prompt:
        context_parts.append(f"[SYSTEM]: {system_prompt}")

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Skip thinking messages
        if role == "thinking":
            continue

        role_label = role.upper()
        context_parts.append(f"[{role_label}]: {content}")

    return "\n\n".join(context_parts)


def process_task(task: dict, fallback_id: str = "unknown") -> dict[str, Any]:
    """Process a single task dict into the preprocessed JSONL format."""
    metadata = task.get("metadata", {})
    task_id = metadata.get("taskId", fallback_id)

    # Build the record for JSONL
    record = {
        "uuid": str(task_id),
        "task_id": task_id,
        # Agent reference - tells NeMo-Gym which agent to route this to
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "multichallenge_simple_agent",
        },
        # Input messages wrapped in responses_create_params (required by ng_collect_rollouts)
        "responses_create_params": {
            "input": build_input_messages(task),
        },
        # Rubric for evaluation
        "rubric": task.get("rubric", []),
        # Pre-built context string for the judge
        "context": build_context_string(task),
        # Full metadata
        "metadata": {
            **metadata,
            "messages": task.get("messages", []),
            "system": task.get("system", None),
            "ground_truth_answer": task.get("ground_truth_answer", None),
        },
    }

    return record


def process_task_file(filepath: Path) -> dict[str, Any]:
    """Process a single task JSON file into JSONL format."""
    with open(filepath, "r", encoding="utf-8") as f:
        task = json.load(f)
    return process_task(task, fallback_id=filepath.stem)


def process_jsonl_file(input_file: Path, output_file: Path) -> int:
    """Process a JSONL file where each line is a task."""
    count = 0
    errors = 0

    print(f"Processing JSONL file: {input_file}")

    with open(input_file, "r", encoding="utf-8") as in_f, open(output_file, "w", encoding="utf-8") as out_f:
        for line_num, line in enumerate(in_f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                record = process_task(task, fallback_id=f"line_{line_num}")
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except json.JSONDecodeError as e:
                print(f"  Warning: Invalid JSON on line {line_num}: {e}")
                errors += 1
            except Exception as e:
                print(f"  Error processing line {line_num}: {e}")
                errors += 1

    print(f"  Wrote {count} records to {output_file}" + (f" ({errors} errors)" if errors else ""))
    return count


def process_split_jsonl(data_dir: Path, split: str, output_dir: Path) -> int:
    """Process a split from a JSONL file."""
    input_file = data_dir / f"{split}.jsonl"
    if not input_file.exists():
        print(f"Warning: JSONL file not found: {input_file}")
        return 0

    output_file = output_dir / f"{split}.jsonl"
    return process_jsonl_file(input_file, output_file)


def process_split_json_dir(data_dir: Path, split: str, output_dir: Path) -> int:
    """Process all JSON files in a split directory."""
    split_dir = data_dir / split
    if not split_dir.exists():
        print(f"Warning: Split directory not found: {split_dir}")
        return 0

    output_file = output_dir / f"{split}.jsonl"
    count = 0

    json_files = sorted(split_dir.glob("*.json"))
    print(f"Processing {len(json_files)} files from {split}...")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for filepath in json_files:
            try:
                record = process_task_file(filepath)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    print(f"Wrote {count} records to {output_file}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert MultiChallenge data to NeMo-Gym JSONL format")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help=f"Directory containing the data (default: {DEFAULT_RAW_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory for preprocessed JSONL files (default: ./data)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["advanced", "vanilla"],
        help="Splits to process (default: advanced vanilla)",
    )
    parser.add_argument(
        "--mode",
        choices=["jsonl", "json-dir"],
        default="jsonl",
        help="Input mode: 'jsonl' reads {split}.jsonl files, 'json-dir' reads {split}/*.json directories (default: jsonl)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Splits: {args.splits}")
    print()

    total = 0
    for split in args.splits:
        if args.mode == "jsonl":
            total += process_split_jsonl(args.data_dir, split, args.output_dir)
        else:
            total += process_split_json_dir(args.data_dir, split, args.output_dir)

    print(f"\nTotal: {total} records processed")


if __name__ == "__main__":
    main()
