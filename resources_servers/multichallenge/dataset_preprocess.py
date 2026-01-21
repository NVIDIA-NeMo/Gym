#!/usr/bin/env python3
"""
Preprocesses MultiChallenge dataset from individual JSON files to JSONL format.

Converts task files from:
  data/{split}/*.json  ->  data/{split}.jsonl

Each output line contains the task data formatted for the simple_agent.
"""

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from typing import Any


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


def process_task_file(filepath: Path) -> dict[str, Any]:
    """Process a single task JSON file into JSONL format."""
    with open(filepath, "r", encoding="utf-8") as f:
        task = json.load(f)
    
    metadata = task.get("metadata", {})
    task_id = metadata.get("taskId", filepath.stem)
    
    # Build the record for JSONL
    record = {
        "uuid": str(task_id),
        "task_id": task_id,
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


def process_split(data_dir: Path, split: str, output_dir: Path) -> int:
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
    parser = argparse.ArgumentParser(
        description="Convert MultiChallenge JSON files to JSONL format"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory containing the split subdirectories (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for JSONL files (default: same as data-dir)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["advanced", "vanilla"],
        help="Splits to process (default: advanced vanilla)",
    )
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = 0
    for split in args.splits:
        total += process_split(args.data_dir, split, output_dir)
    
    print(f"\nTotal: {total} records processed")


if __name__ == "__main__":
    main()
