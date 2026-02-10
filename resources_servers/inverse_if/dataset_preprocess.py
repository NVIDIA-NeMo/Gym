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
Preprocesses the Inverse IF dataset to the format required by NeMo-Gym.

Reads individual JSON task files from the raw data directory and produces
a single JSONL file where each line is a preprocessed task ready for the
NeMo-Gym rollout pipeline.

Key transformations:
- Extracts prompt, reference response, judge template, and judge system prompt
  from the heterogeneous ``messages`` array
- Normalises inconsistent rubric key names (criteria, criteria1, rule, question, …)
  into a uniform ``criteria`` field
- Falls back to parsing ``response_reference`` when the ``rubrics`` array is empty
- Wraps the prompt in ``responses_create_params`` for ``ng_collect_rollouts``
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Hardcoded path for raw Inverse IF data
DEFAULT_RAW_DATA_DIR = Path("/lustre/fsw/portfolios/llmservice/users/mfathi/data/inverse_if")


# ---------------------------------------------------------------------------
# Rubric normalisation
# ---------------------------------------------------------------------------


def _normalise_rubric_item(item: dict) -> dict[str, str]:
    """
    Normalise a single rubric item to ``{"id": ..., "criteria": ...}``.

    The raw data uses inconsistent key names for the criteria text:
      criteria, criteria1, criteria2, criteria 1, rule, question, …

    We extract whichever non-``id`` key is present and map it to ``criteria``.
    """
    criterion_id = item.get("id", "")
    # Find the criteria text: any key that is not "id"
    criteria_text = ""
    for key, value in item.items():
        if key == "id":
            continue
        criteria_text = str(value)
        break  # take the first non-id key

    return {"id": criterion_id, "criteria": criteria_text}


def _extract_rubrics(task: dict) -> list[dict[str, str]]:
    """
    Extract and normalise rubric items from a task.

    Prefers the top-level ``rubrics`` array.  If it is empty, falls back to
    parsing the ``response_reference`` message (a JSON string containing the
    criteria list).  Some response_reference fields are not clean JSON — they
    may have prose preambles or missing array brackets — so we also try
    extracting individual ``{"id": ..., ...}`` objects via regex.
    """
    raw_rubrics = task.get("rubrics", [])

    # Fallback: parse response_reference when rubrics array is empty
    if not raw_rubrics:
        for msg in task.get("messages", []):
            if msg.get("role") == "response_reference":
                content = msg.get("content", "")

                # Try 1: parse as a clean JSON array
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        raw_rubrics = parsed
                        break
                except (json.JSONDecodeError, KeyError):
                    pass

                # Try 2: extract individual JSON objects with an "id" field
                for obj_match in re.finditer(r"\{[^{}]*\"id\"[^{}]*\}", content):
                    try:
                        obj = json.loads(obj_match.group(0))
                        if "id" in obj:
                            raw_rubrics.append(obj)
                    except json.JSONDecodeError:
                        continue
                break

    return [_normalise_rubric_item(item) for item in raw_rubrics]


# ---------------------------------------------------------------------------
# Judge prompt template normalisation
# ---------------------------------------------------------------------------

# Canonical placeholder names used by app.py
_CANONICAL_PLACEHOLDERS = ("prompt", "model_response", "standard_response", "criteria")

# Map every observed variant to the canonical name
_PLACEHOLDER_ALIASES: dict[str, str] = {
    # canonical → canonical (identity)
    "prompt": "prompt",
    "model_response": "model_response",
    "standard_response": "standard_response",
    "criteria": "criteria",
    # typo variant (~64 tasks)
    "model_resposne": "model_response",
    # alternate-name variant (~254 tasks)
    "response": "model_response",
    "response_reference": "criteria",
}


def _normalise_judge_prompt_template(template: str | None) -> str | None:
    """
    Rewrite placeholder names in a judge prompt template to the canonical set.

    Observed variants in the raw data:
      - ``{model_resposne}``    → ``{model_response}``  (typo, ~64 tasks)
      - ``{response}``          → ``{model_response}``  (alias, ~254 tasks)
      - ``{response_reference}`` → ``{criteria}``       (alias, ~254 tasks)

    Returns ``None`` unchanged so callers can still detect missing templates.
    """
    if not template:
        return template

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        canonical = _PLACEHOLDER_ALIASES.get(name)
        if canonical is not None:
            return "{" + canonical + "}"
        # Unknown placeholder — leave as-is
        return match.group(0)

    return re.sub(r"\{(\w+)\}", _replace, template)


# ---------------------------------------------------------------------------
# Message extraction helpers
# ---------------------------------------------------------------------------


def _get_message_by_role(task: dict, role: str) -> str:
    """Return the content of the first message matching *role*, or ``""``."""
    for msg in task.get("messages", []):
        if msg.get("role") == role:
            return msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------


def process_task(task: dict, fallback_id: str = "unknown") -> dict[str, Any]:
    """Convert a single raw task dict into the preprocessed JSONL record."""
    metadata = task.get("metadata", {})
    task_id = metadata.get("task_id", fallback_id)

    prompt = _get_message_by_role(task, "prompt")
    reference_response = _get_message_by_role(task, "response")
    judge_prompt_template = _normalise_judge_prompt_template(
        _get_message_by_role(task, "judge_prompt_template") or None
    )
    judge_system_prompt = _get_message_by_role(task, "judge_system_prompt") or None
    rubric = _extract_rubrics(task)

    record: dict[str, Any] = {
        "uuid": str(task_id),
        "task_id": task_id,
        # Agent reference — tells NeMo-Gym which agent to route this to
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "inverse_if_simple_agent",
        },
        # Input wrapped for ng_collect_rollouts
        "responses_create_params": {
            "input": [{"role": "user", "content": prompt}],
        },
        # Per-task evaluation data
        "rubric": rubric,
        "reference_response": reference_response,
        "prompt": prompt,
        # Per-task judge configuration (None → server will use config defaults)
        "judge_prompt_template": judge_prompt_template,
        "judge_system_prompt": judge_system_prompt,
        # Full metadata for reference
        "metadata": {
            **metadata,
            "response_reference_raw": _get_message_by_role(task, "response_reference"),
        },
    }
    return record


def process_task_file(filepath: Path) -> dict[str, Any]:
    """Process a single JSON task file into a preprocessed record."""
    with open(filepath, "r", encoding="utf-8") as f:
        task = json.load(f)
    return process_task(task, fallback_id=filepath.stem)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def process_directory(data_dir: Path, output_file: Path) -> int:
    """Process all JSON files in *data_dir* and write to *output_file*."""
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"Warning: no JSON files found in {data_dir}")
        return 0

    print(f"Processing {len(json_files)} files from {data_dir} ...")

    count = 0
    errors = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for filepath in json_files:
            try:
                record = process_task_file(filepath)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"  Error processing {filepath.name}: {e}")
                errors += 1

    print(f"  Wrote {count} records to {output_file}" + (f" ({errors} errors)" if errors else ""))
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw Inverse IF JSON files to NeMo-Gym JSONL format"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help=f"Directory containing raw JSON task files (default: {DEFAULT_RAW_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory for preprocessed JSONL files (default: ./data)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="inverse_if.jsonl",
        help="Name of the output JSONL file (default: inverse_if.jsonl)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / args.output_name

    print(f"Input directory:  {args.data_dir}")
    print(f"Output file:      {output_file}")
    print()

    total = process_directory(args.data_dir, output_file)
    print(f"\nTotal: {total} records processed")


if __name__ == "__main__":
    main()
