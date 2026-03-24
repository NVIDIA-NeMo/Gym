# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert the SWE-Multilingual HuggingFace dataset to NeMo-Gym JSONL format.

Usage:
    python scripts/prepare_swe_multilingual.py

Outputs:
    responses_api_agents/swe_agents/data/swe_multilingual_for_sweagent_and_openhands.jsonl
    responses_api_agents/swe_agents/data/swe_multilingual_example.jsonl  (5 entries)

Requirements:
    pip install datasets
"""

import json
import sys
from pathlib import Path


DATASET_NAME = "swe-bench/SWE-bench_Multilingual"
SPLIT = "test"
OUTPUT_PATH = Path("responses_api_agents/swe_agents/data/swe_multilingual_for_sweagent_and_openhands.jsonl")
EXAMPLE_PATH = Path("responses_api_agents/swe_agents/data/swe_multilingual_example.jsonl")
EXAMPLE_COUNT = 5

# Default inference params (same as SWE-Verified)
DEFAULT_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8


def convert_row(row: dict) -> dict:
    """Convert a single HuggingFace SWE-Multilingual row to Gym JSONL format."""
    instance_id = row["instance_id"]
    base_commit = row["base_commit"]
    problem_statement = row["problem_statement"]
    # SWE-Multilingual uses `patch` as the gold/reference patch
    golden_patch = row.get("patch", "")

    # Serialize the full instance as a JSON string for utils.py extract_problem_info()
    instance_dict = {
        "instance_id": instance_id,
        "base_commit": base_commit,
        "dataset_name": DATASET_NAME,
        "split": SPLIT,
        "problem_statement": problem_statement,
        "golden_patch": golden_patch,
        "repo": row.get("repo", ""),
        "patch": row.get("patch", ""),
        "test_patch": row.get("test_patch", ""),
        "hints_text": row.get("hints_text", ""),
        "created_at": str(row.get("created_at", "")),
        "version": row.get("version", ""),
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": row.get("PASS_TO_PASS", []),
    }

    return {
        "responses_create_params": {
            "input": [],
            "metadata": {
                "instance_id": instance_id,
                "base_commit": base_commit,
                "dataset_name": DATASET_NAME,
                "split": SPLIT,
                "problem_statement": problem_statement,
                "golden_patch": golden_patch,
                "instance_dict": json.dumps(instance_dict),
            },
            "model": DEFAULT_MODEL,
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
        },
        "agent_ref": {"type": "responses_api_agents", "name": "swe_multilingual_openhands"},
        # Duplicate raw fields at top level for compatibility / inspection
        "repo": row.get("repo", ""),
        "instance_id": instance_id,
        "base_commit": base_commit,
        "patch": row.get("patch", ""),
        "test_patch": row.get("test_patch", ""),
        "problem_statement": problem_statement,
        "hints_text": row.get("hints_text", ""),
        "created_at": str(row.get("created_at", "")),
        "version": row.get("version", ""),
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": row.get("PASS_TO_PASS", []),
    }


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {DATASET_NAME} (split={SPLIT}) from HuggingFace...")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    print(f"  {len(dataset)} instances loaded.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with OUTPUT_PATH.open("w") as f:
        for row in dataset:
            gym_row = convert_row(row)
            rows.append(gym_row)
            f.write(json.dumps(gym_row) + "\n")

    print(f"Written {len(rows)} rows to {OUTPUT_PATH}")

    # Write example file (first EXAMPLE_COUNT rows)
    with EXAMPLE_PATH.open("w") as f:
        for row in rows[:EXAMPLE_COUNT]:
            f.write(json.dumps(row) + "\n")

    print(f"Written {min(EXAMPLE_COUNT, len(rows))} example rows to {EXAMPLE_PATH}")


if __name__ == "__main__":
    main()
