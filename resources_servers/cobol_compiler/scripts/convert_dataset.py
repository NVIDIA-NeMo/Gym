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

"""Convert DomainForge cobol_multipl_eval.json to NeMo-Gym JSONL format.

Usage:
    python convert_dataset.py \
        --input ~/projects/domainforge/datasets/cobol_multipl_eval.json \
        --output ../data/cobol_multipl_eval.jsonl \
        --system-prompt ../prompts/cobol_basic.txt \
        --example-output ../data/example.jsonl \
        --example-count 5
"""

import argparse
import json
import sys
from pathlib import Path


def convert_problem(problem: dict, system_prompt: str) -> dict:
    """Convert a single DomainForge problem to NeMo-Gym JSONL format."""
    # Build user prompt from problem description + I/O format spec
    user_content = problem["prompt"]
    if problem.get("format_specification"):
        try:
            format_spec = json.loads(problem["format_specification"])
            if "io_format_spec" in format_spec:
                user_content += "\n\n" + format_spec["io_format_spec"]
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        },
        "verifier_metadata": {
            "test_cases": problem["test_cases"],
            "task_id": problem["task_id"],
            "entry_point": problem["entry_point"],
            "category": problem.get("category", "unknown"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert DomainForge COBOL dataset to NeMo-Gym JSONL")
    parser.add_argument("--input", required=True, help="Path to cobol_multipl_eval.json")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--system-prompt", required=True, help="Path to system prompt text file")
    parser.add_argument("--example-output", default=None, help="Path for example.jsonl subset")
    parser.add_argument("--example-count", type=int, default=5, help="Number of examples to include")
    args = parser.parse_args()

    prompt_path = Path(args.system_prompt).expanduser()
    if not prompt_path.exists():
        print(f"Error: system prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text().strip()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems from {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = []
    for problem in problems:
        converted.append(convert_problem(problem, system_prompt))

    with open(output_path, "w") as f:
        for entry in converted:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(converted)} entries to {output_path}")

    if args.example_output:
        example_path = Path(args.example_output)
        example_path.parent.mkdir(parents=True, exist_ok=True)
        with open(example_path, "w") as f:
            for entry in converted[: args.example_count]:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {args.example_count} example entries to {example_path}")


if __name__ == "__main__":
    main()
