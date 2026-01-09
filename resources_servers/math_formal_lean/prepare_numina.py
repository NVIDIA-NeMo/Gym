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

"""Prepare NuminaMath-LEAN dataset for NeMo Gym.

Dataset: https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN

Filters applied:
- author == 'human'
- ground_truth_type in ['complete', 'statement']
- win_rate > 0.01 and win_rate < 0.95

This yields approximately 4394 problems.
"""

import argparse
import json
import re
from pathlib import Path


PROOF_PROMPT_TEMPLATE = """Complete the following Lean 4 code:

```lean4
{header}{informal_prefix}{formal_statement}
  sorry
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""


def _ensure_header_ends_with_by(text: str) -> str:
    """Ensure the formal statement ends with ':= by' and a newline."""
    marker = ":= by"
    idx = text.rfind(marker)
    if idx != -1:
        return text[: idx + len(marker)] + "\n"
    return text


def clean_lean_snippet(text: str | None) -> str | None:
    """Clean up a Lean snippet by removing sorry and ensuring proper ending."""
    if text is None:
        return None
    cleaned = text.replace(" by sorry", " by").replace("by sorry", "by").replace("sorry", "")
    cleaned = _ensure_header_ends_with_by(cleaned)
    return cleaned


def parse_formal_statement(formal_statement: str) -> tuple[str, str, str]:
    """Parse NuminaMath formal_statement into (header, informal_prefix, theorem).

    The formal_statement typically has this structure:
    - import Mathlib
    - optional open statements
    - optional docstring (/-...-/ or /--...--/)
    - theorem declaration ending with := by

    Returns:
        header: Import and open statements
        informal_prefix: The docstring (problem statement)
        theorem: The theorem declaration
    """
    lines = formal_statement.split('\n')
    header_lines = []
    informal_lines = []
    theorem_lines = []

    in_docstring = False
    docstring_started = False
    theorem_started = False

    for line in lines:
        stripped = line.strip()

        # Check for theorem/lemma/example start
        if not theorem_started and re.match(r'^(theorem|lemma|example)\s+', stripped):
            theorem_started = True

        if theorem_started:
            theorem_lines.append(line)
        elif in_docstring:
            informal_lines.append(line)
            # Check for docstring end
            if '-/' in line or '--/' in line:
                in_docstring = False
        elif '/-' in stripped or '/--' in stripped:
            # Docstring start
            in_docstring = True
            docstring_started = True
            informal_lines.append(line)
            # Check if docstring ends on same line
            if '-/' in stripped.split('/-', 1)[-1] or '--/' in stripped.split('/--', 1)[-1]:
                in_docstring = False
        else:
            # Header (imports, opens, etc.)
            if stripped:  # Only add non-empty lines to header
                header_lines.append(line)

    header = '\n'.join(header_lines)
    if header and not header.endswith('\n'):
        header += '\n'
    if header:
        header += '\n'  # Extra newline before docstring/theorem

    informal_prefix = '\n'.join(informal_lines)
    if informal_prefix and not informal_prefix.endswith('\n'):
        informal_prefix += '\n'

    theorem = '\n'.join(theorem_lines)
    theorem = clean_lean_snippet(theorem) or ""

    return header, informal_prefix, theorem


def process_entry(entry: dict) -> dict:
    """Process a NuminaMath entry into NeMo Gym format."""
    uuid = entry.get("uuid", "")
    problem = entry.get("problem", "")
    formal_statement_raw = entry.get("formal_statement", "")
    ground_truth_type = entry.get("ground_truth_type", "")
    problem_type = entry.get("problem_type", "")

    # Parse the formal statement
    header, informal_prefix, theorem = parse_formal_statement(formal_statement_raw)

    # Build prompt
    prompt = PROOF_PROMPT_TEMPLATE.format(
        header=header,
        informal_prefix=informal_prefix,
        formal_statement=theorem,
    )

    gym_entry = {
        "responses_create_params": {
            "input": [{"role": "user", "content": prompt}],
        },
        "header": header,
        "formal_statement": theorem,
        "informal_prefix": informal_prefix,
        "name": uuid,
        "problem": problem,  # Original natural language problem
        "problem_type": problem_type,
        "ground_truth_type": ground_truth_type,
    }

    return gym_entry


def load_and_filter_dataset() -> list[dict]:
    """Load NuminaMath-LEAN from HuggingFace and apply filters."""
    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        raise ImportError("Please install datasets and pandas: pip install datasets pandas")

    print("Loading NuminaMath-LEAN dataset from HuggingFace...")
    ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train")
    df = pd.DataFrame(ds)

    # Extract win_rate from nested rl_data dict
    df['win_rate'] = df['rl_data'].apply(lambda x: x.get('win_rate', 0) if isinstance(x, dict) else 0)

    # Apply filters
    candidates = df[
        (df['author'] == 'human') &
        (df['ground_truth_type'].isin(['complete', 'statement'])) &
        (df['win_rate'] > 0.01) &
        (df['win_rate'] < 0.95)
    ]

    print(f"Filtered to {len(candidates)} problems (from {len(df)} total)")

    return candidates.to_dict('records')


def save_data(data: list, output_file: str) -> None:
    """Save processed data to JSONL file."""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} entries to {output_file}")


def main(output_name: str = "numina", limit: int | None = None) -> None:
    """Main entry point."""
    data_dir = Path(__file__).absolute().parent / "data"

    # Load and filter
    raw_data = load_and_filter_dataset()

    if limit:
        raw_data = raw_data[:limit]
        print(f"Limited to {limit} entries")

    # Process entries
    processed = []
    for entry in raw_data:
        gym_entry = process_entry(entry)
        processed.append(gym_entry)

    # Save
    output_file = str(data_dir / f"{output_name}.jsonl")
    save_data(processed, output_file)

    print("Dataset preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NuminaMath-LEAN dataset for NeMo Gym")
    parser.add_argument(
        "--output-name",
        default="numina",
        help="Output filename (without .jsonl extension)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries (for testing)",
    )
    args = parser.parse_args()

    main(args.output_name, args.limit)
