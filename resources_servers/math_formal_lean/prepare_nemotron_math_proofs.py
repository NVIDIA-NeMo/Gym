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

"""Prepare Nemotron-Math-Proofs-v1 dataset for NeMo Gym.

Dataset: https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1

This script filters the dataset to create a challenging subset for RL training:
- Excludes nano_v3 entries (used for SFT training)
- Selects problems based on difficulty (initial success count from pass@4)

Default subset (9,605 problems):
- Hard (1/4 initial succeeded): 3,904
- Medium (2/4 initial succeeded): 2,450
- Easy (3/4 initial succeeded): 3,251

Difficulty is based on how many of the 4 independent proof attempts (pass@4)
succeeded on the initial prompt without needing error correction.
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


def extract_theorem_name(formal_statement: str | None) -> str | None:
    """Extract theorem name (e.g., 'problem_123456') from formal statement."""
    if not formal_statement:
        return None
    match = re.search(r"theorem\s+(\w+)", formal_statement)
    return match.group(1) if match else None


def get_prompt_type(messages: list | None) -> str:
    """Classify prompt type from messages."""
    if not messages or len(messages) == 0:
        return "no_messages"
    user_content = messages[0].get("content", "")
    if user_content.startswith("Complete the following"):
        return "initial"
    elif "proof attempt" in user_content.lower():
        return "error_correction"
    return "other"


def extract_informal_prefix(formal_statement: str) -> str:
    """Extract docstring/informal prefix from formal statement if present."""
    # Look for /-- ... -/ or /- ... -/ style comments before theorem
    match = re.search(r"(/\*\*.*?\*/|/\-\-.*?\-/|/\-.*?\-/)", formal_statement, re.DOTALL)
    if match:
        return match.group(1) + "\n"
    return ""


def process_entry(theorem_name: str, row: dict) -> dict:
    """Process a theorem into NeMo Gym format."""
    formal_statement = row["formal_statement"]
    lean_header = row["lean_header"] or ""
    problem = row["problem"] or ""

    # Extract informal prefix from formal_statement if present
    informal_prefix = extract_informal_prefix(formal_statement)

    # Clean up formal_statement - ensure it ends with 'by sorry' pattern
    # The formal_statement should already have 'by sorry' at the end
    stmt = formal_statement
    if not re.search(r"by\s+sorry\s*$", stmt):
        # Add 'by sorry' if missing
        stmt = stmt.rstrip() + " := by sorry"

    # Build prompt
    prompt = PROOF_PROMPT_TEMPLATE.format(
        header=lean_header + "\n" if lean_header else "",
        informal_prefix=informal_prefix,
        formal_statement=stmt.replace(" := by sorry", " := by\n"),
    )

    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": prompt}],
        },
        "header": lean_header,
        "formal_statement": stmt,
        "informal_prefix": informal_prefix,
        "name": theorem_name,
        "problem": problem,
        "source": row.get("source", ""),
        "difficulty": row.get("difficulty", ""),
        "initial_success_count": row.get("initial_count", 0),
    }


def load_and_filter_dataset(
    dataset_path: str,
    exclude_nano_v3: bool = True,
    difficulties: list[str] | None = None,
) -> list[dict]:
    """Load Nemotron-Math-Proofs-v1 and filter by difficulty.

    Args:
        dataset_path: Path to saved dataset (from datasets.save_to_disk)
        exclude_nano_v3: Whether to exclude entries used in nano_v3 SFT
        difficulties: List of difficulties to include ['hard', 'medium', 'easy', 'very_easy']
                     Default: ['hard', 'medium', 'easy']

    Returns:
        List of processed entries ready for Gym
    """
    try:
        import pandas as pd
        from datasets import load_from_disk
    except ImportError:
        raise ImportError("Please install datasets and pandas: pip install datasets pandas")

    if difficulties is None:
        difficulties = ["hard", "medium", "easy"]

    print(f"Loading dataset from {dataset_path}...")
    ds = load_from_disk(dataset_path)
    df = pd.DataFrame(ds["lean"])

    print(f"Total entries: {len(df):,}")

    # Extract theorem name
    df["theorem_name"] = df["formal_statement"].apply(extract_theorem_name)

    # Classify prompt type
    df["prompt_type"] = df["messages"].apply(get_prompt_type)

    # Check nano_v3
    df["in_nano_v3"] = df["used_in"].apply(lambda x: "nano_v3" in x if x else False)

    # Group by theorem
    theorem_stats = df.groupby("theorem_name").agg(
        {
            "prompt_type": list,
            "in_nano_v3": "any",
            "formal_statement": "first",
            "lean_header": "first",
            "problem": "first",
            "source": "first",
        }
    )

    # Count initial successes per theorem
    theorem_stats["initial_count"] = theorem_stats["prompt_type"].apply(lambda x: x.count("initial"))
    theorem_stats["error_count"] = theorem_stats["prompt_type"].apply(lambda x: x.count("error_correction"))

    # Filter to solved theorems (at least one initial success)
    solved = theorem_stats[theorem_stats["initial_count"] > 0].copy()
    print(f"Solved theorems: {len(solved):,}")

    # Exclude nano_v3 if requested
    if exclude_nano_v3:
        solved = solved[~solved["in_nano_v3"]]
        print(f"After excluding nano_v3: {len(solved):,}")

    # Assign difficulty based on initial success count
    def assign_difficulty(initial_count: int) -> str:
        if initial_count >= 4:
            return "very_easy"
        elif initial_count == 3:
            return "easy"
        elif initial_count == 2:
            return "medium"
        else:  # 1
            return "hard"

    solved["difficulty"] = solved["initial_count"].apply(assign_difficulty)

    # Print difficulty distribution
    print("\nDifficulty distribution:")
    for diff in ["hard", "medium", "easy", "very_easy"]:
        count = (solved["difficulty"] == diff).sum()
        print(f"  {diff}: {count:,}")

    # Filter by requested difficulties
    filtered = solved[solved["difficulty"].isin(difficulties)]
    print(f"\nSelected difficulties {difficulties}: {len(filtered):,} theorems")

    # Convert to list of dicts
    results = []
    for theorem_name, row in filtered.iterrows():
        entry = process_entry(theorem_name, row.to_dict())
        results.append(entry)

    return results


def save_data(data: list, output_file: str) -> None:
    """Save processed data to JSONL file."""
    import os

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} entries to {output_file}")


def main(
    dataset_path: str = "~/datasets/Nemotron-Math-Proofs-v1",
    output_name: str = "nemotron_math_proofs",
    difficulties: list[str] | None = None,
    include_nano_v3: bool = False,
    limit: int | None = None,
) -> None:
    """Main entry point."""
    data_dir = Path(__file__).absolute().parent / "data"
    dataset_path = str(Path(dataset_path).expanduser())

    # Load and filter
    data = load_and_filter_dataset(
        dataset_path=dataset_path,
        exclude_nano_v3=not include_nano_v3,
        difficulties=difficulties,
    )

    if limit:
        data = data[:limit]
        print(f"Limited to {limit} entries")

    # Save
    output_file = str(data_dir / f"{output_name}.jsonl")
    save_data(data, output_file)

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Nemotron-Math-Proofs-v1 dataset for NeMo Gym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: hard + medium + easy (9,605 problems)
  python prepare_nemotron_math_proofs.py

  # Include very_easy problems too
  python prepare_nemotron_math_proofs.py --difficulties hard medium easy very_easy

  # Only hard problems
  python prepare_nemotron_math_proofs.py --difficulties hard

  # Include nano_v3 entries (not recommended for RL)
  python prepare_nemotron_math_proofs.py --include-nano-v3
        """,
    )
    parser.add_argument(
        "--dataset-path",
        default="~/datasets/Nemotron-Math-Proofs-v1",
        help="Path to saved dataset (default: ~/datasets/Nemotron-Math-Proofs-v1)",
    )
    parser.add_argument(
        "--output-name",
        default="nemotron_math_proofs",
        help="Output filename (without .jsonl extension)",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        choices=["hard", "medium", "easy", "very_easy"],
        default=["hard", "medium", "easy"],
        help="Difficulties to include (default: hard medium easy)",
    )
    parser.add_argument(
        "--include-nano-v3",
        action="store_true",
        help="Include nano_v3 entries (not recommended for RL, used in SFT)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries (for testing)",
    )
    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        output_name=args.output_name,
        difficulties=args.difficulties,
        include_nano_v3=args.include_nano_v3,
        limit=args.limit,
    )
