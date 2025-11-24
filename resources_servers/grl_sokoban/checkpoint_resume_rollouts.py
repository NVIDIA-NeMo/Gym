#!/usr/bin/env python3
"""
Checkpoint and resume helper for rollout collection.
Identifies which prompts have been completed and creates a filtered input file
with remaining prompts for resuming failed runs.
"""

import argparse
import json
from pathlib import Path
from typing import Set


def extract_completed_prompt_ids(rollouts_file: Path) -> Set[int]:
    """Extract level_ids that have been completed from rollouts file.

    Args:
        rollouts_file: Path to existing rollouts JSONL file

    Returns:
        Set of level_ids (prompt IDs) that have been completed
    """
    completed_ids = set()

    if not rollouts_file.exists():
        print(f"No existing rollouts file found at {rollouts_file}")
        return completed_ids

    try:
        with open(rollouts_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    rollout = json.loads(line.strip())
                    # Extract level_id from the rollout's prompt
                    if "prompt" in rollout and "level_id" in rollout["prompt"]:
                        level_id = rollout["prompt"]["level_id"]
                        completed_ids.add(level_id)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON at line {line_num}")
                    continue

        print(f"Found {len(completed_ids)} unique completed prompt IDs")
        return completed_ids

    except Exception as e:
        print(f"Error reading rollouts file: {e}")
        return completed_ids


def count_rollouts_per_prompt(rollouts_file: Path, target_repeats: int = 16) -> dict[str, int]:
    """Count rollouts and determine completion based on sequential processing order.

    Since ng_collect_rollouts processes prompts in order (abc -> aabbcc pattern),
    we can infer which prompts are complete by dividing total rollouts by target_repeats.

    Args:
        rollouts_file: Path to existing rollouts JSONL file
        target_repeats: Target number of rollouts per prompt (default: 16)

    Returns:
        Dictionary with 'total_rollouts' and 'completed_prompts_count'
    """
    if not rollouts_file.exists():
        return {"total_rollouts": 0, "completed_prompts_count": 0}

    try:
        # Count total rollouts
        total_rollouts = 0
        with open(rollouts_file, "r") as f:
            for line in f:
                if line.strip():
                    total_rollouts += 1

        # Calculate completed prompts based on sequential processing
        completed_prompts_count = total_rollouts // target_repeats
        partial_rollouts = total_rollouts % target_repeats

        print("Rollout completion status (based on sequential processing):")
        print(f"  Total rollouts: {total_rollouts}")
        print(f"  Completed prompts (full {target_repeats} rollouts): {completed_prompts_count}")
        if partial_rollouts > 0:
            print(
                f"  Partial progress on prompt {completed_prompts_count + 1}: {partial_rollouts}/{target_repeats} rollouts"
            )

        return {"total_rollouts": total_rollouts, "completed_prompts_count": completed_prompts_count}

    except Exception as e:
        print(f"Error counting rollouts: {e}")
        return {"total_rollouts": 0, "completed_prompts_count": 0}


def create_remaining_prompts_file(
    input_file: Path,
    output_file: Path,
    completed_ids: Set[int],
    rollout_counts: dict[str, int],
    target_repeats: int = 16,
):
    """Create a new input file with only prompts that haven't been fully processed yet.

    Uses sequential processing order to determine which prompts are complete.
    If we have N complete prompts, skip the first N lines and keep the rest.

    Args:
        input_file: Original input prompts file
        output_file: Output file for remaining prompts
        completed_ids: Set of level_ids (unused - kept for compatibility)
        rollout_counts: Dictionary with 'completed_prompts_count' key
        target_repeats: Target number of rollouts per prompt
    """
    total_count = 0
    completed_count = rollout_counts.get("completed_prompts_count", 0)
    remaining_count = 0

    # Skip the first `completed_count` prompts, keep the rest
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for idx, line in enumerate(f_in):
            if line.strip():
                total_count += 1
                # Skip prompts that are already complete (0-indexed)
                if idx >= completed_count:
                    f_out.write(line)
                    remaining_count += 1

    print(f"\nCreated remaining prompts file: {output_file}")
    print(f"  Total prompts in input: {total_count}")
    print(f"  Already completed: {completed_count} prompts ({completed_count * target_repeats} rollouts)")
    print(f"  Remaining to process: {remaining_count} prompts")
    print(f"  Expected new rollouts: ~{remaining_count * target_repeats}")

    return remaining_count


def main():
    parser = argparse.ArgumentParser(description="Create checkpoint file for resuming rollout collection")
    parser.add_argument("--input", type=Path, required=True, help="Original input prompts file (JSONL)")
    parser.add_argument("--rollouts", type=Path, required=True, help="Existing rollouts file (JSONL)")
    parser.add_argument("--output", type=Path, required=True, help="Output file for remaining prompts (JSONL)")
    parser.add_argument(
        "--target-repeats", type=int, default=16, help="Target number of rollouts per prompt (default: 16)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Checkpoint Resume Helper for Rollout Collection")
    print("=" * 60)

    # Count existing rollouts per prompt
    rollout_counts = count_rollouts_per_prompt(args.rollouts, args.target_repeats)

    # Get completed IDs
    completed_ids = set(rollout_counts.keys())

    # Create remaining prompts file
    remaining = create_remaining_prompts_file(
        args.input, args.output, completed_ids, rollout_counts, args.target_repeats
    )

    if remaining == 0:
        print("\n✓ All prompts completed! No remaining work.")
    else:
        print(f"\n→ Resume collection using: {args.output}")
        print(f"   Expected new rollouts: ~{remaining * args.target_repeats}")


if __name__ == "__main__":
    main()
