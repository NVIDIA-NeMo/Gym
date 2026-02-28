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

"""
# Single task with default config
python create_dataset.py --task knights_knaves --size 500 --seed 42 --output data/train.jsonl

# Multiple tasks (creates composite dataset with equal weights)
python create_dataset.py --tasks knights_knaves,syllogism --size 500 --output data/train.jsonl

# All tasks in a category
python create_dataset.py --category logic --size 500 --output data/train.jsonl

# All tasks from all categories
python create_dataset.py --all-tasks --size 5000 --output data/train_all.jsonl

# Single task with custom config
python create_dataset.py --task knights_knaves --size 500 --config '{"n_people": 3}' --output data/train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import reasoning_gym
from reasoning_gym.composite import DatasetSpec


TASK_CATEGORIES = {
    "logic": [
        "knights_knaves",
        "syllogism",
        "propositional_logic",
        "zebra_puzzles",
        "aiw",
        "circuit_logic",
        "self_reference",
    ],
    "arithmetic": [
        "basic_arithmetic",
        "chain_sum",
        "leg_counting",
        "prime_factorization",
        "gcd",
        "lcm",
        "time_intervals",
        "calendar_arithmetic",
        "dice",
        "products",
        "decimal_chain_sum",
        "decimal_arithmetic",
        "count_bits",
        "bitwise_arithmetic",
        "number_format",
        "fraction_simplification",
        "power_function",
        "gsm_symbolic",
    ],
    "algebra": [
        "simple_equations",
        "polynomial_equations",
        "polynomial_multiplication",
        "simple_integration",
        "intermediate_integration",
        "complex_arithmetic",
    ],
    "algorithmic": [
        "word_sorting",
        "string_insertion",
        "sentence_reordering",
        "count_primes",
        "binary_alternation",
        "manipulate_matrix",
        "pool_matrix",
        "base_conversion",
        "jugs",
        "graph_color",
        "spiral_matrix",
        "cryptarithm",
        "ab",
        "spell_backward",
        "letter_counting",
        "letter_jumble",
        "number_sorting",
        "caesar_cipher",
        "rotate_matrix",
        "rotten_oranges",
        "palindrome_generation",
        "palindrome_partitioning",
        "game_of_life",
        "game_of_life_halting",
        "group_anagrams",
        "isomorphic_strings",
        "string_synthesis",
        "binary_matrix",
        "word_sequence_reversal",
        "number_filtering",
        "word_ladder",
        "string_manipulation",
        "ransom_note",
        "string_splitting",
    ],
    "cognition": [
        "figlet_font",
        "needle_haystack",
        "modulo_grid",
        "number_sequence",
        "rubiks_cube",
        "color_cube_rotation",
        "rectangle_count",
    ],
    "arc": [
        "arc_1d",
        "arc_agi",
        "rearc",
    ],
    "code": [
        "bf",
        "codeio",
    ],
    "games": [
        "boxnet",
        "countdown",
        "emoji_mystery",
        "futoshiki",
        "kakurasu",
        "knight_swap",
        "mahjong_puzzle",
        "maze",
        "mini_sudoku",
        "n_queens",
        "puzzle24",
        "rush_hour",
        "sokoban",
        "sudoku",
        "survo",
        "tower_of_hanoi",
        "tsumego",
    ],
    "geometry": [
        "advanced_geometry",
        "simple_geometry",
    ],
    "graphs": [
        "course_schedule",
        "family_relationships",
        "largest_island",
        "quantum_lock",
        "shortest_path",
    ],
    "induction": [
        "acre",
        "list_functions",
    ],
    "probability": [
        "coin_flip",
    ],
}


def format_entry_to_nemo_gym(entry: dict) -> dict:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": entry["question"]}],
        },
        **entry,
    }


def create_single_task_dataset(task_name: str, size: int, seed: int, config: dict = None) -> list[dict]:
    config = config or {}
    config["size"] = size
    config["seed"] = seed

    print(f"Creating {task_name} dataset with {size} samples (seed={seed})...")
    try:
        dataset = reasoning_gym.create_dataset(task_name, **config)
    except Exception as e:
        print(f"Error creating dataset {task_name}: {e}")
        return []

    entries = []
    for i in range(size):
        try:
            entry = dataset[i]
            entries.append(format_entry_to_nemo_gym(entry))
        except Exception as e:
            print(f"Error generating entry {i} for {task_name}: {e}")
            continue

    print(f"Generated {len(entries)} entries for {task_name}")
    return entries


def create_composite_dataset(task_names: list[str], size: int, seed: int, config: dict = None) -> list[dict]:
    config = config or {}

    specs = [DatasetSpec(name=name, weight=1.0, config=config.get(name, {})) for name in task_names]

    print(f"Creating composite dataset with {len(task_names)} tasks: {', '.join(task_names)}")
    print(f"Total samples: {size}, seed: {seed}")

    try:
        dataset = reasoning_gym.create_dataset("composite", size=size, seed=seed, datasets=specs)
    except Exception as e:
        print(f"Error creating composite dataset: {e}")
        return []

    entries = []
    for i in range(size):
        try:
            entry = dataset[i]
            entries.append(format_entry_to_nemo_gym(entry))
        except Exception as e:
            print(f"Error generating entry {i}: {e}")
            continue

    task_counts = {}
    for entry in entries:
        task = entry["metadata"]["source_dataset"]
        task_counts[task] = task_counts.get(task, 0) + 1

    print(f"\nGenerated {len(entries)} total entries")
    print("Task distribution:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Create reasoning_gym datasets in NeMo Gym format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", type=str, help="Single task name (e.g., knights_knaves)")
    task_group.add_argument("--tasks", type=str, help="Comma-separated list of tasks")
    task_group.add_argument(
        "--category", type=str, choices=list(TASK_CATEGORIES.keys()), help="Task category (e.g., logic, arithmetic)"
    )
    task_group.add_argument("--all-tasks", action="store_true", help="Use all tasks from all categories")

    parser.add_argument("--size", type=int, default=500, help="Number of samples to generate (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--config",
        type=str,
        default="{}",
        help="Task-specific config as JSON string (e.g., '{\"n_people\": 3}')",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Output JSONL file path (e.g., data/train_knights_knaves.jsonl)"
    )

    args = parser.parse_args()

    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as e:
        print(f"Error parsing config JSON: {e}")
        sys.exit(1)

    if args.task:
        task_names = [args.task]
    elif args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",")]
    elif args.category:
        task_names = TASK_CATEGORIES[args.category]
    elif args.all_tasks:
        task_names = []
        for category_tasks in TASK_CATEGORIES.values():
            task_names.extend(category_tasks)
    else:
        print("Error: Must specify --task, --tasks, --category, or --all-tasks")
        sys.exit(1)

    if len(task_names) == 1:
        entries = create_single_task_dataset(task_names[0], args.size, args.seed, config)
    else:
        entries = create_composite_dataset(task_names, args.size, args.seed, config)

    if not entries:
        print("Error: No entries generated")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDataset saved to {output_path}")
    print(f"Total entries: {len(entries)}")


if __name__ == "__main__":
    main()
