#!/usr/bin/env python3
"""Generate test examples for GRL Sokoban environment.

This script creates ~500 diverse test examples with varying seeds and room dimensions
for reward profiling as per CONTRIBUTING.md requirements.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List


def generate_sokoban_example(level_id: int, seed: int, dim_room: List[int], num_boxes: int) -> Dict[str, Any]:
    """Generate a single Sokoban test example in the expected JSONL format.

    Args:
        level_id: Unique identifier for the level
        seed: Random seed for reproducible level generation
        dim_room: Room dimensions as [width, height]
        num_boxes: Number of boxes in the puzzle

    Returns:
        Dictionary containing the level configuration and prompt
    """
    return {
        "level_id": level_id,
        "seed": seed,
        "dim_room": dim_room,
        "num_boxes": num_boxes,
        "responses_create_params": {
            "max_tool_calls": 10,
            "input": [
                {
                    "role": "developer",
                    "content": 'You are a Sokoban-solving assistant. IMPORTANT: First call the `step` tool with an empty array [] to see the initial puzzle state. Example: step({"actions": []}). The tool will return the board as a string with symbols: #=wall, _=floor, O=target, X=box, √=box on target, P=player, S=player on target. Then continue calling `step` with valid actions (Up, Down, Left, Right) until the puzzle is solved or you run out of moves. At the end, respond with <answer>Action1 || Action2 || ...</answer> summarizing all moves you executed.',
                },
                {"role": "user", "content": "Call the step tool to see the puzzle, then solve it step by step."},
            ],
            "tools": [
                {
                    "name": "step",
                    "type": "function",
                    "description": "Execute Sokoban moves sequentially. Call with empty array [] to see current puzzle state without moving.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Sequence of actions e.g. ['Up', 'Right']. Use empty array [] to view current state.",
                            }
                        },
                        "required": ["actions"],
                        "additionalProperties": False,
                    },
                }
            ],
        },
    }


def generate_test_examples(num_examples: int = 500, output_file: str = "data/test_examples.jsonl") -> None:
    """Generate diverse test examples for Sokoban environment.

    Args:
        num_examples: Number of examples to generate (default: 500)
        output_file: Output JSONL file path
    """
    examples = []

    # Define parameter ranges for diversity
    room_sizes = [
        [4, 4],  # Tiny square
        [5, 5],  # Small square
        [6, 6],  # Medium square
        [7, 7],  # Large square
        [8, 8],  # Extra large square
        [4, 6],  # Narrow tall
        [6, 4],  # Wide short
        [5, 6],  # Small tall
        [6, 5],  # Small wide
        [5, 7],  # Medium tall
        [7, 5],  # Medium wide
        [6, 7],  # Large tall
        [7, 6],  # Large wide
    ]

    # Primarily use 1 box (most common), but include some harder puzzles
    num_boxes_options = [1, 1, 1, 1, 1, 2, 2, 3]  # Weighted toward 1 box

    # Generate diverse examples
    for i in range(num_examples):
        level_id = i + 1

        # Use level_id as base for seed to ensure reproducibility but diversity
        seed = random.randint(1000, 99999) + i * 97  # Prime offset for better distribution

        # Cycle through room sizes with some randomness
        dim_room = random.choice(room_sizes)

        # Most puzzles should have 1 box, some have more
        num_boxes = random.choice(num_boxes_options)

        # Ensure room is large enough for boxes
        min_room_size = dim_room[0] * dim_room[1]
        if num_boxes >= min_room_size // 3:
            num_boxes = 1  # Fall back to 1 box if room is too small

        example = generate_sokoban_example(level_id, seed, dim_room, num_boxes)
        examples.append(example)

    # Write to JSONL file
    output_path = Path(__file__).parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Generated {num_examples} test examples")
    print(f"Output file: {output_path}")
    print("\nParameter distribution:")
    print(f"  Room sizes: {sorted(set(tuple(e['dim_room']) for e in examples))}")
    print("  Num boxes distribution:")
    box_counts = {}
    for e in examples:
        nb = e["num_boxes"]
        box_counts[nb] = box_counts.get(nb, 0) + 1
    for nb in sorted(box_counts.keys()):
        print(f"    {nb} boxes: {box_counts[nb]} examples ({100 * box_counts[nb] / num_examples:.1f}%)")
    print(f"  Seed range: {min(e['seed'] for e in examples)} - {max(e['seed'] for e in examples)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Sokoban test examples")
    parser.add_argument("--num-examples", type=int, default=500, help="Number of examples to generate (default: 500)")
    parser.add_argument(
        "--output",
        type=str,
        default="data/test_examples.jsonl",
        help="Output JSONL file path (default: data/test_examples.jsonl)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation (default: 42)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    generate_test_examples(args.num_examples, args.output)
