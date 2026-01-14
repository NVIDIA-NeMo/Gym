#!/usr/bin/env python3
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
"""Generate test examples for GRL Tetris environment."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List


def generate_tetris_example(game_id: int, seed: int, dim_board: List[int], box_type: int) -> Dict[str, Any]:
    """Generate a single Tetris test example in the expected JSONL format.

    Args:
        game_id: Unique identifier for the game
        seed: Random seed for reproducible game generation
        dim_board: Board dimensions as [width, height]
        box_type: Type of Tetris pieces (0=single, 1=single, 2=I and -, 3=I, -, and O)

    Returns:
        Dictionary containing the game configuration and prompt
    """
    return {
        "game_id": game_id,
        "seed": seed,
        "dim_board": dim_board,
        "box_type": box_type,
        "responses_create_params": {
            "input": [
                {
                    "role": "developer",
                    "content": "You are a Tetris-playing assistant. IMPORTANT: First call the `step` tool with an empty array [] to see the initial board state and active piece. Example: step({\"actions\": []}). The tool will return an ASCII board using '_' for empty cells, '#' for locked blocks, and 'X' for the active piece. Then continue calling `step` with valid actions (Left, Right, Down) until you clear a line or the board locks out. At the end, respond with <answer>Action1 || Action2 || ...</answer> summarizing all moves you made.",
                },
                {
                    "role": "user",
                    "content": "Call the step tool to see the board, then play Tetris to clear at least one line if possible.",
                },
            ],
            "tools": [
                {
                    "name": "step",
                    "type": "function",
                    "description": "Execute Tetris moves sequentially. Call with empty array [] to see current board state without moving.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Sequence of actions, e.g. ['Left', 'Down']. Use empty array [] to view current state.",
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
    """Generate diverse test examples for Tetris environment.

    Args:
        num_examples: Number of examples to generate (default: 500)
        output_file: Output JSONL file path
    """
    examples = []

    # Define parameter ranges for diversity
    board_sizes = [
        [4, 4],  # Small square
        [5, 5],  # Medium square
        [6, 6],  # Large square
        [4, 6],  # Narrow tall
        [6, 4],  # Wide short
        [5, 6],  # Medium tall
        [6, 5],  # Medium wide
        [4, 5],  # Small tall
        [5, 4],  # Small wide
    ]

    box_types = [0, 1, 2, 3]  # All available piece types

    # Generate diverse examples
    for i in range(num_examples):
        game_id = i + 1

        # Use game_id as base for seed to ensure reproducibility but diversity
        seed = random.randint(10000, 99999) + i * 137  # Prime offset for better distribution

        # Cycle through board sizes with some randomness
        dim_board = random.choice(board_sizes)

        # Distribute box types evenly but with some randomness
        box_type = random.choice(box_types)

        example = generate_tetris_example(game_id, seed, dim_board, box_type)
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
    print(f"  Board sizes: {set(tuple(e['dim_board']) for e in examples)}")
    print(f"  Box types: {set(e['box_type'] for e in examples)}")
    print(f"  Seed range: {min(e['seed'] for e in examples)} - {max(e['seed'] for e in examples)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Tetris test examples")
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
