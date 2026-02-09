#!/usr/bin/env python3
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

"""Generate training and validation data for Wordle NemoGym environment.

This script generates JSONL files with Wordle game prompts for training.

Key design decisions:
- Training data: No target words specified. Server picks randomly from TRAINING_WORDS
  (2,000 words) at runtime. This gives variety across training runs.
- Validation data: Fixed target words from VALIDATION_WORDS (315 words, no overlap
  with training). This ensures reproducible evaluation across training steps.

Usage:
    python generate_data.py --output_dir data/
    python generate_data.py --train_samples 2000 --output_dir data/
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directories to path for imports
gym_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(gym_root))
nemo_rl_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(nemo_rl_root))


# System prompt for Wordle
# Note: Reasoning mode disabled due to compatibility issues with multi-turn tool calling
SYSTEM_PROMPT = """/no_think
You are playing Wordle, a word-guessing game. Your goal is to guess a secret 5-letter word in 6 attempts or fewer.

After each guess, you'll receive feedback:
- G (Green): Letter is correct and in the right position
- Y (Yellow): Letter is in the word but in the wrong position
- _ (Gray): Letter is not in the word

Strategy tips:
- Start with words containing common letters (E, A, R, T, O, I, N, S)
- Use the feedback to narrow down possibilities
- Never repeat a guess
- Place confirmed green letters in their positions
- Include yellow letters in different positions
- Avoid gray (eliminated) letters

IMPORTANT: Always respond with a tool call. Never reply with plain text. After receiving feedback, immediately make your next move by calling a tool."""

# User prompt variations
USER_PROMPTS = [
    "Let's play Wordle! Guess the secret 5-letter word.",
    "Time to play Wordle! Can you guess the mystery word in 6 tries or less?",
    "I'm thinking of a 5-letter word. Use your best strategy to guess it!",
    "Wordle time! Start guessing - you have 6 attempts.",
    "Let's see how quickly you can solve this Wordle puzzle!",
    "Ready to play Wordle? Make your first guess!",
    "I have a secret 5-letter word for you to guess. Good luck!",
    "Wordle challenge: Find the hidden word in as few guesses as possible.",
    "Let's play! Guess the 5-letter word I'm thinking of.",
    "Your Wordle game is ready. What's your opening guess?",
]

# Tool definitions
TOOLS = [
    {
        "type": "function",
        "name": "submit_guess",
        "description": "Submit a 5-letter word guess. Returns feedback for each letter: G (green) = correct position, Y (yellow) = wrong position but in word, _ (gray) = not in word.",
        "parameters": {
            "type": "object",
            "properties": {
                "guess": {
                    "type": "string",
                    "description": "A 5-letter English word to guess"
                }
            },
            "required": ["guess"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "check_word_validity",
        "description": "Check if a word is valid before guessing. This is optional and informational only - it won't affect your game.",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "A word to check for validity"
                }
            },
            "required": ["word"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_game_state",
        "description": "Get the current game state including guesses made, feedback received, and accumulated knowledge about the target word.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    }
]


def create_wordle_entry(
    user_prompt: str,
    word_length: int = 5,
    max_turns: int = 6,
    custom_target: str = None,
) -> dict:
    """Create a single Wordle data entry.

    Args:
        user_prompt: The user message to start the game
        word_length: Length of words (default 5)
        max_turns: Maximum guesses allowed (default 6)
        custom_target: Optional specific target word. If None, server picks randomly.
    """
    entry = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "tools": TOOLS,
            "parallel_tool_calls": False,
            "temperature": 1.0
        },
        "word_length": word_length,
        "max_turns": max_turns,
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "wordle_simple_agent"
        },
    }

    # Include custom_target if specified (for validation with fixed words)
    if custom_target:
        entry["custom_target"] = custom_target

    return entry


def generate_training_data(num_samples: int, seed: int = 42) -> list[dict]:
    """Generate training data WITHOUT target words.

    The server will pick random targets from TRAINING_WORDS (2,000 words)
    at runtime. This ensures variety across training runs.

    Args:
        num_samples: Number of entries to generate
        seed: Random seed for prompt shuffling
    """
    import random
    random.seed(seed)

    entries = []
    for i in range(num_samples):
        # Cycle through user prompts for variety
        user_prompt = USER_PROMPTS[i % len(USER_PROMPTS)]
        entry = create_wordle_entry(user_prompt, custom_target=None)
        entries.append(entry)

    return entries


def generate_validation_data(seed: int = 43) -> list[dict]:
    """Generate validation data WITH fixed target words.

    Uses all 315 words from VALIDATION_WORDS (no overlap with training).
    Each validation entry has a specific target word for reproducible evaluation.

    Args:
        seed: Random seed for prompt assignment
    """
    import random
    random.seed(seed)

    # Import validation words from the proper split
    from resources_servers.wordle.wordle_words import VALIDATION_WORDS

    entries = []
    for i, target_word in enumerate(VALIDATION_WORDS):
        # Cycle through user prompts
        user_prompt = USER_PROMPTS[i % len(USER_PROMPTS)]
        entry = create_wordle_entry(user_prompt, custom_target=target_word)
        entries.append(entry)

    return entries


def save_jsonl(entries: list[dict], filepath: Path) -> None:
    """Save entries to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved {len(entries)} entries to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Wordle training and validation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Word Split:
  - TRAINING_WORDS: 2,000 words (server picks randomly at runtime)
  - VALIDATION_WORDS: 315 words (fixed in JSONL, no overlap with training)

Examples:
  python generate_data.py                          # Default: 1000 train, all 315 val
  python generate_data.py --train_samples 2000     # More training samples
        """
    )
    parser.add_argument("--train_samples", type=int, default=1000,
                        help="Number of training samples (default: 1000)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for JSONL files (default: data)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Generate training data (no target words - server picks from TRAINING_WORDS)
    print(f"Generating {args.train_samples} training samples...")
    print("  - No target words in data (server picks randomly from 2,000 training words)")
    train_data = generate_training_data(args.train_samples, seed=args.seed)
    save_jsonl(train_data, output_dir / "train.jsonl")

    # Generate validation data (fixed target words from VALIDATION_WORDS)
    print(f"\nGenerating validation samples...")
    print("  - Fixed target words from 315 validation words (no overlap with training)")
    val_data = generate_validation_data(seed=args.seed + 1)
    save_jsonl(val_data, output_dir / "validation.jsonl")

    # Generate example data (small subset of validation for quick testing)
    print(f"\nGenerating example samples...")
    example_data = val_data[:5]
    save_jsonl(example_data, output_dir / "example.jsonl")

    print("\nDone!")
    print(f"\nSummary:")
    print(f"  Training:   {len(train_data)} samples (targets picked at runtime from 2,000 words)")
    print(f"  Validation: {len(val_data)} samples (fixed targets, 315 unique words)")
    print(f"  Example:    {len(example_data)} samples")


if __name__ == "__main__":
    main()
