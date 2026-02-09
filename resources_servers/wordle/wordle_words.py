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

"""Wordle word lists for NemoGym resource server.

This module re-exports the word lists from the main NemoRL codebase
for use in the NemoGym Wordle environment.

Contains:
- WORDLE_SOLUTIONS: ~2,315 words that can be the daily answer (5 letters)
- WORDLE_VALID_GUESSES: Set of all ~12,972 words accepted as guesses (5 letters)
- TRAINING_WORDS: 2,000 words for training (randomly sampled targets)
- VALIDATION_WORDS: 315 words for validation (fixed, no overlap with training)
- get_random_target: Function to get a random target word for training

The infrastructure supports variable word lengths, but currently only
5-letter word lists are available.
"""

import random
from typing import Optional

# Import the word lists from the local copy (self-contained, no nemo_rl dependency)
# These are the standard 5-letter Wordle word lists
from wordle_words_data import (
    WORDLE_SOLUTIONS,
    WORDLE_VALID_GUESSES,
)

# =============================================================================
# Train/Validation Split
# =============================================================================
# Split WORDLE_SOLUTIONS into non-overlapping train and validation sets.
# Using a fixed seed ensures the split is reproducible.
# Total: 2,315 words -> Training: 2,000 words, Validation: 315 words

_SPLIT_SEED = 42
_rng = random.Random(_SPLIT_SEED)
_shuffled_solutions = list(WORDLE_SOLUTIONS)
_rng.shuffle(_shuffled_solutions)

TRAINING_WORDS: list[str] = _shuffled_solutions[:2000]
VALIDATION_WORDS: list[str] = _shuffled_solutions[2000:]

# Verify no overlap
assert len(set(TRAINING_WORDS) & set(VALIDATION_WORDS)) == 0, "Train/val overlap detected!"
assert len(TRAINING_WORDS) + len(VALIDATION_WORDS) == len(WORDLE_SOLUTIONS), "Word count mismatch!"

# Re-export for convenience
__all__ = [
    "WORDLE_SOLUTIONS",
    "WORDLE_VALID_GUESSES",
    "TRAINING_WORDS",
    "VALIDATION_WORDS",
    "get_random_target",
    "get_validation_words",
    "is_valid_guess",
]


def get_random_target(word_length: int = 5, seed: Optional[int] = None, use_training_set: bool = True) -> str:
    """Get a random target word for training.

    Args:
        word_length: Length of the target word. Currently only 5 is supported.
        seed: Optional random seed for reproducibility.
        use_training_set: If True, sample from TRAINING_WORDS only (default).
                         If False, sample from all WORDLE_SOLUTIONS.

    Returns:
        A random target word.

    Raises:
        ValueError: If word_length is not 5 (only 5-letter words are currently supported).
    """
    if word_length != 5:
        raise ValueError(f"Only 5-letter words are currently supported, got word_length={word_length}")

    word_pool = TRAINING_WORDS if use_training_set else WORDLE_SOLUTIONS

    if seed is not None:
        rng = random.Random(seed)
        return rng.choice(word_pool)
    return random.choice(word_pool)


def get_validation_words() -> list[str]:
    """Get the fixed list of validation words.

    Returns:
        List of 315 validation words (no overlap with training).
    """
    return VALIDATION_WORDS.copy()


def is_valid_guess(word: str, word_length: int = 5) -> bool:
    """Check if a word is a valid guess.

    Args:
        word: The word to check.
        word_length: Expected length of the word.

    Returns:
        True if the word is valid, False otherwise.
    """
    word = word.lower()

    if len(word) != word_length:
        return False

    if not word.isalpha():
        return False

    return word in WORDLE_VALID_GUESSES
