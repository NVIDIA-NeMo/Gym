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

"""Wordle game logic for NemoGym resource server.

Implements the core Wordle game mechanics including:
- Feedback calculation (Green/Yellow/Gray)
- Knowledge state tracking
- Reward calculation with improved formula
"""

from dataclasses import dataclass, field
from typing import Optional

from resources_servers.wordle.wordle_words import WORDLE_SOLUTIONS, WORDLE_VALID_GUESSES


# =============================================================================
# Reward Constants (Improved)
# =============================================================================

# Win rewards (dynamic based on turns)
WIN_REWARD_BASE = 2.0
WIN_REWARD_PENALTY_PER_TURN = 0.2


# Soft penalties (strategic mistakes)
PENALTY_REPEATED_GUESS = -0.2
PENALTY_IGNORE_GREEN = -0.05
PENALTY_IGNORE_YELLOW = -0.03
PENALTY_USE_ELIMINATED = -0.02

# Invalid guess penalties
PENALTY_WRONG_LENGTH = -0.02
PENALTY_NOT_A_WORD = -0.02

# Loss
LOSS_REWARD = 0.0


def calculate_win_reward(turns_used: int) -> float:
    """Calculate win reward based on number of turns used.

    Formula: reward = 2.0 - 0.2 * (turns_used - 1), capped at turn-3 level.
    Turn 1: 1.6 (lucky, same as turn 3), Turn 2: 1.8, Turn 3: 1.6,
    Turn 4: 1.4, Turn 5: 1.2, Turn 6: 1.0
    """
    effective_turns = max(turns_used, 3)  # Turn-1 luck shouldn't beat turn-2 skill
    return WIN_REWARD_BASE - WIN_REWARD_PENALTY_PER_TURN * (effective_turns - 1)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class WordleGameState:
    """Represents the complete state of a Wordle game."""

    target_word: str
    word_length: int
    max_turns: int
    turn: int = 0
    guesses: list[str] = field(default_factory=list)
    feedback_history: list[list[str]] = field(default_factory=list)
    known_greens: dict[int, str] = field(default_factory=dict)
    known_yellows: set[str] = field(default_factory=set)
    eliminated_letters: set[str] = field(default_factory=set)
    game_over: bool = False
    won: bool = False
    total_reward: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "target_word": self.target_word,
            "word_length": self.word_length,
            "max_turns": self.max_turns,
            "turn": self.turn,
            "guesses": self.guesses,
            "feedback_history": self.feedback_history,
            "known_greens": self.known_greens,
            "known_yellows": list(self.known_yellows),
            "eliminated_letters": list(self.eliminated_letters),
            "game_over": self.game_over,
            "won": self.won,
            "total_reward": self.total_reward,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WordleGameState":
        """Create from dictionary."""
        return cls(
            target_word=data["target_word"],
            word_length=data["word_length"],
            max_turns=data["max_turns"],
            turn=data["turn"],
            guesses=data["guesses"],
            feedback_history=data["feedback_history"],
            known_greens=data["known_greens"],
            known_yellows=set(data["known_yellows"]),
            eliminated_letters=set(data["eliminated_letters"]),
            game_over=data["game_over"],
            won=data["won"],
            total_reward=data["total_reward"],
        )


@dataclass
class GuessResult:
    """Result of processing a guess."""

    valid: bool
    feedback: Optional[list[str]] = None
    feedback_str: Optional[str] = None
    won: bool = False
    game_over: bool = False
    reward: float = 0.0
    reward_breakdown: dict = field(default_factory=dict)
    error: Optional[str] = None
    turn: int = 0
    turns_remaining: int = 0
    target_word: Optional[str] = None  # Only revealed on game over


# =============================================================================
# Game Logic
# =============================================================================


class WordleGameLogic:
    """Static methods for Wordle game rules and mechanics."""

    @staticmethod
    def get_feedback(guess: str, target: str) -> list[str]:
        """Calculate feedback for a guess against the target word.

        Returns a list of feedback characters:
        - 'G' (Green): Correct letter in correct position
        - 'Y' (Yellow): Correct letter in wrong position
        - '_' (Gray): Letter not in the word

        Uses standard Wordle rules where each target letter can only
        match one guess letter (greens take priority over yellows).
        """
        guess = guess.lower()
        target = target.lower()
        word_length = len(target)
        feedback = ["_"] * word_length
        target_remaining = list(target)

        # First pass: mark greens
        for i, (g, t) in enumerate(zip(guess, target)):
            if g == t:
                feedback[i] = "G"
                target_remaining[i] = None  # Mark as used

        # Second pass: mark yellows
        for i, g in enumerate(guess):
            if feedback[i] == "G":
                continue
            if g in target_remaining:
                feedback[i] = "Y"
                # Remove first occurrence from remaining
                idx = target_remaining.index(g)
                target_remaining[idx] = None

        return feedback

    @staticmethod
    def is_valid_word(word: str, word_length: int = 5) -> tuple[bool, str]:
        """Check if a word is valid for guessing.

        Returns:
            (valid, reason) tuple
        """
        word = word.lower()

        if len(word) != word_length:
            return False, f"Word must be {word_length} letters, got {len(word)}"

        if not word.isalpha():
            return False, "Word must contain only letters"

        if word not in WORDLE_VALID_GUESSES:
            return False, f"'{word}' is not a valid English word"

        return True, "Valid word"

    @staticmethod
    def calculate_turn_reward(
        guess: str,
        feedback: list[str],
        state: WordleGameState,
    ) -> tuple[float, dict]:
        """Calculate the reward for a turn based on information gain.

        Returns:
            (total_reward, breakdown_dict)
        """
        reward = 0.0
        breakdown = {}

        # Check for strategic mistakes (soft penalties)

        # Repeated guess
        if guess in state.guesses:
            reward += PENALTY_REPEATED_GUESS
            breakdown["repeated_guess_penalty"] = PENALTY_REPEATED_GUESS

        # Ignoring known greens
        if state.known_greens:
            for pos, letter in state.known_greens.items():
                if pos < len(guess) and guess[pos] != letter:
                    reward += PENALTY_IGNORE_GREEN
                    breakdown["ignore_green_penalty"] = breakdown.get("ignore_green_penalty", 0) + PENALTY_IGNORE_GREEN

        # Ignoring known yellows (should include at least one)
        if state.known_yellows and not any(letter in guess for letter in state.known_yellows):
            reward += PENALTY_IGNORE_YELLOW
            breakdown["ignore_yellow_penalty"] = PENALTY_IGNORE_YELLOW

        # Using eliminated letters
        for letter in guess:
            if letter in state.eliminated_letters:
                reward += PENALTY_USE_ELIMINATED
                breakdown["use_eliminated_penalty"] = breakdown.get("use_eliminated_penalty", 0) + PENALTY_USE_ELIMINATED

        return reward, breakdown

    @staticmethod
    def update_knowledge(
        guess: str,
        feedback: list[str],
        state: WordleGameState,
    ) -> None:
        """Update the game state knowledge based on guess feedback.

        Modifies state in place.
        """
        # First pass: record greens and yellows
        for i, (fb, letter) in enumerate(zip(feedback, guess)):
            if fb == "G":
                state.known_greens[i] = letter
            elif fb == "Y":
                state.known_yellows.add(letter)

        # Second pass: eliminate grays (now all greens/yellows from this guess are known)
        for i, (fb, letter) in enumerate(zip(feedback, guess)):
            if fb == "_":
                if letter not in state.known_greens.values() and letter not in state.known_yellows:
                    state.eliminated_letters.add(letter)

    @staticmethod
    def render_game_state(state: WordleGameState, hide_target: bool = True) -> str:
        """Render the current game state as a human-readable string."""
        lines = []

        # Show guess history
        if state.guesses:
            lines.append("=== Guess History ===")
            for i, (guess, feedback) in enumerate(zip(state.guesses, state.feedback_history)):
                feedback_str = "".join(feedback)
                lines.append(f"Turn {i + 1}: {guess.upper()} -> {feedback_str}")
            lines.append("")

        # Show current knowledge
        lines.append("=== Current Knowledge ===")

        if state.known_greens:
            green_info = ", ".join(f"pos {pos + 1}={letter.upper()}" for pos, letter in sorted(state.known_greens.items()))
            lines.append(f"Confirmed (Green): {green_info}")

        if state.known_yellows:
            yellow_letters = ", ".join(sorted(l.upper() for l in state.known_yellows))
            lines.append(f"In word (Yellow): {yellow_letters}")

        if state.eliminated_letters:
            elim_letters = ", ".join(sorted(l.upper() for l in state.eliminated_letters))
            lines.append(f"Eliminated (Gray): {elim_letters}")

        lines.append("")
        lines.append(f"Turn: {state.turn}/{state.max_turns}")
        lines.append(f"Turns remaining: {state.max_turns - state.turn}")

        if state.game_over:
            lines.append("")
            if state.won:
                lines.append(f"Game Won! Word: {state.target_word.upper()}")
            else:
                lines.append(f"Game Over. The word was: {state.target_word.upper()}")

        return "\n".join(lines)
