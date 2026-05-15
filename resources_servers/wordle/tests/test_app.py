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

"""Unit tests for Wordle NemoGym resource server."""

import sys
from pathlib import Path

import pytest

# Add parent directories to path for imports
gym_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(gym_root))

from resources_servers.wordle.wordle_logic import (
    LOSS_REWARD,
    PENALTY_IGNORE_GREEN,
    PENALTY_IGNORE_YELLOW,
    PENALTY_NOT_A_WORD,
    PENALTY_REPEATED_GUESS,
    PENALTY_USE_ELIMINATED,
    PENALTY_WRONG_LENGTH,
    REWARD_PER_NEW_GRAY,
    REWARD_PER_NEW_GREEN,
    REWARD_PER_NEW_YELLOW,
    REWARD_VALID_ATTEMPT,
    REWARD_YELLOW_TO_GREEN,
    WordleGameLogic,
    WordleGameState,
    calculate_win_reward,
)


class TestWordleGameLogic:
    """Tests for WordleGameLogic class."""

    def test_get_feedback_all_green(self):
        """Test feedback when all letters are correct."""
        feedback = WordleGameLogic.get_feedback("crane", "crane")
        assert feedback == ["G", "G", "G", "G", "G"]

    def test_get_feedback_all_gray(self):
        """Test feedback when no letters match."""
        feedback = WordleGameLogic.get_feedback("xyz", "abc")
        assert feedback == ["_", "_", "_"]

    def test_get_feedback_mixed(self):
        """Test feedback with mixed green, yellow, gray."""
        # Target: "crane", Guess: "clear"
        # c: at position 0 in both -> G (green)
        # l: not in word -> _
        # e: in word at position 4 -> Y (yellow)
        # a: in word at position 2 -> Y (yellow)
        # r: in word at position 1 -> Y (yellow)
        feedback = WordleGameLogic.get_feedback("clear", "crane")
        assert feedback == ["G", "_", "Y", "Y", "Y"]

    def test_get_feedback_double_letter(self):
        """Test feedback with repeated letters."""
        # Target: "speed", Guess: "eerie"
        # e: first e matches second e in speed -> Y
        # e: second e matches third e in speed -> Y
        # r: not in word -> _
        # i: not in word -> _
        # e: no more e's available -> _
        feedback = WordleGameLogic.get_feedback("eerie", "speed")
        assert feedback == ["Y", "Y", "_", "_", "_"]

    def test_get_feedback_green_priority(self):
        """Test that greens take priority over yellows."""
        # Target: "hello", Guess: "llama"
        # l: not at position 0, but l is at position 2 and 3 -> Y
        # l: at position 1? No, hello has l at 2 and 3 -> Y (uses one l)
        # Actually, let's think more carefully:
        # Target: h-e-l-l-o
        # Guess:  l-l-a-m-a
        # l at 0: target[0]=h, not green. Is l in remaining? Yes (pos 2,3) -> Y, remove one l
        # l at 1: target[1]=e, not green. Is l in remaining? Yes (pos 3) -> Y, remove one l
        # a at 2: target[2]=l, not green. Is a in remaining? No -> _
        # m at 3: target[3]=l, not green. Is m in remaining? No -> _
        # a at 4: target[4]=o, not green. Is a in remaining? No -> _
        feedback = WordleGameLogic.get_feedback("llama", "hello")
        assert feedback == ["Y", "Y", "_", "_", "_"]

    def test_get_feedback_exact_green(self):
        """Test green feedback for exact position matches."""
        # Target: "hello", Guess: "jello"
        feedback = WordleGameLogic.get_feedback("jello", "hello")
        assert feedback == ["_", "G", "G", "G", "G"]

    def test_is_valid_word_correct_length(self):
        """Test valid word validation."""
        valid, reason = WordleGameLogic.is_valid_word("crane", 5)
        assert valid is True
        assert reason == "Valid word"

    def test_is_valid_word_wrong_length(self):
        """Test invalid word due to wrong length."""
        valid, reason = WordleGameLogic.is_valid_word("cat", 5)
        assert valid is False
        assert "5 letters" in reason

    def test_is_valid_word_not_in_dictionary(self):
        """Test invalid word not in dictionary."""
        valid, reason = WordleGameLogic.is_valid_word("zzzzz", 5)
        assert valid is False
        assert "not a valid" in reason.lower()

    def test_is_valid_word_non_alpha(self):
        """Test invalid word with non-alphabetic characters."""
        valid, reason = WordleGameLogic.is_valid_word("cra1e", 5)
        assert valid is False
        assert "letters" in reason.lower()


class TestCalculateWinReward:
    """Tests for win reward calculation."""

    def test_win_reward_turn_1(self):
        """Test win reward for turn 1 (best case)."""
        reward = calculate_win_reward(1)
        assert reward == 2.0

    def test_win_reward_turn_6(self):
        """Test win reward for turn 6 (minimum win)."""
        reward = calculate_win_reward(6)
        assert reward == 1.0

    def test_win_reward_decreases(self):
        """Test that win reward decreases with more turns."""
        rewards = [calculate_win_reward(i) for i in range(1, 7)]
        assert rewards == [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]


class TestCalculateTurnReward:
    """Tests for turn reward calculation."""

    def test_valid_attempt_bonus(self):
        """Test that valid attempts get a small bonus."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        feedback = ["_", "_", "_", "_", "_"]  # No matches
        reward, breakdown = WordleGameLogic.calculate_turn_reward("xyzbq", feedback, state)

        # Should have valid attempt reward plus some gray rewards
        assert "valid_attempt" in breakdown
        assert breakdown["valid_attempt"] == REWARD_VALID_ATTEMPT

    def test_new_green_reward(self):
        """Test reward for discovering new green letters."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        feedback = ["G", "_", "_", "_", "_"]  # One green
        reward, breakdown = WordleGameLogic.calculate_turn_reward("cxxxx", feedback, state)

        assert "new_greens" in breakdown
        assert breakdown["new_greens"] == REWARD_PER_NEW_GREEN

    def test_new_yellow_reward(self):
        """Test reward for discovering new yellow letters."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        feedback = ["_", "Y", "_", "_", "_"]  # One yellow
        reward, breakdown = WordleGameLogic.calculate_turn_reward("xrxxx", feedback, state)

        assert "new_yellows" in breakdown
        assert breakdown["new_yellows"] == REWARD_PER_NEW_YELLOW

    def test_yellow_to_green_reward(self):
        """Test reward for converting yellow to green."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        state.known_yellows.add("r")  # R was previously yellow

        feedback = ["_", "G", "_", "_", "_"]  # R is now green
        reward, breakdown = WordleGameLogic.calculate_turn_reward("xrxxx", feedback, state)

        assert "yellow_to_green" in breakdown
        assert breakdown["yellow_to_green"] == REWARD_YELLOW_TO_GREEN

    def test_letter_elimination_reward(self):
        """Test reward for eliminating letters."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        # Guess with letters not in target
        feedback = ["_", "_", "_", "_", "_"]  # All gray
        reward, breakdown = WordleGameLogic.calculate_turn_reward("xyzbq", feedback, state)

        assert "letter_elimination" in breakdown
        # 5 new gray letters discovered
        assert breakdown["letter_elimination"] == 5 * REWARD_PER_NEW_GRAY

    def test_repeated_guess_penalty(self):
        """Test penalty for repeated guess."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        state.guesses.append("hello")  # Previous guess

        feedback = ["_", "_", "_", "_", "_"]
        reward, breakdown = WordleGameLogic.calculate_turn_reward("hello", feedback, state)

        assert "repeated_guess_penalty" in breakdown
        assert breakdown["repeated_guess_penalty"] == PENALTY_REPEATED_GUESS

    def test_ignore_green_penalty(self):
        """Test penalty for ignoring known green positions."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        state.known_greens[0] = "c"  # C confirmed at position 0

        feedback = ["_", "_", "_", "_", "_"]
        reward, breakdown = WordleGameLogic.calculate_turn_reward("xxxxx", feedback, state)

        assert "ignore_green_penalty" in breakdown
        assert breakdown["ignore_green_penalty"] == PENALTY_IGNORE_GREEN

    def test_ignore_yellow_penalty(self):
        """Test penalty for ignoring known yellow letters."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        state.known_yellows.add("r")  # R known to be in word

        feedback = ["_", "_", "_", "_", "_"]
        # Guess without R
        reward, breakdown = WordleGameLogic.calculate_turn_reward("xxxxx", feedback, state)

        assert "ignore_yellow_penalty" in breakdown
        assert breakdown["ignore_yellow_penalty"] == PENALTY_IGNORE_YELLOW

    def test_use_eliminated_penalty(self):
        """Test penalty for using eliminated letters."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        state.eliminated_letters.add("z")  # Z is eliminated

        feedback = ["_", "_", "_", "_", "_"]
        # Guess with Z
        reward, breakdown = WordleGameLogic.calculate_turn_reward("zzzzz", feedback, state)

        assert "use_eliminated_penalty" in breakdown
        # 5 uses of eliminated letter
        assert breakdown["use_eliminated_penalty"] == 5 * PENALTY_USE_ELIMINATED


class TestUpdateKnowledge:
    """Tests for knowledge state updates."""

    def test_update_greens(self):
        """Test updating known green positions."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        feedback = ["G", "_", "_", "_", "_"]
        WordleGameLogic.update_knowledge("cxxxx", feedback, state)

        assert state.known_greens == {0: "c"}

    def test_update_yellows(self):
        """Test updating known yellow letters."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        feedback = ["Y", "_", "_", "_", "_"]
        WordleGameLogic.update_knowledge("rxxxx", feedback, state)

        assert "r" in state.known_yellows

    def test_update_eliminated(self):
        """Test updating eliminated letters."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        feedback = ["_", "_", "_", "_", "_"]
        WordleGameLogic.update_knowledge("xyzbq", feedback, state)

        assert "x" in state.eliminated_letters
        assert "y" in state.eliminated_letters
        assert "z" in state.eliminated_letters
        assert "b" in state.eliminated_letters
        assert "q" in state.eliminated_letters

    def test_no_eliminate_if_yellow(self):
        """Test that letters aren't eliminated if they're yellow elsewhere."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
        )
        # First guess discovers 'a' is yellow
        state.known_yellows.add("a")

        # If 'a' appears as gray in a position, it shouldn't be eliminated
        feedback = ["_", "_", "_", "_", "_"]
        WordleGameLogic.update_knowledge("axzzz", feedback, state)

        # 'a' should NOT be in eliminated (it's in known_yellows)
        assert "a" not in state.eliminated_letters


class TestWordleGameState:
    """Tests for WordleGameState dataclass."""

    def test_to_dict_and_back(self):
        """Test serialization and deserialization."""
        state = WordleGameState(
            target_word="crane",
            word_length=5,
            max_turns=6,
            turn=2,
            guesses=["hello", "world"],
            feedback_history=[["_", "_", "_", "_", "_"], ["_", "_", "_", "_", "_"]],
            known_greens={0: "c"},
            known_yellows={"r", "a"},
            eliminated_letters={"h", "e", "l", "o", "w", "d"},
            game_over=False,
            won=False,
            total_reward=0.5,
        )

        data = state.to_dict()
        restored = WordleGameState.from_dict(data)

        assert restored.target_word == state.target_word
        assert restored.word_length == state.word_length
        assert restored.max_turns == state.max_turns
        assert restored.turn == state.turn
        assert restored.guesses == state.guesses
        assert restored.feedback_history == state.feedback_history
        assert restored.known_greens == state.known_greens
        assert restored.known_yellows == state.known_yellows
        assert restored.eliminated_letters == state.eliminated_letters
        assert restored.game_over == state.game_over
        assert restored.won == state.won
        assert restored.total_reward == state.total_reward


class TestRewardConstants:
    """Tests to verify reward constant values."""

    def test_win_rewards_positive(self):
        """Test that all win rewards are positive."""
        for turn in range(1, 7):
            assert calculate_win_reward(turn) > 0

    def test_loss_reward_zero(self):
        """Test that loss reward is zero."""
        assert LOSS_REWARD == 0.0

    def test_information_rewards_positive(self):
        """Test that information gain rewards are positive."""
        assert REWARD_PER_NEW_GREEN > 0
        assert REWARD_PER_NEW_YELLOW > 0
        assert REWARD_YELLOW_TO_GREEN > 0
        assert REWARD_PER_NEW_GRAY > 0
        assert REWARD_VALID_ATTEMPT > 0

    def test_penalties_negative(self):
        """Test that penalties are negative."""
        assert PENALTY_REPEATED_GUESS < 0
        assert PENALTY_IGNORE_GREEN < 0
        assert PENALTY_IGNORE_YELLOW < 0
        assert PENALTY_USE_ELIMINATED < 0
        assert PENALTY_WRONG_LENGTH < 0
        assert PENALTY_NOT_A_WORD < 0

    def test_yellow_to_green_balance(self):
        """Test that yellow + yellow_to_green equals direct green reward."""
        # Per the plan: yellow(0.03) + yellow_to_green(0.02) = direct_green(0.05)
        assert abs(REWARD_PER_NEW_YELLOW + REWARD_YELLOW_TO_GREEN - REWARD_PER_NEW_GREEN) < 0.001
