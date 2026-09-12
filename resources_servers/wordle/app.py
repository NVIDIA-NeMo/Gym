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

"""Wordle NemoGym Resource Server.

Implements a Wordle game environment for training LLMs with reinforcement learning.
The model learns to play Wordle by making guesses and receiving feedback.

Endpoints:
- POST /seed_session: Initialize a new game with a target word
- POST /submit_guess: Submit a guess and receive feedback
- POST /check_word_validity: Check if a word is valid (soft constraint tool)
- POST /get_game_state: Query current game knowledge state
- POST /verify: Calculate final reward for the game
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field as PydanticField

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.wordle.wordle_words import WORDLE_VALID_GUESSES, get_random_target, is_valid_guess


# =============================================================================
# Reward Constants
# =============================================================================

WIN_REWARD_BASE = 2.0
WIN_REWARD_PENALTY_PER_TURN = 0.2

PENALTY_REPEATED_GUESS = -0.2
PENALTY_IGNORE_GREEN = -0.05
PENALTY_IGNORE_YELLOW = -0.03
PENALTY_USE_ELIMINATED = -0.02
PENALTY_WRONG_LENGTH = -0.02
PENALTY_NOT_A_WORD = -0.02

LOSS_REWARD = 0.0


def calculate_win_reward(turns_used: int) -> float:
    """Calculate win reward based on number of turns used.

    Formula: reward = 2.0 - 0.2 * (turns_used - 1), capped at turn-3 level.
    Turn 1: 1.6 (lucky, same as turn 3), Turn 2: 1.8, Turn 3: 1.6,
    Turn 4: 1.4, Turn 5: 1.2, Turn 6: 1.0
    """
    effective_turns = max(turns_used, 3)
    return WIN_REWARD_BASE - WIN_REWARD_PENALTY_PER_TURN * (effective_turns - 1)


# =============================================================================
# Game State
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

        for i, (g, t) in enumerate(zip(guess, target)):
            if g == t:
                feedback[i] = "G"
                target_remaining[i] = None

        for i, g in enumerate(guess):
            if feedback[i] == "G":
                continue
            if g in target_remaining:
                feedback[i] = "Y"
                idx = target_remaining.index(g)
                target_remaining[idx] = None

        return feedback

    @staticmethod
    def is_valid_word(word: str, word_length: int = 5) -> tuple[bool, str]:
        """Check if a word is valid for guessing."""
        word = word.lower()
        if len(word) != word_length:
            return False, f"Word must be {word_length} letters, got {len(word)}"
        if not word.isalpha():
            return False, "Word must contain only letters"
        if word not in WORDLE_VALID_GUESSES:
            return False, f"'{word}' is not a valid English word"
        return True, "Valid word"

    @staticmethod
    def calculate_turn_reward(guess: str, feedback: list[str], state: WordleGameState) -> tuple[float, dict]:
        """Calculate the penalty for a turn based on strategic mistakes."""
        reward = 0.0
        breakdown = {}

        if guess in state.guesses:
            reward += PENALTY_REPEATED_GUESS
            breakdown["repeated_guess_penalty"] = PENALTY_REPEATED_GUESS

        if state.known_greens:
            for pos, letter in state.known_greens.items():
                if pos < len(guess) and guess[pos] != letter:
                    reward += PENALTY_IGNORE_GREEN
                    breakdown["ignore_green_penalty"] = breakdown.get("ignore_green_penalty", 0) + PENALTY_IGNORE_GREEN

        if state.known_yellows and not any(letter in guess for letter in state.known_yellows):
            reward += PENALTY_IGNORE_YELLOW
            breakdown["ignore_yellow_penalty"] = PENALTY_IGNORE_YELLOW

        for letter in guess:
            if letter in state.eliminated_letters:
                reward += PENALTY_USE_ELIMINATED
                breakdown["use_eliminated_penalty"] = breakdown.get("use_eliminated_penalty", 0) + PENALTY_USE_ELIMINATED

        return reward, breakdown

    @staticmethod
    def update_knowledge(guess: str, feedback: list[str], state: WordleGameState) -> None:
        """Update the game state knowledge based on guess feedback."""
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


# =============================================================================
# Request/Response Models
# =============================================================================


class WordleResourcesServerConfig(BaseResourcesServerConfig):
    pass


class WordleSeedSessionRequest(BaseSeedSessionRequest):
    word_length: int = 5
    max_turns: int = 6
    custom_target: Optional[str] = None


class WordleSeedSessionResponse(BaseSeedSessionResponse):
    word_length: int
    max_turns: int
    message: str


class SubmitGuessRequest(BaseModel):
    guess: str


class SubmitGuessResponse(BaseModel):
    valid: bool
    feedback: Optional[str] = None
    won: bool = False
    game_over: bool = False
    turn: int = 0
    turns_remaining: int = 0
    error: Optional[str] = None
    target_word: Optional[str] = None


class CheckWordValidityRequest(BaseModel):
    word: str


class CheckWordValidityResponse(BaseModel):
    valid: bool
    reason: str


class GetGameStateResponse(BaseModel):
    turn: int
    turns_remaining: int
    guesses: list[str]
    feedback_history: list[str]
    known_greens: Dict[str, str]
    known_yellows: list[str]
    eliminated_letters: list[str]
    game_over: bool
    won: bool


class WordleVerifyRequest(BaseVerifyRequest):
    word_length: int = 5
    max_turns: int = 6
    custom_target: Optional[str] = None


class WordleVerifyResponse(BaseVerifyResponse):
    reward_breakdown: Dict[str, Any] = PydanticField(default_factory=dict)
    game_outcome: str = ""
    turns_used: int = 0
    won: float = 0.0
    turns_if_won: float = 0.0


# =============================================================================
# Resource Server
# =============================================================================


class WordleResourcesServer(SimpleResourcesServer):
    """Wordle game resource server for NemoGym."""

    config: WordleResourcesServerConfig
    session_id_to_state: Dict[str, WordleGameState] = PydanticField(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/submit_guess")(self.submit_guess)
        app.post("/check_word_validity")(self.check_word_validity)
        app.post("/get_game_state")(self.get_game_state)
        return app

    async def seed_session(self, request: Request, body: WordleSeedSessionRequest) -> WordleSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]

        if body.custom_target:
            target_word = body.custom_target.lower()
            if len(target_word) != body.word_length:
                target_word = get_random_target(body.word_length, use_training_set=True)
        else:
            target_word = get_random_target(body.word_length, use_training_set=True)

        state = WordleGameState(
            target_word=target_word,
            word_length=body.word_length,
            max_turns=body.max_turns,
        )
        self.session_id_to_state[session_id] = state

        return WordleSeedSessionResponse(
            word_length=body.word_length,
            max_turns=body.max_turns,
            message=f"Wordle game started! Guess the {body.word_length}-letter word in {body.max_turns} attempts.",
        )

    async def submit_guess(self, request: Request, body: SubmitGuessRequest) -> SubmitGuessResponse:
        session_id = request.session[SESSION_ID_KEY]

        if session_id not in self.session_id_to_state:
            return SubmitGuessResponse(valid=False, error="Game not initialized. Session not found.")

        state = self.session_id_to_state[session_id]

        if state.game_over:
            return SubmitGuessResponse(
                valid=False, game_over=True, won=state.won,
                turn=state.turn, turns_remaining=0,
                error="Game is already over.", target_word=state.target_word.upper(),
            )

        guess = body.guess.lower().strip()

        # All guesses consume a turn, even invalid ones
        state.turn += 1

        if len(guess) != state.word_length:
            state.total_reward += PENALTY_WRONG_LENGTH
            if state.turn >= state.max_turns:
                state.game_over = True
                state.total_reward = LOSS_REWARD
            return SubmitGuessResponse(
                valid=False, turn=state.turn,
                turns_remaining=max(0, state.max_turns - state.turn),
                error=f"Guess must be {state.word_length} letters. Got {len(guess)} letters.",
                game_over=state.game_over,
            )

        if not is_valid_guess(guess, state.word_length):
            state.total_reward += PENALTY_NOT_A_WORD
            if state.turn >= state.max_turns:
                state.game_over = True
                state.total_reward = LOSS_REWARD
            return SubmitGuessResponse(
                valid=False, turn=state.turn,
                turns_remaining=max(0, state.max_turns - state.turn),
                error=f"'{guess}' is not a valid English word.",
                game_over=state.game_over,
            )

        if guess == state.target_word:
            state.won = True
            state.game_over = True
            state.total_reward += calculate_win_reward(state.turn)
            state.total_reward = max(state.total_reward, 0.1)
            state.guesses.append(guess)
            state.feedback_history.append(["G"] * state.word_length)
            return SubmitGuessResponse(
                valid=True, feedback="G" * state.word_length,
                won=True, game_over=True,
                turn=state.turn, turns_remaining=0,
                target_word=state.target_word.upper(),
            )

        feedback = WordleGameLogic.get_feedback(guess, state.target_word)
        feedback_str = "".join(feedback)

        turn_reward, _ = WordleGameLogic.calculate_turn_reward(guess, feedback, state)
        state.total_reward += turn_reward

        WordleGameLogic.update_knowledge(guess, feedback, state)
        state.guesses.append(guess)
        state.feedback_history.append(feedback)

        if state.turn >= state.max_turns:
            state.game_over = True
            state.total_reward = LOSS_REWARD
            return SubmitGuessResponse(
                valid=True, feedback=feedback_str,
                won=False, game_over=True,
                turn=state.turn, turns_remaining=0,
                target_word=state.target_word.upper(),
            )

        return SubmitGuessResponse(
            valid=True, feedback=feedback_str,
            won=False, game_over=False,
            turn=state.turn, turns_remaining=state.max_turns - state.turn,
        )

    async def check_word_validity(self, request: Request, body: CheckWordValidityRequest) -> CheckWordValidityResponse:
        session_id = request.session[SESSION_ID_KEY]
        word_length = 5
        if session_id in self.session_id_to_state:
            word_length = self.session_id_to_state[session_id].word_length
        word = body.word.lower().strip()
        is_valid, reason = WordleGameLogic.is_valid_word(word, word_length)
        return CheckWordValidityResponse(valid=is_valid, reason=reason)

    async def get_game_state(self, request: Request) -> GetGameStateResponse:
        session_id = request.session[SESSION_ID_KEY]

        if session_id not in self.session_id_to_state:
            return GetGameStateResponse(
                turn=0, turns_remaining=6, guesses=[], feedback_history=[],
                known_greens={}, known_yellows=[], eliminated_letters=[],
                game_over=False, won=False,
            )

        state = self.session_id_to_state[session_id]
        feedback_strings = ["".join(fb) for fb in state.feedback_history]
        greens_display = {str(pos + 1): letter.upper() for pos, letter in state.known_greens.items()}

        return GetGameStateResponse(
            turn=state.turn,
            turns_remaining=state.max_turns - state.turn,
            guesses=[g.upper() for g in state.guesses],
            feedback_history=feedback_strings,
            known_greens=greens_display,
            known_yellows=sorted([l.upper() for l in state.known_yellows]),
            eliminated_letters=sorted([l.upper() for l in state.eliminated_letters]),
            game_over=state.game_over,
            won=state.won,
        )

    async def verify(self, request: Request, body: WordleVerifyRequest) -> WordleVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]

        if session_id not in self.session_id_to_state:
            return WordleVerifyResponse(
                **body.model_dump(), reward=0.0,
                reward_breakdown={"error": "No game state found"},
                game_outcome="incomplete", turns_used=0, won=0.0, turns_if_won=0.0,
            )

        state = self.session_id_to_state[session_id]

        if state.won:
            outcome = "win"
        elif state.game_over:
            outcome = "loss"
        else:
            outcome = "incomplete"

        breakdown = {"outcome": outcome, "turns_used": state.turn, "total_reward": state.total_reward}
        if state.won:
            breakdown["win_reward"] = calculate_win_reward(state.turn)

        final_reward = max(state.total_reward, 0.0)

        return WordleVerifyResponse(
            **body.model_dump(), reward=final_reward, reward_breakdown=breakdown,
            game_outcome=outcome, turns_used=state.turn,
            won=1.0 if state.won else 0.0,
            turns_if_won=float(state.turn) if state.won else 0.0,
        )


if __name__ == "__main__":
    WordleResourcesServer.run_webserver()
