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

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.wordle.wordle_logic import (
    LOSS_REWARD,
    PENALTY_NOT_A_WORD,
    PENALTY_WRONG_LENGTH,
    GuessResult,
    WordleGameLogic,
    WordleGameState,
    calculate_win_reward,
)
from resources_servers.wordle.wordle_words import TRAINING_WORDS, get_random_target, is_valid_guess


# =============================================================================
# Configuration
# =============================================================================


class WordleResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the Wordle resource server."""

    pass


# =============================================================================
# Request/Response Models
# =============================================================================


class WordleSeedSessionRequest(BaseSeedSessionRequest):
    """Request to initialize a new Wordle game session."""

    word_length: int = 5
    max_turns: int = 6
    custom_target: Optional[str] = None  # For testing - specify target word


class WordleSeedSessionResponse(BaseSeedSessionResponse):
    """Response after initializing a Wordle game session."""

    word_length: int
    max_turns: int
    message: str


class SubmitGuessRequest(BaseModel):
    """Request to submit a guess."""

    guess: str


class SubmitGuessResponse(BaseModel):
    """Response after submitting a guess."""

    valid: bool
    feedback: Optional[str] = None  # e.g., "GY__G"
    won: bool = False
    game_over: bool = False
    turn: int = 0
    turns_remaining: int = 0
    error: Optional[str] = None
    target_word: Optional[str] = None  # Revealed on game over


class CheckWordValidityRequest(BaseModel):
    """Request to check word validity (soft constraint tool)."""

    word: str


class CheckWordValidityResponse(BaseModel):
    """Response for word validity check."""

    valid: bool
    reason: str


class GetGameStateResponse(BaseModel):
    """Response with current game knowledge state."""

    turn: int
    turns_remaining: int
    guesses: list[str]
    feedback_history: list[str]  # List of "GY__G" strings
    known_greens: Dict[str, str]  # {"1": "a", "3": "e"} (1-indexed for clarity)
    known_yellows: list[str]
    eliminated_letters: list[str]
    game_over: bool
    won: bool


class WordleVerifyRequest(BaseVerifyRequest):
    """Request to verify and calculate final reward."""

    word_length: int = 5
    max_turns: int = 6
    custom_target: Optional[str] = None


class WordleVerifyResponse(BaseVerifyResponse):
    """Response with final reward and breakdown."""

    reward_breakdown: Dict[str, Any] = Field(default_factory=dict)
    game_outcome: str = ""  # "win", "loss", or "incomplete"
    turns_used: int = 0

    # Numeric metrics for easy aggregation (averaging across validation set)
    won: float = 0.0  # 1.0 if won, 0.0 if lost (average = win rate)
    turns_if_won: float = 0.0  # turns_used if won, 0.0 if lost (for avg turns calculation)


# =============================================================================
# Resource Server
# =============================================================================


class WordleResourcesServer(SimpleResourcesServer):
    """Wordle game resource server for NemoGym."""

    config: WordleResourcesServerConfig
    session_id_to_state: Dict[str, WordleGameState] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        """Set up the FastAPI webserver with all endpoints."""
        app = super().setup_webserver()

        app.post("/submit_guess")(self.submit_guess)
        app.post("/check_word_validity")(self.check_word_validity)
        app.post("/get_game_state")(self.get_game_state)

        return app

    async def seed_session(
        self, request: Request, body: WordleSeedSessionRequest
    ) -> WordleSeedSessionResponse:
        """Initialize a new Wordle game session.

        Creates a new game with a target word:
        - If custom_target is provided (validation): uses that exact word
        - If no custom_target (training): picks randomly from TRAINING_WORDS (2,000 words)

        This ensures no overlap between training targets and validation targets.
        """
        session_id = request.session[SESSION_ID_KEY]

        # Get target word
        if body.custom_target:
            # Validation mode: use the specified target word
            target_word = body.custom_target.lower()
            # Validate custom target length
            if len(target_word) != body.word_length:
                target_word = get_random_target(body.word_length, use_training_set=True)
        else:
            # Training mode: pick randomly from TRAINING_WORDS (no overlap with validation)
            target_word = get_random_target(body.word_length, use_training_set=True)

        # Create game state
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

    async def submit_guess(
        self, request: Request, body: SubmitGuessRequest
    ) -> SubmitGuessResponse:
        """Submit a guess and receive feedback.

        This is the main game tool. Submitting a guess will:
        1. Validate the guess (correct length, valid English word)
        2. Calculate feedback (G/Y/_ for each letter)
        3. Update game state
        4. Return result with feedback and game status
        """
        session_id = request.session[SESSION_ID_KEY]

        if session_id not in self.session_id_to_state:
            return SubmitGuessResponse(
                valid=False,
                error="Game not initialized. Session not found.",
            )

        state = self.session_id_to_state[session_id]

        # Check if game is already over
        if state.game_over:
            return SubmitGuessResponse(
                valid=False,
                game_over=True,
                won=state.won,
                turn=state.turn,
                turns_remaining=0,
                error="Game is already over.",
                target_word=state.target_word.upper(),
            )

        guess = body.guess.lower().strip()

        # Validate guess length
        if len(guess) != state.word_length:
            # Small penalty but allow retry (don't consume turn)
            state.total_reward += PENALTY_WRONG_LENGTH
            return SubmitGuessResponse(
                valid=False,
                turn=state.turn,
                turns_remaining=state.max_turns - state.turn,
                error=f"Guess must be {state.word_length} letters. Got {len(guess)} letters.",
            )

        # Validate it's a real word
        if not is_valid_guess(guess, state.word_length):
            # Small penalty but allow retry (don't consume turn)
            state.total_reward += PENALTY_NOT_A_WORD
            return SubmitGuessResponse(
                valid=False,
                turn=state.turn,
                turns_remaining=state.max_turns - state.turn,
                error=f"'{guess}' is not a valid English word.",
            )

        # Valid guess - process it
        state.turn += 1

        # Check for win
        if guess == state.target_word:
            state.won = True
            state.game_over = True
            win_reward = calculate_win_reward(state.turn)
            state.total_reward += win_reward
            state.total_reward = max(state.total_reward, 0.1)  # Wins are always positive
            state.guesses.append(guess)
            state.feedback_history.append(["G"] * state.word_length)

            return SubmitGuessResponse(
                valid=True,
                feedback="G" * state.word_length,
                won=True,
                game_over=True,
                turn=state.turn,
                turns_remaining=0,
                target_word=state.target_word.upper(),
            )

        # Calculate feedback
        feedback = WordleGameLogic.get_feedback(guess, state.target_word)
        feedback_str = "".join(feedback)

        # Calculate turn reward
        turn_reward, _ = WordleGameLogic.calculate_turn_reward(guess, feedback, state)
        state.total_reward += turn_reward

        # Update knowledge state
        WordleGameLogic.update_knowledge(guess, feedback, state)

        # Record guess
        state.guesses.append(guess)
        state.feedback_history.append(feedback)

        # Check for loss (max turns reached)
        if state.turn >= state.max_turns:
            state.game_over = True
            state.total_reward = LOSS_REWARD  # Override with loss reward

            return SubmitGuessResponse(
                valid=True,
                feedback=feedback_str,
                won=False,
                game_over=True,
                turn=state.turn,
                turns_remaining=0,
                target_word=state.target_word.upper(),
            )

        # Game continues
        return SubmitGuessResponse(
            valid=True,
            feedback=feedback_str,
            won=False,
            game_over=False,
            turn=state.turn,
            turns_remaining=state.max_turns - state.turn,
        )

    async def check_word_validity(
        self, request: Request, body: CheckWordValidityRequest
    ) -> CheckWordValidityResponse:
        """Check if a word is valid BEFORE guessing (soft constraint tool).

        This is an informational tool - it does NOT block invalid guesses.
        The model can use this to check words before submitting, but it's
        not required and doesn't affect the game state.
        """
        session_id = request.session[SESSION_ID_KEY]

        # Get word length from session if available, default to 5
        word_length = 5
        if session_id in self.session_id_to_state:
            word_length = self.session_id_to_state[session_id].word_length

        word = body.word.lower().strip()

        # Check validity
        is_valid, reason = WordleGameLogic.is_valid_word(word, word_length)

        return CheckWordValidityResponse(valid=is_valid, reason=reason)

    async def get_game_state(self, request: Request) -> GetGameStateResponse:
        """Query current game knowledge state.

        Returns the accumulated knowledge from all previous guesses:
        - Confirmed letter positions (greens)
        - Letters known to be in the word (yellows)
        - Eliminated letters (grays)
        """
        session_id = request.session[SESSION_ID_KEY]

        if session_id not in self.session_id_to_state:
            # Return empty state if game not initialized
            return GetGameStateResponse(
                turn=0,
                turns_remaining=6,
                guesses=[],
                feedback_history=[],
                known_greens={},
                known_yellows=[],
                eliminated_letters=[],
                game_over=False,
                won=False,
            )

        state = self.session_id_to_state[session_id]

        # Convert feedback history to strings
        feedback_strings = ["".join(fb) for fb in state.feedback_history]

        # Convert known_greens to 1-indexed string keys for clarity
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

    async def verify(
        self, request: Request, body: WordleVerifyRequest
    ) -> WordleVerifyResponse:
        """Calculate final reward for the game.

        Called at the end of an episode to get the total reward.

        Returns metrics that can be aggregated across validation set:
        - reward: Total reward (win: 1.0-2.0, loss: 0.0)
        - won: 1.0 if won, 0.0 if lost (average across val set = win rate)
        - turns_if_won: turns used if won, 0 if lost (sum/num_wins = avg turns to win)
        """
        session_id = request.session[SESSION_ID_KEY]

        if session_id not in self.session_id_to_state:
            # No game state - return zero reward
            return WordleVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                reward_breakdown={"error": "No game state found"},
                game_outcome="incomplete",
                turns_used=0,
                won=0.0,
                turns_if_won=0.0,
            )

        state = self.session_id_to_state[session_id]

        # Determine outcome
        if state.won:
            outcome = "win"
        elif state.game_over:
            outcome = "loss"
        else:
            outcome = "incomplete"

        # Build reward breakdown
        breakdown = {
            "outcome": outcome,
            "turns_used": state.turn,
            "total_reward": state.total_reward,
        }

        if state.won:
            breakdown["win_reward"] = calculate_win_reward(state.turn)

        # Numeric metrics for aggregation
        won_numeric = 1.0 if state.won else 0.0
        turns_if_won = float(state.turn) if state.won else 0.0

        # Minimum reward is 0.0 — losses and incomplete games get zero
        final_reward = max(state.total_reward, 0.0)

        return WordleVerifyResponse(
            **body.model_dump(),
            reward=final_reward,
            reward_breakdown=breakdown,
            game_outcome=outcome,
            turns_used=state.turn,
            won=won_numeric,
            turns_if_won=turns_if_won,
        )


if __name__ == "__main__":
    WordleResourcesServer.run_webserver()
