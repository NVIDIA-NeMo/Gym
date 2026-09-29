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

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Dict, Optional

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall
from resources_servers.gymnasium import GymnasiumServer


GUESSES_PATH = Path(__file__).parent / "data" / "guesses.txt"

WIN_REWARD_BASE = 2.0
WIN_REWARD_PENALTY_PER_TURN = 0.2

PENALTY_REPEATED_GUESS = -0.2
PENALTY_IGNORE_GREEN = -0.05
PENALTY_IGNORE_YELLOW = -0.03
PENALTY_USE_ELIMINATED = -0.02
PENALTY_WRONG_LENGTH = -0.02
PENALTY_NOT_A_WORD = -0.02

LOSS_REWARD = 0.0


def read_words(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python resources_servers/wordle/generate_data.py` first.")
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


@cache
def valid_guesses() -> frozenset[str]:
    return frozenset(read_words(GUESSES_PATH))


def is_valid_guess(word: str, word_length: int = 5) -> bool:
    return len(word) == word_length and word.lower() in valid_guesses()


def calculate_win_reward(turns_used: int) -> float:
    # A turn-1 win is scored like turn 3 so lucky openers are not over-rewarded.
    effective_turns = 3 if turns_used == 1 else turns_used
    return WIN_REWARD_BASE - WIN_REWARD_PENALTY_PER_TURN * (effective_turns - 1)


@dataclass
class WordleGameState:
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


class WordleGameLogic:
    @staticmethod
    def get_feedback(guess: str, target: str) -> list[str]:
        # Each target letter matches at most one guess letter, and greens claim theirs first.
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
        word = word.lower()
        if len(word) != word_length:
            return False, f"Word must be {word_length} letters, got {len(word)}"
        if not word.isalpha():
            return False, "Word must contain only letters"
        if word not in valid_guesses():
            return False, f"'{word}' is not a valid English word"
        return True, "Valid word"

    @staticmethod
    def calculate_turn_reward(guess: str, state: WordleGameState) -> float:
        reward = 0.0
        if guess in state.guesses:
            reward += PENALTY_REPEATED_GUESS
        for pos, letter in state.known_greens.items():
            if pos < len(guess) and guess[pos] != letter:
                reward += PENALTY_IGNORE_GREEN
        if state.known_yellows and not any(letter in guess for letter in state.known_yellows):
            reward += PENALTY_IGNORE_YELLOW
        for letter in guess:
            if letter in state.eliminated_letters:
                reward += PENALTY_USE_ELIMINATED
        return reward

    @staticmethod
    def update_knowledge(guess: str, feedback: list[str], state: WordleGameState) -> None:
        for i, (fb, letter) in enumerate(zip(feedback, guess)):
            if fb == "G":
                state.known_greens[i] = letter
            elif fb == "Y":
                state.known_yellows.add(letter)

        # Grays go last so a letter that is green or yellow elsewhere in this guess is not eliminated.
        for i, (fb, letter) in enumerate(zip(feedback, guess)):
            if fb == "_":
                if letter not in state.known_greens.values() and letter not in state.known_yellows:
                    state.eliminated_letters.add(letter)


class WordleResourcesServerConfig(BaseResourcesServerConfig):
    pass


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


class WordleResourcesServer(GymnasiumServer):
    ray_enabled = False
    config: WordleResourcesServerConfig

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        word_length = metadata.get("word_length", 5)
        max_turns = metadata.get("max_turns", 6)
        custom_target = metadata.get("custom_target")

        # A random fallback would give each rollout in a GRPO group a different word.
        target_word = (custom_target or "").lower()
        if not is_valid_guess(target_word, word_length):
            raise HTTPException(status_code=400, detail=f"Invalid or missing custom_target: {custom_target!r}")

        self.session_state[session_id] = WordleGameState(
            target_word=target_word,
            word_length=word_length,
            max_turns=max_turns,
        )
        return None, {}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        state = self.session_state.get(session_id)
        if state is None:
            raise HTTPException(status_code=400, detail="Session not initialized. Call /reset first.")

        tool_outputs = [
            self.tool_output(item, self._call_tool(state, item))
            for item in action.output
            if item.type == "function_call"
        ]
        info = {
            "game_outcome": "win" if state.won else "loss" if state.game_over else "incomplete",
            "turns_used": state.turn,
            "won": 1.0 if state.won else 0.0,
            "turns_if_won": float(state.turn) if state.won else 0.0,
        }
        if tool_outputs and not state.game_over:
            return None, 0.0, False, False, {**info, "tool_outputs": tool_outputs}

        # The game ended or the model stopped calling tools.
        return None, max(state.total_reward, 0.0), True, False, info

    def _call_tool(self, state: WordleGameState, call: NeMoGymResponseFunctionToolCall) -> dict:
        try:
            args = json.loads(call.arguments or "{}")
            if call.name == "submit_guess":
                return self.submit_guess(state, SubmitGuessRequest.model_validate(args)).model_dump()
            if call.name == "check_word_validity":
                return self.check_word_validity(state, CheckWordValidityRequest.model_validate(args)).model_dump()
            if call.name == "get_game_state":
                return self.get_game_state(state).model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            return {"error": f"Invalid arguments for {call.name}: {e}"}
        return {"error": f"Unknown tool: {call.name}"}

    def submit_guess(self, state: WordleGameState, body: SubmitGuessRequest) -> SubmitGuessResponse:
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

        # All guesses consume a turn, even invalid ones
        state.turn += 1

        error = None
        if len(guess) != state.word_length:
            penalty = PENALTY_WRONG_LENGTH
            error = f"Guess must be {state.word_length} letters. Got {len(guess)} letters."
        elif not is_valid_guess(guess, state.word_length):
            penalty = PENALTY_NOT_A_WORD
            error = f"'{guess}' is not a valid English word."
        if error:
            state.total_reward += penalty
            if state.turn >= state.max_turns:
                state.game_over = True
                state.total_reward = LOSS_REWARD
            return SubmitGuessResponse(
                valid=False,
                turn=state.turn,
                turns_remaining=max(0, state.max_turns - state.turn),
                error=error,
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
                valid=True,
                feedback="G" * state.word_length,
                won=True,
                game_over=True,
                turn=state.turn,
                turns_remaining=0,
                target_word=state.target_word.upper(),
            )

        feedback = WordleGameLogic.get_feedback(guess, state.target_word)
        feedback_str = "".join(feedback)

        state.total_reward += WordleGameLogic.calculate_turn_reward(guess, state)

        WordleGameLogic.update_knowledge(guess, feedback, state)
        state.guesses.append(guess)
        state.feedback_history.append(feedback)

        if state.turn >= state.max_turns:
            state.game_over = True
            state.total_reward = LOSS_REWARD
        return SubmitGuessResponse(
            valid=True,
            feedback=feedback_str,
            game_over=state.game_over,
            turn=state.turn,
            turns_remaining=max(0, state.max_turns - state.turn),
            target_word=state.target_word.upper() if state.game_over else None,
        )

    def check_word_validity(self, state: WordleGameState, body: CheckWordValidityRequest) -> CheckWordValidityResponse:
        word = body.word.lower().strip()
        is_valid, reason = WordleGameLogic.is_valid_word(word, state.word_length)
        return CheckWordValidityResponse(valid=is_valid, reason=reason)

    def get_game_state(self, state: WordleGameState) -> GetGameStateResponse:
        feedback_strings = ["".join(fb) for fb in state.feedback_history]
        greens_display = {str(pos + 1): letter.upper() for pos, letter in state.known_greens.items()}

        return GetGameStateResponse(
            turn=state.turn,
            turns_remaining=state.max_turns - state.turn,
            guesses=[g.upper() for g in state.guesses],
            feedback_history=feedback_strings,
            known_greens=greens_display,
            known_yellows=sorted([letter.upper() for letter in state.known_yellows]),
            eliminated_letters=sorted([letter.upper() for letter in state.eliminated_letters]),
            game_over=state.game_over,
            won=state.won,
        )


if __name__ == "__main__":
    WordleResourcesServer.run_webserver()
