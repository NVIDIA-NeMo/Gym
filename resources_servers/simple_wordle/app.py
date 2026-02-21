"""FastAPI resources-server that exposes a Wordle game for SimpleGameAgent.

Endpoints
---------
POST /get_initial_board   →  returns instructions + fresh game_state
POST /make_move           →  submits a guess, returns feedback / reward
POST /verify              →  simple echo, used by the evaluation harness
"""

from __future__ import annotations

import random
import re
from typing import List, Tuple, Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from nltk.corpus import words
import nltk

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)


# ---------------------------------------------------------------------------
# Internal helpers (pure Wordle logic – no web stuff)
# ---------------------------------------------------------------------------


def _evaluate_guess(secret: str, guess: str) -> List[str]:
    """Return list of G / Y / X exactly like original env."""
    feedback = [None] * len(secret)
    secret_list = list(secret)
    guess_list = list(guess)

    # pass 1 – greens
    for i in range(len(secret)):
        if guess_list[i] == secret_list[i]:
            feedback[i] = "G"
            secret_list[i] = None

    # pass 2 – yellows / wrongs
    for i in range(len(secret)):
        if feedback[i] is None:
            if guess_list[i] in secret_list:
                feedback[i] = "Y"
                secret_list[secret_list.index(guess_list[i])] = None
            else:
                feedback[i] = "X"
    return feedback


def _internal_reward(
    feedback: List[str], milestones: List[int]
) -> Tuple[float, List[int]]:
    """0.5 / word_length for every newly found green."""
    reward = 0.0
    for i, (f, m) in enumerate(zip(feedback, milestones)):
        if f == "G" and m == 0:
            reward += 0.5 / len(milestones)
            milestones[i] = 1
    return reward, milestones.copy()


# ---------------------------------------------------------------------------
# Pydantic models (request / response)
# ---------------------------------------------------------------------------


class GetInitialBoardRequest(BaseModel):
    word_length: int | None = 5
    max_turns: int = 20
    only_real: bool = True


class GetInitialBoardResponse(BaseModel):
    instructions: str
    board_text: str
    game_state: Dict[str, Any]


class MakeMoveRequest(BaseModel):
    game_state: Dict[str, Any]
    guess: str  # e.g. "CRANE"


class MakeMoveResponse(BaseModel):
    success: bool
    message: str
    feedback: List[str] | None = None
    board_text: str
    game_state: Dict[str, Any]
    is_complete: bool
    move_reward: float


class WordleRunRequest(BaseRunRequest, GetInitialBoardRequest):
    pass


class WordleVerifyRequest(WordleRunRequest, BaseVerifyRequest):
    reward: float = 0.0
    total_moves: int = 0
    is_complete: bool = False


class WordleVerifyResponse(BaseVerifyResponse):
    total_moves: int = 0
    is_complete: bool = False


# ---------------------------------------------------------------------------
# The server itself
# ---------------------------------------------------------------------------


class WordleResourcesServerConfig(BaseResourcesServerConfig):
    pass


class WordleResourcesServer(SimpleResourcesServer):
    config: WordleResourcesServerConfig

    # ------------------------------------------------------------------ setup
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/get_initial_board")(self.get_initial_board)
        app.post("/make_move")(self.make_move)
        return app

    # ----------------------------------------------------------- API helpers
    @staticmethod
    def _instructions(word_len: int, max_turns: int) -> str:
        return (
            f"You are playing Wordle.\n"
            f"Guess the secret {word_len}-letter word in <= {max_turns} turns.\n"
            "After each guess you will receive a feedback string of equal length\n"
            "made of characters:\n"
            "  G - correct letter & position\n"
            "  Y - letter is present but misplaced\n"
            "  X - letter not in word\n\n"
            "Use the make_move function to submit your guesses.\n"
        )

    @staticmethod
    def _render_board(history: List[Tuple[str, List[str]]]) -> str:
        """Very simple text board: one line per past guess."""
        if not history:
            return "(no guesses yet)"
        return "\n".join(f"{g}\n{''.join(fb)}" for g, fb in history)

    # ----------------------------------------------------------- /get_initial_board
    async def get_initial_board(
        self, body: GetInitialBoardRequest
    ) -> GetInitialBoardResponse:
        nltk.download("words", quiet=True)

        word_len = body.word_length or random.randint(3, 6)
        vocab = [
            w
            for w in words.words("en-basic")
            if len(w) == word_len and w.isalpha() and w.islower()
        ]
        secret = random.choice(vocab).upper()

        game_state = {
            "secret": secret,
            "word_length": word_len,
            "max_turns": body.max_turns,
            "turn_count": 0,
            "history": [],  # list[(guess, feedback)]
            "milestones": [0] * word_len,
        }

        return GetInitialBoardResponse(
            instructions=self._instructions(word_len, body.max_turns),
            board_text=self._render_board([]),
            game_state=game_state,
        )

    # ----------------------------------------------------------- /make_move
    async def make_move(self, body: MakeMoveRequest) -> MakeMoveResponse:
        gs = body.game_state.copy()  # don’t mutate caller’s dict
        guess_raw = body.guess.strip().upper()

        # turn counter
        gs["turn_count"] += 1

        # parse boxed {...} if the agent included it
        match = re.search(r"\{([A-Z]+)\}", guess_raw)
        if match:
            guess_raw = match.group(1)

        # validate format / length
        if len(guess_raw) != gs["word_length"] or not guess_raw.isalpha():
            return MakeMoveResponse(
                success=False,
                message=f"Invalid guess '{guess_raw}'. It must be {gs['word_length']} letters.",
                board_text=self._render_board(gs["history"]),
                game_state=gs,
                is_complete=False,
                move_reward=0.0,  # Since we want to only have binary 0/1 rewards
            )

        # repeated guess?
        if any(guess_raw == prev for prev, _ in gs["history"]):
            return MakeMoveResponse(
                success=False,
                message=f"You already tried '{guess_raw}'.",
                board_text=self._render_board(gs["history"]),
                game_state=gs,
                is_complete=False,
                move_reward=0.0,  # Since we want to only have binary 0/1 rewards
            )

        # evaluate
        feedback = _evaluate_guess(gs["secret"], guess_raw)
        step_reward, new_milestones = _internal_reward(feedback, gs["milestones"])
        gs["milestones"] = new_milestones
        gs["history"].append((guess_raw, feedback))

        # finished?
        completed = feedback.count("G") == gs["word_length"]
        if completed:
            step_reward += 1.0  # success bonus

        # max turns?
        failed_out = gs["turn_count"] >= gs["max_turns"] and not completed
        is_terminal = completed or failed_out

        msg = (
            f"Correct! Secret word was {gs['secret']}."
            if completed
            else f"Turn {gs['turn_count']}: feedback for '{guess_raw}' is {''.join(feedback)}"
        )
        return MakeMoveResponse(
            success=True,
            message=msg,
            feedback=feedback,
            board_text=self._render_board(gs["history"]),
            game_state=gs,
            is_complete=is_terminal,
            move_reward=1.0 if completed else 0.0,
        )

    # ----------------------------------------------------------- /verify
    async def verify(self, body: WordleVerifyRequest) -> WordleVerifyResponse:
        return WordleVerifyResponse(**body.model_dump())


# ---------------------------------------------------------------------------
# Launch as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    WordleResourcesServer.run_webserver()
