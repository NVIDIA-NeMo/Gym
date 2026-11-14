# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Scrabble move-finding environment in Gymnasium API style.

Each row pins a position (board and rack). The model submits a move with the ``play_move`` tool; an
illegal move gets a reason and uses one of the row's attempts. The episode ends at the first legal move.
Reward: ``score / best_score - invalid_attempt_penalty * illegal_moves``, clipped to [0, 1]. No legal move: 0.
"""

import json
import logging
from functools import cache
from pathlib import Path
from typing import ClassVar, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field, JsonValue, ValidationError, field_validator

from nemo_gym.core import NeMoGymResponse, NeMoGymResponseFunctionToolCall
from nemo_gym.server_runtime import BaseResourcesServerConfig, CheckpointMode
from resources_servers.gymnasium import GymnasiumServer
from resources_servers.scrabble import game
from resources_servers.scrabble.task_data import TaskData


_LOGGER = logging.getLogger(__name__)

DEFAULT_DICTIONARY_PATH = Path(__file__).parent / "data" / "words.txt"


@cache
def load_dictionary(path: str) -> frozenset[str]:
    """Read one uppercase word per line; ``#`` lines are comments."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"{file} not found. Run `python resources_servers/scrabble/generate_data.py` to download the word list."
        )
    return frozenset(
        line.strip().upper() for line in file.read_text().splitlines() if line.strip() and not line.startswith("#")
    )


class ScrabbleResourcesServerConfig(BaseResourcesServerConfig):
    dictionary_path: str = Field(
        default=str(DEFAULT_DICTIONARY_PATH), description="Word list, one word per line. Defaults to ENABLE."
    )
    invalid_attempt_penalty: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Subtracted from the reward for each illegal move before a legal one."
    )


class PlayMoveRequest(BaseModel):
    word: str
    row: int
    col: str
    direction: str

    @field_validator("col", mode="before")
    @classmethod
    def _col_as_letter(cls, value: object) -> object:
        # Accept a 1-based column number as well as a letter, since models mix the two up.
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value)
        if isinstance(value, int) and 1 <= value <= game.BOARD_SIZE:
            return game.COLUMNS[value - 1]
        return value


class ScrabbleSession(BaseModel):
    """Per-session state. Plain JSON, so it is exported as-is for checkpoints."""

    board: list[str]
    rack: str
    best_score: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    illegal_moves: int = 0
    finished: bool = False

    @field_validator("board")
    @classmethod
    def _board(cls, value: list[str]) -> list[str]:
        return game.validate_board(value)

    @field_validator("rack")
    @classmethod
    def _rack(cls, value: str) -> str:
        return game.validate_rack(value)


class ScrabbleResourcesServer(GymnasiumServer):
    config: ScrabbleResourcesServerConfig
    # The rest of an episode depends only on the session fields, which are all exported.
    checkpoint_mode: ClassVar[CheckpointMode] = "exported"

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        # Fail at startup, not on the first move, when the word list is missing.
        words = load_dictionary(self.config.dictionary_path)
        _LOGGER.info("Loaded %d words from %s", len(words), self.config.dictionary_path)

    @property
    def dictionary(self) -> frozenset[str]:
        return load_dictionary(self.config.dictionary_path)

    def export_env_state(self, state: ScrabbleSession) -> JsonValue:
        return state.model_dump(mode="json")

    def restore_env_state(self, state: JsonValue) -> ScrabbleSession:
        try:
            return ScrabbleSession.model_validate(state)
        except ValidationError as error:
            raise ValueError(f"invalid scrabble session state: {error}") from error

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        try:
            task = TaskData.model_validate(metadata)
            session = ScrabbleSession(
                board=task.board, rack=task.rack, best_score=task.best_score, max_attempts=task.max_attempts
            )
        except ValidationError as error:
            raise HTTPException(status_code=400, detail=f"invalid scrabble row: {error}") from error
        self.session_state[session_id] = session
        return None, {}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        session: ScrabbleSession | None = self.session_state.get(session_id)
        if session is None:
            # Never reset, or released (for example by a checkpoint retire): there is no puzzle to continue.
            raise HTTPException(status_code=404, detail="no Scrabble puzzle for this session; call /reset first")

        calls = [item for item in action.output if item.type == "function_call"]
        if not calls:
            return None, 0.0, True, False, self._metrics(session, outcome="no_move")

        tool_outputs = []
        result: game.MoveResult | None = None
        for call in calls:
            if session.finished:
                tool_outputs.append(
                    self.tool_output(call, {"error": "The puzzle is over; no further moves are accepted."})
                )
                continue
            result, output = self._play(session, call)
            tool_outputs.append(self.tool_output(call, output))

        if result is not None and result.valid:
            ratio = min(result.score / session.best_score, 1.0)
            if result.score > session.best_score:
                _LOGGER.warning("Move scored %d, above the row's best_score %d", result.score, session.best_score)
            reward = max(0.0, ratio - self.config.invalid_attempt_penalty * session.illegal_moves)
            return None, reward, True, False, self._metrics(session, outcome="legal_move", result=result)
        if session.finished:
            return None, 0.0, True, False, self._metrics(session, outcome="out_of_attempts")
        return None, 0.0, False, False, {"tool_outputs": tool_outputs}

    def _play(self, session: ScrabbleSession, call: NeMoGymResponseFunctionToolCall) -> tuple[game.MoveResult, dict]:
        """Check one tool call as a move and update the session."""
        if call.name != "play_move":
            result = game.MoveResult(False, error=f"Unknown tool {call.name!r}; the only tool is play_move.")
        else:
            try:
                move = PlayMoveRequest.model_validate(json.loads(call.arguments or "{}"))
            except (json.JSONDecodeError, ValidationError) as error:
                result = game.MoveResult(False, error=f"Invalid arguments for play_move: {error}")
            else:
                col = game.COLUMNS.find(move.col.strip().upper()) if len(move.col.strip()) == 1 else -1
                result = game.evaluate_move(
                    session.board, session.rack, move.word, move.row - 1, col, move.direction.strip().lower(),
                    self.dictionary,
                )  # fmt: skip

        if result.valid:
            session.finished = True
            return result, {"legal": True, "score": result.score, "words": list(result.words)}

        session.illegal_moves += 1
        remaining = session.max_attempts - session.illegal_moves
        session.finished = remaining <= 0
        return result, {"legal": False, "error": result.error, "attempts_remaining": remaining}

    @staticmethod
    def _metrics(session: ScrabbleSession, *, outcome: str, result: game.MoveResult | None = None) -> dict:
        legal = result is not None and result.valid
        score = result.score if legal else 0
        return {
            "outcome": outcome,
            "score": float(score),
            "best_score": float(session.best_score),
            "score_ratio": min(score / session.best_score, 1.0),
            "optimal": 1.0 if score >= session.best_score else 0.0,
            "legal_move": 1.0 if legal else 0.0,
            "illegal_moves": float(session.illegal_moves),
            "bingo": 1.0 if legal and len(result.placed) == game.RACK_SIZE else 0.0,
        }


if __name__ == "__main__":
    ScrabbleResourcesServer.run_webserver()
