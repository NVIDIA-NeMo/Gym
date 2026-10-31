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

"""
Connect Four environment in Gymnasium API style.

Multi-step: the model drops one disc per turn until the game ends. A scripted opponent
(uniform random at depth 0, depth-limited negamax otherwise) replies after every move.
Reward: +1 win, 0 draw, -1 loss or illegal move.
"""

import random
import re
from typing import Optional

from fastapi import HTTPException

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.gymnasium import GymnasiumServer, extract_text


ROWS, COLS = 6, 7
EMPTY, MODEL, OPPONENT = ".", "X", "O"
DEFAULT_OPPONENT_DEPTH = 2
MAX_OPPONENT_DEPTH = 6  # keep in sync with TaskData; search cost grows ~7x per ply
WIN_SCORE = 1_000_000

# Center-first move ordering makes alpha-beta prune far more.
_SEARCH_ORDER = [3, 2, 4, 1, 5, 0, 6]
_ACTION = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)

# Every horizontal, vertical and diagonal line of four cells.
_WINDOWS = [
    [(r + i * dr, c + i * dc) for i in range(4)]
    for r in range(ROWS)
    for c in range(COLS)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
    if 0 <= r + 3 * dr < ROWS and 0 <= c + 3 * dc < COLS
]

Board = list[list[str]]  # board[row][col]; row 0 is the top


def _other(piece: str) -> str:
    return OPPONENT if piece == MODEL else MODEL


def _new_board() -> Board:
    return [[EMPTY] * COLS for _ in range(ROWS)]


def _legal_columns(board: Board) -> list[int]:
    return [c for c in range(COLS) if board[0][c] == EMPTY]


def _drop(board: Board, col: int, piece: str) -> int:
    """Place `piece` in `col` and return the row it landed in."""
    for row in range(ROWS - 1, -1, -1):
        if board[row][col] == EMPTY:
            board[row][col] = piece
            return row
    raise ValueError(f"column {col + 1} is full")


def _connects_four(board: Board, row: int, col: int) -> bool:
    """Whether the disc at (row, col) completes a line of four."""
    piece = board[row][col]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == piece:
                count += 1
                r, c = r + sign * dr, c + sign * dc
        if count >= 4:
            return True
    return False


def _heuristic(board: Board, piece: str) -> int:
    """Static evaluation from `piece`'s point of view: open lines and center control."""
    other = _other(piece)
    score = 3 * sum(board[r][COLS // 2] == piece for r in range(ROWS))
    for window in _WINDOWS:
        cells = [board[r][c] for r, c in window]
        mine, theirs, empty = cells.count(piece), cells.count(other), cells.count(EMPTY)
        if mine == 3 and empty == 1:
            score += 5
        elif mine == 2 and empty == 2:
            score += 2
        if theirs == 3 and empty == 1:
            score -= 4
    return score


def _negamax(board: Board, depth: int, alpha: float, beta: float, piece: str) -> float:
    """Value of the position for `piece`, who is to move. Faster wins score higher."""
    legal = [c for c in _SEARCH_ORDER if board[0][c] == EMPTY]
    if not legal:
        return 0
    if depth == 0:
        return _heuristic(board, piece)
    best = -float("inf")
    for col in legal:
        row = _drop(board, col, piece)
        if _connects_four(board, row, col):
            value = WIN_SCORE + depth
        else:
            value = -_negamax(board, depth - 1, -beta, -alpha, _other(piece))
        board[row][col] = EMPTY
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best


def _opponent_move(board: Board, depth: int, rng: random.Random) -> int:
    """Pick the opponent's column; ties between equally good moves are broken by `rng`."""
    legal = _legal_columns(board)
    if depth <= 0:
        return rng.choice(legal)
    scores = {}
    for col in legal:
        row = _drop(board, col, OPPONENT)
        if _connects_four(board, row, col):
            scores[col] = WIN_SCORE + depth
        else:
            scores[col] = -_negamax(board, depth - 1, -float("inf"), float("inf"), MODEL)
        board[row][col] = EMPTY
    best = max(scores.values())
    return rng.choice([col for col, score in scores.items() if score == best])


def _parse_action(text: str, legal: list[int]) -> tuple[Optional[int], Optional[str]]:
    """The 0-based column the model chose, or (None, reason) when the move is illegal.

    Conflicting tags ("<action>3</action> ... or <action>4</action>") are rejected rather
    than resolved, so an ambiguous reply is never scored as a deliberate move.
    """
    choices = {m.strip() for m in _ACTION.findall(text)}
    if not choices:
        return None, "no_action_tag"
    if len(choices) > 1:
        return None, "multiple_actions"
    choice = choices.pop()
    # ASCII-only and length-bounded so exotic digits and huge strings fail cleanly.
    if not (choice.isascii() and choice.isdigit() and len(choice) <= 2) or not 1 <= int(choice) <= COLS:
        return None, "not_a_column"
    col = int(choice) - 1
    if col not in legal:
        return None, "column_full"
    return col, None


def _render(board: Board) -> str:
    rows = [" ".join(row) for row in board]
    return "\n".join(rows + [" ".join(str(c + 1) for c in range(COLS))])


def _observation(board: Board, opponent_col: Optional[int]) -> str:
    lead = f"Opponent dropped a disc in column {opponent_col + 1}.\n" if opponent_col is not None else ""
    legal = ", ".join(str(c + 1) for c in _legal_columns(board))
    return (
        f"{lead}Board (you are {MODEL}, opponent is {OPPONENT}; discs fall to the lowest empty row):\n"
        f"{_render(board)}\n"
        f"Legal columns: {legal}.\n"
        f"Respond with <action>N</action> where N is the column number."
    )


class ConnectFourEnv(GymnasiumServer):
    ray_enabled = False

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        rng = random.Random(metadata.get("seed"))
        depth = metadata.get("opponent_depth")
        depth = DEFAULT_OPPONENT_DEPTH if depth is None else depth
        if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= MAX_OPPONENT_DEPTH:
            raise HTTPException(
                status_code=400,
                detail=f"opponent_depth must be an integer in [0, {MAX_OPPONENT_DEPTH}], got {depth!r}",
            )
        model_first = metadata.get("model_first")
        model_first = True if model_first is None else model_first
        if not isinstance(model_first, bool):
            raise HTTPException(status_code=400, detail=f"model_first must be a boolean, got {model_first!r}")

        board = _new_board()
        moves: list[int] = []
        opponent_col = None
        if not model_first:
            opponent_col = _opponent_move(board, depth, rng)
            _drop(board, opponent_col, OPPONENT)
            moves.append(opponent_col + 1)

        self.session_state[session_id] = {
            "board": board,
            "rng": rng,
            "depth": depth,
            "model_first": model_first,
            "moves": moves,
        }
        return _observation(board, opponent_col), {}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        state = self.session_state.get(session_id)
        if state is None:
            raise HTTPException(status_code=400, detail="Session not initialized. Call /reset first.")
        board = state["board"]

        col, invalid_reason = _parse_action(extract_text(action), _legal_columns(board))
        if col is None:
            return None, -1.0, True, False, self._final_info(state, "invalid_move", invalid_reason)

        row = _drop(board, col, MODEL)
        state["moves"].append(col + 1)
        if _connects_four(board, row, col):
            return None, 1.0, True, False, self._final_info(state, "win")
        if not _legal_columns(board):
            return None, 0.0, True, False, self._final_info(state, "draw")

        opponent_col = _opponent_move(board, state["depth"], state["rng"])
        row = _drop(board, opponent_col, OPPONENT)
        state["moves"].append(opponent_col + 1)
        if _connects_four(board, row, opponent_col):
            return None, -1.0, True, False, self._final_info(state, "loss")
        if not _legal_columns(board):
            return None, 0.0, True, False, self._final_info(state, "draw")

        return _observation(board, opponent_col), 0.0, False, False, {}

    @staticmethod
    def _final_info(state: dict, result: str, invalid_reason: Optional[str] = None) -> dict:
        info = {
            "result": result,
            "moves": state["moves"],
            "model_first": state["model_first"],
            "opponent_depth": state["depth"],
            "board": _render(state["board"]),
        }
        if invalid_reason is not None:
            info["invalid_reason"] = invalid_reason
        return info


if __name__ == "__main__":
    ConnectFourEnv.run_webserver()
