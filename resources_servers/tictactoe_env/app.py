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

"""TicTacToe environment.

Model plays as X. Opponent (O) is an LLM called via user_model_server.
Falls back to rule-based play if the opponent model returns an invalid move.

Reward: +1 win, 0 draw, -1 loss.
"""

import re
from typing import Dict, Optional

from pydantic import Field

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.example_gymnasium import GymnasiumServer, extract_text


_OPPONENT_SYSTEM_PROMPT = (
    "You are playing TicTacToe as O. Look at the board and pick an empty cell. "
    "Respond with <action>R C</action> where R is row and C is column (1-3)."
)


def _empty_board() -> list:
    return [[" "] * 3 for _ in range(3)]


def _fmt_board(board: list) -> str:
    divider = "\n---+---+---\n"
    return divider.join(" " + " | ".join(row) + " " for row in board)


def _winner(board: list) -> Optional[str]:
    lines = (
        [board[r] for r in range(3)]
        + [[board[r][c] for r in range(3)] for c in range(3)]
        + [[board[i][i] for i in range(3)], [board[i][2 - i] for i in range(3)]]
    )
    for line in lines:
        if line[0] != " " and all(c == line[0] for c in line):
            return line[0]
    return None


def _is_full(board: list) -> bool:
    return all(board[r][c] != " " for r in range(3) for c in range(3))


def _first_empty(board: list) -> Optional[tuple]:
    for r in range(3):
        for c in range(3):
            if board[r][c] == " ":
                return r, c
    return None


def _parse_action(text: str) -> Optional[tuple[int, int]]:
    m = re.search(r"<action>\s*(\d)\s*(\d)\s*</action>", text)
    if not m:
        m = re.search(r"row\s*([1-3])\s*col\s*([1-3])", text)
    if not m:
        return None
    r, c = int(m.group(1)) - 1, int(m.group(2)) - 1
    if 0 <= r <= 2 and 0 <= c <= 2:
        return r, c
    return None


class TicTacToeEnv(GymnasiumServer):
    user_model_server: str = "tictactoe_user_model"
    session_state: Dict[str, dict] = Field(default_factory=dict)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        board = _empty_board()
        self.session_state[session_id] = {"board": board}
        obs = f"New game. You are X.\n\n{_fmt_board(board)}\n\nRespond with <action>R C</action> where R is row and C is column (1-3)."
        return obs, {}

    async def _get_opponent_move(self, board: list) -> Optional[tuple[int, int]]:
        """Call user_model_server for the opponent move. Falls back to first empty cell."""
        board_str = _fmt_board(board)
        try:
            resp = await self.server_client.post(
                server_name=self.user_model_server,
                url_path="/v1/responses",
                json={
                    "input": [
                        {"role": "system", "content": _OPPONENT_SYSTEM_PROMPT},
                        {"role": "user", "content": f"{board_str}\n\nYour move as O:"},
                    ],
                },
            )
            await raise_for_status(resp)
            data = await get_response_json(resp)
            text = ""
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            text = c.get("text", "")
                            break
            move = _parse_action(text)
            if move and board[move[0]][move[1]] == " ":
                return move
        except Exception:
            pass
        return _first_empty(board)

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        state = self.session_state.get(session_id, {})
        board = state.get("board", _empty_board())
        text = extract_text(action)

        move = _parse_action(text)
        if not move:
            return "Invalid format. Use <action>R C</action> where R and C are 1-3.", 0.0, False, False, {}
        r, c = move
        if board[r][c] != " ":
            return "That cell is taken. Pick another with <action>R C</action>.", 0.0, False, False, {}

        board[r][c] = "X"

        if _winner(board) == "X":
            return None, 1.0, True, False, {"result": "win", "board": _fmt_board(board)}
        if _is_full(board):
            return None, 0.0, True, False, {"result": "draw", "board": _fmt_board(board)}

        opp = await self._get_opponent_move(board)
        if opp:
            board[opp[0]][opp[1]] = "O"

        if _winner(board) == "O":
            return None, -1.0, True, False, {"result": "loss", "board": _fmt_board(board)}
        if _is_full(board):
            return None, 0.0, True, False, {"result": "draw", "board": _fmt_board(board)}

        obs = f"{_fmt_board(board)}\n\nRespond with <action>R C</action>."
        return obs, 0.0, False, False, {}


if __name__ == "__main__":
    TicTacToeEnv.run_webserver()
