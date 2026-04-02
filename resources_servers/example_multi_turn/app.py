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

"""Example multi-turn resources server: tic-tac-toe.

The policy model plays as X and the user model plays as O, both using the
make_move tool. The server tracks whose turn it is and checks for a winner
or draw after each move.

By default X goes first, but the JSONL data can specify initial_moves in
verifier_metadata to pre-populate the board (e.g. O opens with a move).
"""

from typing import Any, Dict, List, Optional

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


WINNING_LINES = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
]


class GameState(BaseModel):
    board: List[str] = Field(default_factory=lambda: [" "] * 9)
    game_over: bool = False
    winner: Optional[str] = None
    # X goes first by default; seed_session can override via initial_moves.
    next_mark: str = "X"


def check_winner(board: List[str]) -> Optional[str]:
    for line in WINNING_LINES:
        if board[line[0]] == board[line[1]] == board[line[2]] != " ":
            return board[line[0]]
    return None


def is_draw(board: List[str]) -> bool:
    return " " not in board and check_winner(board) is None


def format_board(board: List[str]) -> str:
    rows = []
    for i in range(3):
        row = " | ".join(board[i * 3 : (i + 1) * 3])
        rows.append(row)
    return "\n-----------\n".join(rows)


class ExampleMultiTurnConfig(BaseResourcesServerConfig):
    pass


class ExampleMultiTurnSeedSessionRequest(BaseSeedSessionRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class MakeMoveRequest(BaseModel):
    position: int


class MakeMoveResponse(BaseModel):
    success: bool
    board: str
    game_over: bool
    winner: Optional[str]
    message: str


class ExampleMultiTurnVerifyRequest(BaseVerifyRequest):
    pass


class ExampleMultiTurnVerifyResponse(BaseVerifyResponse):
    game_result: Optional[str] = None


class ExampleMultiTurnServer(SimpleResourcesServer):
    config: ExampleMultiTurnConfig
    session_id_to_game: Dict[str, GameState] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/make_move")(self.make_move)
        return app

    async def seed_session(self, request: Request, body: ExampleMultiTurnSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        game = GameState()

        initial_moves = (body.verifier_metadata or {}).get("initial_moves", [])
        for move in initial_moves:
            game.board[move["position"]] = move["mark"]
        if initial_moves:
            last_mark = initial_moves[-1]["mark"]
            game.next_mark = "O" if last_mark == "X" else "X"

        self.session_id_to_game[session_id] = game
        return BaseSeedSessionResponse()

    async def make_move(self, request: Request, body: MakeMoveRequest) -> MakeMoveResponse:
        """Place the current player's mark (X or O) at the given position.

        Both the policy model and user model call this same endpoint — the
        server determines which mark to place based on turn order.
        """
        session_id = request.session[SESSION_ID_KEY]
        game = self.session_id_to_game.get(session_id)
        if game is None:
            return MakeMoveResponse(
                success=False,
                board="",
                game_over=False,
                winner=None,
                message="No active game. Call seed_session first.",
            )

        if game.game_over:
            return MakeMoveResponse(
                success=False,
                board=format_board(game.board),
                game_over=True,
                winner=game.winner,
                message=f"Game is already over. Winner: {game.winner or 'Draw'}",
            )

        if body.position < 0 or body.position > 8:
            return MakeMoveResponse(
                success=False,
                board=format_board(game.board),
                game_over=False,
                winner=None,
                message=f"Invalid position {body.position}. Must be 0-8.",
            )

        if game.board[body.position] != " ":
            return MakeMoveResponse(
                success=False,
                board=format_board(game.board),
                game_over=False,
                winner=None,
                message=f"Position {body.position} is already occupied by {game.board[body.position]}.",
            )

        mark = game.next_mark
        game.board[body.position] = mark
        message = f"{mark} placed at position {body.position}."

        winner = check_winner(game.board)
        if winner:
            game.game_over = True
            game.winner = winner
            message += f" {winner} wins!"
        elif is_draw(game.board):
            game.game_over = True
            message += " Game is a draw."
        else:
            # Alternate turns
            game.next_mark = "O" if mark == "X" else "X"
            message += f" {game.next_mark}'s turn."

        return MakeMoveResponse(
            success=True,
            board=format_board(game.board),
            game_over=game.game_over,
            winner=game.winner,
            message=message,
        )

    async def verify(self, request: Request, body: ExampleMultiTurnVerifyRequest) -> ExampleMultiTurnVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        game = self.session_id_to_game.get(session_id)

        if game is None:
            return ExampleMultiTurnVerifyResponse(**body.model_dump(), reward=0.0, game_result="no_game")

        if game.winner == "X":
            reward = 1.0
            game_result = "win"
        elif game.winner == "O":
            reward = 0.0
            game_result = "loss"
        elif game.game_over:
            reward = 0.5
            game_result = "draw"
        else:
            reward = 0.0
            game_result = "incomplete"

        return ExampleMultiTurnVerifyResponse(**body.model_dump(), reward=reward, game_result=game_result)


if __name__ == "__main__":
    ExampleMultiTurnServer.run_webserver()
