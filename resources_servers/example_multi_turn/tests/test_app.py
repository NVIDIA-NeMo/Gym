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

"""Tests for the example multi-turn resources server (tic-tac-toe).

This server provides a simple two-player game environment used as a
reference example for multi-turn agent development. Tests are organized as:

- TestGameLogic: Pure-function tests for board evaluation (winner detection,
  draw detection, board formatting), no server involved.
- TestApp: Basic sanity check that the server can be instantiated.
- TestEndpoints: Integration tests exercising the full HTTP API flow
  (seed_session → make_move → verify) using FastAPI's TestClient.

Board positions are numbered 0-8:
  0 | 1 | 2
  ---------
  3 | 4 | 5
  ---------
  6 | 7 | 8

The server tracks turns automatically: X always goes first (unless initial_moves
are provided in verifier_metadata to pre-populate the board).
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.example_multi_turn.app import (
    ExampleMultiTurnConfig,
    ExampleMultiTurnServer,
    check_winner,
    format_board,
    is_draw,
)


# ---------------------------------------------------------------------------
# Pure game logic tests (no server required)
# ---------------------------------------------------------------------------


class TestGameLogic:
    def test_check_winner_x_row(self) -> None:
        """X wins by filling the top row (positions 0, 1, 2)."""
        board = ["X", "X", "X", " ", " ", " ", " ", " ", " "]
        assert check_winner(board) == "X"

    def test_check_winner_o_column(self) -> None:
        """O wins by filling the left column (positions 0, 3, 6)."""
        board = ["O", " ", " ", "O", " ", " ", "O", " ", " "]
        assert check_winner(board) == "O"

    def test_check_winner_diagonal(self) -> None:
        """X wins by filling the main diagonal (positions 0, 4, 8)."""
        board = ["X", " ", " ", " ", "X", " ", " ", " ", "X"]
        assert check_winner(board) == "X"

    def test_check_winner_anti_diagonal(self) -> None:
        """O wins by filling the anti-diagonal (positions 2, 4, 6)."""
        board = [" ", " ", "O", " ", "O", " ", "O", " ", " "]
        assert check_winner(board) == "O"

    def test_check_winner_none(self) -> None:
        """No winner yet — game still in progress."""
        board = ["X", "O", " ", " ", "X", " ", " ", " ", "O"]
        assert check_winner(board) is None

    def test_is_draw(self) -> None:
        """All squares filled with no winner — draw."""
        board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        assert is_draw(board) is True

    def test_not_draw(self) -> None:
        """Empty squares remain — not a draw."""
        board = ["X", "O", "X", " ", "O", "O", "O", "X", "X"]
        assert is_draw(board) is False

    def test_format_board(self) -> None:
        """Board renders as a human-readable 3x3 grid with pipe separators."""
        board = ["X", "O", " ", " ", "X", " ", " ", " ", "O"]
        result = format_board(board)
        assert "X | O |  " in result
        assert "  | X |  " in result
        assert "  |   | O" in result


# ---------------------------------------------------------------------------
# Server instantiation sanity check
# ---------------------------------------------------------------------------


class TestApp:
    def test_sanity(self) -> None:
        """Verify the server can be constructed without errors."""
        config = ExampleMultiTurnConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        ExampleMultiTurnServer(config=config, server_client=MagicMock(spec=ServerClient))


# ---------------------------------------------------------------------------
# HTTP endpoint integration tests
#
# Each test creates a fresh server + TestClient, seeds a new game session,
# and exercises the endpoints. TestClient handles session cookies automatically.
# ---------------------------------------------------------------------------


class TestEndpoints:
    def _make_server(self) -> tuple:
        """Create a fresh ExampleMultiTurnServer and a FastAPI TestClient."""
        config = ExampleMultiTurnConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = ExampleMultiTurnServer(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)
        return server, client

    def test_seed_and_make_move(self) -> None:
        """Seed a game, make one move — X goes first by default."""
        _, client = self._make_server()
        client.post("/seed_session", json={"responses_create_params": {"input": []}})

        resp = client.post("/make_move", json={"position": 4})
        data = resp.json()
        assert data["success"] is True
        assert data["game_over"] is False
        assert "X placed at position 4" in data["message"]

    def test_alternating_turns(self) -> None:
        """Moves alternate between X and O automatically."""
        _, client = self._make_server()
        client.post("/seed_session", json={"responses_create_params": {"input": []}})

        r1 = client.post("/make_move", json={"position": 0}).json()
        assert "X placed" in r1["message"]

        r2 = client.post("/make_move", json={"position": 4}).json()
        assert "O placed" in r2["message"]

        r3 = client.post("/make_move", json={"position": 8}).json()
        assert "X placed" in r3["message"]

    def test_x_wins(self) -> None:
        """X wins by completing the top row: X=0, O=3, X=1, O=4, X=2."""
        _, client = self._make_server()
        client.post("/seed_session", json={"responses_create_params": {"input": []}})

        client.post("/make_move", json={"position": 0})
        client.post("/make_move", json={"position": 3})
        client.post("/make_move", json={"position": 1})
        client.post("/make_move", json={"position": 4})
        result = client.post("/make_move", json={"position": 2}).json()

        assert result["game_over"] is True
        assert result["winner"] == "X"
        assert "X wins" in result["message"]

    def test_draw(self) -> None:
        """All 9 squares filled with no winner — game ends in a draw.

        Move sequence: X=0, O=1, X=2, O=4, X=3, O=6, X=7, O=8, X=5
        """
        _, client = self._make_server()
        client.post("/seed_session", json={"responses_create_params": {"input": []}})

        for pos in [0, 1, 2, 4, 3, 6, 7, 8, 5]:
            resp = client.post("/make_move", json={"position": pos})

        data = resp.json()
        assert data["game_over"] is True
        assert data["winner"] is None
        assert "draw" in data["message"].lower()

    def test_invalid_position(self) -> None:
        """Position outside 0–8 is rejected with an error message."""
        _, client = self._make_server()
        client.post("/seed_session", json={"responses_create_params": {"input": []}})

        resp = client.post("/make_move", json={"position": 9}).json()
        assert resp["success"] is False
        assert "Invalid position" in resp["message"]

    def test_occupied_position(self) -> None:
        """Placing a mark on an already-occupied square is rejected."""
        _, client = self._make_server()
        client.post("/seed_session", json={"responses_create_params": {"input": []}})

        client.post("/make_move", json={"position": 0})
        resp = client.post("/make_move", json={"position": 0}).json()
        assert resp["success"] is False
        assert "already occupied" in resp["message"]

    def test_move_after_game_over(self) -> None:
        """No moves are allowed after the game has already ended."""
        _, client = self._make_server()
        client.post("/seed_session", json={"responses_create_params": {"input": []}})

        # X wins with top row
        for pos in [0, 3, 1, 4, 2]:
            client.post("/make_move", json={"position": pos})

        resp = client.post("/make_move", json={"position": 8}).json()
        assert resp["success"] is False
        assert "already over" in resp["message"]

    def test_seed_with_initial_moves(self) -> None:
        """Seeding with initial_moves pre-populates the board.

        This allows scenarios where O goes first (e.g. O takes center),
        and the policy model (X) responds to an existing board state.
        After O's initial move, the next make_move should place X.
        """
        _, client = self._make_server()
        client.post(
            "/seed_session",
            json={
                "responses_create_params": {"input": []},
                "verifier_metadata": {"initial_moves": [{"position": 4, "mark": "O"}]},
            },
        )

        resp = client.post("/make_move", json={"position": 0}).json()
        assert "X placed" in resp["message"]
        assert "O" in resp["board"]

    # -----------------------------------------------------------------------
    # /verify endpoint tests
    #
    # The verify endpoint computes a reward based on the final game state:
    #   X wins  → reward 1.0 (game_result="win")
    #   O wins  → reward 0.0 (game_result="loss")
    #   Draw    → reward 0.5 (game_result="draw")
    #   Neither → reward 0.0 (game_result="incomplete")
    # -----------------------------------------------------------------------

    def _verify_body(self) -> dict:
        """Build a minimal BaseVerifyRequest body for the /verify endpoint."""
        return {
            "responses_create_params": {"input": []},
            "response": {
                "id": "resp_test",
                "created_at": 0.0,
                "model": "test",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }

    def test_verify_x_wins(self) -> None:
        """X wins → reward 1.0."""
        _, client = self._make_server()
        client.post("/seed_session", json={"responses_create_params": {"input": []}})

        for pos in [0, 3, 1, 4, 2]:
            client.post("/make_move", json={"position": pos})

        resp = client.post("/verify", json=self._verify_body()).json()
        assert resp["reward"] == 1.0
        assert resp["game_result"] == "win"

    def test_verify_incomplete(self) -> None:
        """Game not finished → reward 0.0, result "incomplete"."""
        _, client = self._make_server()
        client.post("/seed_session", json={"responses_create_params": {"input": []}})

        client.post("/make_move", json={"position": 0})

        resp = client.post("/verify", json=self._verify_body()).json()
        assert resp["reward"] == 0.0
        assert resp["game_result"] == "incomplete"

    def test_verify_o_wins(self) -> None:
        """O wins → reward 0.0 (policy lost)."""
        _, client = self._make_server()
        client.post("/seed_session", json={"responses_create_params": {"input": []}})

        # X=0, O=3, X=1, O=4, X=8, O=5 (O completes middle row)
        for pos in [0, 3, 1, 4, 8, 5]:
            client.post("/make_move", json={"position": pos})

        resp = client.post("/verify", json=self._verify_body()).json()
        assert resp["reward"] == 0.0
        assert resp["game_result"] == "loss"

    def test_verify_draw(self) -> None:
        """Draw → reward 0.5."""
        _, client = self._make_server()
        client.post("/seed_session", json={"responses_create_params": {"input": []}})

        for pos in [0, 1, 2, 4, 3, 6, 7, 8, 5]:
            client.post("/make_move", json={"position": pos})

        resp = client.post("/verify", json=self._verify_body()).json()
        assert resp["reward"] == 0.5
        assert resp["game_result"] == "draw"
