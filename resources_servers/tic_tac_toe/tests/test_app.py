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
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.tic_tac_toe.app import (
    TicTacToeResourcesServer,
    TicTacToeResourcesServerConfig,
    check_winner,
    format_board,
    is_draw,
)


class TestGameLogic:
    def test_check_winner_x_row(self) -> None:
        board = ["X", "X", "X", " ", " ", " ", " ", " ", " "]
        assert check_winner(board) == "X"

    def test_check_winner_o_column(self) -> None:
        board = ["O", " ", " ", "O", " ", " ", "O", " ", " "]
        assert check_winner(board) == "O"

    def test_check_winner_diagonal(self) -> None:
        board = ["X", " ", " ", " ", "X", " ", " ", " ", "X"]
        assert check_winner(board) == "X"

    def test_check_winner_none(self) -> None:
        board = ["X", "O", " ", " ", "X", " ", " ", " ", "O"]
        assert check_winner(board) is None

    def test_is_draw(self) -> None:
        board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        assert is_draw(board) is True

    def test_not_draw(self) -> None:
        board = ["X", "O", "X", " ", "O", "O", "O", "X", "X"]
        assert is_draw(board) is False

    def test_format_board(self) -> None:
        board = ["X", "O", " ", " ", "X", " ", " ", " ", "O"]
        result = format_board(board)
        assert "X | O |  " in result
        assert "  | X |  " in result
        assert "  |   | O" in result


class TestApp:
    def test_sanity(self) -> None:
        config = TicTacToeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        TicTacToeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
