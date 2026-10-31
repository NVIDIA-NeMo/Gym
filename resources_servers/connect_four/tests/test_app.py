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
import random
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText
from nemo_gym.server_utils import ServerClient
from resources_servers.connect_four.app import (
    MODEL,
    OPPONENT,
    ConnectFourEnv,
    _connects_four,
    _drop,
    _new_board,
    _opponent_move,
    _parse_action,
)


def _make_env():
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
    return ConnectFourEnv(config=config, server_client=MagicMock(spec=ServerClient))


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _drawn_board(open_top_cells: int):
    """A board with no four-in-a-row and only the last `open_top_cells` of the top row empty."""
    pattern = ["XXOOXXO", "OOXXOOX", "XXOOXXO", "OOXXOOX", "XXOOXXO", "OOXXOOX"]
    board = [list(row) for row in reversed(pattern)]
    for col in range(7 - open_top_cells, 7):
        board[0][col] = "."
    return board


def _board(*placements: tuple[int, str]):
    """Build a board by dropping (0-based column, piece) pairs in order."""
    board = _new_board()
    for col, piece in placements:
        _drop(board, col, piece)
    return board


class TestBoard:
    def test_disc_falls_to_bottom(self):
        board = _new_board()
        assert _drop(board, 3, MODEL) == 5
        assert _drop(board, 3, OPPONENT) == 4

    def test_full_column_raises(self):
        board = _board(*[(0, MODEL)] * 6)
        with pytest.raises(ValueError):
            _drop(board, 0, MODEL)

    def test_horizontal_win(self):
        board = _board((0, MODEL), (1, MODEL), (2, MODEL), (3, MODEL))
        assert _connects_four(board, 5, 3)

    def test_vertical_win(self):
        board = _board(*[(4, MODEL)] * 4)
        assert _connects_four(board, 2, 4)

    def test_diagonal_win(self):
        # X on (5,0), (4,1), (3,2), (2,3), with O filler underneath.
        board = _board(
            (0, MODEL),
            (1, OPPONENT), (1, MODEL),
            (2, OPPONENT), (2, OPPONENT), (2, MODEL),
            (3, OPPONENT), (3, OPPONENT), (3, OPPONENT), (3, MODEL),
        )  # fmt: skip
        assert _connects_four(board, 2, 3)

    def test_anti_diagonal_win(self):
        board = _board(
            (6, MODEL),
            (5, OPPONENT), (5, MODEL),
            (4, OPPONENT), (4, OPPONENT), (4, MODEL),
            (3, OPPONENT), (3, OPPONENT), (3, OPPONENT), (3, MODEL),
        )  # fmt: skip
        assert _connects_four(board, 2, 3)

    def test_three_is_not_a_win(self):
        board = _board((0, MODEL), (1, MODEL), (2, MODEL))
        assert not _connects_four(board, 5, 2)


class TestParseAction:
    LEGAL = list(range(7))

    def test_valid_column(self):
        assert _parse_action("I'll go center. <action>4</action>", self.LEGAL) == (3, None)

    def test_whitespace_and_case(self):
        assert _parse_action("<ACTION>  7 </ACTION>", self.LEGAL) == (6, None)

    def test_repeated_same_column_is_fine(self):
        assert _parse_action("<action>2</action> so <action>2</action>", self.LEGAL) == (1, None)

    def test_no_tag(self):
        assert _parse_action("column 4", self.LEGAL) == (None, "no_action_tag")

    def test_conflicting_tags_rejected(self):
        assert _parse_action("<action>3</action> or <action>4</action>", self.LEGAL) == (None, "multiple_actions")

    def test_out_of_range(self):
        assert _parse_action("<action>8</action>", self.LEGAL) == (None, "not_a_column")
        assert _parse_action("<action>0</action>", self.LEGAL) == (None, "not_a_column")
        assert _parse_action("<action>center</action>", self.LEGAL) == (None, "not_a_column")

    def test_full_column(self):
        assert _parse_action("<action>1</action>", [1, 2, 3]) == (None, "column_full")

    def test_huge_digit_string_is_not_a_column(self):
        assert _parse_action(f"<action>{'9' * 5000}</action>", self.LEGAL) == (None, "not_a_column")

    def test_non_ascii_digit_is_not_a_column(self):
        assert _parse_action("<action>\u0664</action>", self.LEGAL) == (None, "not_a_column")


class TestOpponent:
    def test_takes_immediate_win(self):
        board = _board((0, OPPONENT), (1, OPPONENT), (2, OPPONENT), (6, MODEL), (6, MODEL))
        assert _opponent_move(board, 1, random.Random(0)) == 3

    def test_blocks_immediate_threat(self):
        board = _board((0, MODEL), (1, MODEL), (2, MODEL), (6, OPPONENT))
        assert _opponent_move(board, 2, random.Random(0)) == 3

    def test_random_opponent_plays_legal_move(self):
        board = _board(*[(c, MODEL if i % 2 else OPPONENT) for c in range(6) for i in range(6)])
        assert _opponent_move(board, 0, random.Random(0)) == 6

    def test_search_leaves_board_unchanged(self):
        board = _board((3, MODEL), (3, OPPONENT))
        snapshot = [row[:] for row in board]
        _opponent_move(board, 4, random.Random(0))
        assert board == snapshot

    def test_search_handles_full_board_after_last_move(self):
        # One cell open; the search must look past the filling move and score the draw.
        board = _drawn_board(open_top_cells=1)
        snapshot = [row[:] for row in board]
        assert _opponent_move(board, 2, random.Random(0)) == 6
        assert board == snapshot


class TestReset:
    @pytest.mark.asyncio
    async def test_model_first_starts_empty(self):
        env = _make_env()
        obs, _ = await env.reset({"seed": 0}, session_id="sid")
        assert env.session_state["sid"]["moves"] == []
        assert "Legal columns: 1, 2, 3, 4, 5, 6, 7." in obs
        assert "<action>N</action>" in obs

    @pytest.mark.asyncio
    async def test_opponent_first_moves_once(self):
        env = _make_env()
        obs, _ = await env.reset({"seed": 0, "model_first": False}, session_id="sid")
        assert len(env.session_state["sid"]["moves"]) == 1
        assert "Opponent dropped a disc in column" in obs

    @pytest.mark.asyncio
    async def test_same_seed_same_game(self):
        env = _make_env()
        await env.reset({"seed": 7, "opponent_depth": 0, "model_first": False}, session_id="a")
        await env.reset({"seed": 7, "opponent_depth": 0, "model_first": False}, session_id="b")
        assert env.session_state["a"]["moves"] == env.session_state["b"]["moves"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("depth", [-1, 7, "2", 2.0, True])
    async def test_rejects_bad_opponent_depth(self, depth):
        env = _make_env()
        with pytest.raises(HTTPException) as exc:
            await env.reset({"opponent_depth": depth}, session_id="sid")
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_non_bool_model_first(self):
        env = _make_env()
        with pytest.raises(HTTPException) as exc:
            await env.reset({"model_first": "yes"}, session_id="sid")
        assert exc.value.status_code == 400


class TestStep:
    @pytest.mark.asyncio
    async def test_model_win(self):
        env = _make_env()
        await env.reset({"seed": 0}, session_id="sid")
        env.session_state["sid"]["board"] = _board((0, MODEL), (1, MODEL), (2, MODEL), (6, OPPONENT), (6, OPPONENT))
        obs, reward, term, trunc, info = await env.step(_response("<action>4</action>"), {}, session_id="sid")
        assert (obs, reward, term, trunc) == (None, 1.0, True, False)
        assert info["result"] == "win"

    @pytest.mark.asyncio
    async def test_model_loss(self):
        env = _make_env()
        await env.reset({"seed": 0, "opponent_depth": 1}, session_id="sid")
        env.session_state["sid"]["board"] = _board((0, OPPONENT), (1, OPPONENT), (2, OPPONENT), (6, MODEL), (6, MODEL))
        _, reward, term, _, info = await env.step(_response("<action>6</action>"), {}, session_id="sid")
        assert (reward, term) == (-1.0, True)
        assert info["result"] == "loss"
        assert info["moves"] == [6, 4]

    @pytest.mark.asyncio
    async def test_game_continues(self):
        env = _make_env()
        await env.reset({"seed": 0}, session_id="sid")
        obs, reward, term, _, info = await env.step(_response("<action>4</action>"), {}, session_id="sid")
        assert (reward, term, info) == (0.0, False, {})
        assert "Opponent dropped a disc in column" in obs
        assert len(env.session_state["sid"]["moves"]) == 2

    @pytest.mark.asyncio
    async def test_illegal_move_forfeits(self):
        env = _make_env()
        await env.reset({"seed": 0}, session_id="sid")
        obs, reward, term, _, info = await env.step(_response("I pick the middle"), {}, session_id="sid")
        assert (obs, reward, term) == (None, -1.0, True)
        assert info["result"] == "invalid_move"
        assert info["invalid_reason"] == "no_action_tag"

    @pytest.mark.asyncio
    async def test_draw_on_full_board(self):
        env = _make_env()
        await env.reset({"seed": 0}, session_id="sid")
        env.session_state["sid"]["board"] = _drawn_board(open_top_cells=1)
        _, reward, term, _, info = await env.step(_response("<action>7</action>"), {}, session_id="sid")
        assert (reward, term) == (0.0, True)
        assert info["result"] == "draw"

    @pytest.mark.asyncio
    async def test_draw_after_opponent_move(self):
        env = _make_env()
        await env.reset({"seed": 0}, session_id="sid")
        env.session_state["sid"]["board"] = _drawn_board(open_top_cells=2)
        _, reward, term, _, info = await env.step(_response("<action>6</action>"), {}, session_id="sid")
        assert (reward, term) == (0.0, True)
        assert info["result"] == "draw"
        assert info["moves"] == [6, 7]

    @pytest.mark.asyncio
    async def test_step_before_reset_is_400(self):
        env = _make_env()
        with pytest.raises(HTTPException) as exc:
            await env.step(_response("<action>4</action>"), {}, session_id="missing")
        assert exc.value.status_code == 400
