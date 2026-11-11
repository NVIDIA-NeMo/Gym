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
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import HTTPException
from omegaconf import DictConfig

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from resources_servers.scrabble import game
from resources_servers.scrabble.app import (
    DEFAULT_DICTIONARY_PATH,
    ScrabbleResourcesServer,
    ScrabbleResourcesServerConfig,
    load_dictionary,
)
from resources_servers.scrabble.generate_data import make_row
from resources_servers.scrabble.task_data import TaskData


WORDS = ["AT", "CA", "CAT", "CATS", "QI", "TA", "AS", "SAT"]
CAT_BOARD = game.apply_move(game.empty_board(), [(7, 7, "C"), (7, 8, "A"), (7, 9, "T")])
EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "example.jsonl"


@pytest.fixture
def dictionary_path(tmp_path) -> str:
    path = tmp_path / "words.txt"
    path.write_text("# test words\n" + "\n".join(WORDS) + "\n")
    return str(path)


def _env(dictionary_path: str, *, checkpointing: bool = False, penalty: float = 0.1) -> ScrabbleResourcesServer:
    server_client = MagicMock(spec=ServerClient)
    if checkpointing:
        server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = ScrabbleResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="scrabble",
        dictionary_path=dictionary_path,
        invalid_attempt_penalty=penalty,
    )
    return ScrabbleResourcesServer(config=config, server_client=server_client)


def _row(**overrides) -> dict:
    # With WORDS, the best move for AST is SAT on G9-I9 for 12: S and T sit on DLs and form CA and AT.
    row = {"position_id": "test-0", "board": CAT_BOARD, "rack": "AST", "best_score": 12, "max_attempts": 3}
    return row | overrides


def _call(word: str, row: int, col, direction: str = "across", name: str = "play_move", call_id: str = "c1") -> dict:
    arguments = json.dumps({"word": word, "row": row, "col": col, "direction": direction})
    return {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name, "arguments": arguments}


def _response(*items: dict) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "resp",
            "object": "response",
            "created_at": 0.0,
            "model": "m",
            "output": list(items),
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def _text(text: str) -> dict:
    return {
        "type": "message",
        "id": "msg",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class TestDictionary:
    def test_missing_word_list_fails_at_startup(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="generate_data.py"):
            _env(str(tmp_path / "absent.txt"))

    def test_comments_are_skipped_and_words_uppercased(self, tmp_path):
        path = tmp_path / "w.txt"
        path.write_text("# header\nqi\n\nZA\n")
        assert load_dictionary(str(path)) == frozenset({"QI", "ZA"})


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_stores_the_pinned_position(self, dictionary_path):
        env = _env(dictionary_path)
        obs, info = await env.reset(_row(), session_id="s")
        assert (obs, info) == (None, {})
        assert env.session_state["s"].board == CAT_BOARD
        assert env.session_state["s"].rack == "AST"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "overrides",
        [{"rack": "AST1"}, {"board": CAT_BOARD[:14]}, {"best_score": 0}, {"board": ["#" * 15] * 15}],
    )
    async def test_reset_rejects_malformed_rows(self, dictionary_path, overrides):
        with pytest.raises(HTTPException) as error:
            await _env(dictionary_path).reset(_row(**overrides), session_id="s")
        assert error.value.status_code == 400


class TestStep:
    @pytest.mark.asyncio
    async def test_best_move_earns_full_reward(self, dictionary_path):
        env = _env(dictionary_path)
        await env.reset(_row(), session_id="s")
        _, reward, terminated, truncated, info = await env.step(_response(_call("SAT", 9, "G")), {}, session_id="s")
        assert (reward, terminated, truncated) == (1.0, True, False)
        assert info["outcome"] == "legal_move"
        assert (info["score"], info["optimal"], info["illegal_moves"]) == (12.0, 1.0, 0.0)

    @pytest.mark.asyncio
    async def test_weaker_move_earns_score_ratio(self, dictionary_path):
        env = _env(dictionary_path)
        await env.reset(_row(), session_id="s")
        _, reward, terminated, _, info = await env.step(_response(_call("CATS", 8, "H")), {}, session_id="s")
        assert terminated and reward == pytest.approx(6 / 12)
        assert info["optimal"] == 0.0

    @pytest.mark.asyncio
    async def test_illegal_move_returns_reason_and_costs_reward(self, dictionary_path):
        env = _env(dictionary_path, penalty=0.25)
        await env.reset(_row(), session_id="s")
        _, reward, terminated, _, info = await env.step(_response(_call("TAC", 8, "H")), {}, session_id="s")
        assert (reward, terminated) == (0.0, False)
        (output,) = info["tool_outputs"]
        assert output["call_id"] == "c1"
        feedback = json.loads(output["output"])
        assert feedback["legal"] is False and feedback["attempts_remaining"] == 2
        assert "already holds C" in feedback["error"]

        _, reward, terminated, _, info = await env.step(_response(_call("SAT", 9, "G")), {}, session_id="s")
        assert terminated and reward == pytest.approx(1.0 - 0.25)
        assert info["illegal_moves"] == 1.0

    @pytest.mark.asyncio
    async def test_running_out_of_attempts_ends_with_zero(self, dictionary_path):
        env = _env(dictionary_path)
        await env.reset(_row(max_attempts=2), session_id="s")
        _, _, terminated, _, _ = await env.step(_response(_call("ZZ", 8, "A")), {}, session_id="s")
        assert not terminated
        _, reward, terminated, _, info = await env.step(_response(_call("ZZ", 8, "A")), {}, session_id="s")
        assert (reward, terminated, info["outcome"]) == (0.0, True, "out_of_attempts")

    @pytest.mark.asyncio
    async def test_reply_without_a_tool_call_ends_with_zero(self, dictionary_path):
        env = _env(dictionary_path)
        await env.reset(_row(), session_id="s")
        _, reward, terminated, _, info = await env.step(_response(_text("I play CATS")), {}, session_id="s")
        assert (reward, terminated, info["outcome"]) == (0.0, True, "no_move")

    @pytest.mark.asyncio
    async def test_unknown_tool_uses_an_attempt(self, dictionary_path):
        env = _env(dictionary_path)
        await env.reset(_row(), session_id="s")
        _, _, terminated, _, info = await env.step(_response(_call("AT", 9, "H", name="pass")), {}, session_id="s")
        assert not terminated
        assert "Unknown tool" in json.loads(info["tool_outputs"][0]["output"])["error"]
        assert env.session_state["s"].illegal_moves == 1

    @pytest.mark.asyncio
    async def test_calls_after_a_legal_move_are_refused(self, dictionary_path):
        env = _env(dictionary_path)
        await env.reset(_row(), session_id="s")
        calls = _response(_call("CATS", 8, "H"), _call("AT", 9, "H", call_id="c2"))
        _, reward, terminated, _, _ = await env.step(calls, {}, session_id="s")
        # The first legal move counts; the second is not played.
        assert terminated and reward == pytest.approx(6 / 12)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("col", [7, "7", "g"])
    async def test_column_number_or_lowercase_letter_is_accepted(self, dictionary_path, col):
        env = _env(dictionary_path)
        await env.reset(_row(), session_id="s")
        _, reward, _, _, _ = await env.step(_response(_call("SAT", 9, col)), {}, session_id="s")
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_step_without_reset_is_404(self, dictionary_path):
        with pytest.raises(HTTPException) as error:
            await _env(dictionary_path).step(_response(_call("SAT", 9, "G")), {}, session_id="nope")
        assert error.value.status_code == 404


class TestGeneratedRows:
    @pytest.mark.asyncio
    async def test_generated_row_round_trips_through_the_server(self, dictionary_path):
        words = frozenset(WORDS)
        best = max(game.generate_moves(CAT_BOARD, "AST", game.build_trie(words), words), key=lambda m: m.score)
        row = make_row("test-1", CAT_BOARD, "AST", best)
        TaskData.model_validate(row)
        assert row["best_score"] == 12

        env = _env(dictionary_path)
        await env.reset(row, session_id="s")
        move = row["best_move"]
        call = _call(move["word"], move["row"], move["col"], move["direction"])
        _, reward, _, _, _ = await env.step(_response(call), {}, session_id="s")
        assert reward == 1.0

    @pytest.mark.skipif(not DEFAULT_DICTIONARY_PATH.exists(), reason="run generate_data.py for the ENABLE word list")
    def test_example_rows_best_moves_score_best_score(self):
        dictionary = load_dictionary(str(DEFAULT_DICTIONARY_PATH))
        rows = [json.loads(line) for line in EXAMPLE_PATH.read_text().splitlines()]
        assert len(rows) == 5
        for row in rows:
            task = TaskData.model_validate(row)
            best = task.best_move
            result = game.evaluate_move(
                task.board,
                task.rack,
                best.word,
                best.row - 1,
                game.COLUMNS.index(best.col),
                best.direction,
                dictionary,
            )
            assert result.valid and result.score == task.best_score, task.position_id


AUTH = {"authorization": "Bearer t"}


def _control(checkpoint_id: str = "c1", **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


def _wire(row: dict, *items: dict | None) -> dict:
    body = {"responses_create_params": {"input": []}, **row}
    if items:
        body["response"] = _response(*items).model_dump(mode="json")
    return body


class TestCheckpointing:
    @pytest.mark.asyncio
    async def test_restored_puzzle_keeps_its_illegal_move_count(self, dictionary_path, tmp_path):
        env = _env(dictionary_path, checkpointing=True)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=env.setup_webserver()), base_url="http://r")
        async with client:
            await client.post("/ng-rollout/r-a1/reset", json=_wire(_row()))
            await client.post("/ng-rollout/r-a1/step", json=_wire(_row(), _call("ZZ", 8, "A")))
            await client.post("/ng-control/v1/checkpoint/prepare", json=_control(), headers=AUTH)
            commit = await client.post(
                "/ng-control/v1/checkpoint/commit", json=_control(checkpoint_dir=str(tmp_path)), headers=AUTH
            )
            assert commit.json()["episode_ids"] == ["r-a1"]
            cookies = dict(client.cookies)

        restored = _env(dictionary_path, checkpointing=True)
        fresh = httpx.AsyncClient(transport=httpx.ASGITransport(app=restored.setup_webserver()), base_url="http://r")
        async with fresh:
            fresh.cookies.update(cookies)
            scope = [{"rollout_id": "r"}, {"rollout_id": "r", "attempt": 1}]
            await fresh.post(
                "/ng-control/v1/checkpoint/restore",
                json=_control("r1", checkpoint_dir=str(tmp_path), episode_ids=scope),
                headers=AUTH,
            )
            await fresh.post("/ng-control/v1/checkpoint/resume", json=_control("r1"), headers=AUTH)
            final = (await fresh.post("/ng-rollout/r-a2/step", json=_wire(_row(), _call("SAT", 9, "G")))).json()

        assert final["terminated"] is True
        assert final["info"]["illegal_moves"] == 1.0
        assert final["reward"] == pytest.approx(0.9)
        assert restored.session_state == {}

    @pytest.mark.parametrize(
        "state",
        [
            {"board": CAT_BOARD, "rack": "AST"},
            {"board": CAT_BOARD[:3], "rack": "AST", "best_score": 10, "max_attempts": 3},
            {"board": CAT_BOARD, "rack": "A!", "best_score": 10, "max_attempts": 3},
        ],
    )
    def test_restore_rejects_malformed_state(self, dictionary_path, state):
        with pytest.raises(ValueError, match="invalid scrabble session state"):
            _env(dictionary_path).restore_env_state(state)
