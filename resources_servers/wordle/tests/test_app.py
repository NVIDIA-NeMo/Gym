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
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.wordle import app
from resources_servers.wordle.app import (
    PENALTY_IGNORE_GREEN,
    PENALTY_IGNORE_YELLOW,
    PENALTY_REPEATED_GUESS,
    PENALTY_USE_ELIMINATED,
    WordleGameLogic,
    WordleGameState,
    WordleResourcesServer,
    WordleResourcesServerConfig,
    calculate_win_reward,
    read_words,
)
from resources_servers.wordle.generate_data import make_row, split_targets


_WORDS = ["abide", "crane", "light", "react", "those"]
_CREATE_PARAMS = NeMoGymResponseCreateParamsNonStreaming(input="Make your first guess.").model_dump(mode="json")


def _state(target: str = "crane") -> WordleGameState:
    return WordleGameState(target_word=target, word_length=5, max_turns=6)


def _response(output: list[dict]) -> dict:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0.0,
        "status": "completed",
        "output": output,
        "model": "test",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _call(name: str, arguments: str, call_id: str = "c1") -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _guess(word: str) -> dict:
    return _call("submit_guess", json.dumps({"guess": word}))


def _text(text: str) -> dict:
    return {
        "id": "msg",
        "role": "assistant",
        "status": "completed",
        "type": "message",
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
    }


@pytest.fixture(autouse=True)
def _words(monkeypatch):
    monkeypatch.setattr(app, "valid_guesses", lambda: frozenset(_WORDS))


@pytest.fixture
def client() -> TestClient:
    config = WordleResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    server = WordleResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return TestClient(server.setup_webserver())


def _reset(client: TestClient, **metadata):
    return client.post("/reset", json={"responses_create_params": _CREATE_PARAMS, **metadata})


def _step(client: TestClient, cookies, *output: dict) -> dict:
    response = client.post(
        "/step", json={"responses_create_params": _CREATE_PARAMS, "response": _response(list(output))}, cookies=cookies
    )
    assert response.status_code == 200
    return response.json()


class TestGameLogic:
    def test_feedback_all_green(self):
        assert WordleGameLogic.get_feedback("crane", "crane") == ["G"] * 5

    def test_feedback_all_gray(self):
        assert WordleGameLogic.get_feedback("light", "crane") == ["_"] * 5

    def test_feedback_mixed(self):
        assert WordleGameLogic.get_feedback("react", "crane") == ["Y", "Y", "G", "Y", "_"]

    def test_feedback_duplicate_letter_counts_once(self):
        # Only one E in the target, so the second E in the guess is gray.
        assert WordleGameLogic.get_feedback("speed", "abide") == ["_", "_", "Y", "_", "Y"]

    def test_feedback_green_takes_priority(self):
        assert WordleGameLogic.get_feedback("geese", "those") == ["_", "_", "_", "G", "G"]

    def test_is_valid_word(self):
        assert WordleGameLogic.is_valid_word("crane") == (True, "Valid word")
        assert WordleGameLogic.is_valid_word("cran")[0] is False
        assert WordleGameLogic.is_valid_word("cr4ne")[0] is False
        assert WordleGameLogic.is_valid_word("zzzzz")[0] is False

    def test_update_knowledge_keeps_yellow_letter_out_of_eliminated(self):
        state = _state("abbey")
        guess = "babes"
        feedback = WordleGameLogic.get_feedback(guess, state.target_word)
        WordleGameLogic.update_knowledge(guess, feedback, state)
        assert state.known_greens == {2: "b", 3: "e"}
        assert state.known_yellows == {"b", "a"}
        assert state.eliminated_letters == {"s"}


class TestReward:
    def test_win_reward_by_turn(self):
        assert calculate_win_reward(1) == pytest.approx(1.6)
        assert calculate_win_reward(2) == pytest.approx(1.8)
        assert calculate_win_reward(3) == pytest.approx(1.6)
        assert calculate_win_reward(4) == pytest.approx(1.4)
        assert calculate_win_reward(6) == pytest.approx(1.0)

    def test_repeated_guess_penalty(self):
        state = _state()
        state.guesses.append("light")
        assert WordleGameLogic.calculate_turn_reward("light", state) == pytest.approx(PENALTY_REPEATED_GUESS)

    def test_ignore_green_and_yellow_penalties(self):
        state = _state()
        state.known_greens = {0: "c", 1: "r"}
        state.known_yellows = {"e"}
        reward = WordleGameLogic.calculate_turn_reward("light", state)
        assert reward == pytest.approx(2 * PENALTY_IGNORE_GREEN + PENALTY_IGNORE_YELLOW)

    def test_use_eliminated_penalty(self):
        state = _state()
        state.eliminated_letters = {"l", "t"}
        reward = WordleGameLogic.calculate_turn_reward("light", state)
        assert reward == pytest.approx(2 * PENALTY_USE_ELIMINATED)

    def test_good_guess_has_no_penalty(self):
        assert WordleGameLogic.calculate_turn_reward("crane", _state()) == 0.0


class TestWordLists:
    def test_read_words_skips_comments(self, tmp_path):
        path = tmp_path / "targets.txt"
        path.write_text("# Copyright notice\n#\ncrane\nlight\n")
        assert read_words(path) == ["crane", "light"]

    def test_missing_word_file_says_how_to_generate(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="generate_data.py"):
            read_words(tmp_path / "guesses.txt")

    def test_split_is_disjoint_and_complete(self):
        targets = [f"w{i:04d}" for i in range(100)]
        train, validation = split_targets(targets)
        assert len(validation) == 15
        assert sorted(train + validation) == targets

    def test_rows_pin_a_target(self):
        row = make_row("crane")
        assert row["custom_target"] == "crane"
        assert [t["name"] for t in row["responses_create_params"]["tools"]] == [
            "submit_guess",
            "check_word_validity",
            "get_game_state",
        ]


class TestServer:
    def test_reset_rejects_missing_target(self, client):
        assert _reset(client).status_code == 400

    def test_reset_rejects_wrong_length_target(self, client):
        assert _reset(client, custom_target="cranes").status_code == 400

    def test_step_before_reset(self, client):
        response = client.post(
            "/step", json={"responses_create_params": _CREATE_PARAMS, "response": _response([_guess("crane")])}
        )
        assert response.status_code == 400

    def test_first_turn_win(self, client):
        cookies = _reset(client, custom_target="CRANE").cookies
        payload = _step(client, cookies, _guess("crane"))
        assert payload["terminated"] is True
        assert payload["reward"] == pytest.approx(1.6)
        assert payload["info"]["game_outcome"] == "win"
        assert payload["info"]["turns_if_won"] == 1.0

    def test_feedback_then_win_with_penalty(self, client):
        cookies = _reset(client, custom_target="crane").cookies

        payload = _step(client, cookies, _guess("light"))
        assert payload["terminated"] is False
        assert payload["reward"] == 0.0
        [tool_output] = payload["info"]["tool_outputs"]
        assert tool_output["call_id"] == "c1"
        assert json.loads(tool_output["output"])["feedback"] == "_____"

        # REACT reuses eliminated T.
        _step(client, cookies, _guess("react"))
        # LIGHT again repeats, drops green A, skips yellows and reuses five eliminated letters.
        _step(client, cookies, _guess("light"))
        payload = _step(client, cookies, _guess("crane"))
        assert payload["terminated"] is True
        penalties = PENALTY_REPEATED_GUESS + PENALTY_IGNORE_GREEN + PENALTY_IGNORE_YELLOW + 6 * PENALTY_USE_ELIMINATED
        expected = calculate_win_reward(4) + penalties
        assert payload["reward"] == pytest.approx(expected)

    def test_loss_after_max_turns(self, client):
        cookies = _reset(client, custom_target="crane", max_turns=2).cookies
        assert _step(client, cookies, _guess("light"))["terminated"] is False
        payload = _step(client, cookies, _guess("zzzzz"))
        assert payload["terminated"] is True
        assert payload["reward"] == 0.0
        assert payload["info"]["game_outcome"] == "loss"

    def test_text_only_turn_ends_incomplete(self, client):
        cookies = _reset(client, custom_target="crane").cookies
        payload = _step(client, cookies, _text("I think the word is crane."))
        assert payload["terminated"] is True
        assert payload["reward"] == 0.0
        assert payload["info"]["game_outcome"] == "incomplete"

    def test_other_tools_do_not_use_a_turn(self, client):
        cookies = _reset(client, custom_target="crane").cookies
        _step(client, cookies, _guess("react"))
        payload = _step(
            client,
            cookies,
            _call("check_word_validity", json.dumps({"word": "qqqqq"}), "c1"),
            _call("get_game_state", "{}", "c2"),
        )
        validity, game_state = [json.loads(o["output"]) for o in payload["info"]["tool_outputs"]]
        assert validity["valid"] is False
        assert game_state["turn"] == 1
        assert game_state["known_greens"] == {"3": "A"}
        assert game_state["known_yellows"] == ["C", "E", "R"]
        assert game_state["eliminated_letters"] == ["T"]

    def test_bad_tool_calls_return_errors(self, client):
        cookies = _reset(client, custom_target="crane").cookies
        payload = _step(
            client,
            cookies,
            _call("submit_guess", "not json", "c1"),
            _call("submit_guess", json.dumps({"word": "crane"}), "c2"),
            _call("give_up", "{}", "c3"),
        )
        assert payload["terminated"] is False
        errors = [json.loads(o["output"])["error"] for o in payload["info"]["tool_outputs"]]
        assert errors[0].startswith("Invalid arguments for submit_guess")
        assert errors[1].startswith("Invalid arguments for submit_guess")
        assert errors[2] == "Unknown tool: give_up"
