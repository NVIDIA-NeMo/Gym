# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import random
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from environments.chess.prepare import EXAMPLE_TASKS, build_row, generated_tasks, mate_in_one_tasks
from nemo_gym.client import ServerClient
from resources_servers.chess.app import ChessResourcesServer, ChessResourcesServerConfig, choose_opponent_move
from resources_servers.chess.chess_rules import STARTING_FEN, Position
from resources_servers.chess.task_data import TaskData


EXAMPLE_PATH = Path(__file__).parents[3] / "environments" / "chess" / "data" / "example.jsonl"
MATE_IN_ONE_FEN = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"
# After 1. f3 e5: if White now plays g4, Black mates with Qh4.
FOOLS_MATE_FEN = "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2"


def _create_params() -> dict:
    return build_row(agent_color="white", opponent="random", seed=0)["responses_create_params"]


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


def _call(arguments: str, name: str = "make_move", call_id: str = "c1") -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _move(uci: str, call_id: str = "c1") -> dict:
    return _call(json.dumps({"move": uci}), call_id=call_id)


def _text(text: str) -> dict:
    return {
        "id": "msg",
        "role": "assistant",
        "status": "completed",
        "type": "message",
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
    }


@pytest.fixture
def client() -> TestClient:
    config = ChessResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    server = ChessResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return TestClient(server.setup_webserver())


class Game:
    """One episode over HTTP, carrying the session cookies the way gymnasium_agent does."""

    def __init__(self, client: TestClient, **task):
        # A client of its own, so its cookie jar holds only this episode's session.
        self.client = TestClient(client.app)
        response = self.client.post("/reset", json={"responses_create_params": _create_params(), **task})
        assert response.status_code == 200, response.text
        self.info = response.json()["info"]
        self.observation = json.loads(response.json()["observation"])

    def step_raw(self, *output: dict):
        return self.client.post(
            "/step",
            json={"responses_create_params": _create_params(), "response": _response(list(output))},
        )

    def step(self, *output: dict) -> dict:
        response = self.step_raw(*output)
        assert response.status_code == 200, response.text
        return response.json()

    def play(self, uci: str) -> dict:
        """Play one move and return the step payload with the parsed tool output under "result"."""
        payload = self.step(_move(uci))
        if not payload["terminated"]:
            payload["result"] = json.loads(payload["info"]["tool_outputs"][0]["output"])
        return payload


class TestReset:
    def test_starting_observation(self, client):
        game = Game(client)
        observation = game.observation
        assert observation["status"] == "ongoing"
        assert (observation["you_play"], observation["turn"]) == ("white", "white")
        assert observation["fen"] == STARTING_FEN
        assert observation["board"][0] == "8 r n b q k b n r"
        assert observation["board"][7] == "1 R N B Q K B N R"
        assert observation["board"][8] == "  a b c d e f g h"
        assert observation["in_check"] is False
        assert observation["castling_rights"] == "KQkq"
        assert observation["en_passant"] is None
        assert (observation["halfmove_clock"], observation["fullmove_number"]) == (0, 1)
        assert observation["recent_moves"] == []
        assert len(observation["legal_moves"]) == 20 and "g1f3" in observation["legal_moves"]
        assert "opponent_move" not in observation
        assert game.info == {"supports_explicit_close": True}

    def test_opponent_moves_first_when_the_model_plays_black(self, client):
        observation = Game(client, agent_color="black", seed=5).observation
        assert (observation["you_play"], observation["turn"]) == ("black", "black")
        assert observation["recent_moves"] == [f"1. {observation['opponent_move']}"]
        assert len(observation["legal_moves"]) == len(Position.from_fen(observation["fen"]).legal_moves)

    def test_same_seed_same_opponent(self, client):
        first = [Game(client, agent_color="black", seed=seed).observation["opponent_move"] for seed in range(8)]
        again = [Game(client, agent_color="black", seed=seed).observation["opponent_move"] for seed in range(8)]
        assert first == again
        assert len(set(first)) > 1

    def test_custom_starting_position(self, client):
        observation = Game(client, fen=MATE_IN_ONE_FEN).observation
        assert observation["fen"] == MATE_IN_ONE_FEN
        assert "d1d8" in observation["legal_moves"]

    @pytest.mark.parametrize(
        "task, message",
        [
            ({"fen": "not a fen"}, "a FEN has 6 space-separated fields"),
            ({"fen": "8/8/8/8/8/8/8/8 w - - 0 1"}, "white must have exactly one king"),
            ({"agent_color": "red"}, "agent_color"),
            ({"opponent": "stockfish"}, "opponent"),
            ({"max_consecutive_invalid_actions": -1}, "max_consecutive_invalid_actions"),
            ({"fen": "k7/1Q6/1K6/8/8/8/8/8 b - - 0 1"}, "the game is already over \\(checkmate\\)"),
        ],
    )
    def test_invalid_tasks_are_rejected(self, client, task, message):
        response = client.post("/reset", json={"responses_create_params": _create_params(), **task})
        assert response.status_code == 400
        assert "Invalid chess task" in response.json()["detail"]
        assert re.search(message, response.json()["detail"])


class TestStep:
    def test_legal_move_gets_a_reply_and_the_new_position(self, client):
        game = Game(client, seed=3)
        payload = game.play("e2e4")
        assert (payload["reward"], payload["terminated"], payload["truncated"]) == (0.0, False, False)
        result = payload["result"]
        assert result["your_move"] == "e2e4"
        assert result["recent_moves"] == ["1. e2e4", f"1... {result['opponent_move']}"]
        assert result["turn"] == "white" and result["fullmove_number"] == 2
        assert result["legal_moves"] == sorted(m.uci() for m in Position.from_fen(result["fen"]).legal_moves)
        assert payload["info"]["tool_outputs"][0]["call_id"] == "c1"

    def test_illegal_move_is_reported_and_changes_nothing(self, client):
        game = Game(client)
        payload = game.play("e2e5")
        result = payload["result"]
        assert payload["reward"] == 0.0 and payload["terminated"] is False
        assert result["error"] == (
            "e2e5 is illegal: the pawn on e2 cannot move to e5. Invalid actions in a row left before you forfeit: 2."
        )
        assert result["fen"] == STARTING_FEN
        assert "your_move" not in result and "opponent_move" not in result

    def test_moving_the_opponents_piece_is_illegal(self, client):
        result = Game(client).play("e7e5")["result"]
        assert "the piece on e7 is black and it is white to move" in result["error"]

    @pytest.mark.parametrize(
        "call, error",
        [
            (_call("{not json"), "Invalid arguments for make_move"),
            (_call(json.dumps({"square": "e4"})), "Invalid arguments for make_move"),
            (_call(json.dumps({"move": "e4"})), "'e4' is not a UCI move"),
            (_call(json.dumps({"move": "e2e4"}), name="resign"), "Unknown tool 'resign'"),
        ],
    )
    def test_invalid_calls_are_reported(self, client, call, error):
        payload = Game(client).step(call)
        result = json.loads(payload["info"]["tool_outputs"][0]["output"])
        assert error in result["error"]
        assert result["fen"] == STARTING_FEN

    def test_turn_without_a_move_gets_a_reminder(self, client):
        payload = Game(client).step(_text("I think e4 is a good move."))
        assert payload["terminated"] is False and payload["info"] == {}
        observation = json.loads(payload["observation"])
        assert observation["error"].startswith("No move was made. Call make_move")
        assert observation["legal_moves"]

    def test_only_the_first_call_of_a_turn_is_played(self, client):
        payload = Game(client).step(_move("e2e4", call_id="a"), _move("d2d4", call_id="b"))
        outputs = payload["info"]["tool_outputs"]
        assert [output["call_id"] for output in outputs] == ["a", "b"]
        assert json.loads(outputs[0]["output"])["your_move"] == "e2e4"
        assert json.loads(outputs[1]["output"]) == {
            "error": "Only one move is played per turn; this call was ignored."
        }

    def test_too_many_invalid_actions_in_a_row_forfeit(self, client):
        game = Game(client, max_consecutive_invalid_actions=1)
        assert game.play("e2e5")["terminated"] is False
        payload = game.step(_text("pass"))
        assert (payload["reward"], payload["terminated"], payload["truncated"]) == (-1.0, True, False)
        assert payload["info"] == {
            "result": "loss",
            "termination": "forfeit",
            "agent_moves": 0,
            "invalid_actions": 2,
            "final_fen": STARTING_FEN,
        }

    def test_a_legal_move_resets_the_invalid_action_count(self, client):
        game = Game(client, max_consecutive_invalid_actions=1)
        assert game.play("e2e5")["result"]["error"].endswith("Invalid actions in a row left before you forfeit: 0.")
        assert game.play("e2e4")["terminated"] is False
        result = game.play("e4e6")["result"]
        assert result["error"].endswith("Invalid actions in a row left before you forfeit: 0.")
        payload = game.step(_text("pass"))
        assert payload["info"]["termination"] == "forfeit"
        assert (payload["info"]["agent_moves"], payload["info"]["invalid_actions"]) == (1, 3)

    def test_zero_allowance_forfeits_on_the_first_invalid_action(self, client):
        payload = Game(client, max_consecutive_invalid_actions=0).play("e1e2")
        assert payload["reward"] == -1.0 and payload["info"]["termination"] == "forfeit"

    def test_step_without_reset_is_not_found(self, client):
        response = client.post("/step", json={"responses_create_params": _create_params(), "response": _response([])})
        assert response.status_code == 404


class TestRewards:
    def test_win_by_checkmate(self, client):
        payload = Game(client, fen=MATE_IN_ONE_FEN, opponent="greedy").play("d1d8")
        assert (payload["reward"], payload["terminated"]) == (1.0, True)
        assert payload["info"] == {
            "result": "win",
            "termination": "checkmate",
            "agent_moves": 1,
            "invalid_actions": 0,
            "final_fen": "3R2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1",
        }

    def test_loss_when_the_opponent_mates(self, client):
        payload = Game(client, fen=FOOLS_MATE_FEN, opponent="greedy").play("g2g4")
        assert payload["reward"] == -1.0
        assert payload["info"]["result"] == "loss" and payload["info"]["termination"] == "checkmate"

    def test_win_as_black_is_scored_for_black(self, client):
        # The model plays Black after 1. f3 e5 2. g4; mating with Qh4 is a win for the model.
        fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
        payload = Game(client, fen=fen, agent_color="black").play("d8h4")
        assert payload["reward"] == 1.0 and payload["info"]["result"] == "win"

    def test_draw_by_stalemate(self, client):
        payload = Game(client, fen="k7/8/1K6/8/8/8/8/2Q5 w - - 0 1").play("c1c7")
        assert payload["reward"] == 0.0
        assert payload["info"]["result"] == "draw" and payload["info"]["termination"] == "stalemate"

    def test_draw_by_insufficient_material(self, client):
        payload = Game(client, fen="8/8/4k3/8/8/8/8/3rK3 w - - 0 1").play("e1d1")
        assert payload["reward"] == 0.0 and payload["info"]["termination"] == "insufficient_material"

    def test_game_lost_before_the_models_first_move(self, client):
        game = Game(client, fen=MATE_IN_ONE_FEN, agent_color="black", opponent="greedy")
        assert game.observation["opponent_move"] == "d1d8"
        assert game.observation["status"] == "checkmate" and game.observation["legal_moves"] == []
        payload = game.play("g8f8")
        assert payload["reward"] == -1.0 and payload["info"]["agent_moves"] == 0


class TestEpisodeLifecycle:
    def test_no_moves_after_the_game_ends(self, client):
        game = Game(client, fen=MATE_IN_ONE_FEN)
        assert game.play("d1d8")["terminated"] is True
        assert game.step_raw(_move("g8h8")).status_code == 404

    def test_close_releases_an_unfinished_game(self, client):
        game = Game(client)
        assert game.client.post("/close", json={}).status_code == 200
        assert game.step_raw(_move("e2e4")).status_code == 404

    def test_sessions_are_isolated(self, client):
        first, second = Game(client, seed=1), Game(client, seed=1, fen=MATE_IN_ONE_FEN)
        assert first.play("e2e4")["terminated"] is False
        assert second.play("d1d8")["reward"] == 1.0
        assert first.play("d2d4")["result"]["your_move"] == "d2d4"

    def test_consecutive_games_on_one_session(self, client):
        game = Game(client, fen=MATE_IN_ONE_FEN)
        assert game.play("d1d8")["reward"] == 1.0
        response = game.client.post("/reset", json={"responses_create_params": _create_params()})
        assert json.loads(response.json()["observation"])["fen"] == STARTING_FEN
        assert game.play("e2e4")["terminated"] is False

    @pytest.mark.parametrize("agent_color", ["white", "black"])
    @pytest.mark.parametrize("opponent", ["random", "greedy", "lookahead"])
    def test_complete_game_from_the_starting_position(self, client, agent_color, opponent):
        # The model's stand-in plays a seeded random legal move every turn until the game ends.
        policy = random.Random(7)
        game = Game(client, agent_color=agent_color, opponent=opponent, seed=11)
        observation = game.observation
        for turn in range(1, 400):
            payload = game.play(policy.choice(observation["legal_moves"]))
            if payload["terminated"]:
                break
            observation = payload["result"]
            assert "error" not in observation and observation["turn"] == agent_color
        info = payload["info"]
        assert payload["terminated"] is True
        assert info["agent_moves"] == turn and info["invalid_actions"] == 0
        assert info["termination"] in {
            "checkmate",
            "stalemate",
            "insufficient_material",
            "fifty_moves",
            "threefold_repetition",
        }
        assert payload["reward"] == {"win": 1.0, "draw": 0.0, "loss": -1.0}[info["result"]]
        assert (info["result"] == "draw") == (info["termination"] != "checkmate")


class TestOpponent:
    def test_greedy_captures_the_most_valuable_piece(self):
        # The knight on d4 can take a pawn on c2 or the queen on e6.
        position = Position.from_fen("4k3/8/4Q3/8/3n4/8/2P5/4K3 b - - 0 1")
        for seed in range(5):
            assert choose_opponent_move(position, random.Random(seed), "greedy").uci() == "d4e6"

    @pytest.mark.parametrize("opponent", ["greedy", "lookahead"])
    def test_takes_a_mate_in_one(self, opponent):
        position = Position.from_fen("3r2k1/8/8/8/8/8/5PPP/6K1 b - - 0 1")
        assert choose_opponent_move(position, random.Random(0), opponent).uci() == "d8d1"

    def test_greedy_takes_a_defended_pawn_with_its_queen_and_lookahead_does_not(self):
        # Qxd2 wins a pawn and loses the queen to Kxd2.
        position = Position.from_fen("3qk3/8/8/8/8/8/3P4/4K3 b - - 0 1")
        assert choose_opponent_move(position, random.Random(0), "greedy").uci() == "d8d2"
        for seed in range(10):
            assert choose_opponent_move(position, random.Random(seed), "lookahead").uci() != "d8d2"

    def test_lookahead_takes_an_undefended_piece(self):
        # The rook on d1 attacks the queen and nothing defends it, so taking it is the only move that keeps the queen.
        position = Position.from_fen("3qk3/8/8/8/8/8/8/3R3K b - - 0 1")
        for seed in range(10):
            assert choose_opponent_move(position, random.Random(seed), "lookahead").uci() == "d8d1"

    def test_random_plays_legal_moves_reproducibly(self):
        position = Position.from_fen(STARTING_FEN)
        moves = [choose_opponent_move(position, random.Random(seed), "random") for seed in range(20)]
        assert all(move in position.legal_moves for move in moves)
        assert moves == [choose_opponent_move(position, random.Random(seed), "random") for seed in range(20)]
        assert len(set(moves)) > 5


class TestTaskData:
    def test_example_rows_start_games(self, client):
        rows = [json.loads(line) for line in EXAMPLE_PATH.read_text().splitlines()]
        assert rows == [build_row(**task) for task in EXAMPLE_TASKS]
        for row in rows:
            TaskData.model_validate(row)
            response = client.post("/reset", json=row)
            assert response.status_code == 200, response.text

    def test_generated_splits_cover_both_colors_and_opponents_with_disjoint_seeds(self):
        train, validation = generated_tasks("train", 40), generated_tasks("validation", 40)
        assert {(task["agent_color"], task["opponent"]) for task in train} == {
            (color, opponent) for color in ("white", "black") for opponent in ("random", "greedy", "lookahead")
        }
        assert not {task["seed"] for task in train} & {task["seed"] for task in validation}

    def test_mate_in_one_tasks_have_a_mate_for_the_side_to_move(self):
        for task in mate_in_one_tasks("train", 5):
            position = Position.from_fen(task["fen"])
            assert task["agent_color"] == position.turn
            assert any(
                (after := position.apply(move)).in_check() and not after.legal_moves for move in position.legal_moves
            )

    def test_mate_in_one_splits_are_disjoint_and_reproducible(self):
        train = mate_in_one_tasks("train", 5)
        assert train == mate_in_one_tasks("train", 5)
        assert not {task["fen"] for task in train} & {task["fen"] for task in mate_in_one_tasks("validation", 3)}
