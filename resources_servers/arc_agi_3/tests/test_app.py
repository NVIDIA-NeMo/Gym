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
import asyncio
import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetricsRequest
from nemo_gym.core import NeMoGymResponse
from nemo_gym.data import load_task_data_schema, validate_jsonl_rows
from resources_servers.arc_agi_3 import create_dataset
from resources_servers.arc_agi_3 import game_server as game_server_module
from resources_servers.arc_agi_3.app import (
    ATTEMPT_REPORT_KEY,
    ArcAgi3ResourcesServer,
    ArcAgi3ResourcesServerConfig,
    ArcAgi3VerifyRequest,
    load_environment_metadata,
)
from resources_servers.arc_agi_3.game_server import GameServer
from resources_servers.arc_agi_3.rhae import game_score


HERE = Path(__file__).resolve().parent
# The real public25 game files, when all are present (create_dataset.py --download-kaggle-competition, or a copy);
# the repo commits only the two example games.
REAL_ENVIRONMENTS = Path(
    os.environ.get("ARC3_ENVIRONMENTS_DIR", str(HERE.parent / "data" / "environment_files"))
).resolve()
needs_games = pytest.mark.skipif(
    len(list(REAL_ENVIRONMENTS.glob("*/*/metadata.json"))) < len(create_dataset.PUBLIC25_VERSIONS),
    reason=f"the public25 game files are not all at {REAL_ENVIRONMENTS}",
)


def write_fake_game(root: Path, game: str, version: str, baseline) -> None:
    game_dir = root / game / version
    game_dir.mkdir(parents=True)
    (game_dir / "metadata.json").write_text(json.dumps({"game_id": f"{game}-{version}", "baseline_actions": baseline}))
    (game_dir / f"{game}.py").write_text("# fake game\n")


def make_server(environments_dir: Path, **overrides) -> ArcAgi3ResourcesServer:
    config = ArcAgi3ResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="arc_agi_3",
        environments_dir=str(environments_dir),
        **{"game_server_dir": str(environments_dir.parent / f"{environments_dir.name}_game_server"), **overrides},
    )
    return ArcAgi3ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def response_stub() -> dict:
    return NeMoGymResponse(
        id="resp_1",
        created_at=0,
        model="policy",
        object="response",
        output=[],
        tool_choice="auto",
        tools=[],
        parallel_tool_calls=True,
    ).model_dump()


def loaded_metadata(root: Path, game: str) -> dict:
    """The loaded game ``game``: a full id, or a short id with one version under ``root``."""
    games = load_environment_metadata(str(root))
    if game in games:
        return games[game]
    (meta,) = [meta for full_game_id, meta in games.items() if full_game_id.split("-", 1)[0] == game]
    return meta


def pinned_metadata(root: Path, game: str) -> dict:
    """A create_dataset.py-style row for the game under ``root`` (its real pins)."""
    loaded = loaded_metadata(root, game)
    return {
        "game_id": loaded["game_id"].split("-", 1)[0],
        "full_game_id": loaded["game_id"],
        "version": loaded["_version"],
        "files_sha256": loaded["_files_sha256"],
        "baseline_actions": loaded["baseline_actions"],
    }


def utc(naive_iso: str) -> str:
    return naive_iso + "+00:00"


def verify_request(root: Path, game_id: str, record, **extra) -> ArcAgi3VerifyRequest:
    # The agent always sends the run clock origin (UTC) and its scorecard; by default here the game's first server
    # event is at the run start (offset 0).
    extra.setdefault("benchmark_start_time", utc((record or {}).get("started_at") or "2026-11-04T12:00:10"))
    extra.setdefault("scorecard_id", "card-1")
    extra.setdefault("game_guid", "guid-1")
    extra.setdefault("scorecard_api_key", "key-1")
    return ArcAgi3VerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": f"Play {game_id}"}]},
        response=response_stub(),
        verifier_metadata=pinned_metadata(root, game_id),
        game_record=record,
        **extra,
    )


def record_for(actions_per_level, levels_completed, base, walls, game_id="fake-0001", **extra) -> dict:
    return {
        "game_id": game_id,
        "number_of_levels": len(base),
        "base_actions_per_level": base,
        "actions_per_level": actions_per_level,
        "levels_completed": levels_completed,
        "state": "gave_up",
        "started_at": "2026-11-04T12:00:10",
        "history": [{"action": {"id": "ACTION1", "data": {}}, "wallclock_seconds": w} for w in walls],
        **extra,
    }


def server_events(actions_per_level, levels_completed, walls, start="2026-11-04T12:00:10") -> list:
    """The game server's recording of a play: make()'s RESET, then one event per action at start + wall seconds."""
    import datetime as dt

    origin = dt.datetime.fromisoformat(utc(start))
    events = [{"timestamp": origin.isoformat(), "data": {"action_input": {"id": "RESET"}, "levels_completed": 0}}]
    level, done_in_level = 0, 0
    for wall in walls:
        done_in_level += 1
        if level < levels_completed and done_in_level == actions_per_level[level]:
            level, done_in_level = level + 1, 0
        when = origin + dt.timedelta(seconds=wall)
        events.append(
            {"timestamp": when.isoformat(), "data": {"action_input": {"id": "ACTION1"}, "levels_completed": level}}
        )
    return events


class FakeGameServer:
    """The game server's two outputs for tests on fake games: closed cards and run recordings."""

    def __init__(self):
        self.cards, self.recordings, self.closed = {}, {}, []
        self.running, self.url, self.out_dir = True, "http://game-server.invalid", Path("/nonexistent")

    def close_card(self, card_id: str, api_key: str) -> dict:
        self.closed.append((card_id, api_key))
        if card_id not in self.cards:
            raise ValueError(f"scorecard {card_id} is unknown to the game server")
        return self.cards[card_id]

    def recording(self, card_id: str, game_id: str, guid: str) -> list:
        key = (card_id, game_id, guid)
        if key not in self.recordings:
            raise FileNotFoundError(f"the game server has no recording {key}")
        return self.recordings[key]


def fake_play(server, record, *, won=False, card_id="card-1", guid="guid-1", runs=None, events=None) -> FakeGameServer:
    """Put on a fake game server the card and recording that a faithful play of ``record`` gives."""
    fake = FakeGameServer()
    apl, levels = record["actions_per_level"], record["levels_completed"]
    base = record["base_actions_per_level"]
    run = {
        "guid": guid,
        "score": game_score(apl, base, len(base), levels),
        "levels_completed": levels,
        "actions": sum(apl),
        "state": "WIN" if won else "NOT_FINISHED",
        "level_actions": apl,
    }
    fake.cards[card_id] = {"card_id": card_id, "environments": [{"id": record["game_id"], "runs": runs or [run]}]}
    walls = [h["wallclock_seconds"] for h in record["history"]]
    recorded = events if events is not None else server_events(apl, levels, walls, record["started_at"])
    fake.recordings[(card_id, record["game_id"], guid)] = recorded
    server._game_server = fake
    return fake


class TestVerify:
    def test_scores_from_the_scorecard(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [10, 20])
        server = make_server(tmp_path)
        # Level 1 in 20 actions (25 points), cleared at 30 min; level 2 not cleared.
        walls = [90.0 * (i + 1) for i in range(25)]
        record = record_for([20, 5], 1, [10, 20], walls)
        fake = fake_play(server, record)
        result = asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))
        assert result.rhae == pytest.approx(25.0 / 3)
        assert result.reward == pytest.approx(25.0 / 300)
        assert result.levels_completed == 1 and result.actions == 25 and result.number_of_levels == 2
        assert getattr(result, "rhae_1h") == pytest.approx(25.0 / 3)
        assert result.timing_valid and result.result_valid and result.scorecard_runs == 1
        assert result.state == "not_won" and result.agent_state == "gave_up" and result.scorecard_id == "card-1"
        # per level, from the server's recording: level 1 cleared by action 20, 5 actions on level 2 after it
        assert result.actions_per_level == [20, 5] and result.level_clear_steps == [20]
        assert fake.closed == [("card-1", "key-1")]
        assert "scorecard_api_key" not in result.model_dump()

    def test_reward_is_the_card_score_of_the_best_run(self, tmp_path: Path) -> None:
        # Two runs of one game (a full reset): the SDK's best-run score and summed actions, the last run's state.
        write_fake_game(tmp_path, "fake", "0001", [10, 20])
        server = make_server(tmp_path)
        record = record_for([12, 0], 0, [10, 20], [float(i + 1) for i in range(12)])
        runs = [
            {"guid": "guid-1", "score": 40.0, "levels_completed": 1, "actions": 10, "state": "NOT_FINISHED"},
            {"guid": "guid-1", "score": 0.0, "levels_completed": 0, "actions": 2, "state": "NOT_FINISHED"},
        ]
        events = server_events([10, 2], 1, [float(i + 1) for i in range(12)])
        fake_play(server, record, runs=runs, events=events)
        result = asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))
        assert (
            result.rhae == 40.0
            and result.actions == 12
            and result.scorecard_runs == 2
            and result.levels_completed == 1
        )

    def test_rhae_at_hour_excludes_late_levels(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [10, 20])
        server = make_server(tmp_path)
        walls = [400.0 * (i + 1) for i in range(20)]  # the clearing action is at 8000 s
        record = record_for([20, 0], 1, [10, 20], walls)
        fake_play(server, record)
        result = asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))
        assert getattr(result, "rhae_1h") == 0.0 and getattr(result, "rhae_2h") == 0.0
        assert result.rhae > 0

    def test_missing_record_scores_zero_and_closes_its_card(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [10])
        server = make_server(tmp_path)
        fake = FakeGameServer()
        fake.cards["card-1"] = {"card_id": "card-1", "environments": []}
        server._game_server = fake
        request = verify_request(tmp_path, "fake", None, agent_error="GameStartError: boom")
        result = asyncio.run(server.verify(request))
        assert result.reward == 0.0 and result.state == "not_started" and result.agent_state == "not_started"
        assert not result.result_valid and getattr(result, "rhae_1h") is None
        assert fake.closed == [("card-1", "key-1")]
        no_card = verify_request(tmp_path, "fake", None, scorecard_id=None, scorecard_api_key=None)
        assert asyncio.run(server.verify(no_card)).state == "not_started"

    def test_baseline_mismatch_raises(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [10, 20])
        record = record_for([1, 0], 0, [11, 20], [1.0])
        with pytest.raises(ValueError, match="baselines"):
            asyncio.run(make_server(tmp_path).verify(verify_request(tmp_path, "fake", record)))

    def test_one_worker_only(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [10])
        with pytest.raises(ValueError, match="num_workers must be 1"):
            make_server(tmp_path, num_workers=2)

    def test_unknown_game_raises(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [10])
        with pytest.raises(KeyError):
            make_server(tmp_path).loaded_game("nope")

    def test_persistence_error_keeps_the_score_but_flags_the_result(self, tmp_path: Path) -> None:
        # F1: a completed game whose snapshot could not be saved keeps its record and score.
        write_fake_game(tmp_path, "fake", "0001", [10])
        server = make_server(tmp_path)
        record = record_for([10], 1, [10], [float(i + 1) for i in range(10)], state="won")
        fake_play(server, record, won=True)
        request = verify_request(tmp_path, "fake", record, persistence_error="OSError: disk full")
        result = asyncio.run(server.verify(request))
        assert result.rhae == pytest.approx(100.0) and result.persistence_error == "OSError: disk full"
        assert result.state == "won" and not result.result_valid

    def test_model_that_never_answered_is_masked(self, tmp_path: Path) -> None:
        # A dead model endpoint: the game ran (and its card is checked) but without the model, so the result is
        # invalid and the rollout masked as a model request error, not a valid score of 0.
        write_fake_game(tmp_path, "fake", "0001", [10])
        server = make_server(tmp_path)
        record = record_for([], 0, [10], [])
        fake_play(server, record)
        error = "every model call of the game failed (4 calls; last: HTTP 404)"
        result = asyncio.run(server.verify(verify_request(tmp_path, "fake", record, model_error=error)))
        assert result.actions == 0 and not result.result_valid and result.model_error == error
        assert result.mask_sample and result.failure_kind == "model_request_error" and result.failure_reason == error
        assert server.compute_metrics([[result.model_dump()]])["model_errors"] == 1
        played = record_for([3], 0, [10], [1.0, 2.0, 3.0])
        fake_play(server, played)
        valid = asyncio.run(server.verify(verify_request(tmp_path, "fake", played)))
        assert valid.result_valid and not valid.mask_sample and valid.failure_kind is None

    def test_the_report_comes_in_the_agents_response(self, tmp_path: Path) -> None:
        # An agent-session episode: the environment server verifies the activation's reply, which carries the agent's
        # report of the game; the report's harness fields stay in the rollout, and the reply loses the report.
        write_fake_game(tmp_path, "fake", "0001", [10])
        server = make_server(tmp_path)
        record = record_for([3], 0, [10], [1.0, 2.0, 3.0])
        fake_play(server, record)
        plain = verify_request(tmp_path, "fake", record)
        report = {
            **{
                k: getattr(plain, k)
                for k in ("scorecard_id", "game_guid", "scorecard_api_key", "benchmark_start_time")
            },
            "game_record": record,
            "harness": "arc3-kaggle-kit",
            "agent_call_log": [{"index": 0}],
            "verifier_metadata": {"game_id": "other"},  # a report cannot replace the task's own fields
        }
        request = ArcAgi3VerifyRequest(
            responses_create_params=plain.responses_create_params,
            response={**response_stub(), ATTEMPT_REPORT_KEY: report},
            verifier_metadata=plain.verifier_metadata,
        )
        result = asyncio.run(server.verify(request))
        assert result.result_valid and result.actions == 3 and result.scorecard_id == "card-1"
        assert result.harness == "arc3-kaggle-kit" and result.agent_call_log == [{"index": 0}]
        assert result.verifier_metadata == plain.verifier_metadata
        assert ATTEMPT_REPORT_KEY not in result.response.model_dump()
        assert "scorecard_api_key" not in result.model_dump() and "game_record" not in result.model_dump()

    def test_cancelled_attempt_is_not_a_valid_result(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [10])
        server = make_server(tmp_path)
        record = record_for([3], 0, [10], [1.0, 2.0, 3.0], state="cancelled")
        fake_play(server, record)
        result = asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))
        assert result.agent_state == "cancelled" and not result.result_valid

    def test_official_report_with_an_action_count(self, tmp_path: Path) -> None:
        # The official harness reports an action count instead of a history.
        write_fake_game(tmp_path, "fake", "0001", [10])
        server = make_server(tmp_path)
        record = record_for([3], 0, [10], [1.0, 2.0, 3.0])
        fake_play(server, record)
        report = dict(record, history=[], actions=3)
        assert asyncio.run(server.verify(verify_request(tmp_path, "fake", report))).actions == 3
        with pytest.raises(ValueError, match="actions 3 != the agent's 4"):
            asyncio.run(server.verify(verify_request(tmp_path, "fake", dict(report, actions=4))))

    def test_compute_metrics(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [10])
        server = make_server(tmp_path)
        valid = {
            "rhae": 10.0,
            "rhae_1h": 5.0,
            "rhae_2h": 10.0,
            "levels_completed": 1,
            "timing_valid": True,
            "result_valid": True,
            "agent_state": "gave_up",
            "game_id": "aa01",
            "verifier_metadata": {"full_game_id": "aa01-0001"},
        }
        other = dict(
            valid,
            rhae=30.0,
            rhae_1h=0.0,
            rhae_2h=20.0,
            levels_completed=2,
            game_id="bb02",
            verifier_metadata={"full_game_id": "bb02-0001"},
        )
        metrics = server.compute_metrics([[valid], [other]])
        assert metrics["rhae"] == 20.0 and metrics["rhae_1h"] == 2.5 and metrics["rhae_2h"] == 15.0
        assert metrics["received_games"] == 2 and metrics["levels_completed"] == 3
        # No expected workload: a statistic of the received games, not a complete benchmark score.
        assert (
            metrics["all_received_valid"] and metrics["complete"] is None and metrics["score_over"] == "received_games"
        )
        invalid = dict(other, timing_valid=False, result_valid=False)
        metrics = server.compute_metrics([[valid], [invalid]])
        assert metrics["rhae"] is None and metrics["rhae_1h"] is None and not metrics["all_received_valid"]
        assert metrics["timing_invalid"] == 1 and metrics["valid_results"] == 1 and "replay_mismatches" not in metrics
        assert server.get_key_metrics(metrics)["rhae"] is None
        assert server.compute_metrics([]) == {}
        untimed = dict(valid, rhae_1h=None)  # e.g. no run clock origin: the end score stays, the hour score does not
        metrics = server.compute_metrics([[valid], [untimed]])
        assert metrics["rhae"] == 10.0 and metrics["rhae_1h"] is None

    def test_compute_metrics_with_expected_games(self, tmp_path: Path) -> None:
        # F2 residual: one valid winner received out of an expected 25 is not a complete benchmark score.
        write_fake_game(tmp_path, "fake", "0001", [10])
        server = make_server(tmp_path, expected_games=25)
        winner = {"rhae": 100.0, "rhae_1h": 100.0, "rhae_2h": 100.0, "result_valid": True, "agent_state": "won"}

        def game(index: int, **row) -> dict:
            return {
                **winner,
                "game_id": f"g{index:03d}",
                "verifier_metadata": {"full_game_id": f"g{index:03d}-v1"},
                **row,
            }

        metrics = server.compute_metrics([[game(0)]])
        assert metrics["complete"] is False and metrics["rhae"] is None and metrics["rhae_1h"] is None
        assert metrics["expected_games"] == 25 and metrics["received_games"] == 1
        assert metrics["score_over"] == "expected_games" and metrics["all_received_valid"]
        metrics = server.compute_metrics([[game(i)] for i in range(25)])
        assert metrics["complete"] is True and metrics["rhae"] == 100.0

    def test_compute_metrics_counts_distinct_games_and_weighs_them_equally(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [10])
        server = make_server(tmp_path, expected_games=2)

        def row(full_game_id: str, rhae: float) -> dict:
            return {
                "rhae": rhae,
                "rhae_1h": rhae,
                "rhae_2h": rhae,
                "result_valid": True,
                "agent_state": "gave_up",
                "game_id": full_game_id.split("-")[0],
                "verifier_metadata": {"full_game_id": full_game_id},
            }

        # two task groups of the same game do not make two games: the other expected game is missing
        metrics = server.compute_metrics([[row("aa01-v1", 0.0)], [row("aa01-v1", 0.0)]])
        assert metrics["received_games"] == 1 and metrics["received_rows"] == 2
        assert metrics["complete"] is False and metrics["rhae"] is None
        # uneven repeats: game A twice at 0, game B once at 100 -> the equal-game mean 50, not the row mean 33.33
        metrics = server.compute_metrics([[row("aa01-v1", 0.0), row("aa01-v1", 0.0)], [row("bb02-v1", 100.0)]])
        assert metrics["complete"] is True and metrics["received_games"] == 2
        assert metrics["rhae"] == 50.0 and metrics["rhae_1h"] == 50.0 and metrics["rhae_2h"] == 50.0

    def test_aggregate_metrics_counts_masked_rows(self, tmp_path: Path) -> None:
        # The shared aggregation drops masked rows before compute_metrics; completeness must still see them.
        write_fake_game(tmp_path, "fake", "0001", [10])
        server = make_server(tmp_path, expected_games=2)

        def row(task: int, rollout: int, full_game_id: str, rhae: float, **extra) -> dict:
            return {
                TASK_INDEX_KEY_NAME: task,
                ROLLOUT_INDEX_KEY_NAME: rollout,
                "reward": rhae / 100.0,
                "rhae": rhae,
                "rhae_1h": rhae,
                "rhae_2h": rhae,
                "timing_valid": True,
                "result_valid": True,
                "agent_state": "gave_up",
                "game_id": full_game_id.split("-")[0],
                "verifier_metadata": {"full_game_id": full_game_id},
                **extra,
            }

        masked = dict(result_valid=False, model_error="every model call failed", mask_sample=True)
        valid_a, valid_b = row(0, 0, "aa01-v1", 0.0), row(1, 0, "bb02-v1", 100.0)
        rows = [valid_a, valid_b, row(0, 1, "aa01-v1", 0.0, **masked)]
        metrics = asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows)))
        agent = metrics.agent_metrics
        assert agent["received_rows"] == 3 and agent["model_errors"] == 1 and agent["complete"] is False
        assert agent["rhae"] is None and agent["rhae_1h"] is None and agent["rhae_2h"] is None
        assert metrics.key_metrics["complete"] is False and metrics.key_metrics["rhae"] is None
        # the shared path's coverage counters and quality metrics stay (computed from the measured rows)
        assert agent["coverage/masked_rollouts"] == 1 and agent["coverage/measured_rollouts"] == 2
        assert agent["mean/reward"] == pytest.approx(0.5)
        # every row masked: still reported, not a vanished game
        all_masked = asyncio.run(
            server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[row(0, 0, "aa01-v1", 0.0, **masked)]))
        )
        assert all_masked.agent_metrics["received_rows"] == 1 and all_masked.agent_metrics["model_errors"] == 1
        assert all_masked.agent_metrics["complete"] is False and all_masked.key_metrics["rhae"] is None
        # healthy rows only: the equal-game mean over uneven repeats is unchanged
        healthy = asyncio.run(
            server.aggregate_metrics(
                AggregateMetricsRequest(verify_responses=[valid_a, row(0, 1, "aa01-v1", 0.0), valid_b])
            )
        )
        assert healthy.agent_metrics["complete"] is True and healthy.agent_metrics["rhae"] == 50.0


class TestAuthority:
    """The scorecard is the source of truth: any disagreement with the agent's report fails loudly."""

    def setup(self, tmp_path: Path):
        write_fake_game(tmp_path, "fake", "0001", [10, 20])
        server = make_server(tmp_path)
        record = record_for([10, 2], 1, [10, 20], [float(i + 1) for i in range(12)])
        fake = fake_play(server, record)
        return server, record, fake

    @pytest.mark.parametrize(
        "change, match",
        [
            ({"levels_completed": 0}, "levels 1 != the agent's 0"),
            ({"state": "won"}, "state NOT_FINISHED != the agent's 'won'"),
            ({"history": [{"wallclock_seconds": 1.0}] * 11}, "actions 12 != the agent's 11"),
        ],
    )
    def test_report_mismatches_raise(self, tmp_path: Path, change: dict, match: str) -> None:
        server, record, _ = self.setup(tmp_path)
        with pytest.raises(ValueError, match=match):
            asyncio.run(server.verify(verify_request(tmp_path, "fake", dict(record, **change))))

    def test_guid_mismatch_raises(self, tmp_path: Path) -> None:
        server, record, _ = self.setup(tmp_path)
        with pytest.raises(ValueError, match="run guids"):
            asyncio.run(server.verify(verify_request(tmp_path, "fake", record, game_guid="other-guid")))

    def test_card_with_another_or_two_games_raises(self, tmp_path: Path) -> None:
        server, record, fake = self.setup(tmp_path)
        games = fake.cards["card-1"]["environments"]
        fake.cards["card-1"]["environments"] = games + [dict(games[0], id="other-0001")]
        with pytest.raises(ValueError, match="not exactly fake-0001"):
            asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))
        fake.cards["card-1"]["environments"] = [dict(games[0], id="other-0001")]
        with pytest.raises(ValueError, match="not exactly fake-0001"):
            asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))

    def test_unknown_card_or_no_card_raises(self, tmp_path: Path) -> None:
        server, record, _ = self.setup(tmp_path)
        with pytest.raises(ValueError, match="unknown"):
            asyncio.run(server.verify(verify_request(tmp_path, "fake", record, scorecard_id="card-2")))
        for missing in ("scorecard_id", "game_guid", "scorecard_api_key"):
            with pytest.raises(ValueError, match="unverifiable"):
                asyncio.run(server.verify(verify_request(tmp_path, "fake", record, **{missing: None})))

    def test_recording_that_differs_from_its_card_raises(self, tmp_path: Path) -> None:
        server, record, fake = self.setup(tmp_path)
        events = fake.recordings[("card-1", "fake-0001", "guid-1")]
        fake.recordings[("card-1", "fake-0001", "guid-1")] = events[:-1]
        with pytest.raises(ValueError, match="recording .* differs from its own scorecard"):
            asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))


class TestServerTiming:
    """RHAE@T from the game server's own action times; bad or missing times give no timed score (F4, F7)."""

    def play(self, tmp_path: Path, walls, events=None):
        write_fake_game(tmp_path, "fake", "0001", [1])
        server = make_server(tmp_path)
        record = record_for([1], 1, [1], walls)
        fake_play(server, record, events=events)
        return server, record

    def test_honest_late_clear_scores_zero_in_time(self, tmp_path: Path) -> None:
        server, record = self.play(tmp_path, [7201.0])
        result = asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))
        assert result.timing_valid and getattr(result, "rhae_1h") == 0.0 and getattr(result, "rhae_2h") == 0.0

    @pytest.mark.parametrize(
        "events, match",
        [
            ([], "empty"),
            (
                [{"timestamp": "2026-11-04T12:00:10+00:00", "data": {"action_input": {"id": "ACTION1"}}}],
                "initial RESET",
            ),
            ([{"timestamp": "2026-11-04T12:00:10", "data": {"action_input": {"id": "RESET"}}}], "no UTC offset"),
            ([{"data": {"action_input": {"id": "RESET"}}}], "no valid timestamp"),
        ],
    )
    def test_bad_recordings_give_no_timed_score(self, tmp_path: Path, events, match: str) -> None:
        server, record = self.play(tmp_path, [5.0], events=events)
        result = asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))
        assert not result.timing_valid and not result.result_valid and match in result.timing_error
        assert getattr(result, "rhae_1h") is None and getattr(result, "rhae_2h") is None
        assert result.rhae == pytest.approx(100.0)  # the card's score does not depend on time

    def test_out_of_order_times_and_missing_recording(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [2])
        server = make_server(tmp_path)
        record = record_for([2], 1, [2], [5.0, 6.0])
        fake = fake_play(server, record, events=server_events([2], 1, [5.0, 4.0]))
        result = asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))
        assert not result.timing_valid and "out of order" in result.timing_error
        fake.recordings.clear()
        result = asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))
        assert not result.timing_valid and "no recording" in result.timing_error

    def test_action_after_the_last_level_is_refused(self, tmp_path: Path) -> None:
        events = server_events([1], 1, [5.0]) + [
            {
                "timestamp": "2026-11-04T12:00:20+00:00",
                "data": {"action_input": {"id": "ACTION1"}, "levels_completed": 1},
            }
        ]
        server, record = self.play(tmp_path, [5.0], events=events)
        result = asyncio.run(server.verify(verify_request(tmp_path, "fake", record)))
        assert not result.timing_valid and "after all 1 levels" in result.timing_error

    def test_run_clock_offset(self, tmp_path: Path) -> None:
        # F7: a game whose first server event is 10 s after the run start, clearing at 3595 s of its own play: 3605 s
        # on the run clock, so it does not count at 1 h; a run start after the game's first event is refused.
        server, record = self.play(tmp_path, [3595.0])
        on_run_clock = asyncio.run(
            server.verify(verify_request(tmp_path, "fake", record, benchmark_start_time="2026-11-04T12:00:00+00:00"))
        )
        assert on_run_clock.clock_offset_seconds == pytest.approx(10.0) and getattr(on_run_clock, "rhae_1h") == 0.0
        assert getattr(on_run_clock, "rhae_2h") == pytest.approx(100.0)
        early = asyncio.run(
            server.verify(verify_request(tmp_path, "fake", record, benchmark_start_time="2026-11-04T12:00:20+00:00"))
        )
        assert not early.timing_valid and "before the run start" in early.timing_error
        naive = asyncio.run(
            server.verify(verify_request(tmp_path, "fake", record, benchmark_start_time="2026-11-04T12:00:00"))
        )
        assert not naive.timing_valid and "no UTC offset" in naive.timing_error

    def test_missing_run_clock_origin_gives_no_timed_score(self, tmp_path: Path) -> None:
        # F7 residual: without the origin there is no per-game-clock fallback (it would give 100 at 1 h here).
        server, record = self.play(tmp_path, [3595.0])
        request = verify_request(tmp_path, "fake", record, benchmark_start_time=None)
        result = asyncio.run(server.verify(request))
        assert getattr(result, "rhae_1h") is None and getattr(result, "rhae_2h") is None
        assert result.clock_offset_seconds is None and "run clock origin" in result.clock_error
        assert result.rhae == pytest.approx(100.0) and result.timing_valid


class TestPins:
    """F5: the dataset row's pins must match the loaded game, and the record must be for that exact game."""

    def test_pin_mismatches_rejected(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [1])
        server = make_server(tmp_path)
        record = record_for([1], 1, [1], [1.0])
        for key, value in (
            ("version", "0002"),
            ("files_sha256", {"fake.py": "0" * 64}),
        ):
            request = verify_request(tmp_path, "fake", record)
            request.verifier_metadata[key] = value
            with pytest.raises(ValueError, match=key):
                asyncio.run(server.verify(request))
        request = verify_request(tmp_path, "fake", record)
        request.verifier_metadata["full_game_id"] = "fake-0002"  # a game (version) that is not loaded
        with pytest.raises(KeyError, match="fake-0002"):
            asyncio.run(server.verify(request))
        request = verify_request(tmp_path, "fake", record)
        del request.verifier_metadata["files_sha256"]
        with pytest.raises(ValueError, match="no files_sha256 pin"):
            asyncio.run(server.verify(request))
        request = verify_request(tmp_path, "fake", record)
        del request.verifier_metadata["full_game_id"]
        with pytest.raises(ValueError, match="no full_game_id pin"):
            asyncio.run(server.verify(request))

    def test_changed_game_file_rejected(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [1])
        request = verify_request(tmp_path, "fake", record_for([1], 1, [1], [1.0]))
        (tmp_path / "fake" / "0001" / "fake.py").write_text("# changed in place\n")
        with pytest.raises(ValueError, match="files_sha256"):
            asyncio.run(make_server(tmp_path).verify(request))

    def test_record_for_another_game_rejected(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "fake", "0001", [1])
        record = record_for([1], 1, [1], [1.0], game_id="other-0001")
        with pytest.raises(ValueError, match="not the pinned"):
            asyncio.run(make_server(tmp_path).verify(verify_request(tmp_path, "fake", record)))


class TestEnvironmentFiles:
    def test_load_metadata(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "aa11", "v1", [3, 4])
        assert load_environment_metadata(str(tmp_path))["aa11-v1"]["baseline_actions"] == [3, 4]

    def test_two_versions_of_a_game(self, tmp_path: Path) -> None:
        # Synthetic sets hold several versions of one game name: each is its own game, verified on its own pins.
        write_fake_game(tmp_path, "aa11", "v1", [3])
        write_fake_game(tmp_path, "aa11", "v2", [3, 4])
        games = load_environment_metadata(str(tmp_path))
        assert sorted(games) == ["aa11-v1", "aa11-v2"] and games["aa11-v2"]["_version"] == "v2"
        server = make_server(tmp_path)
        for full_game_id, base in (("aa11-v1", [3]), ("aa11-v2", [3, 4])):
            record = record_for([3] + [0] * (len(base) - 1), 1, base, [1.0, 2.0, 3.0], game_id=full_game_id)
            fake_play(server, record)
            result = asyncio.run(server.verify(verify_request(tmp_path, full_game_id, record)))
            assert result.game_id == "aa11" and result.number_of_levels == len(base) and result.result_valid

    def test_two_copies_of_a_game_rejected(self, tmp_path: Path) -> None:
        write_fake_game(tmp_path, "aa11", "v1", [3])
        write_fake_game(tmp_path, "aa11x", "v1", [3])
        (tmp_path / "aa11x" / "v1" / "metadata.json").write_text(json.dumps({"game_id": "aa11-v1"}))
        with pytest.raises(ValueError, match="Two copies"):
            load_environment_metadata(str(tmp_path))

    def test_python_path_reaches_the_game_server(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Runtimes that games import (param25's bundled arcgames) go first on the server's PYTHONPATH.
        write_fake_game(tmp_path, "aa11", "v1", [3])
        server = make_server(tmp_path, python_path=["/runtimes/param25/src"])
        started = {}

        class Exited:
            pid, returncode = 1, 1

            def __init__(self, args, env, **kwargs):
                started.update(env=env)

            def poll(self):
                return 1

        monkeypatch.setattr(game_server_module.subprocess, "Popen", Exited)
        monkeypatch.setenv("PYTHONPATH", "/site")
        with pytest.raises(RuntimeError, match="exited"):
            server.game_server.start(1)
        assert started["env"]["PYTHONPATH"] == os.pathsep.join(["/runtimes/param25/src", "/site"])

    def test_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_environment_metadata(str(tmp_path / "none"))
        with pytest.raises(FileNotFoundError):
            load_environment_metadata(str(tmp_path))


class TestDataset:
    def test_build_and_check(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(create_dataset, "PUBLIC25_VERSIONS", {"aa11": "v1", "bb22": "v2"})
        write_fake_game(tmp_path, "aa11", "v1", [3, 4])
        write_fake_game(tmp_path, "bb22", "v2", [5])
        rows = create_dataset.build_rows(tmp_path)
        assert [r["verifier_metadata"]["game_id"] for r in rows] == ["aa11", "bb22"]
        assert rows[1]["verifier_metadata"]["dispatch_index"] == 1
        assert rows[0]["verifier_metadata"]["number_of_levels"] == 2
        # the agent session's context: the same pinned game
        meta = rows[1]["verifier_metadata"]
        assert rows[1]["agent_context"] == {
            "agent": {"values": {"game": {k: meta[k] for k in create_dataset.AGENT_GAME_KEYS}}}
        }
        out = tmp_path / "out.jsonl"
        create_dataset.write_jsonl(rows, out)
        create_dataset.check_rows(tmp_path, out)
        (tmp_path / "bb22" / "v2" / "bb22.py").write_text("# changed\n")
        with pytest.raises(ValueError, match="differ"):
            create_dataset.check_rows(tmp_path, out)

    def test_version_pin(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(create_dataset, "PUBLIC25_VERSIONS", {"aa11": "v9"})
        write_fake_game(tmp_path, "aa11", "v1", [3])
        with pytest.raises(ValueError, match="pinned public25 versions"):
            create_dataset.build_rows(tmp_path)

    def test_committed_example_rows(self) -> None:
        rows = [json.loads(line) for line in (HERE.parent / "data" / "example.jsonl").read_text().splitlines()]
        assert [r["verifier_metadata"]["game_id"] for r in rows] == list(create_dataset.EXAMPLE_GAMES)
        for row in rows:
            meta = row["verifier_metadata"]
            assert create_dataset.PUBLIC25_VERSIONS[meta["game_id"]] == meta["version"]
            assert meta["number_of_levels"] == len(meta["baseline_actions"])
            assert row["agent_context"]["agent"]["values"]["game"] == {
                k: meta[k] for k in create_dataset.AGENT_GAME_KEYS
            }
        # The committed example games are the pinned files.
        create_dataset.check_rows(HERE.parent / "data" / "environment_files", HERE.parent / "data" / "example.jsonl")


class TestTaskData:
    """``task_data.py`` describes the rows ``create_dataset.py`` writes and the verifier and both agents read."""

    adapter = load_task_data_schema(HERE.parent)

    def report(self, rows: list[dict]):
        return validate_jsonl_rows("arc_agi_3", self.adapter, "rows", [json.dumps(row) for row in rows])

    def test_committed_and_generated_rows_validate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        committed = HERE.parent / "data" / "example.jsonl"
        assert validate_jsonl_rows("arc_agi_3", self.adapter, str(committed), committed.read_text().splitlines()).clean
        monkeypatch.setattr(create_dataset, "PUBLIC25_VERSIONS", {"aa11": "v1"})
        write_fake_game(tmp_path, "aa11", "v1", [3, 4])
        write_fake_game(tmp_path, "sg0005", "aee9edbb", [5])
        write_fake_game(tmp_path, "sg0005x", "2b3d7198", [6])  # several versions of one game are separate rows
        (tmp_path / "sg0005x" / "2b3d7198" / "metadata.json").write_text(
            json.dumps({"game_id": "sg0005-2b3d7198", "baseline_actions": [6]})
        )
        rows = create_dataset.build_rows(tmp_path, ["aa11", "sg0005", "sg0005x"])
        assert self.report(rows).clean
        # dispatch_index is optional (the agents default it to 0)
        meta = dict(rows[0]["verifier_metadata"])
        del meta["dispatch_index"]
        assert self.report([{**rows[0], "verifier_metadata": meta}]).clean
        # an undeclared field validates (extra="allow") but the drift gate reports it
        report = self.report([{**rows[0], "verifier_metadata": dict(meta, comment="kept")}])
        assert report.error_rows == 0 and report.unknown_keys == {"comment": 1}

    @pytest.mark.parametrize("pin", ["game_id", "full_game_id", "version", "files_sha256"])
    def test_required_pins_match_the_verifier(self, tmp_path: Path, pin: str) -> None:
        # A row without a pin the schema requires is one the verifier refuses too, so the two cannot drift.
        write_fake_game(tmp_path, "fake", "0001", [10])
        server = make_server(tmp_path)
        record = record_for([3], 0, [10], [1.0, 2.0, 3.0], game_id="fake-0001")
        fake_play(server, record)
        request = verify_request(tmp_path, "fake", record)
        assert self.report([{"verifier_metadata": request.verifier_metadata}]).clean
        assert asyncio.run(server.verify(request)).result_valid
        del request.verifier_metadata[pin]
        assert not self.report([{"verifier_metadata": request.verifier_metadata}]).clean
        with pytest.raises((ValueError, KeyError)):
            asyncio.run(server.verify(request))

    def test_malformed_pins_fail(self) -> None:
        row = json.loads((HERE.parent / "data" / "example.jsonl").read_text().splitlines()[0])
        meta = row["verifier_metadata"]
        for bad in ({"files_sha256": {"cn04.py": "not-a-sha"}}, {"files_sha256": {}}, {"full_game_id": ""}):
            assert not self.report([{**row, "verifier_metadata": {**meta, **bad}}]).clean
        # a pin written top-level is invisible to the server (it reads verifier_metadata only)
        assert not self.report([{**meta, "responses_create_params": row["responses_create_params"]}]).clean

    @pytest.mark.parametrize("config", ["arc_agi_3.yaml", "arc_agi_3_official.yaml"])
    def test_both_harness_configs_resolve_this_schema(self, config: str) -> None:
        import yaml

        cfg = yaml.safe_load((HERE.parent / "configs" / config).read_text())
        (agent,) = [body for v in cfg.values() for body in (v.get("responses_api_agents") or {}).values()]
        assert agent["datasets"] and agent["resources_server"]["type"] == "resources_servers"
        owner = cfg[agent["resources_server"]["name"]]["resources_servers"]
        assert list(owner) == ["arc_agi_3"]
        assert self.adapter is not None  # resources_servers/arc_agi_3/task_data.py, which collation loads for them


def test_two_versions_of_a_game_on_the_real_game_server(tmp_path: Path) -> None:
    # Two versions of one name (as in the synthetic sets): the committed ls20 game as ls20-9607627b and ls20-0000beef.
    # Asked by full id, the server gives and starts exactly that version.
    import arc_agi
    import requests

    source = HERE.parent / "data" / "environment_files" / "ls20" / "9607627b"
    for version in ("9607627b", "0000beef"):
        game_dir = tmp_path / "games" / "ls20" / version
        game_dir.mkdir(parents=True)
        (game_dir / "ls20.py").write_bytes((source / "ls20.py").read_bytes())
        meta = json.loads((source / "metadata.json").read_text())
        (game_dir / "metadata.json").write_text(json.dumps({**meta, "game_id": f"ls20-{version}"}))
    server = make_server(tmp_path / "games", game_server_dir=str(tmp_path / "gs"))
    server.game_server.start(2)
    try:
        url, key = server.game_server.url, "versions-key"
        os.environ["ARC_API_KEY"] = key
        client = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.ONLINE, arc_base_url=url, logger=logging.getLogger("test_client")
        )
        for full_game_id in ("ls20-0000beef", "ls20-9607627b"):
            assert requests.get(f"{url}/api/games/{full_game_id}", timeout=5).json()["game_id"] == full_game_id
            card = client.create_scorecard()
            started = requests.post(
                f"{url}/api/cmd/RESET",
                json={"game_id": full_game_id, "card_id": card},
                headers={"X-Api-Key": key},
                timeout=30,
            ).json()
            assert started["game_id"] == full_game_id and started["guid"]
    finally:
        server.game_server.stop()


@needs_games
class TestRealGameServer:
    """The official game server on the real public25 files: play over HTTP, close, verify."""

    @pytest.fixture
    def server(self, tmp_path: Path):
        server = make_server(REAL_ENVIRONMENTS, game_server_dir=str(tmp_path / "gs"))
        server.game_server.start(len(load_environment_metadata(str(REAL_ENVIRONMENTS))))
        yield server
        server.game_server.stop()

    def play(self, server, game: str, actions, key: str = "rollout-key"):
        """Play ``actions`` through the server as a harness would (an ONLINE Arcade); return (card, guid, record)."""
        import datetime as dt

        import arc_agi
        from arcengine import GameAction

        os.environ["ARC_API_KEY"] = key
        client = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.ONLINE,
            arc_base_url=server.game_server.url,
            logger=logging.getLogger("test_client"),
        )
        card = client.create_scorecard()
        started = dt.datetime.now(dt.timezone.utc)
        env = client.make(loaded_metadata(REAL_ENVIRONMENTS, game)["game_id"], scorecard_id=card)
        state = env.observation_space
        apl = [0] * state.win_levels
        for name in actions:
            level = state.levels_completed
            state = env.step(GameAction[name], data={})
            apl[level] += 1
        meta = loaded_metadata(REAL_ENVIRONMENTS, game)
        record = {
            "game_id": meta["game_id"],
            "number_of_levels": len(meta["baseline_actions"]),
            "base_actions_per_level": meta["baseline_actions"],
            "actions_per_level": apl,
            "levels_completed": state.levels_completed,
            "state": "gave_up",
            "actions": len(actions),
            "started_at": started.replace(tzinfo=None).isoformat(),
        }
        return client, card, state.guid, record, started

    def request(self, card, guid, record, started, key="rollout-key", **extra):
        return verify_request(
            REAL_ENVIRONMENTS,
            "ls20",
            record,
            scorecard_id=card,
            game_guid=guid,
            scorecard_api_key=key,
            benchmark_start_time=started.isoformat(),
            **extra,
        )

    def test_play_close_verify_and_tampering(self, server) -> None:
        client, card, guid, record, started = self.play(
            server, "ls20", ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET", "ACTION1"]
        )
        result = asyncio.run(server.verify(self.request(card, guid, record, started)))
        assert result.actions == 6 and result.scorecard_runs == 1 and result.result_valid
        assert (
            result.timing_valid and 0 <= result.clock_offset_seconds < 5 and getattr(result, "rhae_1h") == result.rhae
        )
        assert (server.game_server.out_dir / "cards" / f"{card}.json").is_file()  # closed by the verifier
        with pytest.raises(ValueError, match="levels 0 != the agent's 1"):
            asyncio.run(server.verify(self.request(card, guid, dict(record, levels_completed=1), started)))
        with pytest.raises(ValueError, match="state NOT_FINISHED != the agent's 'won'"):
            asyncio.run(server.verify(self.request(card, guid, dict(record, state="won"), started)))

    def test_card_closed_by_its_harness_is_read_from_the_server_copy(self, server) -> None:
        client, card, guid, record, started = self.play(server, "ls20", ["ACTION1", "ACTION2"], key="other-key")
        client.close_scorecard(card)  # as taaf's finish_game does
        result = asyncio.run(server.verify(self.request(card, guid, record, started, key="other-key")))
        assert result.actions == 2 and result.result_valid

    def test_clients_see_the_human_baselines(self, server) -> None:
        # arc_agi's server leaves them out; the kit's records and upstream's per-level budgets need them.
        import arc_agi
        import requests

        meta = loaded_metadata(REAL_ENVIRONMENTS, "ls20")
        info = requests.get(f"{server.game_server.url}/api/games/ls20", timeout=5).json()
        listed = requests.get(f"{server.game_server.url}/api/games", timeout=5).json()
        assert info["baseline_actions"] == meta["baseline_actions"]
        assert all(game["baseline_actions"] for game in listed) and len(listed) == 25
        assert "private_tags" not in info and "level_tags" not in info
        assert requests.get(f"{server.game_server.url}/api/games/nope", timeout=5).status_code == 404
        os.environ["ARC_API_KEY"] = "baseline-key"
        client = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.ONLINE,
            arc_base_url=server.game_server.url,
            logger=logging.getLogger("test_client"),
        )
        env = client.make(meta["game_id"], scorecard_id=client.create_scorecard())
        assert list(env.environment_info.baseline_actions) == meta["baseline_actions"]

    def test_game_server_endpoint_and_errors(self, server, tmp_path: Path) -> None:
        info = asyncio.run(server.game_server_info())
        assert info["url"] == server.game_server.url and info["games"] == 25
        written = json.loads((server.game_server.out_dir / "game_server.json").read_text())
        assert written["url"] == server.game_server.url
        assert server.game_server.url in (server.game_server.out_dir / "game_server.log").read_text()
        with pytest.raises(ValueError, match="unknown"):
            server.game_server.close_card("no-such-card", "k")
        with pytest.raises(FileNotFoundError):
            server.game_server.recording("no-such-card", "ls20-9607627b", "g")
        server.game_server.stop()
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            asyncio.run(server.game_server_info())
        wrong = GameServer(str(REAL_ENVIRONMENTS), tmp_path / "wrong")
        with pytest.raises(RuntimeError, match="expected 3"):
            wrong.start(expected_games=3)
        assert not wrong.running  # stopped, not left running


def test_game_count_mismatch_stops_the_server(tmp_path: Path) -> None:
    """On the two committed example games: a start that expects another count raises and leaves no server running."""
    server = GameServer(str(HERE.parent / "data" / "environment_files"), tmp_path / "gs")
    with pytest.raises(RuntimeError, match="serves 2 games, expected 25"):
        server.start(expected_games=25)
    pid = json.loads((server.out_dir / "game_server.json").read_text())["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


class TestEnsureFlask:
    def test_present(self) -> None:
        from resources_servers.arc_agi_3 import setup_arc_agi

        setup_arc_agi.ensure_flask()

    def test_missing_without_pip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from resources_servers.arc_agi_3 import setup_arc_agi

        monkeypatch.setattr(setup_arc_agi.importlib.util, "find_spec", lambda name: None)
        with pytest.raises(RuntimeError, match="no pip"):
            setup_arc_agi.ensure_flask()


class TestDatasetCli:
    def test_main_writes_and_checks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        games = tmp_path / "games"
        monkeypatch.setattr(create_dataset, "PUBLIC25_VERSIONS", {"aa11": "v1", "bb22": "v2"})
        monkeypatch.setattr(create_dataset, "EXAMPLE_GAMES", ("bb22",))
        write_fake_game(games, "aa11", "v1", [3])
        write_fake_game(games, "bb22", "v2", [4, 5])
        out, example = tmp_path / "p.jsonl", tmp_path / "e.jsonl"
        create_dataset.main(["--environments-dir", str(games), "--output", str(out), "--example-output", str(example)])
        assert len(out.read_text().splitlines()) == 2 and len(example.read_text().splitlines()) == 1
        create_dataset.main(["--environments-dir", str(games), "--check", str(out)])
        assert "matches" in capsys.readouterr().out

    def test_kaggle_download(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAGGLE_API_TOKEN", "test-token")
        pages = {
            "": {
                "files": [{"name": "environment_files/aa11/v1/metadata.json"}, {"name": "other.csv"}],
                "nextPageToken": "p2",
            },
            "p2": {"files": [{"name": "environment_files/aa11/v1/aa11.py"}]},
        }
        seen_tokens = []

        def fake_get(url: str, token: str) -> bytes:
            seen_tokens.append(token)
            if "/data/list/" in url:
                return json.dumps(pages["p2" if "pageToken=p2" in url else ""]).encode()
            return b"{}" if "metadata" in url else b"# game\n"

        monkeypatch.setattr(create_dataset, "_kaggle_get", fake_get)
        root = create_dataset.download_competition_environment_files(tmp_path)
        assert root == tmp_path / "environment_files"
        assert (root / "aa11" / "v1" / "aa11.py").read_text() == "# game\n"
        assert not (tmp_path / "other.csv").exists() and set(seen_tokens) == {"test-token"}

    def test_kaggle_token_never_follows_a_redirect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Kaggle redirects a file download to another origin (cloud storage), which must not receive the token."""
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:  # /api/... redirects to /bucket/... on another origin (localhost, not 127.0.0.1)
                seen.append((self.path.split("/")[1], self.headers.get("Authorization")))
                redirect = self.path.startswith("/api/")
                self.send_response(302 if redirect else 200)
                if redirect:
                    self.send_header("Location", f"http://localhost:{self.server.server_address[1]}/bucket/f")
                self.send_header("Content-Length", "0" if redirect else "7")
                self.end_headers()
                if not redirect:
                    self.wfile.write(b"payload")

        for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(create_dataset.urllib.request, "_opener", None)  # rebuilt without proxies
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            kaggle_api = f"http://127.0.0.1:{server.server_address[1]}/api/v1"
            monkeypatch.setattr(create_dataset, "KAGGLE_API", kaggle_api)
            url = f"{kaggle_api}/competitions/data/download/c/f"
            assert create_dataset._kaggle_get(url, "dummy-token") == b"payload"
        finally:
            server.shutdown()
            server.server_close()
        assert seen == [("api", "Bearer dummy-token"), ("bucket", None)]
        with pytest.raises(ValueError, match="refusing"):
            create_dataset._kaggle_get("https://storage.example/x", "dummy-token")

    def test_kaggle_token_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.setattr(create_dataset.Path, "home", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="KAGGLE_API_TOKEN"):
            create_dataset._kaggle_token()


class TestEnsureFlaskInstall:
    def test_installs_with_pip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from resources_servers.arc_agi_3 import setup_arc_agi

        specs = {"flask": None, "pip": object()}
        monkeypatch.setattr(setup_arc_agi.importlib.util, "find_spec", lambda name: specs.get(name))
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setattr(setup_arc_agi.subprocess, "run", fake_run)
        monkeypatch.setattr(setup_arc_agi.importlib, "import_module", lambda name: None)
        setup_arc_agi.ensure_flask()
        assert commands[0][1:4] == ["-m", "pip", "install"] and "flask==3.1.3" in commands[0]

    def test_pip_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from resources_servers.arc_agi_3 import setup_arc_agi

        monkeypatch.setattr(
            setup_arc_agi.importlib.util, "find_spec", lambda name: object() if name == "pip" else None
        )
        monkeypatch.setattr(
            setup_arc_agi.subprocess, "run", lambda cmd, **kw: MagicMock(returncode=1, stderr="no net")
        )
        with pytest.raises(RuntimeError, match="no net"):
            setup_arc_agi.ensure_flask()
