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
"""Worker lifecycle with small stand-in workers (no harness, no game server): reports, a dead worker, a harness error,
a worker that outlives the play window, and shutdown."""

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import AgentContext
from responses_api_agents.arc_agi_3_official_agent import app as official_app
from responses_api_agents.arc_agi_3_official_agent.app import ArcAgi3OfficialAgent

from .test_official_game import write_game
from .test_official_unit import make_agent


# A stand-in worker: like worker.py, it writes result.json when it ends, also on SIGTERM.
WORKER = """
import json, os, signal, sys, time
settings = json.load(open(sys.argv[sys.argv.index("--settings") + 1]))
report = {"guid": "guid-1", "actions": 4, "levels_completed": 0, "state": "NOT_FINISHED", "exit_reason": None,
          "error": None, "settings": settings}
def finish(*_):
    report.setdefault("stopped", "SIGTERM")
    json.dump(report, open("result.json", "w"))
    sys.exit(0)
signal.signal(signal.SIGTERM, finish)
MODE
"""


def agent_with_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, **overrides) -> ArcAgi3OfficialAgent:
    worker = tmp_path / "fake_worker.py"
    worker.write_text(WORKER.replace("MODE", mode))
    agent = make_agent(
        monkeypatch,
        tmp_path,
        environments_dir=str(tmp_path / "games"),
        output_dir=str(tmp_path / "out"),
        run_name="life",
        cancel_drain_s=5,
        **overrides,
    )
    monkeypatch.setattr(official_app, "WORKER", worker)
    monkeypatch.setattr(official_app, "venv_python", lambda root: Path(sys.executable))
    monkeypatch.setattr(
        ArcAgi3OfficialAgent,
        "model_server_config",
        lambda self: {"uses_reasoning_parser": True, "propagate_context_overflow_errors": True},
    )

    async def open_card(self, key):
        return f"card-{key}"

    async def close_card(self, card_id):
        self._closed_cards.append(card_id)

    monkeypatch.setattr(ArcAgi3OfficialAgent, "_open_card", open_card)
    monkeypatch.setattr(ArcAgi3OfficialAgent, "_close_card", close_card)
    agent._closed_cards = []
    return agent


@pytest.fixture
def meta(tmp_path: Path) -> dict:
    return write_game(tmp_path / "games")


def test_a_finished_worker_report_becomes_the_record(monkeypatch, tmp_path: Path, meta: dict) -> None:
    mode = 'report.update(exit_reason="ACTION_BUDGET", action_budget="level"); json.dump(report, open("result.json", "w"))'
    agent = agent_with_worker(monkeypatch, tmp_path, mode)
    result = asyncio.run(agent.play_game(meta, 0))
    record = result["record"]
    assert result["scorecard_id"] == "card-0-ls20-v1" and result["guid"] == "guid-1"
    assert record["state"] == "gave_up" and record["solver_note"] == "harness exit: ACTION_BUDGET"
    assert record["stop_reason"] == "level_action_budget"
    assert record["actions"] == 4 and record["levels_completed"] == 0 and record["base_actions_per_level"] == [3]
    settings = json.loads((agent._run.run_dir / "games" / "0-ls20-v1" / "result.json").read_text())["settings"]
    assert settings["card_id"] == "card-0-ls20-v1" and settings["game_id"] == "ls20-v1"
    assert settings["game_server_url"] == "http://game-server.invalid"
    assert settings["benchmark_config"]["client"]["base_url"].endswith("/official/0-ls20-v1/v1")
    assert settings["game_api_key"] == agent._game_api_key and settings["model_api_key"] == "gym-local"
    assert settings["reset_timeout_s"] == 120.0
    assert agent._attempts == {} and result["key"] == "0-ls20-v1" and result["call_log"].calls == []
    # A repeat of the row (same dispatch index) gets its own route, game directory and scorecard.
    again = asyncio.run(agent.play_game(meta, 0))
    assert again["key"] == "0-ls20-v1-p1" and again["scorecard_id"] == "card-0-ls20-v1-p1"
    settings = json.loads((agent._run.run_dir / "games" / "0-ls20-v1-p1" / "result.json").read_text())["settings"]
    assert settings["benchmark_config"]["client"]["base_url"].endswith("/official/0-ls20-v1-p1/v1")
    assert agent._closed_cards == []  # reported games hand their cards to the verifier, which closes them


def test_a_harness_error_crashes_the_game_with_its_score(monkeypatch, tmp_path: Path, meta: dict) -> None:
    mode = 'report.update(error="API_ERROR: RuntimeError: boom"); json.dump(report, open("result.json", "w"))'
    result = asyncio.run(agent_with_worker(monkeypatch, tmp_path, mode).play_game(meta, 0))
    assert result["record"]["state"] == "crashed" and result["solver_note"] == "API_ERROR: RuntimeError: boom"
    assert result["record"]["stop_reason"] == "crashed"


def test_a_worker_dead_without_a_report_fails_loudly(monkeypatch, tmp_path: Path, meta: dict) -> None:
    mode = "sys.stderr.write('segfault in the harness'); sys.stderr.flush(); os._exit(3)"
    agent = agent_with_worker(monkeypatch, tmp_path, mode)
    with pytest.raises(RuntimeError, match=r"exited \(3\) without a game report.*segfault"):
        asyncio.run(agent.play_game(meta, 0))
    assert agent._closed_cards == ["card-0-ls20-v1"] and not agent._workers and not agent._attempts


def test_a_worker_that_cannot_launch_releases_its_card(monkeypatch, tmp_path: Path, meta: dict) -> None:
    agent = agent_with_worker(monkeypatch, tmp_path, "pass")
    monkeypatch.setattr(official_app, "venv_python", lambda root: tmp_path / "no-such-python")
    with pytest.raises(FileNotFoundError):
        asyncio.run(agent.play_game(meta, 0))
    assert (
        agent._closed_cards == ["card-0-ls20-v1"] and not agent._workers and not agent._attempts and not agent._in_play
    )


def test_a_settings_failure_releases_its_card(monkeypatch, tmp_path: Path, meta: dict) -> None:
    agent = agent_with_worker(monkeypatch, tmp_path, "pass")

    def broken_settings(*_):
        raise OSError("disk full")

    monkeypatch.setattr(ArcAgi3OfficialAgent, "_worker_settings", broken_settings)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(agent.play_game(meta, 0))
    assert agent._closed_cards == ["card-0-ls20-v1"]


def test_a_worker_outliving_the_window_is_stopped(monkeypatch, tmp_path: Path, meta: dict) -> None:
    agent = agent_with_worker(monkeypatch, tmp_path, "time.sleep(600)", play_seconds=1.0, window_grace_s=1.0)
    started = time.monotonic()
    result = asyncio.run(agent.play_game(meta, 0))
    assert time.monotonic() - started < 15
    assert result["record"]["state"] == "gave_up" and "after the play window" in result["record"]["solver_note"]
    assert result["record"]["stop_reason"] == "play_window"


def test_shutdown_cancels_games_in_play(monkeypatch, tmp_path: Path, meta: dict) -> None:
    agent = agent_with_worker(monkeypatch, tmp_path, "time.sleep(600)")

    async def main():
        task = asyncio.ensure_future(agent.play_game(meta, 0))
        await asyncio.sleep(2)
        assert len(agent._workers) == 1
        await agent.shutdown()
        return await task

    result = asyncio.run(main())
    assert result["record"]["state"] == "cancelled"
    with pytest.raises(RuntimeError, match="shutting down"):
        asyncio.run(agent._ensure_run())


def test_cancelling_the_run_stops_the_worker(monkeypatch, tmp_path: Path, meta: dict) -> None:
    agent = agent_with_worker(monkeypatch, tmp_path, "time.sleep(600)")

    async def main():
        task = asyncio.ensure_future(agent.play_game(meta, 0))
        await asyncio.sleep(2)
        process = next(iter(agent._workers))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return process

    process = asyncio.run(main())
    assert process.returncode is not None and agent._attempts == {}
    # the worker is stopped first, then the card closed; the cancellation still propagates
    assert agent._closed_cards == ["card-0-ls20-v1"] and not agent._workers


def test_shutdown_refuses_a_game_still_opening_its_card(monkeypatch, tmp_path: Path, meta: dict) -> None:
    agent = agent_with_worker(monkeypatch, tmp_path, "time.sleep(600)")

    async def main():
        opening, release = asyncio.Event(), asyncio.Event()

        async def open_card(self, key):
            opening.set()
            await release.wait()
            return f"card-{key}"

        monkeypatch.setattr(ArcAgi3OfficialAgent, "_open_card", open_card)
        game = asyncio.ensure_future(agent.play_game(meta, 0))
        await opening.wait()
        shutdown = asyncio.ensure_future(agent.shutdown())
        await asyncio.sleep(0.2)
        assert not shutdown.done()  # it waits for the launch in flight
        release.set()
        await asyncio.wait_for(shutdown, 10)
        with pytest.raises(RuntimeError, match="shutting down"):
            await game

    asyncio.run(main())
    assert agent._closed_cards == ["card-0-ls20-v1"]
    assert not agent._workers and not agent._attempts and not agent._in_play


def test_shutdown_stops_a_worker_still_launching(monkeypatch, tmp_path: Path, meta: dict) -> None:
    agent = agent_with_worker(monkeypatch, tmp_path, "time.sleep(600)")
    launched = []

    async def main():
        launching, release = asyncio.Event(), asyncio.Event()
        create = asyncio.create_subprocess_exec

        async def slow_create(*args, **kwargs):
            launching.set()
            await release.wait()
            process = await create(*args, **kwargs)
            launched.append(process)
            return process

        monkeypatch.setattr(official_app.asyncio, "create_subprocess_exec", slow_create)
        game = asyncio.ensure_future(agent.play_game(meta, 0))
        await launching.wait()
        shutdown = asyncio.ensure_future(agent.shutdown())
        await asyncio.sleep(0.2)
        assert not shutdown.done()  # it waits for the launch in flight
        release.set()
        await asyncio.wait_for(shutdown, 15)
        assert game.done()  # the game ended before shutdown returned
        return game.exception() or game.result()

    outcome = asyncio.run(main())
    assert launched[0].returncode is not None and not agent._workers and not agent._attempts and not agent._in_play
    # A worker stopped before it could install its SIGTERM handler leaves no report: the attempt fails and its card is
    # closed. Otherwise it reports, and the game ends as cancelled with its card for the verifier.
    if isinstance(outcome, RuntimeError):
        assert "without a game report" in str(outcome) and agent._closed_cards == ["card-0-ls20-v1"]
    else:
        assert outcome["record"]["state"] == "cancelled" and agent._closed_cards == []


def test_per_game_clock_gives_a_later_game_its_own_window(monkeypatch, tmp_path: Path, meta: dict) -> None:
    """run_clock per_game: a game that starts after the first game's window ended still gets play_seconds, from its
    own start (the shared clock refuses it: test_no_game_after_the_window)."""
    finish = 'json.dump(report, open("result.json", "w"))'
    agent = agent_with_worker(monkeypatch, tmp_path, finish, play_seconds=1.0, run_clock="per_game")
    first = asyncio.run(agent.play_game(meta, 0))
    time.sleep(1.2)  # past the first game's window
    before = time.time()
    second = asyncio.run(agent.play_game(meta, 1))
    settings = json.loads((agent._run.run_dir / "games" / second["key"] / "result.json").read_text())["settings"]
    assert before + 0.5 < settings["deadline_epoch"] <= time.time() + 1.0  # its own window, not the first's
    assert first["clock_origin"] < second["clock_origin"]  # RHAE@T from each game's own start
    assert agent._run.start_time_utc is None  # no shared run clock


def test_no_game_after_the_window(monkeypatch, tmp_path: Path, meta: dict) -> None:
    from responses_api_agents.arc_agi_3_agent.harness_runtime import GameStartError

    agent = agent_with_worker(monkeypatch, tmp_path, "pass", play_seconds=0.01)
    asyncio.run(agent._ensure_run())
    agent._run.start_clock()
    time.sleep(0.05)
    with pytest.raises(GameStartError, match="window has ended"):
        asyncio.run(agent.play_game(meta, 0))


def test_close_after_an_abandoned_activation_releases_worker_and_card(monkeypatch, tmp_path: Path, meta: dict) -> None:
    """The session contract with a real worker: a repeated seed starts nothing, and the close of a session whose
    activation the environment server gave up on stops the worker and closes the card."""
    from .test_official_unit import GAME, PARAMS, http_request

    agent = agent_with_worker(monkeypatch, tmp_path, "time.sleep(600)")
    game = {**meta, "dispatch_index": GAME["dispatch_index"]}

    async def main():
        session: dict = {}
        episode = EpisodeId(rollout_id="9-1")
        seed = AgentSeedSessionRequest(
            agent_session_id="agent-1",
            episode_id=episode,
            task_id=TaskId(taskset="arc_agi_3", task_id="ls20"),
            agent_context=AgentContext(values={"game": game}),
        )
        await agent.seed_agent_session(http_request(session), seed)
        await agent.seed_agent_session(http_request(session), seed)
        activation = asyncio.ensure_future(agent.responses(http_request(session), PARAMS))
        for _ in range(500):
            if agent._workers:
                break
            await asyncio.sleep(0.01)
        (process,) = agent._workers
        activation.cancel()
        closed = await agent.close_agent_session(
            http_request(session), AgentCloseSessionRequest(agent_session_id="agent-1", episode_id=episode)
        )
        return process, closed

    process, closed = asyncio.run(main())
    assert process.returncode is not None and not agent._workers and not agent._attempts
    assert agent._closed_cards == ["card-0-ls20-v1"] and closed.cleanup_confirmed
