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
"""HarnessRuntime lifecycle with fake games and solver (no harness needed): saves, pins, cancellation, shutdown."""

import asyncio
import datetime as dt
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.arc_agi_3_agent import harness_runtime
from responses_api_agents.arc_agi_3_agent.harness_runtime import (
    GameStartError,
    HarnessRuntime,
    atomic_write_json,
)


class FakeRun:
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.state = "playing"
        self.final_score = None
        self.started_at = dt.datetime(2026, 11, 4, 12, 0, 0)
        self.history = []

    def to_json_dict(self):
        return {
            "game_id": self.game_id,
            "state": self.state,
            "history": list(self.history),
            "started_at": self.started_at.isoformat(),
            "solver_note": None,
        }


class FakeGame:
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.game_run = FakeRun(game_id)
        self._scorecard_id = "card-x"
        self.current_state = SimpleNamespace(raw=SimpleNamespace(guid="guid-x"))

    def finish_game(self):
        if self.game_run.state == "playing":
            self.game_run.state = "gave_up"
        self.game_run.final_score = 0.0


class FakeTransport:
    def __init__(self):
        self.cancelled, self.all_cancelled, self.bound = [], False, []

    def bind_game(self, key, progress=None):
        self.bound.append(key)
        self.progress = progress

    def cancel(self, key):
        self.cancelled.append(key)

    def cancel_all(self):
        self.all_cancelled = True


def write_game(root: Path, game="ls20", version="v1") -> dict:
    game_dir = root / game / version
    game_dir.mkdir(parents=True)
    (game_dir / "metadata.json").write_text("{}")
    (game_dir / f"{game}.py").write_text("# game\n")
    sha = {f"{game}.py": hashlib.sha256(b"# game\n").hexdigest()}
    return {"game_id": game, "full_game_id": f"{game}-{version}", "version": version, "files_sha256": sha}


def make_runtime(tmp_path: Path, play, play_seconds=100.0, save_every_s=0.0) -> HarnessRuntime:
    runtime = HarnessRuntime(
        install_root=tmp_path / "harness",
        environments_dir=tmp_path / "games",
        run_dir=tmp_path / "run",
        env={},
        play_seconds=play_seconds,
        concurrency=2,
        num_attempts=2,
        save_request_logs=False,
        mem_lean=False,
        telemetry=False,
        seed=None,
        save_every_s=save_every_s,
        game_server_url="http://game-server.invalid",
        api_key="key-x",
        cancel_drain_s=5.0,
    )
    runtime.run_dir.mkdir(parents=True)
    runtime.solver = SimpleNamespace(
        _worker_pool=ThreadPoolExecutor(max_workers=2),
        _play_one=play,
        _stop_event=threading.Event(),
        soft_end_time=None,
        teardown_calls=0,
    )
    runtime.solver._teardown = lambda: setattr(runtime.solver, "teardown_calls", runtime.solver.teardown_calls + 1)
    runtime._session = object()

    def start_game(full_game_id):
        game = FakeGame(full_game_id)
        with runtime._games_lock:
            if runtime._start_time is None:
                runtime._start_time = game.game_run.started_at
                runtime.solver.soft_end_time = runtime._start_time + dt.timedelta(seconds=play_seconds)
            runtime._games.append(game)
        return game

    runtime._start_game = start_game
    return runtime


def quick_play(game, index, pass_index, local_server):
    game.game_run.history.append({"wallclock_seconds": 1.0})


class TestSaves:
    """F1."""

    def test_concurrent_atomic_writes_never_collide(self, tmp_path: Path) -> None:
        path = tmp_path / "benchmark.json"
        errors = []

        def write(i):
            try:
                for j in range(50):
                    atomic_write_json(path, {"writer": i, "j": j, "pad": "x" * 2000})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors and json.loads(path.read_text())["j"] == 49
        assert not list(tmp_path.glob("*.tmp"))

    def test_failed_final_save_keeps_the_completed_record(self, tmp_path: Path, monkeypatch) -> None:
        meta = write_game(tmp_path / "games")
        runtime = make_runtime(tmp_path, quick_play)

        def broken_save():
            raise OSError("No space left on device")

        monkeypatch.setattr(runtime, "save_benchmark_json", broken_save)
        attempt = asyncio.run(runtime.play_game(meta, 0, FakeTransport()))
        assert attempt.record["state"] == "gave_up" and len(attempt.record["history"]) == 1
        assert attempt.persistence_error == "OSError: No space left on device" and attempt.play_error is None

    def test_saver_survives_failures(self, tmp_path: Path, monkeypatch) -> None:
        runtime = make_runtime(tmp_path, quick_play, save_every_s=0.01)
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("transient")

        monkeypatch.setattr(runtime, "save_benchmark_json", flaky)

        async def main():
            runtime._ensure_saver()
            await asyncio.sleep(0.2)
            runtime._saver.cancel()

        asyncio.run(main())
        assert runtime.save_failures == 1 and runtime.last_save_error == "OSError: transient" and len(calls) > 2

    def test_snapshot_carries_run_config_digest(self, tmp_path: Path) -> None:
        runtime = make_runtime(tmp_path, quick_play)
        digest = runtime.write_run_config({"agent": {"api_key": "REDACTED"}})
        runtime.save_benchmark_json()
        bench = json.loads((runtime.run_dir / "benchmark.json").read_text())
        assert bench["run_config_sha256"] == digest
        assert hashlib.sha256((runtime.run_dir / "run_config.json").read_bytes()).hexdigest() == digest


class TestPins:
    """F5 (agent side): the files to be played must be the dataset's pinned version."""

    def test_good_pins(self, tmp_path: Path) -> None:
        meta = write_game(tmp_path / "games")
        assert make_runtime(tmp_path, quick_play).check_game_pins(meta) == "ls20-v1"

    @pytest.mark.parametrize(
        "change",
        [{"version": "v2"}, {"files_sha256": {"ls20.py": "0" * 64}}, {"full_game_id": "ls20-v9"}, {"version": None}],
    )
    def test_bad_pins_refuse_to_start(self, tmp_path: Path, change) -> None:
        meta = dict(write_game(tmp_path / "games"), **change)
        with pytest.raises(GameStartError):
            asyncio.run(make_runtime(tmp_path, quick_play).play_game(meta, 0, FakeTransport()))


class TestClockAndCancel:
    def test_shared_run_deadline(self, tmp_path: Path) -> None:
        # F7: the first start fixes the run clock and the play deadline for every game.
        meta = write_game(tmp_path / "games")
        runtime = make_runtime(tmp_path, quick_play, play_seconds=7200.0)
        attempt = asyncio.run(runtime.play_game(meta, 0, FakeTransport()))
        assert runtime.start_time == dt.datetime(2026, 11, 4, 12, 0, 0)
        assert runtime.solver.soft_end_time == dt.datetime(2026, 11, 4, 14, 0, 0)
        # The verifier gets the origin in UTC, and the attempt's scorecard and game run guid.
        assert dt.datetime.fromisoformat(runtime.start_time_utc).utcoffset() == dt.timedelta(0)
        assert attempt.scorecard_id == "card-x" and attempt.guid == "guid-x"

    def test_real_start_plays_through_the_game_server_and_stamps_the_origin_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The kit's taaf GameAPI plays through the official game server (ONLINE), and the run clock's origin is taken
        # before the first game's start (its server RESET), so every game's first server event is after it.
        import sys
        import types

        specs, games = [], []

        class GameAPI(FakeGame):
            def __init__(self, env_name, arcade_spec):
                super().__init__(env_name)
                specs.append(arcade_spec)
                games.append(self)

            def start_game(self, session):
                time.sleep(0.02)
                self.game_run.started_at = dt.datetime.now()

        fake_taaf = types.ModuleType("taaf.game_api")
        fake_taaf.ArcadeSpec = lambda **kwargs: kwargs
        fake_taaf.GameAPI = GameAPI
        parent = types.ModuleType("taaf")
        parent.game_api = fake_taaf
        monkeypatch.setitem(sys.modules, "taaf", parent)
        monkeypatch.setitem(sys.modules, "taaf.game_api", fake_taaf)
        runtime = make_runtime(tmp_path, quick_play, play_seconds=60.0)
        before = dt.datetime.now()
        game = harness_runtime.HarnessRuntime._start_game(runtime, "ls20-v1")
        import arc_agi

        assert specs == [
            {
                "operation_mode": arc_agi.OperationMode.ONLINE,
                "arc_base_url": "http://game-server.invalid",
                "environments_dir": str(tmp_path / "games"),
            }
        ]
        assert before <= runtime.start_time < game.game_run.started_at
        assert runtime.solver.soft_end_time == runtime.start_time + dt.timedelta(seconds=60)
        harness_runtime.HarnessRuntime._start_game(runtime, "ls20-v1")
        assert runtime.start_time < games[1].game_run.started_at and len(runtime._games) == 2

    def test_cancelling_one_attempt_stops_only_that_game(self, tmp_path: Path) -> None:
        # F6: the cancelled attempt ends as cancelled and its worker exits; the shared stop event is not set.
        meta = write_game(tmp_path / "games")
        started = threading.Event()

        def long_play(game, index, pass_index, local_server):
            started.set()
            while game.game_run.state == "playing":
                time.sleep(0.01)

        runtime = make_runtime(tmp_path, long_play)
        transport = FakeTransport()

        async def main():
            task = asyncio.ensure_future(runtime.play_game(meta, 3, transport))
            await asyncio.to_thread(started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(main())
        game = runtime._games[0]
        assert game.game_run.state == "cancelled" and transport.cancelled == ["3:ls20-v1:1"]
        assert not runtime.solver._stop_event.is_set() and not runtime._inflight
        assert json.loads((runtime.run_dir / "benchmark.json").read_text())["game_runs"][0]["state"] == "cancelled"

    def test_shutdown_stops_drains_and_closes(self, tmp_path: Path) -> None:
        meta = write_game(tmp_path / "games")
        started = threading.Event()

        def until_stopped(game, index, pass_index, local_server):
            started.set()
            while not runtime.solver._stop_event.is_set():
                time.sleep(0.01)

        runtime = make_runtime(tmp_path, until_stopped, save_every_s=60.0)
        transport = FakeTransport()

        async def main():
            task = asyncio.ensure_future(runtime.play_game(meta, 0, transport))
            await asyncio.to_thread(started.wait, 5)
            await runtime.shutdown(transport)
            attempt = await task
            return attempt

        attempt = asyncio.run(main())
        assert runtime.closed and runtime.solver._stop_event.is_set() and transport.all_cancelled
        assert attempt.record["state"] == "cancelled" and runtime.solver.teardown_calls == 1
        assert runtime._saver.cancelled()
        with pytest.raises(GameStartError, match="shutting down"):
            harness_runtime.HarnessRuntime._start_game(runtime, "ls20-v1")

    def test_play_exception_keeps_the_record(self, tmp_path: Path) -> None:
        meta = write_game(tmp_path / "games")

        def broken(game, index, pass_index, local_server):
            raise RuntimeError("sandbox died")

        attempt = asyncio.run(make_runtime(tmp_path, broken).play_game(meta, 0, FakeTransport()))
        assert attempt.play_error == "RuntimeError: sandbox died" and attempt.record["game_id"] == "ls20-v1"

    def test_play_binds_the_games_progress(self, tmp_path: Path) -> None:
        # Each model call records the game's step (actions taken + 1) and level (levels completed + 1) when sent.
        meta = write_game(tmp_path / "games")
        seen = []

        def play(game, index, pass_index, local_server):
            game.current_state.levels_completed = 1
            seen.append(transport.progress())
            game.game_run.history.append({"wallclock_seconds": 1.0})
            seen.append(transport.progress())

        transport = FakeTransport()
        asyncio.run(make_runtime(tmp_path, play).play_game(meta, 0, transport))
        assert seen == [(1, 2), (2, 2)]


def test_cancel_during_start_finishes_the_game_as_cancelled(tmp_path: Path) -> None:
    meta = write_game(tmp_path / "games")
    runtime = make_runtime(tmp_path, quick_play)
    release = threading.Event()
    original_start = runtime._start_game

    def slow_start(full_game_id):
        release.wait(5)
        return original_start(full_game_id)

    runtime._start_game = slow_start

    async def main():
        task = asyncio.ensure_future(runtime.play_game(meta, 0, FakeTransport()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        for _ in range(100):
            if runtime._games:
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.05)

    asyncio.run(main())
    assert runtime._games and runtime._games[0].game_run.state == "cancelled"


class TestInheritedEnv:
    """The configuration alone sets the harness: inherited harness settings it does not set are removed."""

    def test_stray_settings_are_removed_and_named(self, tmp_path: Path, monkeypatch) -> None:
        runtime = make_runtime(tmp_path, quick_play)
        runtime.env = {"LOCAL_ANALYZER_TEMPERATURE": "0.7", "ARC3_CONTEXT_KEEP_TOKENS": "24576"}
        # Start from no harness settings (an earlier in-process harness start or the shell may have set some);
        # monkeypatch restores them after the test.
        for name in harness_runtime.inherited_harness_env(dict(os.environ), {}) + ["ARC3_CONTEXT_KEEP_TOKENS"]:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("LOCAL_ANALYZER_SEED", "1234")  # the fixed-request-seed trap
        monkeypatch.setenv("ARC3_TURN_COMPACTION", "0")
        monkeypatch.setenv("LOCAL_ANALYZER_TEMPERATURE", "1.5")
        monkeypatch.setenv("ARC3_SMOKE_PLAY_SECONDS", "45")
        monkeypatch.setenv("ARC3_ENVIRONMENTS_DIR", "/games")
        monkeypatch.setenv("UNRELATED_SETTING", "x")
        cleared = runtime.export_env()
        assert cleared == ["ARC3_TURN_COMPACTION", "LOCAL_ANALYZER_SEED"]
        assert "LOCAL_ANALYZER_SEED" not in os.environ and "ARC3_TURN_COMPACTION" not in os.environ
        assert os.environ["LOCAL_ANALYZER_TEMPERATURE"] == "0.7"
        assert os.environ["ARC3_SMOKE_PLAY_SECONDS"] == "45" and os.environ["ARC3_ENVIRONMENTS_DIR"] == "/games"
        assert os.environ["UNRELATED_SETTING"] == "x"

    def test_configured_names_are_kept(self) -> None:
        environ = {"ONLY_RESET_LEVELS": "false", "TAAF_RUN_AS_SUBMISSION": "1", "PATH": "/bin"}
        assert harness_runtime.inherited_harness_env(environ, {"ONLY_RESET_LEVELS": "true"}) == [
            "TAAF_RUN_AS_SUBMISSION"
        ]
