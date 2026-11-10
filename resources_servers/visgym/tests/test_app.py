# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import anyio
import numpy as np
import pytest
from fastapi import HTTPException, Request
from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.visgym import app as visgym_app
from resources_servers.visgym.schemas import (
    VisGymAgentVerifyRequest,
    VisGymCloseRequest,
    VisGymNeMoGymResponse,
    VisGymResourcesServerConfig,
    VisGymSeedSessionRequest,
    VisGymStepRequest,
)


class StubVisGymEnv:
    def __init__(self, done_after: int = 2, raise_on_step: bool = False) -> None:
        self.done_after = done_after
        self.raise_on_step = raise_on_step
        self.step_count = 0
        self.closed = False
        self.actions: list[str] = []

    def get_prompt(self) -> str:
        return "You are navigating a visual maze."

    def reset(self, *, seed: int | None = None, init_state: dict[str, Any] | None = None):
        self.step_count = 0
        return np.zeros((4, 4, 3), dtype=np.uint8), {
            "seed": seed,
            "init_state": init_state,
            "env_feedback": None,
        }

    def step(self, action: str):
        if self.raise_on_step:
            raise ValueError("boom")
        self.actions.append(action)
        self.step_count += 1
        done = self.step_count >= self.done_after
        reward = 1.0 if done else 0.0
        return (
            np.full((4, 4, 3), 255 if done else 128, dtype=np.uint8),
            reward,
            done,
            False,
            {"env_feedback": "Action executed successfully.", "turn": self.step_count},
        )

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class DistanceStubVisGymEnv(StubVisGymEnv):
    def reset(self, *, seed: int | None = None, init_state: dict[str, Any] | None = None):
        observation, info = super().reset(seed=seed, init_state=init_state)
        info["distance"] = 3.0
        return observation, info

    def step(self, action: str):
        self.actions.append(action)
        self.step_count += 1
        return (
            np.full((4, 4, 3), 128, dtype=np.uint8),
            0.0,
            False,
            False,
            {"distance": 2.0, "env_feedback": "Action executed successfully."},
        )


class ClosingDistanceStubVisGymEnv(StubVisGymEnv):
    """Distance closes from 8 to 0 over four moves, then a stop action solves it."""

    def reset(self, *, seed: int | None = None, init_state: dict[str, Any] | None = None):
        observation, info = super().reset(seed=seed, init_state=init_state)
        self._distance = 8.0
        info["distance"] = self._distance
        return observation, info

    def step(self, action: str):
        self.actions.append(action)
        self.step_count += 1
        if action == "('stop', 'stop')":
            return (
                np.full((4, 4, 3), 255, dtype=np.uint8),
                1.0,
                True,
                False,
                {"distance": self._distance, "env_feedback": "Solved."},
            )
        self._distance = max(0.0, self._distance - 2.0)
        return (
            np.full((4, 4, 3), 128, dtype=np.uint8),
            0.0,
            False,
            False,
            {"distance": self._distance, "env_feedback": "Moved closer."},
        )


class RetreatingStubVisGymEnv(StubVisGymEnv):
    """Distance starts at 4 and grows to 10 on the first step -- the agent moved away."""

    def reset(self, *, seed: int | None = None, init_state: dict[str, Any] | None = None):
        observation, info = super().reset(seed=seed, init_state=init_state)
        info["distance"] = 4.0
        return observation, info

    def step(self, action: str):
        self.actions.append(action)
        self.step_count += 1
        return (
            np.full((4, 4, 3), 128, dtype=np.uint8),
            0.0,
            False,
            False,
            {"distance": 10.0, "env_feedback": "Moved away."},
        )


class StubGym:
    def __init__(self, env: StubVisGymEnv) -> None:
        self.env = env
        self.make_calls: list[tuple[str, dict[str, Any]]] = []

    def make(self, env_id: str, **kwargs: Any) -> StubVisGymEnv:
        self.make_calls.append((env_id, kwargs))
        return self.env


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "env_id": "maze_2d/easy",
        "env_kwargs": {"maze_width": 9, "maze_height": 9},
        "seed": 1234,
        "task_id": "stub_seed1234",
        "act_grammar_regex": r"^.+$",
        "horizon_cap": None,
        "task_metadata": {"category": "stub"},
        "responses_create_params": {
            "model": "policy_model",
            "input": [],
            "temperature": 0.7,
            "max_output_tokens": 1024,
            "tools": [],
        },
    }
    row.update(overrides)
    return row


def _server(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    **config_overrides: Any,
) -> visgym_app.VisGymResourcesServer:
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    config = VisGymResourcesServerConfig(
        name="visgym_test",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        task_jsonl_fpaths=[str(path)],
        **config_overrides,
    )
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="0.0.0.0", port=0),
        global_config_dict=OmegaConf.create({}),
    )
    return visgym_app.VisGymResourcesServer(config=config, server_client=server_client)


def _request() -> Request:
    return MagicMock(spec=Request)


def _verify_request(env_id: str) -> VisGymAgentVerifyRequest:
    response = VisGymNeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        env_id=env_id,
    )
    return VisGymAgentVerifyRequest(
        response=response,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )


def _assert_no_session_state(server: visgym_app.VisGymResourcesServer) -> None:
    assert server.env_id_to_failure == {}
    assert server.env_id_to_env == {}
    assert server.env_id_to_task_row == {}
    assert server.env_id_to_total_reward == {}
    assert server.env_id_to_turn_count == {}
    assert server.env_id_to_reward_state == {}
    assert server.env_id_to_task_success == {}
    assert server.env_id_to_operation == {}
    assert server.background_close_tasks == set()


@pytest.mark.asyncio
async def test_unknown_abort_close_tombstones_id_against_delayed_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nemo_gym.server_runtime import base_resources_server

    now = [100.0]
    monkeypatch.setattr(base_resources_server, "monotonic", lambda: now[0])
    monkeypatch.setattr(visgym_app, "monotonic", lambda: now[0])
    env = StubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()], cleanup_tombstone_ttl_seconds=10.0, session_close_retry_window_seconds=10.0)
    env_id = "reordered-cleanup-session"

    closed = await server.close(
        _request(),
        VisGymCloseRequest(env_id=env_id, discard_reward=True),
    )

    assert closed.success is True
    assert server.cleanup_tombstone_expiry[env_id] == 110.0
    with pytest.raises(HTTPException) as exc_info:
        await server.seed_session(
            _request(),
            VisGymSeedSessionRequest(env_id=env_id, task_idx=0),
        )
    assert exc_info.value.status_code == 409
    assert stub_gym.make_calls == []
    _assert_no_session_state(server)

    now[0] = 111.0
    seeded = await server.seed_session(
        _request(),
        VisGymSeedSessionRequest(env_id=env_id, task_idx=0),
    )
    assert seeded.env_id == env_id
    assert env_id not in server.cleanup_tombstone_expiry
    await server.close(_request(), VisGymCloseRequest(env_id=env_id, discard_reward=True))
    _assert_no_session_state(server)


@pytest.mark.parametrize(
    "field, value",
    [
        ("num_workers", 2),
        ("return_transitions", True),
        ("cleanup_tombstone_ttl_seconds", 0),
    ],
)
def test_stateful_resource_config_rejects_unsafe_process_or_output_modes(field: str, value: object) -> None:
    config = {
        "name": "visgym_test",
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        field: value,
    }
    with pytest.raises(ValueError):
        VisGymResourcesServerConfig.model_validate(config)


def test_headless_defaults_force_matplotlib_agg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "TkAgg")

    visgym_app._ensure_headless_defaults()

    assert visgym_app.os.environ["MPLBACKEND"] == "Agg"


def test_base_matplotlib_canvas_supports_print_to_buffer() -> None:
    from matplotlib.figure import Figure

    visgym_app._ensure_matplotlib_canvas_compatibility()
    figure = Figure(figsize=(2, 1), dpi=10)

    buffer, dimensions = figure.canvas.print_to_buffer()

    assert dimensions == (20, 10)
    assert len(buffer) == 20 * 10 * 4


def test_fetch_state_renderer_is_deterministic_and_headless() -> None:
    env = MagicMock()
    env.has_object = True
    env.goal = np.array([1.45, 0.85, 0.65])
    obs = {
        "observation": np.array([1.25, 0.7, 0.55, 1.35, 0.75, 0.42]),
        "achieved_goal": np.array([1.35, 0.75, 0.42]),
        "desired_goal": env.goal,
    }

    first = visgym_app._render_fetch_state(env, obs)
    second = visgym_app._render_fetch_state(env, obs)

    assert first.shape == (128, 128, 3)
    assert first.dtype == np.uint8
    assert first.flags.writeable
    assert np.array_equal(first, second)
    assert np.unique(first.reshape(-1, 3), axis=0).shape[0] >= 8


def test_fetch_pick_info_rewards_approach_and_goal_progress() -> None:
    env = MagicMock()
    env.unwrapped.data.site.return_value.xpos = np.array([1.0, 2.0, 3.0])
    info = visgym_app.VisGymResourcesServer._augment_info(
        env,
        {
            "achieved_goal": np.array([1.0, 2.0, 4.0]),
            "desired_goal": np.array([1.0, 5.0, 4.0]),
        },
        "fetch_pick_and_place/easy",
    )

    assert info["fetch_pick_distance"] == pytest.approx(4.0)


def test_sliding_block_info_includes_total_block_distance() -> None:
    env = MagicMock()
    env.unwrapped.blocks = {
        1: {"position": (1, 4)},
        2: {"position": (3, 2)},
    }
    env.unwrapped.target_blocks = {
        1: {"position": (2, 1)},
        2: {"position": (3, 5)},
    }

    info = visgym_app.VisGymResourcesServer._augment_info(
        env,
        {"env_feedback": None},
        "sliding_block/easy",
    )

    assert info["sliding_distance"] == 7.0


@pytest.mark.asyncio
async def test_seed_session_uses_task_idx_and_emits_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = StubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])

    response = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    assert response.env_id in server.env_id_to_env
    assert stub_gym.make_calls == [("maze_2d/easy", {"maze_width": 9, "maze_height": 9})]
    assert response.obs[0].content[0]["type"] == "input_text"
    assert response.obs[0].content[1]["type"] == "input_image"
    assert response.obs[0].env_info["env_id"] == "maze_2d/easy"


@pytest.mark.asyncio
async def test_seed_uses_client_generated_env_id_and_rejects_collision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = StubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    body = VisGymSeedSessionRequest(env_id="client-session-id", task_idx=0)

    response = await server.seed_session(_request(), body)

    assert response.env_id == "client-session-id"
    with pytest.raises(HTTPException) as exc_info:
        await server.seed_session(_request(), body)
    assert exc_info.value.status_code == 409
    assert len(stub_gym.make_calls) == 1


@pytest.mark.asyncio
async def test_fetch_construction_serializes_process_environment_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class ConcurrentGym:
        def __init__(self) -> None:
            self.active = 0
            self.overlap = False
            self.lock = threading.Lock()

        def make(self, _env_id: str, **_kwargs: Any) -> StubVisGymEnv:
            with self.lock:
                self.active += 1
                self.overlap = self.overlap or self.active > 1
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return StubVisGymEnv()

    stub_gym = ConcurrentGym()
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    monkeypatch.setattr(visgym_app, "_install_fetch_headless_renderer", lambda: None)
    server = _server(tmp_path, [_row(env_id="fetch_reach/easy")])

    await asyncio.gather(
        server.seed_session(_request(), VisGymSeedSessionRequest(env_id="fetch-a", task_idx=0)),
        server.seed_session(_request(), VisGymSeedSessionRequest(env_id="fetch-b", task_idx=0)),
    )

    assert stub_gym.overlap is False


@pytest.mark.asyncio
async def test_fetch_gl_environment_is_hidden_from_non_fetch_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BlockingFetchGym:
        def __init__(self) -> None:
            self.fetch_started = threading.Event()
            self.release_fetch = threading.Event()
            self.maze_started = threading.Event()
            self.maze_gl_environment: tuple[str | None, str | None] | None = None

        def make(self, env_id: str, **_kwargs: Any) -> StubVisGymEnv:
            if env_id.startswith("fetch_"):
                assert visgym_app.os.environ.get("MUJOCO_GL") == "glfw"
                assert visgym_app.os.environ.get("PYOPENGL_PLATFORM") is None
                self.fetch_started.set()
                assert self.release_fetch.wait(timeout=2), "test did not release Fetch construction"
            else:
                self.maze_gl_environment = (
                    visgym_app.os.environ.get("MUJOCO_GL"),
                    visgym_app.os.environ.get("PYOPENGL_PLATFORM"),
                )
                self.maze_started.set()
            return StubVisGymEnv()

    stub_gym = BlockingFetchGym()
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    monkeypatch.setattr(visgym_app, "_install_fetch_headless_renderer", lambda: None)
    server = _server(
        tmp_path,
        [
            _row(env_id="fetch_reach/easy"),
            _row(env_id="maze_2d/easy"),
        ],
    )

    fetch_seed = asyncio.create_task(
        server.seed_session(
            _request(),
            VisGymSeedSessionRequest(env_id="fetch-session", task_idx=0),
        )
    )
    assert await asyncio.to_thread(stub_gym.fetch_started.wait, 2)
    maze_seed = asyncio.create_task(
        server.seed_session(
            _request(),
            VisGymSeedSessionRequest(env_id="maze-session", task_idx=1),
        )
    )
    await asyncio.sleep(0)
    assert stub_gym.maze_started.is_set() is False

    stub_gym.release_fetch.set()
    await asyncio.wait_for(asyncio.gather(fetch_seed, maze_seed), timeout=2)

    assert stub_gym.maze_gl_environment == ("egl", "egl")


@pytest.mark.asyncio
async def test_seed_session_observation_failure_closes_without_publishing_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = StubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)

    def fail_observation(*args: Any, **kwargs: Any) -> None:
        raise ValueError("observation serialization failed")

    monkeypatch.setattr(visgym_app, "observation_to_user_message", fail_observation)
    server = _server(tmp_path, [_row()])

    with pytest.raises(HTTPException, match="observation serialization failed"):
        await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_partial_seed_close_failure_stays_reserved_for_abort_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailOnceCloseEnv(StubVisGymEnv):
        def __init__(self) -> None:
            super().__init__()
            self.close_attempts = 0

        def close(self) -> None:
            self.close_attempts += 1
            if self.close_attempts == 1:
                raise RuntimeError("transient close failure")
            super().close()

    env = FailOnceCloseEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)

    def fail_observation(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("observation serialization failed")

    monkeypatch.setattr(visgym_app, "observation_to_user_message", fail_observation)
    server = _server(tmp_path, [_row()])
    env_id = "partial-seed-close-retry"

    with pytest.raises(HTTPException, match="observation serialization failed"):
        await server.seed_session(
            _request(),
            VisGymSeedSessionRequest(env_id=env_id, task_idx=0),
        )

    assert env.closed is False
    assert server.env_id_to_env[env_id] is env
    assert server.env_id_to_operation[env_id].close_requested is True

    closed = await server.close(
        _request(),
        VisGymCloseRequest(env_id=env_id, discard_reward=True),
    )

    assert closed.success is True
    assert env.close_attempts == 2
    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_seed_session_cancellation_closes_without_publishing_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = StubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)

    def cancel_observation(*args: Any, **kwargs: Any) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(visgym_app, "observation_to_user_message", cancel_observation)
    server = _server(tmp_path, [_row()])

    with pytest.raises(asyncio.CancelledError):
        await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_seed_cancellation_waits_for_constructor_and_closes_returned_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BlockingGym:
        def __init__(self, env: StubVisGymEnv) -> None:
            self.env = env
            self.make_started = threading.Event()
            self.release_make = threading.Event()

        def make(self, _env_id: str, **_kwargs: Any) -> StubVisGymEnv:
            self.make_started.set()
            assert self.release_make.wait(timeout=2), "test did not release blocked constructor"
            return self.env

    env = StubVisGymEnv()
    stub_gym = BlockingGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    seed_task = asyncio.create_task(
        server.seed_session(
            _request(),
            VisGymSeedSessionRequest(env_id="cancelled-constructor", task_idx=0),
        )
    )
    assert await asyncio.to_thread(stub_gym.make_started.wait, 2)

    seed_task.cancel()
    await asyncio.sleep(0)
    assert seed_task.done() is False

    stub_gym.release_make.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(seed_task, timeout=2)

    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_abort_close_waits_for_failing_constructor_and_retires_empty_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BlockingFailingGym:
        def __init__(self) -> None:
            self.make_started = threading.Event()
            self.release_make = threading.Event()

        def make(self, _env_id: str, **_kwargs: Any) -> None:
            self.make_started.set()
            assert self.release_make.wait(timeout=2), "test did not release blocked constructor"
            raise ValueError("constructor failed")

    stub_gym = BlockingFailingGym()
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    env_id = "constructor-failure-session"
    seed_task = asyncio.create_task(
        server.seed_session(
            _request(),
            VisGymSeedSessionRequest(env_id=env_id, task_idx=0),
        )
    )
    assert await asyncio.to_thread(stub_gym.make_started.wait, 2)

    close_task = asyncio.create_task(
        server.close(
            _request(),
            VisGymCloseRequest(env_id=env_id, discard_reward=True),
        )
    )
    await asyncio.sleep(0)
    assert server.env_id_to_operation[env_id].close_requested is True
    assert close_task.done() is False

    stub_gym.release_make.set()
    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(seed_task, timeout=2)
    closed = await asyncio.wait_for(close_task, timeout=2)

    assert exc_info.value.status_code == 500
    assert closed.success is True
    assert env_id in server.cleanup_tombstone_expiry
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_abort_close_waits_for_blocked_reset_and_prevents_late_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BlockingResetEnv(StubVisGymEnv):
        def __init__(self) -> None:
            super().__init__()
            self.reset_started = threading.Event()
            self.release_reset = threading.Event()

        def reset(self, *, seed: int | None = None, init_state: dict[str, Any] | None = None):
            self.reset_started.set()
            assert self.release_reset.wait(timeout=2), "test did not release blocked reset"
            return super().reset(seed=seed, init_state=init_state)

    env = BlockingResetEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    env_id = "blocked-reset-session"
    seed_task = asyncio.create_task(
        server.seed_session(
            _request(),
            VisGymSeedSessionRequest(env_id=env_id, task_idx=0),
        )
    )
    assert await asyncio.to_thread(env.reset_started.wait, 2)

    close_task = asyncio.create_task(
        server.close(
            _request(),
            VisGymCloseRequest(env_id=env_id, discard_reward=True),
        )
    )
    await asyncio.sleep(0)
    assert server.env_id_to_operation[env_id].close_requested is True
    assert close_task.done() is False

    env.release_reset.set()
    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(seed_task, timeout=2)
    closed = await asyncio.wait_for(close_task, timeout=2)

    assert exc_info.value.status_code == 409
    assert closed.success is True
    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_abort_close_waits_for_blocked_step_without_native_overlap_or_ghost_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BlockingStepEnv(StubVisGymEnv):
        def __init__(self) -> None:
            super().__init__()
            self.step_started = threading.Event()
            self.release_step = threading.Event()
            self.in_step = False
            self.close_during_step = False

        def step(self, action: str):
            self.in_step = True
            self.step_started.set()
            try:
                assert self.release_step.wait(timeout=2), "test did not release blocked step"
                return super().step(action)
            finally:
                self.in_step = False

        def close(self) -> None:
            self.close_during_step = self.in_step
            super().close()

    env = BlockingStepEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(
        _request(),
        VisGymSeedSessionRequest(env_id="blocked-step-session", task_idx=0),
    )
    step_task = asyncio.create_task(
        server.step(
            _request(),
            VisGymStepRequest(env_id=seeded.env_id, action_string="('stop', 'stop')"),
        )
    )
    assert await asyncio.to_thread(env.step_started.wait, 2)

    close_task = asyncio.create_task(
        server.close(
            _request(),
            VisGymCloseRequest(env_id=seeded.env_id, discard_reward=True),
        )
    )
    await asyncio.sleep(0)
    assert server.env_id_to_operation[seeded.env_id].close_requested is True
    assert close_task.done() is False

    env.release_step.set()
    stepped = await asyncio.wait_for(step_task, timeout=2)
    closed = await asyncio.wait_for(close_task, timeout=2)

    assert stepped.reward == 0.0
    assert closed.success is True
    assert env.close_during_step is False
    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_step_cancellation_keeps_native_lock_until_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BlockingStepEnv(StubVisGymEnv):
        def __init__(self) -> None:
            super().__init__()
            self.step_started = threading.Event()
            self.release_step = threading.Event()
            self.in_step = False
            self.close_during_step = False

        def step(self, action: str):
            self.in_step = True
            self.step_started.set()
            try:
                assert self.release_step.wait(timeout=2), "test did not release blocked step"
                return super().step(action)
            finally:
                self.in_step = False

        def close(self) -> None:
            self.close_during_step = self.in_step
            super().close()

    env = BlockingStepEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(
        _request(),
        VisGymSeedSessionRequest(env_id="cancelled-step-session", task_idx=0),
    )
    step_task = asyncio.create_task(
        server.step(
            _request(),
            VisGymStepRequest(env_id=seeded.env_id, action_string="move"),
        )
    )
    assert await asyncio.to_thread(env.step_started.wait, 2)

    step_task.cancel()
    close_task = asyncio.create_task(
        server.close(
            _request(),
            VisGymCloseRequest(env_id=seeded.env_id, discard_reward=True),
        )
    )
    await asyncio.sleep(0)
    assert step_task.done() is False
    assert close_task.done() is False

    env.release_step.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(step_task, timeout=2)
    closed = await asyncio.wait_for(close_task, timeout=2)

    assert closed.success is True
    assert env.close_during_step is False
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_anyio_cancel_scope_keeps_native_lock_until_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BlockingStepEnv(StubVisGymEnv):
        def __init__(self) -> None:
            super().__init__()
            self.step_started = threading.Event()
            self.release_step = threading.Event()
            self.in_step = False
            self.close_during_step = False

        def step(self, action: str):
            self.in_step = True
            self.step_started.set()
            try:
                assert self.release_step.wait(timeout=2), "test did not release blocked step"
                return super().step(action)
            finally:
                self.in_step = False

        def close(self) -> None:
            self.close_during_step = self.in_step
            super().close()

    env = BlockingStepEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(
        _request(),
        VisGymSeedSessionRequest(env_id="anyio-cancelled-step-session", task_idx=0),
    )
    cancel_scope = anyio.CancelScope()

    async def step_under_cancel_scope() -> None:
        with cancel_scope:
            await server.step(
                _request(),
                VisGymStepRequest(env_id=seeded.env_id, action_string="move"),
            )

    step_task = asyncio.create_task(step_under_cancel_scope())
    assert await asyncio.to_thread(env.step_started.wait, 2)

    cancel_scope.cancel()
    close_task = asyncio.create_task(
        server.close(
            _request(),
            VisGymCloseRequest(env_id=seeded.env_id, discard_reward=True),
        )
    )
    await asyncio.sleep(0)
    assert step_task.done() is False
    assert close_task.done() is False

    env.release_step.set()
    await asyncio.wait_for(step_task, timeout=2)
    closed = await asyncio.wait_for(close_task, timeout=2)

    assert closed.success is True
    assert env.close_during_step is False
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_cancelled_abort_close_finishes_native_close_before_session_id_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BlockingCloseEnv(StubVisGymEnv):
        def __init__(self) -> None:
            super().__init__()
            self.close_started = threading.Event()
            self.release_close = threading.Event()
            self.close_finished = threading.Event()

        def close(self) -> None:
            self.close_started.set()
            assert self.release_close.wait(timeout=2), "test did not release blocked close"
            super().close()
            self.close_finished.set()

    env = BlockingCloseEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    env_id = "blocked-close-session"
    await server.seed_session(
        _request(),
        VisGymSeedSessionRequest(env_id=env_id, task_idx=0),
    )
    close_task = asyncio.create_task(
        server.close(
            _request(),
            VisGymCloseRequest(env_id=env_id, discard_reward=True),
        )
    )
    assert await asyncio.to_thread(env.close_started.wait, 2)

    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    assert env_id in server.env_id_to_operation
    assert env_id in server.env_id_to_env
    with pytest.raises(HTTPException) as exc_info:
        await server.seed_session(
            _request(),
            VisGymSeedSessionRequest(env_id=env_id, task_idx=0),
        )
    assert exc_info.value.status_code == 409

    env.release_close.set()
    assert await asyncio.to_thread(env.close_finished.wait, 2)

    async def background_cleanup_finished() -> None:
        while server.env_id_to_operation or server.background_close_tasks:
            await asyncio.sleep(0)

    await asyncio.wait_for(background_cleanup_finished(), timeout=2)
    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_verify_before_close_is_rejected_without_draining_reward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = StubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(
        _request(),
        VisGymSeedSessionRequest(env_id="verify-before-close", task_idx=0),
    )

    with pytest.raises(HTTPException) as exc_info:
        await server.verify(_request(), _verify_request(seeded.env_id))

    assert exc_info.value.status_code == 409
    assert server.env_id_to_total_reward[seeded.env_id] == 0.0
    assert server.env_id_to_task_success[seeded.env_id] is False
    assert env.closed is False
    await server.close(
        _request(),
        VisGymCloseRequest(env_id=seeded.env_id, discard_reward=True),
    )
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_native_close_failure_preserves_session_for_retry_and_verify(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailOnceCloseEnv(StubVisGymEnv):
        def __init__(self) -> None:
            super().__init__(done_after=1)
            self.close_attempts = 0

        def close(self) -> None:
            self.close_attempts += 1
            if self.close_attempts == 1:
                raise RuntimeError("transient close failure")
            super().close()

    env = FailOnceCloseEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(
        _request(),
        VisGymSeedSessionRequest(env_id="close-retry-session", task_idx=0),
    )
    await server.step(
        _request(),
        VisGymStepRequest(env_id=seeded.env_id, action_string="('stop', 'stop')"),
    )

    first_close = await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))

    assert first_close.success is False
    assert server.env_id_to_env[seeded.env_id] is env
    assert seeded.env_id in server.env_id_to_operation
    assert server.env_id_to_total_reward[seeded.env_id] == 1.0
    assert server.env_id_to_task_success[seeded.env_id] is True
    with pytest.raises(HTTPException) as exc_info:
        await server.verify(_request(), _verify_request(seeded.env_id))
    assert exc_info.value.status_code == 409

    second_close = await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(_request(), _verify_request(seeded.env_id))

    assert second_close.success is True
    assert verified.reward == 1.0
    assert verified.task_success is True
    assert env.close_attempts == 2
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_step_accumulates_reward_and_verify_drains(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = StubVisGymEnv(done_after=1)
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    stepped = await server.step(
        _request(),
        VisGymStepRequest(env_id=seeded.env_id, action_string="('stop', 'stop')"),
    )
    await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(_request(), _verify_request(seeded.env_id))

    assert env.actions == ["('stop', 'stop')"]
    assert stepped.done is True
    assert stepped.reward == 1.0
    assert verified.reward == 1.0
    assert verified.task_success is True
    assert verified.mask_sample is False
    assert verified.failure_kind is None
    assert verified.response.metadata["training_reward"] == "1.0"
    # NeMo-RL reads responses_create_params off the rollout result to rebuild
    # the initial prompt; dropping it fails postprocessing after the episode.
    assert verified.responses_create_params is not None
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_unknown_verify_is_not_successful(tmp_path: Path) -> None:
    server = _server(tmp_path, [_row()])

    verified = await server.verify(_request(), _verify_request("unknown-session"))

    assert verified.reward == 0.0
    assert verified.task_success is None
    assert verified.mask_sample is True
    assert verified.failure_kind == "session_lost"
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_positive_nonterminal_reward_is_not_task_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class PositiveNonterminalEnv(StubVisGymEnv):
        def step(self, action: str):
            self.actions.append(action)
            self.step_count += 1
            return np.zeros((4, 4, 3), dtype=np.uint8), 1.0, False, False, {}

    env = PositiveNonterminalEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row(horizon_cap=1)])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    stepped = await server.step(
        _request(),
        VisGymStepRequest(env_id=seeded.env_id, action_string="('move', 0)"),
    )
    await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(_request(), _verify_request(seeded.env_id))

    assert stepped.reward == 1.0
    assert stepped.horizon_terminated is True
    assert verified.task_success is False
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_unknown_registry_id_is_rejected_or_reward_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class DenseRewardEnv(StubVisGymEnv):
        def reset(self, *, seed: int | None = None, init_state: dict[str, Any] | None = None):
            observation, info = super().reset(seed=seed, init_state=init_state)
            info["distance"] = 2.0
            return observation, info

        def step(self, action: str):
            self.actions.append(action)
            self.step_count += 1
            return np.zeros((4, 4, 3), dtype=np.uint8), 0.6, False, False, {"distance": 1.5}

    env = DenseRewardEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    row = _row(
        env_id="InvertedPendulum-v5-vlm",
        task_metadata={
            "reward_shaping": {
                "type": "distance_delta",
                "info_key": "distance",
                "weight": 0.3,
            }
        },
    )

    strict_server = _server(tmp_path, [row])
    with pytest.raises(HTTPException, match="outside VisGym's audited") as exc_info:
        await strict_server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))
    assert exc_info.value.status_code == 400
    _assert_no_session_state(strict_server)

    diagnostic_server = _server(
        tmp_path,
        [row],
        require_audited_binary_success=False,
    )
    seeded = await diagnostic_server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))
    stepped = await diagnostic_server.step(
        _request(),
        VisGymStepRequest(env_id=seeded.env_id, action_string="0"),
    )
    await diagnostic_server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
    verified = await diagnostic_server.verify(_request(), _verify_request(seeded.env_id))
    assert stepped.reward == 0.6
    assert verified.reward == 0.6
    assert verified.task_success is None
    _assert_no_session_state(diagnostic_server)


@pytest.mark.asyncio
async def test_audited_env_rejects_nonbinary_terminal_reward_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class NonBinaryTerminalEnv(StubVisGymEnv):
        def step(self, action: str):
            self.actions.append(action)
            self.step_count += 1
            return np.zeros((4, 4, 3), dtype=np.uint8), 0.5, True, False, {}

    env = NonBinaryTerminalEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    stepped = await server.step(_request(), VisGymStepRequest(env_id=seeded.env_id, action_string="('stop', 'stop')"))
    assert stepped.done is True
    await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
    failed = await server.verify(_request(), _verify_request(seeded.env_id))
    assert failed.mask_sample is True
    assert failed.failure_kind == "visgym:simulator_failed"
    assert "success contract has drifted" in failed.failure_reason
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_distance_delta_reward_shaping_normalizes_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Shaping is a fraction of the episode's own starting distance.

    initial=3.0, current=2.0 -> progress = (3-2)/3 = 1/3. With weight=0.3 the
    shaped delta is 0.3 * 1/3 = 0.1 -- the same number the old raw-units
    "scale" design produced here by coincidence, but this design also caps
    the episode total at 1.0 regardless of the environment's distance units
    (see test_distance_delta_full_episode_reward_stays_within_unit_interval).
    """
    env = DistanceStubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    row = _row(
        task_metadata={
            "reward_shaping": {
                "type": "distance_delta",
                "info_key": "distance",
                "weight": 0.3,
            }
        }
    )
    server = _server(tmp_path, [row])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    stepped = await server.step(
        _request(),
        VisGymStepRequest(env_id=seeded.env_id, action_string="('move', 0)"),
    )
    await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(_request(), _verify_request(seeded.env_id))

    assert stepped.reward == pytest.approx(0.1)
    assert stepped.obs[0].env_info["raw_env_reward"] == 0.0
    assert stepped.obs[0].env_info["training_step_reward"] == pytest.approx(0.1)
    assert verified.reward == pytest.approx(0.1)
    assert verified.task_success is False
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_distance_delta_full_episode_reward_stays_within_unit_interval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A solved episode's total must be exactly 1.0, never the old design's 1.8.

    The old "raw_reward + scale * (previous - current)" design added shaping
    on top of the terminal reward with no shared ceiling, so an environment
    like maze_2d (scale=0.1, typical initial distance 8) could total 1.8. This
    normalizes progress to the episode's own starting distance and mixes it
    with the terminal reward as a convex combination, so a fully-closed,
    successfully-stopped episode telescopes to exactly
    (1 - weight) * 1.0 + weight * 1.0 == 1.0.
    """
    env = ClosingDistanceStubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    row = _row(
        task_metadata={
            "reward_shaping": {
                "type": "distance_delta",
                "info_key": "distance",
                "weight": 0.3,
            }
        }
    )
    server = _server(tmp_path, [row])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    rewards = []
    for action in ("('move', 0)", "('move', 0)", "('move', 0)", "('move', 0)", "('stop', 'stop')"):
        stepped = await server.step(_request(), VisGymStepRequest(env_id=seeded.env_id, action_string=action))
        rewards.append(stepped.reward)

    assert stepped.done is True
    assert sum(rewards) == pytest.approx(1.0)
    assert max(rewards) < 1.0, "no single step should pay the full terminal reward plus shaping on top"


@pytest.mark.asyncio
async def test_distance_delta_moving_away_from_goal_does_not_go_negative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unsolved episode that ends farther from the goal than it started must floor at 0.

    The old design summed scale * (initial - final) with no floor, so ending
    farther away than the start (final > initial) produced a negative
    episode total. Progress is clipped to [0, 1] at every step here, so it
    cannot happen.
    """
    env = RetreatingStubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    row = _row(
        horizon_cap=1,
        task_metadata={
            "reward_shaping": {
                "type": "distance_delta",
                "info_key": "distance",
                "weight": 0.3,
            }
        },
    )
    server = _server(tmp_path, [row])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    stepped = await server.step(
        _request(),
        VisGymStepRequest(env_id=seeded.env_id, action_string="('move', 0)"),
    )

    assert stepped.reward == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_shaping_weight_above_ceiling_is_clamped_and_warns_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    env = DistanceStubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    row = _row(
        task_metadata={
            "reward_shaping": {
                "type": "distance_delta",
                "info_key": "distance",
                "weight": 0.9,
            }
        }
    )
    server = _server(tmp_path, [row])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    with caplog.at_level(logging.WARNING):
        first = await server.step(
            _request(),
            VisGymStepRequest(env_id=seeded.env_id, action_string="('move', 0)"),
        )
        second = await server.step(
            _request(),
            VisGymStepRequest(env_id=seeded.env_id, action_string="('move', 0)"),
        )

    # initial=3.0, current=2.0 (the stub env's distance is fixed, not
    # cumulative) -> progress=1/3 on every step; weight clamped from 0.9 to
    # the 0.5 ceiling, so shaped_delta = 0.5 * 1/3 on the first step and 0.0
    # once progress stops changing.
    assert first.reward == pytest.approx(0.5 / 3)
    assert second.reward == pytest.approx(0.0)
    warned = [r for r in caplog.records if "clamped" in r.getMessage()]
    assert len(warned) == 1, "expected exactly one clamp warning across repeated steps"


def test_matchstick_info_exposes_binary_solution_distance() -> None:
    incorrect = visgym_app.VisGymResourcesServer._augment_info(None, {"is_correct": False}, "matchstick_equation/easy")
    correct = visgym_app.VisGymResourcesServer._augment_info(None, {"is_correct": True}, "matchstick_equation/easy")

    assert incorrect["matchstick_distance"] == 1.0
    assert correct["matchstick_distance"] == 0.0


@pytest.mark.asyncio
async def test_horizon_cap_terminates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = StubVisGymEnv(done_after=99)
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row(horizon_cap=1)])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    stepped = await server.step(
        _request(),
        VisGymStepRequest(env_id=seeded.env_id, action_string="('move', 0)"),
    )

    assert stepped.done is True
    assert stepped.horizon_terminated is True


@pytest.mark.asyncio
async def test_step_exception_masks_infrastructure_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = StubVisGymEnv(raise_on_step=True)
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    stepped = await server.step(
        _request(),
        VisGymStepRequest(env_id=seeded.env_id, action_string="bad"),
    )

    assert stepped.done is True
    assert stepped.reward == 0.0
    await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
    failed = await server.verify(_request(), _verify_request(seeded.env_id))
    assert failed.mask_sample is True
    assert failed.failure_kind == "visgym:simulator_failed"
    assert "boom" in failed.failure_reason
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_unknown_env_id_raises(tmp_path: Path) -> None:
    server = _server(tmp_path, [_row()])

    with pytest.raises(HTTPException):
        await server.step(_request(), VisGymStepRequest(env_id="missing", action_string="x"))


@pytest.mark.asyncio
async def test_close_closes_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = StubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    response = await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))

    assert response.success is True
    assert env.closed is True


@pytest.mark.asyncio
async def test_abort_close_discards_unverified_reward(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = StubVisGymEnv(done_after=1)
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))
    await server.step(
        _request(),
        VisGymStepRequest(env_id=seeded.env_id, action_string="('stop', 'stop')"),
    )

    response = await server.close(
        _request(),
        VisGymCloseRequest(env_id=seeded.env_id, discard_reward=True),
    )

    assert response.success is True
    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_relative_asset_paths_resolve_against_search_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A repo-relative sample_dir must reach the environment as an absolute path.

    Manifests carry asset directories relative to the Gym root so that one row
    works from a checkout, a code snapshot and a container mount alike. Baking
    an absolute path into the data instead is how those rows end up running on
    exactly one machine.
    """
    env = StubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    relative = "resources_servers/visgym/data/requested_env_assets/images"
    server = _server(tmp_path, [_row(env_kwargs={"sample_dir": relative})])

    await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    _, kwargs = stub_gym.make_calls[0]
    assert Path(kwargs["sample_dir"]).is_absolute()
    assert kwargs["sample_dir"].endswith(relative)


@pytest.mark.asyncio
async def test_shaping_with_unknown_info_key_warns_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Shaping that names a key the environment never reports must say so.

    It is otherwise indistinguishable from no shaping at all: every step pays
    the raw reward, the run merely learns slowly, and nothing in the logs
    points at the inert shaped term.
    """
    env = StubVisGymEnv()
    stub_gym = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub_gym)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub_gym)
    server = _server(
        tmp_path,
        [
            _row(
                task_metadata={
                    "reward_shaping": {
                        "type": "distance_delta",
                        "info_key": "not_a_real_key",
                        "weight": 0.5,
                    }
                }
            )
        ],
    )
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))

    with caplog.at_level(logging.WARNING):
        first = await server.step(_request(), VisGymStepRequest(env_id=seeded.env_id, action_string="a"))
        second = await server.step(_request(), VisGymStepRequest(env_id=seeded.env_id, action_string="b"))

    warned = [r for r in caplog.records if "reward_shaping is configured" in r.getMessage()]
    assert len(warned) == 1, "expected exactly one warning across repeated steps"
    assert "not_a_real_key" in warned[0].getMessage()
    # The raw environment reward still flows through untouched.
    assert (first.reward, second.reward) == (0.0, 1.0)


def test_mental_rotation_3d_render_and_close_are_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent render and close must not interleave.

    The environment drives matplotlib's global pyplot registry: it renders with
    plt.figure and its close() calls plt.close("all"). Letting a teardown run
    inside another session's render drops that session's axes and raises
    `KeyError: <Axes3D: >` -- intermittently, and only under concurrency, which
    is why it survived every single-threaded probe and only appeared twice in a
    16-node run.
    """
    import threading as _threading
    import types

    overlaps = []
    active = {"render": False}
    barrier_lock = _threading.Lock()

    class FakeEnv:
        image_size = (8, 8)

        def _render(self, rotation):
            with barrier_lock:
                if active["render"]:
                    overlaps.append("render-in-render")
                active["render"] = True
            time.sleep(0.01)
            with barrier_lock:
                active["render"] = False
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self):
            with barrier_lock:
                if active["render"]:
                    overlaps.append("close-during-render")

    module = types.ModuleType("gymnasium.envs.mental_rotation_3d_cube.mental_rotation_3d_cube")
    module.MentalRotation3DCubeEnv = FakeEnv
    monkeypatch.setattr(visgym_app.importlib, "import_module", lambda name: module)

    visgym_app._install_mental_rotation_3d_renderer_compatibility()

    env = FakeEnv()
    threads = [
        _threading.Thread(target=lambda: FakeEnv._render(env, None)),
        _threading.Thread(target=lambda: FakeEnv.close(env)),
        _threading.Thread(target=lambda: FakeEnv._render(env, None)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert overlaps == [], f"figure operations interleaved: {overlaps}"


def test_typed_session_http_conformance(monkeypatch, tmp_path) -> None:
    from nemo_gym.server_runtime import ResourcesSeedSessionRequest, check_resources_session_contract

    stub = StubGym(StubVisGymEnv())
    monkeypatch.setattr(visgym_app, "gym", stub)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub)
    server = _server(tmp_path, [_row()])
    session_id = "typed-conformance"
    check_resources_session_contract(
        server.setup_webserver(),
        ResourcesSeedSessionRequest.model_validate(
            {
                "resources_session_id": session_id,
                "episode_id": {"rollout_id": "contract-rollout"},
                "task_id": {"taskset": "smoke", "task_id": "task-1"},
                "task_data": VisGymSeedSessionRequest(task_idx=0).model_dump(mode="json"),
            }
        ),
        keeps_state=True,
        holds_state=server.has_session_state,
        verify_body=_verify_request(session_id).model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_verification_retry_is_bound_and_abort_invalidates(monkeypatch, tmp_path) -> None:
    stub = StubGym(StubVisGymEnv())
    monkeypatch.setattr(visgym_app, "gym", stub)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))
    await server.step(_request(), VisGymStepRequest(env_id=seeded.env_id, action_string="0"))
    await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
    body = _verify_request(seeded.env_id)
    original = body.model_dump(mode="json")
    first = await server.verify(_request(), body)
    assert body.model_dump(mode="json") == original
    assert (await server.verify(_request(), body)).model_dump() == first.model_dump()
    changed = body.model_copy(deep=True)
    changed.response.id = "different-response"
    refused = await server.verify(_request(), changed)
    assert refused.mask_sample is True
    assert refused.failure_kind == "visgym:verification_mismatch"
    # A mismatched retry must not consume the valid receipt.
    assert (await server.verify(_request(), body)).model_dump() == first.model_dump()
    await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id, discard_reward=True))
    lost = await server.verify(_request(), body)
    assert lost.mask_sample is True
    assert lost.failure_kind == "session_lost"
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_verification_retry_expires(monkeypatch, tmp_path) -> None:
    from nemo_gym.server_runtime import base_resources_server

    stub = StubGym(StubVisGymEnv())
    monkeypatch.setattr(visgym_app, "gym", stub)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub)
    server = _server(tmp_path, [_row()])
    now = [100.0]
    monkeypatch.setattr(base_resources_server, "monotonic", lambda: now[0])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))
    await server.step(_request(), VisGymStepRequest(env_id=seeded.env_id, action_string="0"))
    await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
    first = await server.verify(_request(), _verify_request(seeded.env_id))
    assert first.mask_sample is False
    now[0] += server.config.session_close_retry_window_seconds + 1
    expired = await server.verify(_request(), _verify_request(seeded.env_id))
    assert expired.mask_sample is True
    assert expired.failure_kind == "session_lost"


def test_legacy_routes_cannot_bypass_typed_ownership(monkeypatch, tmp_path) -> None:
    from fastapi.testclient import TestClient

    env = StubVisGymEnv()
    stub = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub)
    server = _server(tmp_path, [_row()])
    with TestClient(server.setup_webserver()) as client:
        response = client.post(
            "/seed_session",
            json={
                "resources_session_id": "owned-session",
                "episode_id": {"rollout_id": "owner"},
                "task_id": {"taskset": "smoke", "task_id": "task-1"},
                "task_data": VisGymSeedSessionRequest(task_idx=0).model_dump(mode="json"),
            },
        )
        assert response.status_code == 200, response.text
        owner_cookie = dict(client.cookies)
        client.cookies.clear()
        assert client.post("/step", json={"env_id": "owned-session", "action_string": "0"}).status_code == 409
        assert client.post("/close", json={"env_id": "owned-session", "discard_reward": True}).status_code == 409
        assert client.post("/verify", json=_verify_request("owned-session").model_dump(mode="json")).status_code == 409
        assert server.has_session_state("owned-session")
        client.cookies.update(owner_cookie)
        assert client.post("/close", json={"env_id": "owned-session", "discard_reward": True}).status_code == 200
        _assert_no_session_state(server)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [ValueError, RuntimeError, MemoryError, "nan", "inf"])
async def test_simulator_fault_masks_whole_episode_without_restepping(monkeypatch, tmp_path, fault) -> None:
    class FailingEnv(StubVisGymEnv):
        calls = 0

        def step(self, action):
            self.calls += 1
            if isinstance(fault, type):
                raise fault("provider failed")
            result = list(super().step(action))
            result[1] = float(fault)
            return tuple(result)

    env = FailingEnv()
    stub = StubGym(env)
    monkeypatch.setattr(visgym_app, "gym", stub)
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))
    for _ in range(2):
        result = await server.step(_request(), VisGymStepRequest(env_id=seeded.env_id, action_string="0"))
        assert result.done is True
        assert result.reward == 0.0
    assert env.calls == 1
    await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
    failed = await server.verify(_request(), _verify_request(seeded.env_id))
    assert failed.mask_sample is True
    assert failed.reward == 0.0
    assert failed.task_success is None
    assert failed.failure_kind == "visgym:simulator_failed"
    assert ("provider failed" if isinstance(fault, type) else "nonfinite") in failed.failure_reason
    _assert_no_session_state(server)
