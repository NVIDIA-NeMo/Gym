# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import HTTPException, Request
from omegaconf import OmegaConf

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
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
    verified = await server.verify(_request(), _verify_request(seeded.env_id))

    assert env.actions == ["('stop', 'stop')"]
    assert stepped.done is True
    assert stepped.reward == 1.0
    assert verified.reward == 1.0
    assert verified.response.metadata["training_reward"] == "1.0"
    # NeMo-RL reads responses_create_params off the rollout result to rebuild
    # the initial prompt; dropping it fails postprocessing after the episode.
    assert verified.responses_create_params is not None


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
    verified = await server.verify(_request(), _verify_request(seeded.env_id))

    assert stepped.reward == pytest.approx(0.1)
    assert stepped.obs[0].env_info["raw_env_reward"] == 0.0
    assert stepped.obs[0].env_info["training_step_reward"] == pytest.approx(0.1)
    assert verified.reward == pytest.approx(0.1)


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
async def test_step_exception_returns_recovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    assert stepped.done is False
    assert stepped.reward == 0.0
    assert "Invalid action" in stepped.obs[0].content[0]["text"]


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
