# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from resources_servers.visgym import app as visgym_app
from resources_servers.visgym.schemas import VisGymCloseRequest, VisGymSeedSessionRequest, VisGymStepRequest
from resources_servers.visgym.tests.conftest import requires_visgym
from resources_servers.visgym.tests.test_app import StubGym, StubVisGymEnv, _request, _row, _server, _verify_request


@requires_visgym
@pytest.mark.asyncio
async def test_wrapped_maze_uses_requested_initial_state(tmp_path: Path) -> None:
    import gymnasium as gym

    reference = gym.make("maze_2d/easy", maze_width=5, maze_height=5)
    try:
        reference.reset(seed=7)
        base = reference.unwrapped
        initial = {
            "maze_map": base.maze_map.tolist(),
            "agent_location": base._agent_location.tolist(),
            "target_location": base._target_location.tolist(),
        }
    finally:
        reference.close()
    server = _server(tmp_path, [_row(seed=1, env_kwargs={"maze_width": 5, "maze_height": 5}, init_state=initial)])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))
    try:
        base = server.env_id_to_env[seeded.env_id].unwrapped
        np.testing.assert_array_equal(base.maze_map, initial["maze_map"])
        np.testing.assert_array_equal(base._agent_location, initial["agent_location"])
        stepped = await server.step(_request(), VisGymStepRequest(env_id=seeded.env_id, action_string="('move', 0)"))
        assert stepped.obs
        assert seeded.env_id not in server.env_id_to_failure
    finally:
        await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id, discard_reward=True))


@requires_visgym
@pytest.mark.asyncio
@pytest.mark.parametrize("grammar,masked", [(r"\('move', \(\d+, [0-3]\)\)", False), (None, True)])
async def test_sliding_block_invalid_direction_remains_a_scored_model_failure(tmp_path: Path, grammar, masked) -> None:
    server = _server(
        tmp_path,
        [_row(env_id="sliding_block/easy", env_kwargs={}, act_grammar_regex=grammar)],
        reward_mode="task_success",
    )
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))
    try:
        result = await server.step(
            _request(), VisGymStepRequest(env_id=seeded.env_id, action_string="('move', (1, 'up'))")
        )
        assert result.done and result.reward == 0
        await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
        grade = await server.verify(_request(), _verify_request(seeded.env_id))
        assert grade.mask_sample is masked
        assert grade.reward == 0
        assert grade.task_success is (None if masked else False)
    finally:
        await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id, discard_reward=True))


def test_failed_typed_construction_can_retry_same_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FailOnceGym(StubGym):
        attempts = 0

        def make(self, env_id, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("transient construction failure")
            return super().make(env_id, **kwargs)

    stub = FailOnceGym(StubVisGymEnv())
    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: stub)
    server = _server(tmp_path, [_row()])
    body = {
        "resources_session_id": "retry-construction",
        "episode_id": {"rollout_id": "retry"},
        "task_id": {"taskset": "smoke", "task_id": "task"},
        "task_data": VisGymSeedSessionRequest(task_idx=0).model_dump(mode="json"),
    }
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=body).status_code == 500
        retry = client.post("/seed_session", json=body)
        assert retry.status_code == 200, retry.text
        assert stub.attempts == 2
        assert client.post("/close", json={"env_id": "retry-construction", "discard_reward": True}).status_code == 200


@pytest.mark.asyncio
async def test_mixed_pyplot_families_serialize_reset_and_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    active = 0
    overlaps = []
    counter = threading.Lock()

    class FigureEnv(StubVisGymEnv):
        def draw(self):
            nonlocal active
            with counter:
                active += 1
                if active > 1:
                    overlaps.append(active)
            time.sleep(0.03)
            with counter:
                active -= 1

        def reset(self, **kwargs):
            self.draw()
            return super().reset(**kwargs)

        def close(self):
            self.draw()
            super().close()

    class FigureGym:
        def make(self, *args, **kwargs):
            return FigureEnv()

    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: FigureGym())
    monkeypatch.setattr(visgym_app, "_install_mental_rotation_3d_renderer_compatibility", lambda: None)
    server = _server(tmp_path, [_row(env_id="matchstick_rotation/easy"), _row(env_id="mental_rotation_3d_cube/easy")])

    async def episode(index):
        seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=index))
        await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id, discard_reward=True))

    await asyncio.gather(*(episode(i % 2) for i in range(8)))
    assert overlaps == [], f"Global pyplot operations overlapped: {overlaps}"


@requires_visgym
@pytest.mark.asyncio
async def test_real_mixed_pyplot_frames_match_serial_reference(tmp_path: Path) -> None:
    rows = [
        _row(env_id="matchstick_rotation/easy", env_kwargs={}, seed_key="seed"),
        _row(env_id="mental_rotation_3d_cube/easy", env_kwargs={}, seed_key="seed"),
    ]
    server = _server(tmp_path, rows)

    async def episode(index: int) -> list[str]:
        seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=index))
        try:
            step = await server.step(
                _request(), VisGymStepRequest(env_id=seeded.env_id, action_string="('stop', 'stop')")
            )
            assert seeded.env_id not in server.env_id_to_failure
            assert step.done
            return [
                part["image_url"]
                for message in [*seeded.obs, *step.obs]
                for part in message.model_dump(mode="json")["content"]
                if part["type"] == "input_image"
            ]
        finally:
            await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id, discard_reward=True))

    reference = [await episode(index) for index in range(2)]
    assert all(reference)
    frames = await asyncio.gather(*(episode(index % 2) for index in range(8)))
    for index, images in enumerate(frames):
        assert images == reference[index % 2], f"Rendered frames changed under mixed concurrency: episode {index}"


@pytest.mark.asyncio
async def test_runtime_failure_is_masked_even_for_out_of_grammar_action(monkeypatch, tmp_path: Path) -> None:
    class BrokenEnv(StubVisGymEnv):
        def step(self, action: str):
            raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(visgym_app, "_ensure_visgym_importable", lambda: StubGym(BrokenEnv()))
    server = _server(tmp_path, [_row(act_grammar_regex="allowed")])
    seeded = await server.seed_session(_request(), VisGymSeedSessionRequest(task_idx=0))
    await server.step(_request(), VisGymStepRequest(env_id=seeded.env_id, action_string="bad-action"))
    await server.close(_request(), VisGymCloseRequest(env_id=seeded.env_id))
    grade = await server.verify(_request(), _verify_request(seeded.env_id))
    assert grade.mask_sample
    assert grade.failure_kind == "visgym:simulator_failed"


def test_task_schema_reuses_fields_without_leaking_dispatcher_metadata() -> None:
    from resources_servers.visgym.schemas import VisGymTaskRow
    from resources_servers.visgym.task_data import TaskData

    data = _row(task_idx=7, _ng_rollout_id="private-rollout", agent_ref="private-agent")
    task = TaskData.model_validate(data)
    session = VisGymTaskRow.model_validate(data)
    expected = task.model_dump(include=set(VisGymTaskRow.model_fields) - {"responses_create_params"})
    assert session.model_dump(exclude={"responses_create_params"}) == expected
    assert not {"task_idx", "_ng_rollout_id", "agent_ref"} & session.model_dump().keys()
