# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock

import anyio
import pytest
from fastapi import HTTPException, Request
from omegaconf import OmegaConf


upstream_gym_v = pytest.importorskip(
    "gym_v",
    reason="upstream gym_v is installed only in the isolated resources-server venv",
)
if not hasattr(upstream_gym_v, "Observation") or not hasattr(upstream_gym_v, "make"):
    pytest.skip("the imported gym_v is not the upstream runtime", allow_module_level=True)
Observation = upstream_gym_v.Observation

from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.gym_v import app as gymv_app
from resources_servers.gym_v.schemas import (
    GymVAgentVerifyRequest,
    GymVCloseRequest,
    GymVNeMoGymResponse,
    GymVResourcesServerConfig,
    GymVSeedSessionRequest,
    GymVStepRequest,
    GymVTaskRow,
)


class StubGymVEnv:
    description = "Choose one action."

    def __init__(
        self,
        *,
        raise_on_step: bool = False,
        reward: float = 1.0,
        terminated: bool = True,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.raise_on_step = raise_on_step
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info or {}
        self.closed = False

    def reset(self, *, seed: int | None = None):
        return {"agent_0": Observation(image=None, text=f"seed={seed}")}, {"agent_0": {}}

    def step(self, actions: dict[str, str]):
        if self.raise_on_step:
            raise ValueError(f"invalid action: {actions['agent_0']}")
        return (
            {"agent_0": Observation(image=None, text="next")},
            {"agent_0": self.reward},
            {"__all__": self.terminated},
            {"__all__": self.truncated},
            {"agent_0": self.info},
        )

    def close(self) -> None:
        self.closed = True


def _row(**overrides: Any) -> GymVTaskRow:
    values: dict[str, Any] = {
        "env_id": "Games/FrozenLake-v0",
        "env_kwargs": {},
        "seed": 7,
        "horizon_cap": 2,
        "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input=[]),
    }
    values.update(overrides)
    return GymVTaskRow.model_validate(values)


def _server(
    monkeypatch: pytest.MonkeyPatch,
    env: StubGymVEnv,
    **config_overrides: Any,
) -> gymv_app.GymVResourcesServer:
    monkeypatch.setattr(gymv_app.gym_v, "make", lambda *_args, **_kwargs: env)
    config = GymVResourcesServerConfig(
        name="gym_v_test",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        task_jsonl_fpaths=[],
        **config_overrides,
    )
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="0.0.0.0", port=0),
        global_config_dict=OmegaConf.create({}),
    )
    return gymv_app.GymVResourcesServer(config=config, server_client=server_client)


def _request() -> Request:
    return MagicMock(spec=Request)


def _assert_no_session_state(server: gymv_app.GymVResourcesServer) -> None:
    assert server.env_id_to_failure == {}
    assert server.env_id_to_env == {}
    assert server.env_id_to_task_row == {}
    assert server.env_id_to_total_reward == {}
    assert server.env_id_to_task_success == {}
    assert server.env_id_to_turn_count == {}
    assert server.env_id_to_operation == {}
    assert server.background_close_tasks == set()


def _verify_request(env_id: str) -> GymVAgentVerifyRequest:
    response = GymVNeMoGymResponse(
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
    return GymVAgentVerifyRequest(
        response=response,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )


@pytest.mark.asyncio
async def test_unknown_abort_close_tombstones_id_against_delayed_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_gym.server_runtime import base_resources_server

    now = [100.0]
    monkeypatch.setattr(base_resources_server, "monotonic", lambda: now[0])
    monkeypatch.setattr(gymv_app, "monotonic", lambda: now[0])
    env = StubGymVEnv()
    server = _server(monkeypatch, env, cleanup_tombstone_ttl_seconds=10.0, session_close_retry_window_seconds=10.0)
    env_id = "reordered-cleanup-session"

    closed = await server.close(
        _request(),
        GymVCloseRequest(env_id=env_id, discard_reward=True),
    )

    assert closed.success is True
    assert server.cleanup_tombstone_expiry[env_id] == 110.0
    with pytest.raises(HTTPException) as exc_info:
        await server.seed_session(
            _request(),
            GymVSeedSessionRequest(env_id=env_id, task_row=_row()),
        )
    assert exc_info.value.status_code == 409
    assert env.closed is False
    _assert_no_session_state(server)

    # Tombstones are not permanent ownership claims. Once the configured
    # network-reordering window has elapsed, the otherwise-unused ID is valid.
    now[0] = 111.0
    seeded = await server.seed_session(
        _request(),
        GymVSeedSessionRequest(env_id=env_id, task_row=_row()),
    )
    assert seeded.env_id == env_id
    assert env_id not in server.cleanup_tombstone_expiry
    await server.close(_request(), GymVCloseRequest(env_id=env_id, discard_reward=True))
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_seed_uses_client_generated_env_id_and_rejects_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    env = StubGymVEnv()
    server = _server(monkeypatch, env)
    env_id = "client-session-id"
    body = GymVSeedSessionRequest(env_id=env_id, task_row=_row())

    seeded = await server.seed_session(_request(), body)

    assert seeded.env_id == env_id
    with pytest.raises(HTTPException) as exc_info:
        await server.seed_session(_request(), body)
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_seed_observation_failure_closes_without_publishing_state(monkeypatch: pytest.MonkeyPatch) -> None:
    env = StubGymVEnv()
    server = _server(monkeypatch, env)

    def fail_observation(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("observation serialization failed")

    monkeypatch.setattr(gymv_app, "observation_to_user_message", fail_observation)

    with pytest.raises(HTTPException, match="observation serialization failed"):
        await server.seed_session(_request(), GymVSeedSessionRequest(env_id="known-id", task_row=_row()))

    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_partial_seed_close_failure_stays_reserved_for_abort_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailOnceCloseEnv(StubGymVEnv):
        def __init__(self) -> None:
            super().__init__()
            self.close_attempts = 0

        def close(self) -> None:
            self.close_attempts += 1
            if self.close_attempts == 1:
                raise RuntimeError("transient close failure")
            super().close()

    env = FailOnceCloseEnv()
    server = _server(monkeypatch, env)
    env_id = "partial-seed-close-retry"

    def fail_observation(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("observation serialization failed")

    monkeypatch.setattr(gymv_app, "observation_to_user_message", fail_observation)

    with pytest.raises(HTTPException, match="observation serialization failed"):
        await server.seed_session(
            _request(),
            GymVSeedSessionRequest(env_id=env_id, task_row=_row()),
        )

    assert env.closed is False
    assert server.env_id_to_env[env_id] is env
    assert server.env_id_to_operation[env_id].close_requested is True

    closed = await server.close(
        _request(),
        GymVCloseRequest(env_id=env_id, discard_reward=True),
    )

    assert closed.success is True
    assert env.close_attempts == 2
    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_seed_cancellation_closes_without_publishing_state(monkeypatch: pytest.MonkeyPatch) -> None:
    env = StubGymVEnv()
    server = _server(monkeypatch, env)

    def cancel_observation(*_args: Any, **_kwargs: Any) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(gymv_app, "observation_to_user_message", cancel_observation)

    with pytest.raises(asyncio.CancelledError):
        await server.seed_session(_request(), GymVSeedSessionRequest(env_id="known-id", task_row=_row()))

    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_seed_cancellation_waits_for_constructor_and_closes_returned_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_started = threading.Event()
    release_make = threading.Event()
    env = StubGymVEnv()
    server = _server(monkeypatch, env)

    def blocked_make(*_args: Any, **_kwargs: Any) -> StubGymVEnv:
        make_started.set()
        assert release_make.wait(timeout=2), "test did not release blocked constructor"
        return env

    monkeypatch.setattr(gymv_app.gym_v, "make", blocked_make)
    seed_task = asyncio.create_task(
        server.seed_session(
            _request(),
            GymVSeedSessionRequest(env_id="cancelled-constructor", task_row=_row()),
        )
    )
    assert await asyncio.to_thread(make_started.wait, 2)

    seed_task.cancel()
    await asyncio.sleep(0)
    assert seed_task.done() is False

    release_make.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(seed_task, timeout=2)

    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_abort_close_waits_for_failing_constructor_and_retires_empty_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_started = threading.Event()
    release_make = threading.Event()
    env = StubGymVEnv()
    server = _server(monkeypatch, env)

    def fail_make(*_args: Any, **_kwargs: Any) -> None:
        make_started.set()
        assert release_make.wait(timeout=2), "test did not release blocked constructor"
        raise ValueError("constructor failed")

    monkeypatch.setattr(gymv_app.gym_v, "make", fail_make)
    env_id = "constructor-failure-session"
    seed_task = asyncio.create_task(
        server.seed_session(
            _request(),
            GymVSeedSessionRequest(env_id=env_id, task_row=_row()),
        )
    )
    assert await asyncio.to_thread(make_started.wait, 2)

    close_task = asyncio.create_task(
        server.close(
            _request(),
            GymVCloseRequest(env_id=env_id, discard_reward=True),
        )
    )
    await asyncio.sleep(0)
    assert server.env_id_to_operation[env_id].close_requested is True
    assert close_task.done() is False

    release_make.set()
    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(seed_task, timeout=2)
    closed = await asyncio.wait_for(close_task, timeout=2)

    assert exc_info.value.status_code == 500
    assert closed.success is True
    assert env.closed is False
    assert env_id in server.cleanup_tombstone_expiry
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_abort_close_waits_for_blocked_reset_and_prevents_late_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingResetEnv(StubGymVEnv):
        def __init__(self) -> None:
            super().__init__()
            self.reset_started = threading.Event()
            self.release_reset = threading.Event()

        def reset(self, *, seed: int | None = None):
            self.reset_started.set()
            assert self.release_reset.wait(timeout=2), "test did not release blocked reset"
            return super().reset(seed=seed)

    env = BlockingResetEnv()
    server = _server(monkeypatch, env)
    env_id = "blocked-reset-session"
    seed_task = asyncio.create_task(
        server.seed_session(
            _request(),
            GymVSeedSessionRequest(env_id=env_id, task_row=_row()),
        )
    )
    assert await asyncio.to_thread(env.reset_started.wait, 2)

    close_task = asyncio.create_task(
        server.close(
            _request(),
            GymVCloseRequest(env_id=env_id, discard_reward=True),
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
) -> None:
    class BlockingStepEnv(StubGymVEnv):
        def __init__(self) -> None:
            super().__init__()
            self.step_started = threading.Event()
            self.release_step = threading.Event()
            self.in_step = False
            self.close_during_step = False

        def step(self, actions: dict[str, str]):
            self.in_step = True
            self.step_started.set()
            try:
                assert self.release_step.wait(timeout=2), "test did not release blocked step"
                return super().step(actions)
            finally:
                self.in_step = False

        def close(self) -> None:
            self.close_during_step = self.in_step
            super().close()

    env = BlockingStepEnv()
    server = _server(monkeypatch, env)
    seeded = await server.seed_session(
        _request(),
        GymVSeedSessionRequest(env_id="blocked-step-session", task_row=_row()),
    )
    step_task = asyncio.create_task(
        server.step(
            _request(),
            GymVStepRequest(env_id=seeded.env_id, action_string="valid"),
        )
    )
    assert await asyncio.to_thread(env.step_started.wait, 2)

    close_task = asyncio.create_task(
        server.close(
            _request(),
            GymVCloseRequest(env_id=seeded.env_id, discard_reward=True),
        )
    )
    await asyncio.sleep(0)
    assert server.env_id_to_operation[seeded.env_id].close_requested is True
    assert close_task.done() is False

    env.release_step.set()
    stepped = await asyncio.wait_for(step_task, timeout=2)
    closed = await asyncio.wait_for(close_task, timeout=2)

    assert stepped.reward == 1.0
    assert closed.success is True
    assert env.close_during_step is False
    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_step_cancellation_keeps_native_lock_until_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingStepEnv(StubGymVEnv):
        def __init__(self) -> None:
            super().__init__()
            self.step_started = threading.Event()
            self.release_step = threading.Event()
            self.in_step = False
            self.close_during_step = False

        def step(self, actions: dict[str, str]):
            self.in_step = True
            self.step_started.set()
            try:
                assert self.release_step.wait(timeout=2), "test did not release blocked step"
                return super().step(actions)
            finally:
                self.in_step = False

        def close(self) -> None:
            self.close_during_step = self.in_step
            super().close()

    env = BlockingStepEnv()
    server = _server(monkeypatch, env)
    seeded = await server.seed_session(
        _request(),
        GymVSeedSessionRequest(env_id="cancelled-step-session", task_row=_row()),
    )
    step_task = asyncio.create_task(
        server.step(
            _request(),
            GymVStepRequest(env_id=seeded.env_id, action_string="[up]"),
        )
    )
    assert await asyncio.to_thread(env.step_started.wait, 2)

    step_task.cancel()
    close_task = asyncio.create_task(
        server.close(
            _request(),
            GymVCloseRequest(env_id=seeded.env_id, discard_reward=True),
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
) -> None:
    class BlockingStepEnv(StubGymVEnv):
        def __init__(self) -> None:
            super().__init__()
            self.step_started = threading.Event()
            self.release_step = threading.Event()
            self.in_step = False
            self.close_during_step = False

        def step(self, actions: dict[str, str]):
            self.in_step = True
            self.step_started.set()
            try:
                assert self.release_step.wait(timeout=2), "test did not release blocked step"
                return super().step(actions)
            finally:
                self.in_step = False

        def close(self) -> None:
            self.close_during_step = self.in_step
            super().close()

    env = BlockingStepEnv()
    server = _server(monkeypatch, env)
    seeded = await server.seed_session(
        _request(),
        GymVSeedSessionRequest(env_id="anyio-cancelled-step-session", task_row=_row()),
    )
    cancel_scope = anyio.CancelScope()

    async def step_under_cancel_scope() -> None:
        with cancel_scope:
            await server.step(
                _request(),
                GymVStepRequest(env_id=seeded.env_id, action_string="valid"),
            )

    step_task = asyncio.create_task(step_under_cancel_scope())
    assert await asyncio.to_thread(env.step_started.wait, 2)

    cancel_scope.cancel()
    close_task = asyncio.create_task(
        server.close(
            _request(),
            GymVCloseRequest(env_id=seeded.env_id, discard_reward=True),
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
) -> None:
    class BlockingCloseEnv(StubGymVEnv):
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
    server = _server(monkeypatch, env)
    env_id = "blocked-close-session"
    await server.seed_session(
        _request(),
        GymVSeedSessionRequest(env_id=env_id, task_row=_row()),
    )
    close_task = asyncio.create_task(
        server.close(
            _request(),
            GymVCloseRequest(env_id=env_id, discard_reward=True),
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
            GymVSeedSessionRequest(env_id=env_id, task_row=_row()),
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
async def test_verify_before_close_is_rejected_without_draining_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    env = StubGymVEnv()
    server = _server(monkeypatch, env)
    seeded = await server.seed_session(
        _request(),
        GymVSeedSessionRequest(env_id="verify-before-close", task_row=_row()),
    )

    with pytest.raises(HTTPException) as exc_info:
        await server.verify(_request(), _verify_request(seeded.env_id))

    assert exc_info.value.status_code == 409
    assert server.env_id_to_total_reward[seeded.env_id] == 0.0
    assert env.closed is False
    await server.close(
        _request(),
        GymVCloseRequest(env_id=seeded.env_id, discard_reward=True),
    )
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_native_close_failure_preserves_session_for_retry_and_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailOnceCloseEnv(StubGymVEnv):
        def __init__(self) -> None:
            super().__init__()
            self.close_attempts = 0

        def close(self) -> None:
            self.close_attempts += 1
            if self.close_attempts == 1:
                raise RuntimeError("transient close failure")
            super().close()

    env = FailOnceCloseEnv()
    server = _server(monkeypatch, env)
    seeded = await server.seed_session(
        _request(),
        GymVSeedSessionRequest(env_id="close-retry-session", task_row=_row()),
    )
    await server.step(
        _request(),
        GymVStepRequest(env_id=seeded.env_id, action_string="valid"),
    )

    first_close = await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))

    assert first_close.success is False
    assert server.env_id_to_env[seeded.env_id] is env
    assert seeded.env_id in server.env_id_to_operation
    assert server.env_id_to_total_reward[seeded.env_id] == 1.0
    with pytest.raises(HTTPException) as exc_info:
        await server.verify(_request(), _verify_request(seeded.env_id))
    assert exc_info.value.status_code == 409

    second_close = await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(_request(), _verify_request(seeded.env_id))

    assert second_close.success is True
    assert verified.reward == 1.0
    assert verified.task_success is None
    assert env.close_attempts == 2
    _assert_no_session_state(server)


@pytest.mark.parametrize(
    "criterion,reward,expected_success",
    [
        ({"type": "terminal_step_reward", "comparison": "equal", "threshold": 1.0}, 2 / 3, False),
        ({"type": "terminal_step_reward", "comparison": "equal", "threshold": 1.0}, 1.0, True),
        ({"type": "terminal_step_reward", "comparison": "greater_than", "threshold": 0.0}, 0.784, True),
    ],
)
@pytest.mark.asyncio
async def test_terminal_step_reward_success_is_distinct_from_training_reward(
    monkeypatch: pytest.MonkeyPatch,
    criterion: dict[str, Any],
    reward: float,
    expected_success: bool,
) -> None:
    env = StubGymVEnv(reward=reward, terminated=True)
    server = _server(monkeypatch, env)
    seeded = await server.seed_session(
        _request(),
        GymVSeedSessionRequest(task_row=_row(success_criterion=criterion)),
    )

    stepped = await server.step(
        _request(),
        GymVStepRequest(env_id=seeded.env_id, action_string="valid"),
    )
    await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(_request(), _verify_request(seeded.env_id))

    assert stepped.reward == reward
    assert verified.reward == reward
    assert verified.task_success is expected_success
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_required_success_contract_uses_allowlist_and_rejects_unknown_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowlist = {
        "Games/FrozenLake-v0": {
            "type": "terminal_step_reward",
            "comparison": "equal",
            "threshold": 1.0,
        }
    }
    server = _server(
        monkeypatch,
        StubGymVEnv(reward=2 / 3),
        require_success_criterion=True,
        success_criteria_by_env_id=allowlist,
    )
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_row=_row()))
    await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="valid"))
    await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(_request(), _verify_request(seeded.env_id))
    assert verified.reward == 2 / 3
    assert verified.task_success is False
    _assert_no_session_state(server)

    with pytest.raises(HTTPException, match="no audited binary success criterion") as exc_info:
        await server.seed_session(
            _request(),
            GymVSeedSessionRequest(task_row=_row(env_id="Geometry/ConvexHull-v0")),
        )
    assert exc_info.value.status_code == 400
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_server_horizon_does_not_turn_positive_nonterminal_reward_into_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = StubGymVEnv(reward=1.0, terminated=False, truncated=False)
    server = _server(monkeypatch, env)
    seeded = await server.seed_session(
        _request(),
        GymVSeedSessionRequest(
            task_row=_row(
                horizon_cap=1,
                success_criterion={
                    "type": "terminal_step_reward",
                    "comparison": "equal",
                    "threshold": 1.0,
                },
            )
        ),
    )

    stepped = await server.step(
        _request(),
        GymVStepRequest(env_id=seeded.env_id, action_string="valid"),
    )
    await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(_request(), _verify_request(seeded.env_id))

    assert stepped.done is True
    assert stepped.horizon_terminated is True
    assert verified.reward == 1.0
    assert verified.task_success is False
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_info_bool_success_criterion_requires_a_boolean_at_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    criterion = {"type": "info_bool", "path": ["correct"]}
    server = _server(monkeypatch, StubGymVEnv(reward=0.0, info={"correct": True}))
    seeded = await server.seed_session(
        _request(),
        GymVSeedSessionRequest(task_row=_row(success_criterion=criterion)),
    )
    await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="valid"))
    await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(_request(), _verify_request(seeded.env_id))
    assert verified.task_success is True
    _assert_no_session_state(server)

    missing_server = _server(monkeypatch, StubGymVEnv(reward=0.0))
    missing_seed = await missing_server.seed_session(
        _request(),
        GymVSeedSessionRequest(task_row=_row(success_criterion=criterion)),
    )
    stepped = await missing_server.step(_request(), GymVStepRequest(env_id=missing_seed.env_id, action_string="valid"))
    assert stepped.done is True
    await missing_server.close(_request(), GymVCloseRequest(env_id=missing_seed.env_id))
    failed = await missing_server.verify(_request(), _verify_request(missing_seed.env_id))
    assert failed.mask_sample is True
    assert failed.failure_kind == "gym_v:simulator_failed"
    assert "criterion path" in failed.failure_reason
    _assert_no_session_state(missing_server)


@pytest.mark.asyncio
async def test_invalid_actions_consume_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    # Providers return ordinary nonterminal zero-reward results for invalid actions.
    env = StubGymVEnv(reward=0.0, terminated=False)
    server = _server(monkeypatch, env)
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_row=_row(horizon_cap=2)))

    first = await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="bad"))
    second = await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="still bad"))

    assert first.done is False
    assert first.horizon_terminated is False
    assert second.done is True
    assert second.horizon_terminated is True
    assert server.env_id_to_turn_count[seeded.env_id] == 2


@pytest.mark.asyncio
async def test_abort_close_discards_unverified_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    env = StubGymVEnv()
    server = _server(monkeypatch, env)
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_row=_row()))
    stepped = await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="valid"))

    closed = await server.close(
        _request(),
        GymVCloseRequest(env_id=seeded.env_id, discard_reward=True),
    )

    assert stepped.reward == 1.0
    assert closed.success is True
    assert env.closed is True
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_missing_reward_session_is_masked_not_a_model_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(monkeypatch, StubGymVEnv())
    verified = await server.verify(_request(), _verify_request("missing-session"))
    assert verified.mask_sample is True
    assert verified.failure_kind == "session_lost"
    assert verified.task_success is None
    assert verified.response.env_id == "missing-session"


@pytest.mark.asyncio
async def test_real_unsuccessful_episode_remains_an_unmasked_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(monkeypatch, StubGymVEnv(reward=0.0))
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_row=_row()))
    await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="valid"))
    await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(_request(), _verify_request(seeded.env_id))
    assert verified.reward == 0.0
    assert verified.mask_sample is False
    assert verified.failure_kind is None
    duplicate = await server.verify(_request(), _verify_request(seeded.env_id))
    assert duplicate.model_dump() == verified.model_dump()


def test_typed_session_http_conformance(monkeypatch) -> None:
    from nemo_gym.server_runtime import ResourcesSeedSessionRequest, check_resources_session_contract

    server = _server(monkeypatch, StubGymVEnv())
    session_id = "typed-conformance"
    check_resources_session_contract(
        server.setup_webserver(),
        ResourcesSeedSessionRequest.model_validate(
            {
                "resources_session_id": session_id,
                "episode_id": {"rollout_id": "contract-rollout"},
                "task_id": {"taskset": "smoke", "task_id": "task-1"},
                "task_data": GymVSeedSessionRequest(task_row=_row()).model_dump(mode="json"),
            }
        ),
        keeps_state=True,
        holds_state=server.has_session_state,
        verify_body=_verify_request(session_id).model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_verification_retry_is_bound_and_abort_invalidates(monkeypatch) -> None:
    server = _server(monkeypatch, StubGymVEnv())
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_row=_row()))
    await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="0"))
    await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))
    body = _verify_request(seeded.env_id)
    original = body.model_dump(mode="json")
    first = await server.verify(_request(), body)
    assert body.model_dump(mode="json") == original
    assert (await server.verify(_request(), body)).model_dump() == first.model_dump()
    changed = body.model_copy(deep=True)
    changed.response.id = "different-response"
    refused = await server.verify(_request(), changed)
    assert refused.mask_sample is True
    assert refused.failure_kind == "gym_v:verification_mismatch"
    # A mismatched retry must not consume the valid receipt.
    assert (await server.verify(_request(), body)).model_dump() == first.model_dump()
    await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id, discard_reward=True))
    lost = await server.verify(_request(), body)
    assert lost.mask_sample is True
    assert lost.failure_kind == "session_lost"
    _assert_no_session_state(server)


@pytest.mark.asyncio
async def test_verification_retry_expires(monkeypatch) -> None:
    from nemo_gym.server_runtime import base_resources_server

    server = _server(monkeypatch, StubGymVEnv())
    now = [100.0]
    monkeypatch.setattr(base_resources_server, "monotonic", lambda: now[0])
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_row=_row()))
    await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="0"))
    await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))
    first = await server.verify(_request(), _verify_request(seeded.env_id))
    assert first.mask_sample is False
    now[0] += server.config.session_close_retry_window_seconds + 1
    expired = await server.verify(_request(), _verify_request(seeded.env_id))
    assert expired.mask_sample is True
    assert expired.failure_kind == "session_lost"


def test_legacy_routes_cannot_bypass_typed_ownership(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    env = StubGymVEnv()
    server = _server(monkeypatch, env)
    with TestClient(server.setup_webserver()) as client:
        response = client.post(
            "/seed_session",
            json={
                "resources_session_id": "owned-session",
                "episode_id": {"rollout_id": "owner"},
                "task_id": {"taskset": "smoke", "task_id": "task-1"},
                "task_data": GymVSeedSessionRequest(task_row=_row()).model_dump(mode="json"),
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
async def test_simulator_fault_masks_whole_episode_without_restepping(monkeypatch, fault) -> None:
    class FailingEnv(StubGymVEnv):
        calls = 0

        def step(self, action):
            self.calls += 1
            if isinstance(fault, type):
                raise fault("provider failed")
            result = list(super().step(action))
            result[1] = {"agent_0": float(fault)}
            return tuple(result)

    env = FailingEnv()
    server = _server(monkeypatch, env)
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_row=_row()))
    for _ in range(2):
        result = await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="0"))
        assert result.done is True
        assert result.reward == 0.0
    assert env.calls == 1
    await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))
    failed = await server.verify(_request(), _verify_request(seeded.env_id))
    assert failed.mask_sample is True
    assert failed.reward == 0.0
    assert failed.task_success is None
    assert failed.failure_kind == "gym_v:simulator_failed"
    assert ("provider failed" if isinstance(fault, type) else "nonfinite") in failed.failure_reason
    _assert_no_session_state(server)
