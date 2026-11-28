# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import inspect
import io
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request
from PIL import Image

from nemo_gym.server_utils import ServerClient
from resources_servers.webarena.actor_pool import BrowserActorPool
from resources_servers.webarena.app import WebArenaResourcesServer
from resources_servers.webarena.schemas import (
    WebArenaAgentVerifyRequest,
    WebArenaCloseRequest,
    WebArenaFinalizeRequest,
    WebArenaFinalizeResponse,
    WebArenaNeMoGymResponse,
    WebArenaResourcesServerConfig,
    WebArenaSeedSessionRequest,
    WebArenaStepRequest,
    WebArenaTaskRow,
)


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 16), color="blue").save(output, format="PNG")
    return output.getvalue()


class _RemoteMethod:
    def __init__(self, function: Any):
        self.function = function

    async def remote(self, *args: Any, **kwargs: Any) -> Any:
        result = self.function(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result


class _FakeActor:
    def __init__(self, task: WebArenaTaskRow):
        self.task = task
        self.reset = _RemoteMethod(self._reset)
        self.step = _RemoteMethod(self._step)
        self.evaluate = _RemoteMethod(self._evaluate)
        self.evaluation_calls = 0

    def _reset(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "task": task,
            "screenshot": _png(),
            "tab_context": {
                "current_tab_id": 0,
                "tab_count": 1,
                "tabs": [{"tab_id": 0, "title": "Test", "url": "http://example"}],
            },
        }

    def _step(self, call: dict[str, Any]) -> dict[str, Any]:
        return {
            "call_id": call["call_id"],
            "function_output": "terminate completed successfully",
            "done": True,
            "reward": 0.0,
            "termination_status": "success",
            "answer": "done",
            "evaluator_details": {},
        }

    def _evaluate(self, *, status: str | None, answer: str | None) -> dict[str, Any]:
        self.evaluation_calls += 1
        return {
            "reward": 1.0,
            "termination_status": status,
            "answer": answer,
            "evaluator_details": {"message": "string_match: score=1.0"},
        }


class _FakePool:
    def __init__(self, actor: _FakeActor):
        self.actor = actor
        self.released: list[str] = []
        self.retired: list[str] = []

    async def checkout(self, env_id: str) -> _FakeActor:
        return self.actor

    async def release(self, env_id: str) -> None:
        self.released.append(env_id)

    async def retire(self, env_id: str) -> None:
        self.retired.append(env_id)


def _task() -> WebArenaTaskRow:
    return WebArenaTaskRow(
        task_id="webarena-test",
        sites=["map"],
        start_urls=["http://example"],
        intent="Return done.",
        eval={
            "eval_types": ["string_match"],
            "reference_answers": {"exact_match": "done"},
        },
    )


def _server(
    task: WebArenaTaskRow,
    *,
    actor: _FakeActor | None = None,
    **config_overrides: Any,
) -> tuple[WebArenaResourcesServer, _FakePool]:
    config = WebArenaResourcesServerConfig(
        name="webarena",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        num_workers=1,
        **config_overrides,
    )
    server = WebArenaResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )
    pool = _FakePool(actor or _FakeActor(task))
    server._pool = pool
    return server, pool


def _request() -> Request:
    return MagicMock(spec=Request)


async def _wait_for_cleanup(server: WebArenaResourcesServer, env_id: str) -> None:
    cleanup_task = await server._terminal_operations.cleanup_task_for(env_id)
    assert cleanup_task is not None
    await cleanup_task


@pytest.mark.asyncio
async def test_seed_terminate_verify_before_close_retains_reward() -> None:
    task = _task()
    server, pool = _server(task)
    seeded = await server.seed_session(
        _request(),
        WebArenaSeedSessionRequest(request_id="seed-1", task=task),
    )
    stepped = await server.step(
        _request(),
        WebArenaStepRequest(
            env_id=seeded.env_id,
            action_request_id="step-1",
            action=[
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "terminate",
                    "arguments": '{"status":"success","answer":"done"}',
                }
            ],
        ),
    )
    response = WebArenaNeMoGymResponse(
        id="response-1",
        created_at=0,
        model="policy-model",
        object="response",
        output=stepped.obs,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        env_id=seeded.env_id,
        group_id=task.task_id,
        seed_obs=seeded.obs,
    )
    verify_request = WebArenaAgentVerifyRequest(
        verify_request_id="verify-1",
        responses_create_params={"input": [], "parallel_tool_calls": False},
        response=response,
    )
    first = await server.verify(_request(), verify_request)
    retried = await server.verify(_request(), verify_request)
    closed = await server.close(
        _request(),
        WebArenaCloseRequest(env_id=seeded.env_id, close_request_id="close-1"),
    )

    assert stepped.done is True
    assert closed.success is True
    assert pool.released == [seeded.env_id]
    assert first.reward == retried.reward == 1.0
    assert first.answer == "done"
    assert len(first.response.seed_obs) == 1
    assert server._pool.actor.evaluation_calls == 1
    assert seeded.env_id not in server._sessions


@pytest.mark.asyncio
async def test_verify_evaluates_live_browser_without_terminate() -> None:
    task = _task()
    server, pool = _server(task)
    seeded = await server.seed_session(
        _request(),
        WebArenaSeedSessionRequest(request_id="seed-max-steps", task=task),
    )
    response = WebArenaNeMoGymResponse(
        id="response-max-steps",
        created_at=0,
        model="policy-model",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        env_id=seeded.env_id,
        group_id=task.task_id,
        seed_obs=seeded.obs,
    )
    verified = await server.verify(
        _request(),
        WebArenaAgentVerifyRequest(
            verify_request_id="verify-max-steps",
            responses_create_params={"input": [], "parallel_tool_calls": False},
            response=response,
        ),
    )
    assert verified.reward == 1.0
    assert pool.actor.evaluation_calls == 1
    assert pool.released == []

    await server.close(
        _request(),
        WebArenaCloseRequest(env_id=seeded.env_id, close_request_id="close-max-steps"),
    )
    assert pool.released == [seeded.env_id]


def test_finalize_schema_is_compact() -> None:
    request = WebArenaFinalizeRequest(env_id="env-1", finalize_request_id="finalize-1")
    response = WebArenaFinalizeResponse(
        env_id="env-1",
        finalize_request_id="finalize-1",
        reward=1.0,
        evaluator_details={"score": 1.0},
        termination_status="success",
        answer="done",
    )

    assert set(request.model_dump()) == {"env_id", "finalize_request_id"}
    assert set(response.model_dump()) == {
        "env_id",
        "finalize_request_id",
        "reward",
        "evaluator_details",
        "termination_status",
        "answer",
    }


@pytest.mark.asyncio
async def test_finalize_same_id_is_cached_after_session_removal() -> None:
    task = _task()
    server, pool = _server(task)
    seeded = await server.seed_session(
        _request(),
        WebArenaSeedSessionRequest(request_id="seed-finalize", task=task),
    )
    finalize_request = WebArenaFinalizeRequest(
        env_id=seeded.env_id,
        finalize_request_id="finalize-stable",
    )

    first = await server.finalize(_request(), finalize_request)
    retried = await server.finalize(_request(), finalize_request)
    await _wait_for_cleanup(server, seeded.env_id)

    assert first == retried
    assert first.reward == 1.0
    assert pool.actor.evaluation_calls == 1
    assert pool.released == [seeded.env_id]
    assert pool.retired == []
    assert seeded.env_id not in server._sessions


@pytest.mark.asyncio
async def test_finalize_survives_caller_cancellation_and_retry_joins() -> None:
    task = _task()
    actor = _FakeActor(task)
    evaluation_started = asyncio.Event()
    finish_evaluation = asyncio.Event()

    async def evaluate(*, status: str | None, answer: str | None) -> dict[str, Any]:
        actor.evaluation_calls += 1
        evaluation_started.set()
        await finish_evaluation.wait()
        return {
            "reward": 1.0,
            "termination_status": status,
            "answer": answer,
            "evaluator_details": {},
        }

    actor.evaluate = _RemoteMethod(evaluate)
    server, pool = _server(task, actor=actor)
    seeded = await server.seed_session(
        _request(),
        WebArenaSeedSessionRequest(request_id="seed-cancel", task=task),
    )
    finalize_request = WebArenaFinalizeRequest(
        env_id=seeded.env_id,
        finalize_request_id="finalize-cancel",
    )
    caller = asyncio.create_task(server.finalize(_request(), finalize_request))
    await evaluation_started.wait()

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    retry = asyncio.create_task(server.finalize(_request(), finalize_request))
    await asyncio.sleep(0)
    assert not retry.done()

    finish_evaluation.set()
    finalized = await retry
    await _wait_for_cleanup(server, seeded.env_id)
    assert finalized.reward == 1.0
    assert actor.evaluation_calls == 1
    assert pool.released == [seeded.env_id]
    assert seeded.env_id not in server._sessions


@pytest.mark.asyncio
async def test_close_joins_active_finalize_without_conflict_or_leaked_lease() -> None:
    task = _task()
    actor = _FakeActor(task)
    evaluation_started = asyncio.Event()
    finish_evaluation = asyncio.Event()

    async def evaluate(*, status: str | None, answer: str | None) -> dict[str, Any]:
        actor.evaluation_calls += 1
        evaluation_started.set()
        await finish_evaluation.wait()
        return {
            "reward": 0.5,
            "termination_status": status,
            "answer": answer,
            "evaluator_details": {},
        }

    actor.evaluate = _RemoteMethod(evaluate)
    server, pool = _server(task, actor=actor)
    seeded = await server.seed_session(
        _request(),
        WebArenaSeedSessionRequest(request_id="seed-race", task=task),
    )
    finalizing = asyncio.create_task(
        server.finalize(
            _request(),
            WebArenaFinalizeRequest(
                env_id=seeded.env_id,
                finalize_request_id="finalize-race",
            ),
        )
    )
    await evaluation_started.wait()
    closing = asyncio.create_task(
        server.close(
            _request(),
            WebArenaCloseRequest(env_id=seeded.env_id, close_request_id="close-race"),
        )
    )
    await asyncio.sleep(0)
    assert not closing.done()

    finish_evaluation.set()
    finalized, closed = await asyncio.gather(finalizing, closing)
    assert finalized.reward == 0.5
    assert closed.success is True
    assert pool.released == [seeded.env_id]
    assert pool.retired == []
    assert seeded.env_id not in server._sessions


@pytest.mark.asyncio
async def test_finalize_returns_reward_before_cleanup_completes() -> None:
    task = _task()
    server, pool = _server(task)
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()

    async def release(env_id: str) -> None:
        cleanup_started.set()
        await finish_cleanup.wait()
        pool.released.append(env_id)

    pool.release = release
    seeded = await server.seed_session(
        _request(),
        WebArenaSeedSessionRequest(request_id="seed-background-cleanup", task=task),
    )

    finalized = await server.finalize(
        _request(),
        WebArenaFinalizeRequest(
            env_id=seeded.env_id,
            finalize_request_id="finalize-background-cleanup",
        ),
    )
    await cleanup_started.wait()

    assert finalized.reward == 1.0
    assert pool.released == []
    assert seeded.env_id in server._sessions

    closing_registry = asyncio.create_task(server._terminal_operations.close())
    await asyncio.sleep(0)
    assert not closing_registry.done()

    finish_cleanup.set()
    await closing_registry
    await _wait_for_cleanup(server, seeded.env_id)
    assert pool.released == [seeded.env_id]
    assert seeded.env_id not in server._sessions


@pytest.mark.asyncio
async def test_finalize_timeout_retires_uncertain_actor_once() -> None:
    task = _task()
    actor = _FakeActor(task)

    async def evaluate(*, status: str | None, answer: str | None) -> dict[str, Any]:
        actor.evaluation_calls += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    actor.evaluate = _RemoteMethod(evaluate)
    server, pool = _server(task, actor=actor, browser_evaluation_timeout_seconds=0.01)
    seeded = await server.seed_session(
        _request(),
        WebArenaSeedSessionRequest(request_id="seed-timeout", task=task),
    )

    with pytest.raises(HTTPException) as exc_info:
        await server.finalize(
            _request(),
            WebArenaFinalizeRequest(
                env_id=seeded.env_id,
                finalize_request_id="finalize-timeout",
            ),
        )

    assert "finalization failed" in exc_info.value.detail
    assert actor.evaluation_calls == 1
    assert pool.released == []
    assert pool.retired == [seeded.env_id]
    assert seeded.env_id not in server._sessions


@pytest.mark.asyncio
async def test_pool_release_cancellation_removes_uncertain_lease_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_started = asyncio.Event()
    shutdown_calls = 0
    killed: list[Any] = []
    monkeypatch.setattr(
        "resources_servers.webarena.actor_pool.ray.kill",
        lambda actor, *, no_restart: killed.append(actor),
    )

    class PoolActor:
        def __init__(self) -> None:
            self.health = _RemoteMethod(lambda: {"healthy": True})
            self.close_episode = _RemoteMethod(self._close_episode)
            self.shutdown = _RemoteMethod(self._shutdown)

        async def _close_episode(self) -> bool:
            cleanup_started.set()
            await asyncio.Event().wait()
            return True

        def _shutdown(self) -> None:
            nonlocal shutdown_calls
            shutdown_calls += 1

    config = WebArenaResourcesServerConfig(
        name="webarena",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        num_workers=1,
        initial_browser_workers=0,
        max_browser_workers=1,
        browser_startup_stagger_seconds=0,
    )
    pool = BrowserActorPool(config, worker_factory=PoolActor)
    actor = await pool.checkout("env-uncertain")
    releasing = asyncio.create_task(pool.release("env-uncertain"))
    await cleanup_started.wait()

    releasing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await releasing
    await pool.retire("env-uncertain")

    assert pool.leased_count == 0
    assert pool.size == 0
    assert pool._idle.empty()
    assert shutdown_calls == 0
    assert killed == [actor]


@pytest.mark.asyncio
async def test_pool_does_not_reuse_actor_before_cleanup_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    monkeypatch.setattr(
        "resources_servers.webarena.actor_pool.ray.kill",
        lambda actor, *, no_restart: None,
    )

    class PoolActor:
        def __init__(self) -> None:
            self.health = _RemoteMethod(lambda: {"healthy": True})
            self.close_episode = _RemoteMethod(self._close_episode)
            self.shutdown = _RemoteMethod(lambda: None)

        async def _close_episode(self) -> bool:
            cleanup_started.set()
            await finish_cleanup.wait()
            return True

    actor = PoolActor()
    config = WebArenaResourcesServerConfig(
        name="webarena",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        num_workers=1,
        initial_browser_workers=0,
        max_browser_workers=1,
        browser_startup_stagger_seconds=0,
    )
    pool = BrowserActorPool(config, worker_factory=lambda: actor)
    assert await pool.checkout("env-first") is actor

    releasing = asyncio.create_task(pool.release("env-first"))
    await cleanup_started.wait()
    waiting_checkout = asyncio.create_task(pool.checkout("env-second"))
    await asyncio.sleep(0)
    assert not waiting_checkout.done()

    finish_cleanup.set()
    await releasing
    assert await waiting_checkout is actor
