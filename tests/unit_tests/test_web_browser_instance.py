# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import threading
from typing import Any

import pytest

from nemo_gym.web import browser_instance
from nemo_gym.web.browser_instance import (
    TRANSPORT,
    BrowserInstanceUnavailableError,
    RemoteWebBackend,
)
from nemo_gym.web.browser_instance_pool import LocalBrowserInstancePoolProvider
from nemo_gym.web.browser_session import BrowserSessionHandle, BrowserSessionSpec, create_browser_session_provider
from nemo_gym.web.models import WebAction, WebBenchmark, WebObservation, WebStepResult, WebTask, WebVerifierResult
from nemo_gym.web.visual_browser import VisualBrowserDriverConfig


CALLS: list[tuple[str, Any]] = []


class FakeBackend:
    """Records calls; ``task_id`` values steer failures."""

    def __init__(
        self, config: VisualBrowserDriverConfig, session_id: str, artifacts: Any, lease: BrowserSessionHandle
    ):
        CALLS.append(("init", (session_id, lease.transport, lease.owner_pid, type(config).__name__)))
        self.task: WebTask | None = None

    def reset(self, task: WebTask) -> tuple[WebObservation, dict[str, Any]]:
        if task.task_id == "precondition":
            raise ValueError("start url is not allowed")
        self.task = task
        CALLS.append(("reset", task.task_id))
        return WebObservation(url="https://example.com/", metadata={"step": 0}), {"proxy_enabled": False}

    def observe(self) -> WebObservation:
        return WebObservation(url="https://example.com/observed")

    def step(self, action: WebAction) -> WebStepResult:
        CALLS.append(("step", action.name))
        return WebStepResult(
            observation=WebObservation(url="https://example.com/after"),
            execution_ok=True,
            terminated=action.terminal,
            info={"runtime_status": "done" if action.terminal else "running"},
        )

    def evaluate(self, final_answer: str | None = None) -> WebVerifierResult:
        return WebVerifierResult(
            valid_sample=False,
            failure_kind="web:external_judge_required",
            metadata={"final_answer": final_answer or ""},
        )

    def close(self) -> None:
        CALLS.append(("close", self.task.task_id if self.task else None))
        if self.task is not None and self.task.task_id == "close-fails":
            raise RuntimeError("browser handles retained")


FACTORY = f"{__name__}:FakeBackend"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def instance():
    CALLS.clear()
    port = _free_port()
    result: dict[str, int] = {}
    thread = threading.Thread(
        target=lambda: result.setdefault("exit", browser_instance.serve(host="127.0.0.1", port=port, token="secret")),
        daemon=True,
    )
    thread.start()
    handle = BrowserSessionHandle(
        session_id="rollout-1",
        provider_name="test",
        transport=TRANSPORT,
        endpoint=f"http://127.0.0.1:{port}",
        metadata={"token": "secret", "request_timeout_seconds": 10},
    )
    yield handle, thread, result


def _backend(handle: BrowserSessionHandle) -> RemoteWebBackend:
    config = VisualBrowserDriverConfig(host="127.0.0.1", port=0, entrypoint="app.py", name="visual_browser")
    return RemoteWebBackend(handle=handle, config=config, session_id="rollout-1", backend_factory=FACTORY)


def _task(task_id: str) -> WebTask:
    return WebTask(benchmark=WebBenchmark.WEBVOYAGER, task_id=task_id, start_urls=["https://example.com/"])


def test_remote_backend_runs_the_factory_backend_inside_the_instance(instance) -> None:
    handle, _thread, _result = instance
    backend = _backend(handle)

    observation, info = backend.reset(_task("Amazon--1"))
    observed = backend.observe()
    step = backend.step(WebAction(name="computer", script="", terminal=False))
    verdict = backend.evaluate("42")
    backend.close()

    assert observation.url == "https://example.com/" and info == {"proxy_enabled": False}
    assert observed.url == "https://example.com/observed"
    assert step.info["runtime_status"] == "running" and step.observation.url == "https://example.com/after"
    assert verdict.failure_kind == "web:external_judge_required" and verdict.metadata["final_answer"] == "42"
    # The instance builds its backend from a local-process lease owned by itself.
    assert CALLS[0] == ("init", ("rollout-1", "local_process", os.getpid(), "VisualBrowserDriverConfig"))
    assert [name for name, _ in CALLS[1:]] == ["reset", "step", "close"]


def test_a_backend_value_error_stays_a_value_error(instance) -> None:
    handle, _thread, _result = instance
    with pytest.raises(ValueError, match="start url is not allowed"):
        _backend(handle).reset(_task("precondition"))
    # The failed reset released the backend: the instance accepts a new session.
    _backend(handle).reset(_task("Amazon--2"))


def test_wrong_token_and_dead_endpoint_are_infrastructure_failures(instance) -> None:
    handle, _thread, _result = instance
    handle.metadata["token"] = "wrong"
    with pytest.raises(BrowserInstanceUnavailableError, match="Unauthorized"):
        _backend(handle).reset(_task("Amazon--3"))
    dead = BrowserSessionHandle(transport=TRANSPORT, endpoint=f"http://127.0.0.1:{_free_port()}", metadata={})
    with pytest.raises(BrowserInstanceUnavailableError, match="unreachable"):
        _backend(dead).reset(_task("Amazon--3"))


def test_a_failed_close_retires_the_instance(instance) -> None:
    handle, thread, result = instance
    backend = _backend(handle)
    backend.reset(_task("close-fails"))
    with pytest.raises(RuntimeError, match="browser handles retained"):
        backend.close()
    thread.join(timeout=10)
    assert result["exit"] == 1


def test_calls_without_a_session_are_rejected(instance) -> None:
    handle, _thread, _result = instance
    with pytest.raises(BrowserInstanceUnavailableError, match="no active session"):
        _backend(handle).step(WebAction(name="computer", script=""))


def test_visual_config_allows_several_sessions_only_with_an_instance_provider() -> None:
    base = dict(host="127.0.0.1", port=0, entrypoint="app.py", name="visual_browser")
    with pytest.raises(ValueError, match="local_process allows max_sessions=1"):
        VisualBrowserDriverConfig(**base, max_sessions=4)
    config = VisualBrowserDriverConfig(
        **base, max_sessions=4, browser_session_provider={"local_instance_pool": {"size": 4, "port_base": 40000}}
    )
    assert config.max_sessions == 4


def test_visual_browser_factory_routes_instance_leases_to_the_remote_backend() -> None:
    from resources_servers.visual_browser.backend import visual_browser_backend_factory
    from resources_servers.visual_browser.config import VisualBrowserResourcesServerConfig

    config = VisualBrowserResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="app.py", name="vb")
    handle = BrowserSessionHandle(transport=TRANSPORT, endpoint="http://127.0.0.1:1", metadata={"token": "t"})
    backend = visual_browser_backend_factory(config, "rollout-1", None, handle)
    assert isinstance(backend, RemoteWebBackend)


@pytest.mark.asyncio
async def test_pool_lends_each_slot_once_and_restarts_unhealthy_slots(monkeypatch) -> None:
    provider = create_browser_session_provider({"local_instance_pool": {"size": 2, "port_base": 41000}})
    assert isinstance(provider, LocalBrowserInstancePoolProvider)
    launched: list[int] = []
    healthy = {0: True, 1: True}
    monkeypatch.setattr("shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(provider, "_launch", lambda slot: launched.append(slot.index))
    monkeypatch.setattr(provider, "_idle_and_healthy", lambda slot: healthy[slot.index])
    monkeypatch.setattr(provider, "_stop", lambda slot: None)
    await provider.start()

    first = await provider.acquire(BrowserSessionSpec(metadata={"rollout_session_id": "a"}))
    second = await provider.acquire(BrowserSessionSpec(metadata={"rollout_session_id": "b"}))
    assert {first.metadata["slot"], second.metadata["slot"]} == {0, 1}
    assert first.transport == TRANSPORT and first.endpoint.endswith(str(41000 + first.metadata["slot"]))

    healthy[first.metadata["slot"]] = False
    await provider.release(first)
    await provider.release(second)
    await provider.release(second)  # idempotent
    assert launched == [0, 1, first.metadata["slot"]]
    third = await provider.acquire(BrowserSessionSpec(metadata={"rollout_session_id": "c"}))
    fourth = await provider.acquire(BrowserSessionSpec(metadata={"rollout_session_id": "d"}))
    assert {third.metadata["slot"], fourth.metadata["slot"]} == {0, 1}
    await provider.close()
